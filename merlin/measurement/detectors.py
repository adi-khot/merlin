from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import cast

import perceval as pcvl
import torch

from ..utils.combinadics import Combinadics


class DetectorTransform(torch.nn.Module):
    """
    Linear map applying per-mode detector rules to a Fock probability vector.

    Args:
        simulation_keys: Iterable describing the raw Fock states produced by the
            simulator (as tuples or lists of integers).
        detectors: One detector per optical mode. Each detector must expose the
            :meth:`detect` method from :class:`perceval.Detector`.
        dtype: Optional torch dtype for the transform matrix. Defaults to
            ``torch.float32``.
        device: Optional device used to stage the transform matrix.
        partial_measurement: When ``True``, only the modes whose detector entry is
            not ``None`` are measured. The transform then operates on complex
            amplitudes and returns per-outcome dictionaries (see :meth:`forward`).
    """

    def __init__(
        self,
        simulation_keys: Iterable[Sequence[int]] | torch.Tensor,
        detectors: Sequence[pcvl.Detector | None],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        partial_measurement: bool = False,
    ) -> None:
        super().__init__()

        if simulation_keys is None:
            raise ValueError("simulation_keys must contain at least one Fock state.")

        self._dtype = dtype or torch.float32
        device_obj = torch.device(device) if device is not None else None

        self._simulation_keys = self._normalize_keys(simulation_keys)
        if not self._simulation_keys:
            raise ValueError("simulation_keys must contain at least one Fock state.")
        self._n_modes = len(self._simulation_keys[0])
        self._total_photons = sum(self._simulation_keys[0])

        if any(len(key) != self._n_modes for key in self._simulation_keys):
            raise ValueError("All simulation keys must have the same number of modes.")

        if len(detectors) != self._n_modes:
            raise ValueError(
                f"Expected {self._n_modes} detectors, received {len(detectors)}."
            )

        self._partial_measurement = bool(partial_measurement)

        normalized_detectors: list[pcvl.Detector | None] = []
        for detector in detectors:
            if detector is None:
                if not self._partial_measurement:
                    raise ValueError(
                        "DetectorTransform with partial_measurement=False requires a detector for every mode."
                    )
                normalized_detectors.append(None)
                continue
            normalized_detectors.append(detector)

        if not self._partial_measurement and not all(
            isinstance(det, pcvl.Detector) for det in normalized_detectors
        ):
            raise ValueError(
                "All detectors must be provided when partial_measurement=False."
            )

        self._detectors: tuple[pcvl.Detector | None, ...] = tuple(normalized_detectors)
        self._response_cache: dict[
            tuple[int, int], list[tuple[tuple[int, ...], float]]
        ] = {}

        if self._partial_measurement:
            (
                self._measured_modes,
                self._unmeasured_modes,
                self._remaining_offsets,
                self._remaining_combinadics,
                self._remaining_dim,
                self._remaining_dims,
            ) = self._build_partial_metadata()
            self._remaining_keys_cache: list[tuple[int, ...]] | None = None
            self._detector_keys = []
            self._is_identity = False
            placeholder = torch.empty((0, 0), dtype=self._dtype)
            self.register_buffer("_matrix", placeholder, persistent=False)
        else:
            matrix, detector_keys, is_identity = self._build_transform(device_obj)
            self._detector_keys = detector_keys
            self._is_identity = is_identity
            if not is_identity:
                self.register_buffer("_matrix", matrix)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_keys(
        keys: Iterable[Sequence[int]] | torch.Tensor,
    ) -> list[tuple[int, ...]]:
        """
        Convert raw simulator keys into a canonical tuple-based representation.
        """
        if isinstance(keys, torch.Tensor):
            if keys.ndim != 2:
                raise ValueError("simulation_keys tensor must have shape (N, M).")
            return [tuple(int(v) for v in row.tolist()) for row in keys]

        normalized: list[tuple[int, ...]] = []
        for key in keys:
            normalized.append(tuple(int(v) for v in key))
        return normalized

    def _build_partial_metadata(
        self,
    ) -> tuple[
        tuple[int, ...],
        tuple[int, ...],
        dict[int, int],
        dict[int, Combinadics],
        int,
        dict[int, int],
    ]:
        """
        Prepare bookkeeping for partial measurement mode.
        """
        measured_modes = tuple(
            idx for idx, detector in enumerate(self._detectors) if detector is not None
        )
        unmeasured_modes = tuple(
            idx for idx, detector in enumerate(self._detectors) if detector is None
        )

        if not unmeasured_modes:
            offsets = {0: 0}
            combinadics_map: dict[int, Combinadics] = {}
            dims_map = {0: 1}
            return (
                measured_modes,
                unmeasured_modes,
                offsets,
                combinadics_map,
                1,
                dims_map,
            )

        offsets: dict[int, int] = {}
        combinadics_map: dict[int, Combinadics] = {}
        dims_map: dict[int, int] = {}
        remaining_dim = 0
        counts_present: set[int] = set()
        for sim_key in self._simulation_keys:
            counts_present.add(sum(sim_key[idx] for idx in unmeasured_modes))

        for remaining_n in sorted(counts_present):
            combinator = Combinadics("fock", remaining_n, len(unmeasured_modes))
            size = combinator.compute_space_size()
            if size == 0:
                continue
            offsets[remaining_n] = remaining_dim
            combinadics_map[remaining_n] = combinator
            dims_map[remaining_n] = size
            remaining_dim += size

        if remaining_dim == 0:
            offsets[0] = 0
            dims_map[0] = 1
            remaining_dim = 1

        return (
            measured_modes,
            unmeasured_modes,
            offsets,
            combinadics_map,
            remaining_dim,
            dims_map,
        )

    def _remaining_position(
        self, remaining_key: tuple[int, ...]
    ) -> tuple[int, int, int, int]:
        remaining_n = sum(remaining_key)
        offset = self._remaining_offsets.get(remaining_n)
        if offset is None:
            raise KeyError(
                f"Unsupported remaining photon count {remaining_n} for key {remaining_key}."
            )

        if not self._unmeasured_modes:
            return offset, remaining_n, 0, 1

        combinator = self._remaining_combinadics.get(remaining_n)
        if combinator is None:
            raise KeyError(
                f"No combinator configured for remaining photon count {remaining_n}."
            )
        local_index = combinator.fock_to_index(remaining_key)
        local_dim = self._remaining_dims.get(remaining_n)
        if local_dim is None:
            raise KeyError(
                f"No local dimension configured for remaining photon count {remaining_n}."
            )
        return offset + local_index, remaining_n, local_index, local_dim

    def _full_measurement_key(
        self, per_mode_counts: tuple[int, ...]
    ) -> tuple[int | None, ...]:
        full_key: list[int | None] = [None] * self._n_modes
        for idx, mode in enumerate(self._measured_modes):
            value = per_mode_counts[idx] if idx < len(per_mode_counts) else None
            full_key[mode] = value
        return tuple(full_key)

    def _detector_response(
        self, mode: int, photon_count: int
    ) -> list[tuple[tuple[int, ...], float]]:
        """
        Return the detection distribution for a single mode and photon count.

        Results are cached because detector configurations rarely change within a
        layer's lifetime.
        """
        cache_key = (mode, photon_count)
        if cache_key in self._response_cache:
            return self._response_cache[cache_key]

        detector = self._detectors[mode]
        if detector is None:
            raise RuntimeError(
                f"No detector configured for mode {mode} in partial measurement path."
            )
        raw = detector.detect(photon_count)

        responses: list[tuple[tuple[int, ...], float]] = []

        if isinstance(raw, pcvl.BasicState):
            responses = [(tuple(int(v) for v in raw), 1.0)]
        else:
            bs_distribution_type = getattr(pcvl, "BSDistribution", None)
            if bs_distribution_type is not None and isinstance(
                raw, bs_distribution_type
            ):
                iterator: Iterable = raw.items()
            elif isinstance(raw, dict):
                iterator = raw.items()
            else:
                iterator = getattr(raw, "items", None)
                if callable(iterator):
                    iterator = iterator()
                else:
                    raise TypeError(
                        f"Unsupported detector response type: {type(raw)!r}"
                    )

            responses = [
                (tuple(int(v) for v in state), float(prob)) for state, prob in iterator
            ]

        if not responses:
            raise ValueError(
                f"Detector {detector!r} returned an empty distribution for {photon_count} photon(s)."
            )

        self._response_cache[cache_key] = responses
        return responses

    def _build_transform(
        self, device: torch.device | None
    ) -> tuple[torch.Tensor | None, list[tuple[int, ...]], bool]:
        """
        Construct the detection transform matrix and associated classical keys.
        """
        detector_key_to_index: dict[tuple[int, ...], int] = {}
        detector_keys: list[tuple[int, ...]] = []
        row_entries: list[dict[int, float]] = []

        for sim_key in self._simulation_keys:
            per_mode = [
                self._detector_response(mode, count)
                for mode, count in enumerate(sim_key)
            ]

            combined: dict[int, float] = {}

            for outcomes in itertools.product(*per_mode):
                outcome_values: list[int] = []
                probability = 1.0
                for partial_state, partial_prob in outcomes:
                    outcome_values.extend(partial_state)
                    probability *= partial_prob

                if probability == 0.0:
                    continue

                outcome_tuple = tuple(outcome_values)
                column_index = detector_key_to_index.get(outcome_tuple)
                if column_index is None:
                    column_index = len(detector_keys)
                    detector_key_to_index[outcome_tuple] = column_index
                    detector_keys.append(outcome_tuple)

                combined[column_index] = combined.get(column_index, 0.0) + probability

            row_entries.append(combined)

        is_identity = self._check_identity(detector_keys, row_entries)

        if is_identity:
            return (
                None,
                [tuple(int(v) for v in key) for key in self._simulation_keys],
                True,
            )

        rows = len(self._simulation_keys)
        cols = len(detector_keys)

        row_indices: list[int] = []
        col_indices: list[int] = []
        values: list[float] = []

        for row_idx, entries in enumerate(row_entries):
            for col_idx, prob in entries.items():
                row_indices.append(row_idx)
                col_indices.append(col_idx)
                values.append(prob)

        if not values:
            raise RuntimeError(
                "Detector transform construction produced an empty matrix; check detector responses."
            )

        if device is not None:
            indices = torch.tensor(
                [row_indices, col_indices],
                dtype=torch.long,
                device=device,
            )
            value_tensor = torch.tensor(values, dtype=self._dtype, device=device)
        else:
            indices = torch.tensor([row_indices, col_indices], dtype=torch.long)
            value_tensor = torch.tensor(values, dtype=self._dtype)
        matrix = torch.sparse_coo_tensor(
            indices,
            value_tensor,
            size=(rows, cols),
        ).coalesce()

        return matrix, detector_keys, False

    def _check_identity(
        self,
        detector_keys: list[tuple[int, ...]],
        row_entries: list[dict[int, float]],
    ) -> bool:
        """
        Determine if the detectors correspond to ideal PNR detection.
        """
        if len(detector_keys) != len(self._simulation_keys):
            return False

        for row_idx, (sim_key, entries) in enumerate(
            zip(self._simulation_keys, row_entries, strict=True)
        ):
            if len(entries) != 1:
                return False
            ((col_idx, prob),) = entries.items()
            if col_idx != row_idx:
                return False
            if not math.isclose(prob, 1.0, rel_tol=1e-12, abs_tol=1e-12):
                return False
            if detector_keys[col_idx] != sim_key:
                return False
        return True

    def _forward_partial(
        self, amplitudes: torch.Tensor
    ) -> list[dict[tuple[int, ...], list[tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Apply partial detector measurement to a complex amplitude tensor.
        """
        if amplitudes.dim() == 0:
            raise ValueError("Amplitude tensor must have at least one dimension.")
        if amplitudes.shape[-1] != len(self._simulation_keys):
            raise ValueError(
                "Amplitude tensor does not match the detector transform basis. "
                f"Expected last dimension {len(self._simulation_keys)}, "
                f"received {amplitudes.shape[-1]}."
            )
        if not torch.is_complex(amplitudes):
            raise TypeError(
                "Partial measurement expects a complex-valued amplitude tensor."
            )

        batch_shape = amplitudes.shape[:-1]
        flattened = amplitudes.reshape(-1, amplitudes.shape[-1])
        batch_size = flattened.shape[0]
        branch_amplitudes: dict[
            tuple[
                tuple[int, ...],
                int,
                tuple[int, ...],
                tuple[int | None, ...],
            ],
            torch.Tensor,
        ] = {}
        branch_probabilities: dict[
            tuple[
                tuple[int, ...],
                int,
                tuple[int, ...],
                tuple[int | None, ...],
            ],
            torch.Tensor,
        ] = {}

        for state_index, sim_key in enumerate(self._simulation_keys):
            amplitude_column = flattened[:, state_index]
            actual_counts = tuple(sim_key[mode] for mode in self._measured_modes)
            remaining_key = tuple(sim_key[idx] for idx in self._unmeasured_modes)
            (
                _,
                remaining_n,
                local_index,
                local_dim,
            ) = self._remaining_position(remaining_key)

            per_mode = [
                self._detector_response(mode, sim_key[mode])
                for mode in self._measured_modes
            ]

            for outcomes in itertools.product(*per_mode):
                measurement_values: list[int] = []
                per_mode_measured: list[int] = []
                probability = 1.0
                for partial_state, partial_prob in outcomes:
                    measurement_values.extend(partial_state)
                    per_mode_measured.append(
                        int(partial_state[0]) if partial_state else 0
                    )
                    probability *= partial_prob

                if probability == 0.0:
                    continue

                measurement_key = tuple(measurement_values)
                full_key = self._full_measurement_key(tuple(per_mode_measured))
                branch_key = (measurement_key, remaining_n, actual_counts, full_key)
                bucket = branch_amplitudes.get(branch_key)
                probability_bucket = branch_probabilities.get(branch_key)
                if bucket is None:
                    bucket = torch.zeros(
                        (batch_size, local_dim),
                        dtype=flattened.dtype,
                        device=flattened.device,
                    )
                    branch_amplitudes[branch_key] = bucket
                    probability_bucket = torch.zeros(
                        batch_size,
                        dtype=flattened.real.dtype,
                        device=flattened.device,
                    )
                    branch_probabilities[branch_key] = probability_bucket
                else:
                    assert probability_bucket is not None

                probability_bucket += amplitude_column.abs().pow(2) * probability

                if local_dim > 0:
                    weight = math.sqrt(probability)
                    bucket[:, local_index] += amplitude_column * weight

        formatted: dict[
            tuple[tuple[int | None, ...], int],
            list[tuple[torch.Tensor, torch.Tensor]],
        ] = {}
        for branch_key, tensor in branch_amplitudes.items():
            _, remaining_n, _, full_key = branch_key
            combined_key = (full_key, remaining_n)
            local_dim = tensor.shape[-1]
            amplitudes_shape = batch_shape + (local_dim,)
            amplitudes_view = tensor.reshape(amplitudes_shape)
            probabilities = branch_probabilities[branch_key]
            if batch_shape:
                probabilities = probabilities.reshape(batch_shape)
            else:
                probabilities = probabilities.reshape(())

            if local_dim > 0:
                normalization = probabilities.sqrt()
                if amplitudes_view.ndim == normalization.ndim + 1:
                    normalization = normalization.unsqueeze(-1)
                normalization = normalization.to(amplitudes_view.dtype)
                zero_mask = normalization == 0
                safe_norm = torch.where(
                    zero_mask,
                    torch.ones_like(normalization),
                    normalization,
                )
                amplitudes_view = torch.where(
                    zero_mask,
                    amplitudes_view,
                    amplitudes_view / safe_norm,
                )

            formatted.setdefault(combined_key, []).append((
                probabilities,
                amplitudes_view,
            ))

        max_remaining = (
            max((key[1] for key in formatted.keys()), default=0) if formatted else 0
        )
        organized: list[
            dict[tuple[int, ...], list[tuple[torch.Tensor, torch.Tensor]]]
        ] = [{} for _ in range(max_remaining + 1)]
        for (measurement_key, remaining_n), entries in formatted.items():
            organized[remaining_n][measurement_key] = entries
        return organized

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def output_keys(self) -> list[tuple[int, ...]]:
        """Return the classical detection outcome keys."""
        if self._partial_measurement:
            if self._remaining_keys_cache is None:
                if not self._unmeasured_modes:
                    self._remaining_keys_cache = [()]
                else:
                    keys: list[tuple[int, ...]] = []
                    for remaining_n in sorted(self._remaining_offsets):
                        combinator = self._remaining_combinadics.get(remaining_n)
                        if combinator is None:
                            continue
                        keys.extend(combinator.enumerate_states())
                    self._remaining_keys_cache = keys
            return list(self._remaining_keys_cache)
        return self._detector_keys

    @property
    def output_size(self) -> int:
        """Number of classical outcomes produced by the detectors."""
        if self._partial_measurement:
            return self._remaining_dim
        return len(self._detector_keys)

    @property
    def is_identity(self) -> bool:
        """Whether the transform reduces to the identity (ideal PNR detectors)."""
        return self._is_identity

    def forward(
        self, tensor: torch.Tensor
    ) -> (
        torch.Tensor
        | list[dict[tuple[int, ...], list[tuple[torch.Tensor, torch.Tensor]]]]
    ):
        """
        Apply the detector transform.

        Args:
            tensor: Probability distribution (complete mode) or amplitudes
                (partial measurement). The last dimension must match the simulator
                basis.

        Returns:
            - Complete mode: real probability tensor expressed in the detector basis.
            - Partial mode: list indexed by remaining photon count. Each entry is a
              dictionary whose keys are full-length mode tuples (unmeasured modes set
              to ``None``) and whose values are lists of
              (probability, normalized remaining-mode amplitudes) pairs – one per
              perfect measurement branch.
        """
        if self._partial_measurement:
            return self._forward_partial(tensor)

        if torch.is_complex(tensor) or not torch.is_floating_point(tensor):
            raise TypeError(
                "Complete detector measurement expects a real-valued probability tensor."
            )

        if self._is_identity:
            return tensor

        matrix: torch.Tensor = cast(torch.Tensor, self._matrix)  # type: ignore[attr-defined]
        if tensor.dtype != matrix.dtype:
            raise TypeError(
                "Detector transform dtype mismatch: "
                f"distribution={tensor.dtype}, transform={matrix.dtype}"
            )
        if tensor.device != matrix.device:
            raise RuntimeError(
                "Detector transform device mismatch: "
                f"distribution={tensor.device}, transform={matrix.device}"
            )

        original_shape = tensor.shape
        last_dim = original_shape[-1]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.reshape(-1, last_dim)

        transformed = torch.sparse.mm(tensor, matrix)

        if len(original_shape) == 1:
            return transformed.squeeze(0)
        return transformed.reshape(*original_shape[:-1], transformed.shape[-1])

    def row(
        self,
        index: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Return a single detector transform row as a dense tensor.
        """
        if self._partial_measurement:
            raise RuntimeError("row() is not available when partial_measurement=True.")
        if index < 0 or index >= len(self._simulation_keys):
            raise IndexError(f"Row index {index} out of bounds.")

        matrix = cast(torch.Tensor, self._matrix)  # type: ignore[attr-defined]
        matrix_device = matrix.device
        matrix_dtype = matrix.dtype

        target_device = torch.device(device) if device is not None else matrix_device
        target_dtype = dtype or matrix_dtype

        output_dim = len(self._detector_keys)

        if self._is_identity:
            row = torch.zeros(output_dim, dtype=target_dtype, device=target_device)
            row[index] = 1.0
            return row

        indices = matrix.indices()
        values = matrix.values()
        mask = indices[0] == index

        row = torch.zeros(output_dim, dtype=target_dtype, device=target_device)
        if mask.any():
            cols = indices[1, mask]
            row_vals = values[mask]
            if row_vals.dtype != target_dtype or row_vals.device != target_device:
                row_vals = row_vals.to(dtype=target_dtype, device=target_device)
            row[cols] = row_vals
        return row

    @property
    def partial_measurement(self) -> bool:
        """Return True when the transform runs in partial measurement mode."""
        return self._partial_measurement

    def remaining_basis(self, remaining_n: int | None = None) -> list[tuple[int, ...]]:
        """
        Return the ordered Fock-state basis for the unmeasured modes.

        Args:
            remaining_n: Optional photon count used to select a specific block.
                When omitted, the method returns the concatenation of every
                remaining-mode basis enumerated during detector initialisation.

        Returns:
            List of tuples describing the photon distribution over the unmeasured modes.
        """
        if not self._partial_measurement:
            raise RuntimeError(
                "remaining_basis() is only available when partial_measurement=True."
            )

        if remaining_n is None:
            if self._remaining_keys_cache is None:
                _ = self.output_keys  # triggers cache population
            return list(self._remaining_keys_cache or [])

        combinator = self._remaining_combinadics.get(remaining_n)
        if combinator is None:
            return []
        return list(combinator.enumerate_states())


def resolve_detectors(
    experiment: pcvl.Experiment, n_modes: int
) -> tuple[list[pcvl.Detector], bool]:
    """
    Build a per-mode detector list from a Perceval experiment.

    Args:
        experiment: Perceval experiment carrying detector configuration.
        n_modes: Number of photonic modes to cover.

    Returns:
        normalized: list[pcvl.Detector]
            List of detectors (defaulting to ideal PNR where unspecified),
        empty_detectors: bool
            If True, no Detector was defined in experiment. If False, at least one Detector was defined in experiement.
    """
    empty_detectors = True
    detectors_attr = getattr(experiment, "detectors", None)
    normalized: list[pcvl.Detector] = []

    for mode in range(n_modes):
        detector = None
        if detectors_attr is not None:
            try:
                detector = detectors_attr[mode]  # type: ignore[index]
            except (KeyError, IndexError, TypeError):
                getter = getattr(detectors_attr, "get", None)
                if callable(getter):
                    detector = getter(mode, None)
        if detector is None:
            detector = pcvl.Detector.pnr()
        else:
            empty_detectors = False  # At least one Detector was defined in experiment
            if not hasattr(detector, "detect"):
                raise TypeError(
                    f"Detector at mode {mode} does not implement a 'detect' method."
                )
        normalized.append(detector)

    return normalized, empty_detectors
