from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import cast

import perceval as pcvl
import torch


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
    """

    def __init__(
        self,
        simulation_keys: Sequence[Sequence[int]] | torch.Tensor,
        detectors: Sequence[pcvl.Detector],
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if simulation_keys is None or len(simulation_keys) == 0:
            raise ValueError("simulation_keys must contain at least one Fock state.")

        self._dtype = dtype or torch.float32
        device_obj = torch.device(device) if device is not None else None

        self._simulation_keys = self._normalize_keys(simulation_keys)
        self._n_modes = len(self._simulation_keys[0])

        if any(len(key) != self._n_modes for key in self._simulation_keys):
            raise ValueError("All simulation keys must have the same number of modes.")

        if len(detectors) != self._n_modes:
            raise ValueError(
                f"Expected {self._n_modes} detectors, received {len(detectors)}."
            )

        self._detectors: tuple[pcvl.Detector, ...] = tuple(detectors)
        self._response_cache: dict[
            tuple[int, int], list[tuple[tuple[int, ...], float]]
        ] = {}

        matrix, detector_keys, is_identity = self._build_transform(device_obj)

        self._detector_keys: list[tuple[int, ...]] = detector_keys
        self._is_identity = is_identity

        if not is_identity:
            self.register_buffer("_matrix", matrix)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_keys(
        keys: Sequence[Sequence[int]] | torch.Tensor,
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def output_keys(self) -> list[tuple[int, ...]]:
        """Return the classical detection outcome keys."""
        return self._detector_keys

    @property
    def output_size(self) -> int:
        """Number of classical outcomes produced by the detectors."""
        return len(self._detector_keys)

    @property
    def is_identity(self) -> bool:
        """Whether the transform reduces to the identity (ideal PNR detectors)."""
        return self._is_identity

    def forward(self, distribution: torch.Tensor) -> torch.Tensor:
        """
        Apply the detector transform to a probability distribution.

        Args:
            distribution: Probability tensor with the simulator basis as its last
                dimension.

        Returns:
            Tensor: Distribution expressed in the detector basis.
        """
        if self._is_identity:
            return distribution

        matrix: torch.Tensor = cast(torch.Tensor, self._matrix)  # type: ignore[attr-defined]
        if distribution.dtype != matrix.dtype:
            raise TypeError(
                "Detector transform dtype mismatch: "
                f"distribution={distribution.dtype}, transform={matrix.dtype}"
            )
        if distribution.device != matrix.device:
            raise RuntimeError(
                "Detector transform device mismatch: "
                f"distribution={distribution.device}, transform={matrix.device}"
            )

        original_shape = distribution.shape
        last_dim = original_shape[-1]
        if distribution.dim() == 1:
            distribution = distribution.unsqueeze(0)
        else:
            distribution = distribution.reshape(-1, last_dim)

        # Direct multiplication: distribution (batch, input_dim) @ matrix (input_dim, output_dim)
        transformed = torch.sparse.mm(distribution, matrix)

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
