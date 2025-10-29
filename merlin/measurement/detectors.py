from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence

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
        self._device = torch.device(device) if device is not None else None

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

        matrix, detector_keys, is_identity = self._build_transform()

        self._detector_keys: list[tuple[int, ...]] = detector_keys
        self._is_identity = is_identity

        if is_identity:
            buffer_kwargs = {}
            if self._device is not None:
                buffer_kwargs["device"] = self._device
            buffer = torch.empty(
                (0, 0),
                dtype=self._dtype,
                **buffer_kwargs,
            )
            self.register_buffer("_matrix", buffer, persistent=False)
        else:
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
        self,
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
        device_kwargs = {}
        if self._device is not None:
            device_kwargs["device"] = self._device

        matrix = torch.zeros(
            (rows, cols),
            dtype=self._dtype,
            **device_kwargs,
        )
        for row_idx, entries in enumerate(row_entries):
            for col_idx, prob in entries.items():
                matrix[row_idx, col_idx] = prob

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

        matrix = self._matrix  # type: ignore[attr-defined]
        if distribution.dtype != matrix.dtype:
            matrix = matrix.to(distribution.dtype)
        if distribution.device != matrix.device:
            matrix = matrix.to(distribution.device)

        return distribution @ matrix

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)

        dtype = kwargs.get("dtype")
        device = kwargs.get("device")

        if dtype is None and len(args) > 0 and isinstance(args[0], torch.dtype):
            dtype = args[0]
        if device is None and len(args) > 0 and isinstance(args[0], torch.device):
            device = args[0]

        if dtype is not None:
            self._dtype = dtype
        if device is not None:
            self._device = device

        return result


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
