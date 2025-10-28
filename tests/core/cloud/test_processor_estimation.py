"""
Shot-estimation tests for MerlinProcessor. These use the platform's estimator,
so they do not submit any cloud jobs.

Focus:
- Returns one integer per input row (single and batch).
- Monotonicity: requesting more desired samples should not reduce estimated shots.
- Does not modify MerlinProcessor job history (no cloud jobs created).
"""

from __future__ import annotations

import torch

from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.core.merlin_processor import MerlinProcessor
from merlin.sampling.strategies import OutputMappingStrategy


def _make_layer(m: int, n: int, input_size: int, no_bunching: bool = True) -> QuantumLayer:
    b = CircuitBuilder(n_modes=m)
    b.add_rotations(trainable=True, name="theta")
    b.add_angle_encoding(modes=list(range(input_size)), name="px")
    if m >= 3:
        b.add_entangling_layer()
    return QuantumLayer(
        input_size=input_size,
        output_size=None,
        builder=b,
        n_photons=n,
        no_bunching=no_bunching,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    ).eval()


class TestShotEstimation:
    def test_single_and_batch_estimates(self, remote_processor):
        q = _make_layer(6, 2, input_size=2, no_bunching=True).eval()
        proc = MerlinProcessor(remote_processor)

        # Single vector
        x_single = torch.rand(2)
        est_single = proc.estimate_required_shots_per_input(q, x_single, desired_samples_per_input=2_000)
        assert isinstance(est_single, list) and len(est_single) == 1
        assert isinstance(est_single[0], int) and est_single[0] >= 0

        # Batch
        X = torch.rand(5, 2)
        est_batch = proc.estimate_required_shots_per_input(q, X, desired_samples_per_input=2_000)
        assert isinstance(est_batch, list) and len(est_batch) == 5
        assert all(isinstance(v, int) and v >= 0 for v in est_batch)

    def test_monotonic_with_desired_samples(self, remote_processor):
        q = _make_layer(5, 2, input_size=2, no_bunching=True).eval()
        proc = MerlinProcessor(remote_processor)

        X = torch.rand(4, 2)
        est_lo = proc.estimate_required_shots_per_input(q, X, desired_samples_per_input=1_000)
        est_hi = proc.estimate_required_shots_per_input(q, X, desired_samples_per_input=5_000)

        # Asking for more desired samples should not reduce the shot estimate
        assert len(est_lo) == len(est_hi) == 4
        for a, b in zip(est_lo, est_hi, strict=False):
            assert b >= a  # monotonic non-decreasing

    def test_no_jobs_created(self, remote_processor):
        q = _make_layer(6, 2, input_size=2, no_bunching=True).eval()
        proc = MerlinProcessor(remote_processor)
        before = len(proc.get_job_history())

        _ = proc.estimate_required_shots_per_input(q, torch.rand(3, 2), desired_samples_per_input=3_000)

        after = len(proc.get_job_history())
        assert after == before, "Estimation must not submit any remote jobs"
