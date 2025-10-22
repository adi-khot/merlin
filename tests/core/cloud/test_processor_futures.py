"""
Futures & cloud execution tests for MerlinProcessor.

Focus:
- forward_async returns torch Future with helpers (cancel_remote, status, job_ids)
- Timeout & cancellation behavior (best-effort; skip if backend too fast)
- resume(job_id, ...) yields same-shaped mapped tensor
- concurrent futures
- pipeline with two quantum layers accrues >=2 job_ids

Requires cloud; auto-skips if no token via `remote_processor` fixture.
"""

import time
import concurrent.futures as _cf

import pytest
import torch
import torch.nn as nn

from merlin.core.merlin_processor import MerlinProcessor
from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.sampling.strategies import OutputMappingStrategy


def _spin_until(pred, timeout_s=10.0, sleep_s=0.02):
    start = time.time()
    while not pred():
        if time.time() - start > timeout_s:
            return False
        time.sleep(sleep_s)
    return True


def _make_layer_6m2p_raw():
    builder = CircuitBuilder(n_modes=6)
    builder.add_rotations(trainable=True, name="theta")
    builder.add_angle_encoding(modes=[0, 1], name="px")
    builder.add_entangling_layer()

    return QuantumLayer(
        input_size=2,
        output_size=None,
        builder=builder,
        n_photons=2,
        no_bunching=True,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    ).eval()


class TestFuturesCloud:
    def test_forward_async_future_and_helpers(self, remote_processor):
        layer = _make_layer_6m2p_raw()
        proc = MerlinProcessor(remote_processor)
        fut = proc.forward_async(layer, torch.rand(3, 2), shots=1500)

        assert isinstance(fut, torch.futures.Future)
        assert hasattr(fut, "cancel_remote")
        assert hasattr(fut, "status")
        assert hasattr(fut, "job_ids")

        _spin_until(lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=10.0)
        out = fut.wait()
        assert out.shape == (3, 15)

    def test_timeout_sets_timeouterror(self, remote_processor):
        layer = _make_layer_6m2p_raw()
        proc = MerlinProcessor(remote_processor)
        fut = proc.forward_async(layer, torch.rand(8, 2), shots=50000, timeout=0.03)

        done_in_time = _spin_until(lambda: fut.done(), timeout_s=2.0)
        if not done_in_time:
            with pytest.raises(TimeoutError):
                fut.wait()
        else:
            try:
                _ = fut.value()
            except Exception:
                with pytest.raises(TimeoutError):
                    fut.wait()

    def test_cancel_remote_cancelled_error(self, remote_processor):
        layer = _make_layer_6m2p_raw()
        proc = MerlinProcessor(remote_processor, timeout=None)
        fut = proc.forward_async(layer, torch.rand(8, 2), shots=40000, timeout=None)

        _spin_until(lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=10.0)
        if fut.done():
            pytest.skip("Backend finished too quickly to test cancellation")

        fut.cancel_remote()
        with pytest.raises(_cf.CancelledError):
            fut.wait()

    def test_resume_attaches_and_maps(self, remote_processor):
        layer = _make_layer_6m2p_raw()
        proc = MerlinProcessor(remote_processor)

        fut = proc.forward_async(layer, torch.rand(2, 2), shots=2000)
        _spin_until(lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=10.0)

        if len(fut.job_ids) == 0:
            out = fut.wait()
            assert out.shape == (2, 15)
            return

        job_id = fut.job_ids[0]
        resumed = proc.resume(job_id, layer=layer, batch_size=2, shots=2000)
        out1 = fut.wait()
        out2 = resumed.wait()
        assert out1.shape == out2.shape == (2, 15)

    def test_multiple_concurrent_futures(self, remote_processor):
        layer = _make_layer_6m2p_raw()
        proc = MerlinProcessor(remote_processor)
        xs = [torch.rand(2, 2) for _ in range(4)]
        futs = [proc.forward_async(layer, x, shots=1500) for x in xs]

        for f in futs:
            _spin_until(lambda: len(f.job_ids) > 0 or f.done(), timeout_s=10.0)

        outs = [f.wait() for f in futs]
        for y in outs:
            assert y.shape == (2, 15)

    def test_two_quantum_layers_pipeline_has_2_jobs(self, remote_processor):
        # q1: 4m,2p -> 6; q2: 5m,2p -> 10
        b1 = CircuitBuilder(n_modes=4)
        b1.add_rotations(trainable=True, name="t1")
        b1.add_angle_encoding(modes=[0], name="px")
        q1 = QuantumLayer(1, None, builder=b1, n_photons=2,
                          no_bunching=True, output_mapping_strategy=OutputMappingStrategy.NONE).eval()

        b2 = CircuitBuilder(n_modes=5)
        b2.add_rotations(trainable=True, name="t2")
        b2.add_angle_encoding(modes=[0, 1], name="px")
        q2 = QuantumLayer(2, None, builder=b2, n_photons=2,
                          no_bunching=True, output_mapping_strategy=OutputMappingStrategy.NONE).eval()

        model = nn.Sequential(
            nn.Linear(3, 1),
            q1,
            nn.Linear(6, 2),
            q2,
            nn.Linear(10, 3),
            nn.Softmax(dim=-1),
        ).eval()

        proc = MerlinProcessor(remote_processor)
        fut = proc.forward_async(model, torch.rand(4, 3), shots=2000)
        _spin_until(lambda: len(fut.job_ids) >= 2 or fut.done(), timeout_s=20.0)
        y = fut.wait()
        assert y.shape == (4, 3)
        assert len(fut.job_ids) >= 2
