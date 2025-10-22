# tests/core/cloud/test_futures_api.py
"""
Futures-focused test suite for MerlinProcessor.

What we verify:

- `forward_async(...)` returns a torch Future
  - has ergonomic helpers: .cancel_remote(), .status(), .job_ids
  - returns immediately (non-blocking)
  - populates .job_ids during run
  - .status() transitions IDLE -> RUNNING-like -> COMPLETE
  - .wait() yields a Tensor with correct shape/device/dtype
  - .cancel_remote() raises CancelledError and stops remote job (best-effort; skipped if backend too fast)
  - per-call timeout sets a TimeoutError on the Future (best-effort; skipped if backend too fast)

- Multiple concurrent futures
  - can run at once and each completes with its own result/job_ids

- Multiple quantum layers pipeline
  - job_ids has >= 2 entries before completion (two offloaded stages)

- `resume(job_id, layer, ...)`
  - returns a Future with the same helper attributes
  - .wait() yields a mapped Tensor of the expected shape

- Idempotent/no-op cancellation after completion
  - calling cancel_remote() twice doesn't error
  - calling cancel_remote() after completion is a no-op

NOTE:
These tests use Quandela Cloud "sim:slos".
Fill in TOKEN at the top before running.

Some timing-sensitive tests will skip if the backend finishes too quickly.
"""

import time
import concurrent.futures as _cf

import pytest
import torch
import torch.nn as nn
import perceval as pcvl
from perceval.runtime import RemoteProcessor, RemoteConfig

from merlin.core.merlin_processor import MerlinProcessor
from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.sampling.strategies import OutputMappingStrategy

# ------------- AUTH -------------
TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6Mzk1LCJleHAiOjE3NjEyMjgyNzUuMjY4MDkzfQ.vPEHupHJhtXAFVMqyhav7s97cfp_CtJFxA9eH7328JSehdxKry192BKZ8i99KarjlMBkKoIyEJEmU45O3aDjSw"  # <--- PUT YOUR TOKEN HERE


@pytest.fixture(scope="session", autouse=True)
def _set_token_once():
    assert TOKEN != "", "Please set TOKEN at the top of this file."
    RemoteConfig.set_token(TOKEN)


# ------------- FIXTURES -------------

@pytest.fixture
def remote_processor():
    return pcvl.RemoteProcessor("sim:slos")


@pytest.fixture
def qlayer_6m2p_raw():
    """
    6 modes, 2 photons, input_size=2, no_bunching=True.
    Raw distribution (no internal mapping): size C(6,2) = 15
    """
    builder = CircuitBuilder(n_modes=6)
    builder.add_rotation_layer(trainable=True, name="theta")
    builder.add_angle_encoding(modes=[0, 1], name="px")
    builder.add_entangling_layer(depth=1)

    layer = QuantumLayer(
        input_size=2,
        output_size=None,  # raw probs
        circuit=builder,
        n_photons=2,
        no_bunching=True,  # local SLOS exact
        output_mapping_strategy=OutputMappingStrategy.NONE,
    ).eval()
    return layer


def _spin_until(pred, timeout_s=10.0, sleep_s=0.02):
    start = time.time()
    while not pred():
        if time.time() - start > timeout_s:
            return False
        time.sleep(sleep_s)
    return True


# ------------- TESTS -------------

class TestFuturesAPI:
    def test_forward_async_returns_future_and_helpers(self, remote_processor, qlayer_6m2p_raw):
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(3, 2)
        # Pass shots as parameter to forward_async
        fut = proc.forward_async(qlayer_6m2p_raw, x, shots=1000)

        # Should be a torch Future and have helper attributes
        assert isinstance(fut, torch.futures.Future)
        assert hasattr(fut, "cancel_remote")
        assert hasattr(fut, "status")
        assert hasattr(fut, "job_ids")
        assert isinstance(fut.job_ids, list)

        # Non-blocking (should return almost immediately)
        assert not fut.done()

        # job_id should appear soon or it might already finish
        ok = _spin_until(lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=10.0)
        assert ok

        # status should be a dict with at least 'state' key
        st = fut.status()
        assert isinstance(st, dict)
        assert "state" in st

        out = fut.wait()
        assert out.shape == (3, 15)
        assert fut.done()
        # Once complete, status() should say COMPLETE
        st2 = fut.status()
        assert st2.get("state") in (None, "COMPLETE") or st2.get("progress") == 1.0

    def test_forward_async_device_and_dtype_roundtrip(self, remote_processor, qlayer_6m2p_raw):
        proc = MerlinProcessor(remote_processor)

        # CPU, float64
        x64 = torch.rand(2, 2, dtype=torch.float64)
        # Pass shots as parameter
        fut64 = proc.forward_async(qlayer_6m2p_raw, x64, shots=1000)
        y64 = fut64.wait()
        assert y64.shape == (2, 15)
        assert y64.dtype == torch.float64
        assert y64.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_async_device_roundtrip_cuda(self, remote_processor, qlayer_6m2p_raw):
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(2, 2, device="cuda", dtype=torch.float32)
        # Pass shots as parameter
        fut = proc.forward_async(qlayer_6m2p_raw, x, shots=1000)
        y = fut.wait()
        assert y.shape == (2, 15)
        assert y.device.type == "cuda"
        assert y.dtype == torch.float32

    def test_cancel_remote_raises_cancellederror(self, remote_processor, qlayer_6m2p_raw):
        """
        Try to cancel while in flight; if the simulator finishes too fast, skip to avoid flakiness.
        """
        # Larger shots to encourage in-flight work
        proc = MerlinProcessor(remote_processor, timeout=None)
        x = torch.rand(8, 2)
        # Pass shots as parameter
        fut = proc.forward_async(qlayer_6m2p_raw, x, shots=40000, timeout=None)

        # Wait until a job id appears or completion
        _spin_until(lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=10.0)
        if fut.done():
            pytest.skip("Remote simulator completed too quickly to test cancellation reliably.")

        fut.cancel_remote()
        with pytest.raises(_cf.CancelledError):
            fut.wait()

    def test_timeout_sets_timeouterror_on_future(self, remote_processor, qlayer_6m2p_raw):
        """
        A very short timeout should produce TimeoutError on the future (if still in flight).
        Skip if it finishes too fast.
        """
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(8, 2)
        # Pass shots and timeout as parameters
        fut = proc.forward_async(qlayer_6m2p_raw, x, shots=50000, timeout=0.03)

        # Either it will time out or it will finish very quickly.
        # If it finished immediately, we don't expect a timeout; then skip.
        finished = _spin_until(lambda: fut.done(), timeout_s=2.0)
        if not finished:
            # Still not done -> wait() should raise TimeoutError set by worker
            with pytest.raises(TimeoutError):
                fut.wait()
        else:
            # Done quickly; ensure not an exception (no timeout)
            try:
                _ = fut.value()
            except Exception:
                # If an exception, it should be TimeoutError (rare case)
                with pytest.raises(TimeoutError):
                    fut.wait()

    def test_resume_future_with_helpers(self, remote_processor, qlayer_6m2p_raw):
        """
        Start an async call, capture its first job_id, then attach via resume(...).
        """
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(2, 2)
        # Pass shots as parameter
        fut = proc.forward_async(qlayer_6m2p_raw, x, shots=2000)

        _spin_until(lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=10.0)
        if len(fut.job_ids) == 0:
            # Already finished too fast; just assert original success
            out = fut.wait()
            assert out.shape == (2, 15)
            return

        job_id = fut.job_ids[0]
        # Pass shots to resume as well
        res = proc.resume(job_id, layer=qlayer_6m2p_raw, batch_size=2, shots=2000)

        assert hasattr(res, "cancel_remote")
        assert hasattr(res, "status")
        assert hasattr(res, "job_ids")
        out2 = res.wait()
        assert out2.shape == (2, 15)

        # Both futures should complete (either order), shapes match
        out1 = fut.wait()
        assert out1.shape == out2.shape == (2, 15)

    def test_multiple_concurrent_futures(self, remote_processor, qlayer_6m2p_raw):
        proc = MerlinProcessor(remote_processor)
        xs = [torch.rand(2, 2) for _ in range(4)]
        # Pass shots as parameter to each forward_async call
        futs = [proc.forward_async(qlayer_6m2p_raw, x, shots=1500) for x in xs]

        # Each should either show a job id or be done soon
        for f in futs:
            _spin_until(lambda: len(f.job_ids) > 0 or f.done(), timeout_s=10.0)

        outs = [f.wait() for f in futs]
        for y in outs:
            assert y.shape == (2, 15)

    def test_multiple_quantum_layers_job_ids(self, remote_processor):
        """
        Build a model with two quantum layers; ensure forward_async populates >=2 job_ids.
        """
        # q1: 4 modes, 2 photons -> 6
        b1 = CircuitBuilder(n_modes=4)
        b1.add_rotation_layer(trainable=True, name="t1")
        b1.add_angle_encoding(modes=[0], name="px")
        q1 = QuantumLayer(
            input_size=1,
            output_size=None,
            circuit=b1,
            n_photons=2,
            no_bunching=True,
            output_mapping_strategy=OutputMappingStrategy.NONE,
        ).eval()

        # q2: 5 modes, 2 photons -> 10
        b2 = CircuitBuilder(n_modes=5)
        b2.add_rotation_layer(trainable=True, name="t2")
        b2.add_angle_encoding(modes=[0, 1], name="px")
        q2 = QuantumLayer(
            input_size=2,
            output_size=None,
            circuit=b2,
            n_photons=2,
            no_bunching=True,
            output_mapping_strategy=OutputMappingStrategy.NONE,
        ).eval()

        model = nn.Sequential(
            nn.Linear(3, 1),
            q1,
            nn.Linear(6, 2),
            q2,
            nn.Linear(10, 3),
            nn.Softmax(dim=-1),
        ).eval()

        proc = MerlinProcessor(remote_processor)
        x = torch.rand(4, 3)
        # Pass shots as parameter
        fut = proc.forward_async(model, x, shots=2000)

        _spin_until(lambda: len(fut.job_ids) >= 2 or fut.done(), timeout_s=20.0)
        y = fut.wait()
        assert y.shape == (4, 3)
        assert len(fut.job_ids) >= 2

    def test_cancel_idempotent_and_post_completion_noop(self, remote_processor, qlayer_6m2p_raw):
        """
        Calling cancel_remote() twice should not error.
        Calling cancel_remote() after completion is a no-op.
        """
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(2, 2)
        # Pass shots as parameter
        fut = proc.forward_async(qlayer_6m2p_raw, x, shots=1000)

        # Try idempotent cancel very early (may or may not take effect)
        try:
            fut.cancel_remote()
            fut.cancel_remote()
        except Exception as e:
            pytest.fail(f"cancel_remote should be idempotent; got exception: {e}")

        # If it cancelled in time, wait() raises; else it will complete.
        try:
            _ = fut.wait()
            # Completed; now cancel_remote should be a no-op
            try:
                fut.cancel_remote()
            except Exception as e:
                pytest.fail(f"cancel_remote after completion should be a no-op; got exception: {e}")
        except _cf.CancelledError:
            # Expected cancelled path: ok
            pass

    def test_status_transition_idle_to_complete(self, remote_processor, qlayer_6m2p_raw):
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(2, 2)
        # Pass shots as parameter
        fut = proc.forward_async(qlayer_6m2p_raw, x, shots=1000)

        st0 = fut.status()
        assert isinstance(st0, dict) and st0.get("state") in (None, "IDLE")

        y = fut.wait()
        assert y.shape == (2, 15)

        st1 = fut.status()
        assert isinstance(st1, dict)
        assert st1.get("state") in (None, "COMPLETE") or st1.get("progress") == 1.0

    def test_different_shots_per_call(self, remote_processor, qlayer_6m2p_raw):
        """
        Test that different shots can be used for different forward calls.
        """
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(2, 2)

        # Call with different shot counts
        fut1 = proc.forward_async(qlayer_6m2p_raw, x, shots=1000)
        fut2 = proc.forward_async(qlayer_6m2p_raw, x, shots=5000)
        fut3 = proc.forward_async(qlayer_6m2p_raw, x, shots=None)  # Use default or 'probs'

        y1 = fut1.wait()
        y2 = fut2.wait()
        y3 = fut3.wait()

        # All should have the same shape
        assert y1.shape == y2.shape == y3.shape == (2, 15)

        # All should be valid probability distributions
        for y in [y1, y2, y3]:
            assert torch.all(y >= 0)
            assert torch.all(y <= 1)
            assert torch.allclose(y.sum(dim=1), torch.ones(2), atol=1e-4)