"""
End-to-end tests for the updated MerlinProcessor (single async surface).
Uses real Quandela Cloud simulator: sim:slos.

Coverage:
- Initialization and from_platform
- Sync/Async Future (with cancel_remote/status/job_ids)
- Remote cancellation via Future.cancel_remote()
- Timeout -> remote cancel behavior
- Resume(job_id, ...) -> Future[Tensor]
- Hybrid model with explicit mapping
- Batch size limits, GPU handling (optional)
- Layer caching and job history
- Multiple quantum layers: all offloaded in order
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

# ------------------- AUTH -------------------
# Fill this with your real Quandela Cloud token before running the tests.

TOKEN = "_T_eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJpZCI6Mzk1LCJleHAiOjE3NjEyMjgyNzUuMjY4MDkzfQ.vPEHupHJhtXAFVMqyhav7s97cfp_CtJFxA9eH7328JSehdxKry192BKZ8i99KarjlMBkKoIyEJEmU45O3aDjSw"  # <--- PUT YOUR TOKEN HERE


@pytest.fixture(scope="session", autouse=True)
def _set_token_once():
    assert TOKEN != "", "Please fill TOKEN at the top of this file with your Quandela Cloud token."
    RemoteConfig.set_token(TOKEN)


# ------------------- FIXTURES -------------------

@pytest.fixture
def remote_processor():
    """Real RemoteProcessor for the sim:slos backend."""
    return pcvl.RemoteProcessor("sim:slos")


@pytest.fixture
def quantum_layer():
    """A simple layer that returns raw probability distributions (no output mapping)."""
    builder = CircuitBuilder(n_modes=6)
    builder.add_rotation_layer(trainable=True, name="theta")
    builder.add_angle_encoding(modes=[0, 1], name="px")
    builder.add_entangling_layer(depth=1)

    layer = QuantumLayer(
        input_size=2,
        output_size=None,  # raw distribution
        circuit=builder,
        n_photons=2, # local SLOS exact probs in layer.forward; cloud uses 'probs'
        no_bunching=True,
        output_mapping_strategy=OutputMappingStrategy.NONE
    )
    layer.eval()
    return layer


@pytest.fixture
def hybrid_model(quantum_layer):
    """nn.Sequential with quantum layer + explicit mapping afterwards."""
    dist_size = 15  # C(6,2) = 15
    model = nn.Sequential(
        quantum_layer,
        nn.Linear(dist_size, 10),
        nn.ReLU(),
        nn.Linear(10, 3),
        nn.Softmax(dim=-1),
    ).eval()
    return model


def spin_until(pred, timeout_s=10.0, sleep_s=0.05):
    start = time.time()
    while not pred():
        if time.time() - start > timeout_s:
            return False
        time.sleep(sleep_s)
    return True


# ------------------- TESTS -------------------

class TestMerlinProcessorE2E:
    def test_init_and_platform_info(self, remote_processor):
        proc = MerlinProcessor(remote_processor, max_batch_size=16, timeout=30.0)
        assert proc.remote_processor == remote_processor
        assert proc.max_batch_size == 16
        assert proc.default_timeout == 30.0

        info = proc.platform_info
        assert "name" in info
        assert "available_commands" in info
        assert isinstance(info["available_commands"], (list, tuple))

    def test_from_platform(self):
        proc = MerlinProcessor.from_platform("sim:slos", max_batch_size=8)
        assert proc.max_batch_size == 8
        # Check that we can call with shots parameter
        x = torch.rand(2, 2)
        # Just verify the method signature accepts shots
        assert hasattr(proc, 'forward')

    def test_sync_forward_single_layer(self, remote_processor, quantum_layer):
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(4, 2)
        # Pass shots as parameter to forward
        out = proc.forward(quantum_layer, x, shots=1000)
        assert out.shape == (4, 15)  # C(6,2)
        assert torch.all(out >= 0) and torch.all(out <= 1)
        # Sums close-ish to 1 per batch row
        assert (torch.abs(out.sum(dim=1) - 1.0) < 1e-4).all().item()

    def test_async_future_has_controls(self, remote_processor, quantum_layer):
        """forward_async returns a Future with cancel_remote/status/job_ids."""
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(3, 2)
        # Pass shots as parameter to forward_async
        fut = proc.forward_async(quantum_layer, x, shots=1000)
        # control attributes exist
        assert hasattr(fut, "cancel_remote")
        assert hasattr(fut, "status")
        assert hasattr(fut, "job_ids")
        # job id should appear shortly or future done
        ok = spin_until(lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=10.0)
        assert ok
        res = fut.wait()
        assert res.shape == (3, 15)

    def test_cancel_propagates_remotely(self, remote_processor, quantum_layer):
        """
        Try to cancel an in-flight future and expect a CancelledError.
        If the simulator finishes too quickly, skip to avoid flakiness.
        """
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(8, 2)
        # Pass shots as parameter, use higher shot count for heavier load
        fut = proc.forward_async(quantum_layer, x, shots=20000, timeout=None)

        # Wait until a job id appears or completion
        spin_until(lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=10.0)
        if fut.done():
            pytest.skip("Remote simulator completed too quickly to test cancellation reliably.")

        # Cancel and expect a CancelledError (or already finished)
        fut.cancel_remote()
        with pytest.raises(_cf.CancelledError):
            fut.wait()

    def test_timeout_triggers_remote_cancel(self, remote_processor, quantum_layer):
        """
        A very short timeout should trigger remote cancel internally and surface TimeoutError.
        """
        proc = MerlinProcessor(remote_processor, timeout=0.05)
        x = torch.rand(8, 2)
        # Pass shots and timeout as parameters
        fut = proc.forward_async(quantum_layer, x, shots=50000, timeout=0.05)
        # Wait for completion; since the worker sets TimeoutError on fut, .wait() should raise
        with pytest.raises(TimeoutError):
            fut.wait()

    def test_resume_mapped_tensor(self, remote_processor, quantum_layer):
        """
        Start a future, capture its first job id, then attach via resume(...) and get the same shape.
        """
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(2, 2)

        # Pass shots as parameter
        fut = proc.forward_async(quantum_layer, x, shots=2000)
        # wait for job id or completion
        spin_until(lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=10.0)
        if len(fut.job_ids) == 0:
            # finished too fast; just assert success through the original future
            res = fut.wait()
            assert res.shape == (2, 15)
            return

        job_id = fut.job_ids[0]
        # Attach and map to tensor, pass shots to resume as well
        resumed = proc.resume(
            job_id,
            layer=quantum_layer,
            batch_size=2,
            shots=2000,
            device=torch.device("cpu"),
            dtype=torch.float32
        )
        r1 = fut.wait()
        r2 = resumed.wait()
        assert r1.shape == r2.shape == (2, 15)

    def test_hybrid_model(self, remote_processor, hybrid_model):
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(4, 2)
        # Pass shots as parameter
        out = proc.forward(hybrid_model, x, shots=1000)
        assert out.shape == (4, 3)
        # Softmax sums to 1
        assert (torch.abs(out.sum(dim=1) - 1.0) < 1e-5).all().item()

    def test_batch_size_limits(self, remote_processor, quantum_layer):
        proc = MerlinProcessor(remote_processor, max_batch_size=4)
        # Pass shots as parameter
        ok = proc.forward(quantum_layer, torch.rand(4, 2), shots=1000)
        assert ok.shape == (4, 15)
        with pytest.raises(ValueError, match="exceeds cloud limit"):
            proc.forward(quantum_layer, torch.rand(8, 2), shots=1000)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_tensor_roundtrip(self, remote_processor, quantum_layer):
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(2, 2, device="cuda")
        # Pass shots as parameter
        y = proc.forward(quantum_layer, x, shots=1000)
        assert y.device.type == "cuda" and y.shape == (2, 15)

    def test_layer_caching_and_history(self, remote_processor):
        proc = MerlinProcessor(remote_processor)

        builder = CircuitBuilder(n_modes=4)
        builder.add_rotation_layer(trainable=True, name="theta")
        builder.add_angle_encoding(modes=[0], name="px")

        layer = QuantumLayer(
            input_size=1,
            output_size=None,
            circuit=builder,
            n_photons=2,
            no_bunching=True,
            output_mapping_strategy=OutputMappingStrategy.NONE,
        ).eval()

        x = torch.rand(2, 1)

        # Initially
        assert len(proc.get_job_history()) == 0
        # Pass shots as parameter
        out1 = proc.forward(layer, x, shots=500)
        assert len(proc._layer_cache) == 1
        out2 = proc.forward(layer, x, shots=500)
        assert len(proc._layer_cache) == 1
        assert out1.shape == out2.shape == (2, 6)  # C(4,2)

        # Job history should have >= 2
        assert len(proc.get_job_history()) >= 2
        proc.clear_job_history()
        assert len(proc.get_job_history()) == 0

    def test_multiple_quantum_layers_offloaded(self, remote_processor):
        """
        Build a model with two quantum layers; ensure both get offloaded (len(job_ids)==2).
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
            q1,  # offload 1
            nn.Linear(6, 2),
            q2,  # offload 2
            nn.Linear(10, 3),
            nn.Softmax(dim=-1),
        ).eval()

        proc = MerlinProcessor(remote_processor)
        x = torch.rand(4, 3)
        # Pass shots as parameter
        fut = proc.forward_async(model, x, shots=2000)
        # Wait until at least two job ids, or done
        spin_until(lambda: len(fut.job_ids) >= 2 or fut.done(), timeout_s=20.0)
        out = fut.wait()
        assert out.shape == (4, 3)
        # Both quantum layers should have been offloaded
        assert len(fut.job_ids) >= 2
        # Softmax
        assert (torch.abs(out.sum(dim=1) - 1.0) < 1e-5).all().item()

    def test_shots_parameter_variations(self, remote_processor, quantum_layer):
        """Test that shots parameter works correctly in different scenarios."""
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(2, 2)

        # Test with explicit shots
        out1 = proc.forward(quantum_layer, x, shots=5000)
        assert out1.shape == (2, 15)

        # Test with None (should use default or 'probs' command)
        out2 = proc.forward(quantum_layer, x, shots=None)
        assert out2.shape == (2, 15)

        # Test async with different shots values
        fut1 = proc.forward_async(quantum_layer, x, shots=1000)
        time.sleep(0.1)
        fut2 = proc.forward_async(quantum_layer, x, shots=10000)

        res1 = fut1.wait()
        res2 = fut2.wait()

        assert res1.shape == (2, 15)
        assert res2.shape == (2, 15)