# tests/core/cloud/test_nobunching_bunching_and_gpu.py
"""
Validation tests for:
- No-bunching vs bunching output sizes (local QuantumLayer and remote MerlinProcessor)
- Forward-pass equivalence between local QuantumLayer and remote sim:slos
- GPU workflows:
    * local QuantumLayer on CUDA
    * remote QuantumLayer via MerlinProcessor with CUDA inputs/outputs
    * hybrid nn.Sequential with classical layers around a QuantumLayer on CUDA
- Training-mode guard (MerlinProcessor requires eval mode)
"""

import math
import time

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


def _make_layer(n_modes: int, n_photons: int, input_size: int, no_bunching: bool) -> QuantumLayer:
    """
    Build a simple circuit exposing angle encoding + a rotation layer (trainable)
    and returning raw distribution (no internal mapping).
    """
    builder = CircuitBuilder(n_modes=n_modes)
    builder.add_rotation_layer(trainable=True, name="theta")
    builder.add_angle_encoding(modes=list(range(input_size)), name="px")
    # a little entanglement for non-trivial outputs if there is room
    if n_modes >= 3:
        builder.add_entangling_layer(depth=1)

    layer = QuantumLayer(
        input_size=input_size,
        output_size=None,  # raw distribution
        circuit=builder,
        n_photons=n_photons, # local SLOS exact probs
        no_bunching=no_bunching,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    )
    layer.eval()
    return layer


def _expected_size(n_modes: int, n_photons: int, no_bunching: bool) -> int:
    if no_bunching:
        # combinations without repetition: C(m, n)
        from math import comb
        return comb(n_modes, n_photons)
    else:
        # combinations with repetition ("stars and bars"): C(m+n-1, n)
        from math import comb
        return comb(n_modes + n_photons - 1, n_photons)


# ------------------- TESTS -------------------

class TestNoBunchingBunchingAndGPU:
    @pytest.mark.parametrize(
        "n_modes,n_photons,input_size,no_bunching",
        [
            (4, 2, 2, True),  # C(4,2) = 6
            (4, 2, 2, False),  # C(4+2-1,2) = C(5,2) = 10
            (5, 2, 2, True),  # C(5,2) = 10
            (5, 3, 2, False),  # C(5+3-1,3) = C(7,3) = 35
            (6, 2, 2, True),  # C(6,2) = 15
        ],
    )
    def test_sizes_match_local_and_remote(self, remote_processor, n_modes, n_photons, input_size, no_bunching):
        """
        For a variety of (m, n, no_bunching), verify the output distribution size matches the
        combinatorial expectation both locally and when offloaded via MerlinProcessor.
        """
        layer = _make_layer(n_modes, n_photons, input_size, no_bunching)
        B = 3
        x = torch.rand(B, input_size)

        # Local (exact SLOS)
        local_out = layer(x)
        exp = _expected_size(n_modes, n_photons, no_bunching)
        assert local_out.shape == (B, exp)

        # Remote - shots is now a parameter to forward, not constructor
        proc = MerlinProcessor(remote_processor)
        remote_out = proc.forward(layer, x, shots=2000)  # shots ignored if 'probs' is available
        assert remote_out.shape == (B, exp)

    @pytest.mark.parametrize(
        "n_modes,n_photons,input_size,no_bunching",
        [
            (6, 2, 2, True),  # 15
            (5, 2, 2, False),  # 14
        ],
    )
    def test_forward_equivalence_high_shots(self, remote_processor, n_modes, n_photons, input_size, no_bunching):
        """
        Forward-pass equivalence between:
            - local QuantumLayer (SLOS exact probs, shots=0)
            - remote sim:slos via MerlinProcessor (prefers 'probs'→ exact; else 'sample_count'→ approx)
        With 'probs' the results should match ~exactly; with 'sample_count' they should be close
        given sufficiently large shots.
        """
        layer = _make_layer(n_modes, n_photons, input_size, no_bunching)
        B = 5
        x = torch.rand(B, input_size)

        local_out = layer(x)  # exact
        exp = _expected_size(n_modes, n_photons, no_bunching)
        assert local_out.shape == (B, exp)

        # Use high shots to reduce variance if the backend falls back to counts
        proc = MerlinProcessor(remote_processor)
        # Pass shots as parameter to forward
        remote_out = proc.forward(layer, x, shots=500000)
        assert remote_out.shape == (B, exp)

        # Tolerance: if the backend served 'probs', these are essentially exact; if counts, allow slack
        # We'll try a fairly tight tolerance first and relax if needed
        if torch.allclose(local_out, remote_out, atol=1e-4, rtol=1e-4):
            assert True
        else:
            # Allow a looser bound (counts -> normalized → small mismatch)
            assert torch.allclose(local_out, remote_out, atol=3e-2, rtol=3e-2), \
                f"Local vs Remote differ more than tolerance:\nlocal={local_out}\nremote={remote_out}"

        # Row-wise sums ~ 1
        assert torch.allclose(remote_out.sum(dim=1), torch.ones(B), atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_local_quantumlayer(self):
        """
        Run QuantumLayer locally on CUDA and verify device and shapes.
        """
        n_modes, n_photons, input_size, no_bunching = 5, 2, 2, True
        layer = _make_layer(n_modes, n_photons, input_size, no_bunching).to("cuda")
        B = 4
        x = torch.rand(B, input_size, device="cuda")

        y = layer(x)
        exp = _expected_size(n_modes, n_photons, no_bunching)
        assert y.shape == (B, exp)
        assert y.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_remote_quantumlayer_roundtrip(self, remote_processor):
        """
        Provide CUDA input to MerlinProcessor.forward; output should come back on CUDA with correct shape.
        """
        n_modes, n_photons, input_size, no_bunching = 6, 2, 2, True
        layer = _make_layer(n_modes, n_photons, input_size, no_bunching)
        proc = MerlinProcessor(remote_processor)

        B = 3
        x = torch.rand(B, input_size, device="cuda")
        # Pass shots as parameter to forward
        y = proc.forward(layer, x, shots=2000)
        exp = _expected_size(n_modes, n_photons, no_bunching)
        assert y.shape == (B, exp)
        assert y.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_hybrid_sequential(self, remote_processor):
        """
        nn.Sequential with classical layers around a QuantumLayer on CUDA.
        MerlinProcessor offloads QuantumLayer(s) but preserves device for I/O tensors.
        """
        n_modes, n_photons, input_size, no_bunching = 5, 2, 2, True
        dist_size = _expected_size(n_modes, n_photons, no_bunching)

        qlayer = _make_layer(n_modes, n_photons, input_size, no_bunching).eval()
        model = nn.Sequential(
            nn.Linear(input_size, input_size).to("cuda"),
            qlayer,
            nn.Linear(dist_size, 4).to("cuda"),
            nn.Softmax(dim=-1).to("cuda"),
        ).eval()

        proc = MerlinProcessor(remote_processor)

        B = 6
        x = torch.rand(B, input_size, device="cuda")
        # Pass shots as parameter to forward
        y = proc.forward(model, x, shots=2000)

        assert y.shape == (B, 4)
        assert y.device.type == "cuda"
        # softmax sanity: sums ~ 1
        assert torch.allclose(y.sum(dim=1), torch.ones(B, device="cuda"), atol=1e-5, rtol=1e-5)

    def test_training_mode_guard(self, remote_processor):
        """
        MerlinProcessor requires eval mode for remote execution.
        Verify that calling forward on a training QuantumLayer raises.
        """
        n_modes, n_photons, input_size, no_bunching = 5, 2, 2, True
        qlayer = _make_layer(n_modes, n_photons, input_size, no_bunching)
        qlayer.train()

        proc = MerlinProcessor(remote_processor)
        x = torch.rand(2, input_size)

        # Pass shots as parameter to forward
        with pytest.raises(RuntimeError, match="requires `.eval\\(\\)` mode"):
            proc.forward(qlayer, x, shots=1000)

        # Restore eval to ensure later tests not affected (defensive)
        qlayer.eval()

    def test_shots_none_behavior(self, remote_processor):
        """
        Test that passing shots=None works correctly (should use 'probs' if available,
        or fall back to DEFAULT_SHOTS_PER_CALL).
        """
        n_modes, n_photons, input_size, no_bunching = 4, 2, 2, True
        layer = _make_layer(n_modes, n_photons, input_size, no_bunching)
        proc = MerlinProcessor(remote_processor)

        B = 2
        x = torch.rand(B, input_size)

        # Pass shots=None explicitly
        y = proc.forward(layer, x, shots=None)
        exp = _expected_size(n_modes, n_photons, no_bunching)
        assert y.shape == (B, exp)

        # Verify probabilities sum to ~1
        assert torch.allclose(y.sum(dim=1), torch.ones(B), atol=1e-4, rtol=1e-4)