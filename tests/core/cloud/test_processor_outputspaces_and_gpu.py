"""
Output-space sizing (local & cloud) and GPU roundtrips.

Focus:
- Local QuantumLayer output sizes match combinatorial expectations
- Cloud offload returns the same sized distribution (optional; auto-skips)
- GPU: local forward on CUDA; cloud roundtrip preserves device (auto-skips if no CUDA)
"""

import pytest
import torch
from math import comb

from merlin.core.merlin_processor import MerlinProcessor
from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.sampling.strategies import OutputMappingStrategy


def _make_layer(m: int, n: int, input_size: int, no_bunching: bool) -> QuantumLayer:
    builder = CircuitBuilder(n_modes=m)
    builder.add_rotations(trainable=True, name="theta")
    builder.add_angle_encoding(modes=list(range(input_size)), name="px")
    if m >= 3:
        builder.add_entangling_layer()

    return QuantumLayer(
        input_size=input_size,
        output_size=None,
        builder=builder,
        n_photons=n,
        no_bunching=no_bunching,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    ).eval()


class TestOutputSpacesAndGPU:
    @pytest.mark.parametrize(
        "m,n,input_size,no_bunching",
        [
            (4, 2, 2, True),   # C(4,2)=6
            (4, 2, 2, False),  # C(5,2)=10
            (5, 3, 3, True),   # C(5,3)=10
            (5, 3, 3, False),  # C(7,3)=35
            (6, 2, 2, True),   # C(6,2)=15
        ],
    )
    def test_local_distribution_size(self, m, n, input_size, no_bunching):
        layer = _make_layer(m, n, input_size, no_bunching)
        B = 3
        y = layer(torch.rand(B, input_size))
        expected = comb(m, n) if no_bunching else comb(m + n - 1, n)
        assert y.shape == (B, expected)
        assert torch.allclose(y.sum(dim=1), torch.ones(B), atol=1e-5)

    def test_cloud_distribution_size_matches(self, remote_processor):
        m, n, input_size, no_bunching = 6, 2, 2, True
        layer = _make_layer(m, n, input_size, no_bunching)
        B = 4
        x = torch.rand(B, input_size)
        expected = comb(m, n)

        proc = MerlinProcessor(remote_processor)
        y = proc.forward(layer, x, shots=2000)
        assert y.shape == (B, expected)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_local_cuda(self):
        layer = _make_layer(5, 2, 2, True).to("cuda")
        y = layer(torch.rand(3, 2, device="cuda"))
        assert y.device.type == "cuda"
        assert y.shape[1] == comb(5, 2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cloud_cuda_roundtrip(self, remote_processor):
        layer = _make_layer(6, 2, 2, True)
        proc = MerlinProcessor(remote_processor)
        x = torch.rand(2, 2, device="cuda")
        y = proc.forward(layer, x, shots=1500)
        assert y.device.type == "cuda"
        assert y.shape[1] == comb(6, 2)
