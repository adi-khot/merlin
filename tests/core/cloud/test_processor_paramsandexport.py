"""
Export & parameter-mapping tests (local-first; cloud optional).

Focus:
- export_config applies current trainable params to the exported circuit
- no_bunching reduces state space vs bunching
- local vs remote shape/normalization consistency (cloud optional)
"""

import torch
import pytest
from math import comb

from merlin.algorithms import QuantumLayer
from merlin.core.merlin_processor import MerlinProcessor
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.sampling.strategies import OutputMappingStrategy


def _make_layer(m: int, n: int, input_size: int, trainable=True, no_bunching=True) -> QuantumLayer:
    builder = CircuitBuilder(n_modes=m)
    if trainable:
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


class TestExportAndParams:
    def test_export_config_includes_trained_thetas(self):
        q = _make_layer(5, 3, 3, trainable=True, no_bunching=True)
        before = {n: p.clone() for n, p in q.named_parameters()}

        q.train()
        opt = torch.optim.Adam(q.parameters(), lr=0.05)
        X = torch.randn(6, 3)
        for _ in range(4):
            opt.zero_grad()
            y = q(X).sum()
            y.backward()
            opt.step()
        q.eval()

        changed = any(
            not torch.allclose(p, before[n], atol=1e-6)
            for n, p in q.named_parameters()
        )
        assert changed, "Expected trainable parameters to update"

        cfg = q.export_config()
        exported = cfg["circuit"]
        names = [p.name for p in exported.get_parameters()]
        assert any("theta" in n for n in names)

    def test_no_bunching_reduces_states(self):
        m, n = 5, 3
        layer_b = _make_layer(m, n, input_size=3, trainable=False, no_bunching=False)
        layer_nb = _make_layer(m, n, input_size=3, trainable=False, no_bunching=True)

        X = torch.randn(4, 3)
        y_b = layer_b(X)
        y_nb = layer_nb(X)

        assert y_b.shape[1] == comb(m + n - 1, n)
        assert y_nb.shape[1] == comb(m, n)
        assert y_nb.shape[1] < y_b.shape[1]
        assert torch.allclose(y_b.sum(dim=1), torch.ones(4), atol=1e-5)
        assert torch.allclose(y_nb.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_local_vs_remote_shape_and_norm(self, remote_processor):
        q = _make_layer(5, 2, input_size=2, trainable=True, no_bunching=True)
        X = torch.randn(3, 2)

        y_local = q(X)  # exact
        proc = MerlinProcessor(remote_processor)
        y_remote = proc.forward(q, X, shots=20000)

        assert y_local.shape == y_remote.shape
        assert torch.allclose(y_local.sum(dim=1), torch.ones(3), atol=1e-5)
        assert torch.allclose(y_remote.sum(dim=1), torch.ones(3), atol=1e-3)
