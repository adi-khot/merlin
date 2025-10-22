"""
Introspection & execution-policy tests for MerlinProcessor <-> QuantumLayer.

Focus:
- Offload by default when not forced
- Forced local simulation (no offload) and correct shape
- Mixed nn.Sequential with one offloaded, one local
- Temporary override via context manager as_simulation()

Cloud use is optional via `remote_processor` fixture (auto-skips if no token).
"""

import time
import pytest
import torch
import torch.nn as nn

from merlin.core.merlin_processor import MerlinProcessor
from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.sampling.strategies import OutputMappingStrategy


def _make_layer(n_modes: int, n_photons: int, input_size: int, no_bunching: bool = True) -> QuantumLayer:
    builder = CircuitBuilder(n_modes=n_modes)
    builder.add_rotations(trainable=True, name="theta")
    builder.add_angle_encoding(modes=list(range(input_size)), name="px")
    if n_modes >= 3:
        builder.add_entangling_layer()

    return QuantumLayer(
        input_size=input_size,
        output_size=None,   # raw distribution
        builder=builder,
        n_photons=n_photons,
        no_bunching=no_bunching,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    ).eval()


def _wait(fut, timeout_s=30.0):
    end = time.time() + timeout_s
    while not fut.done():
        if time.time() >= end:
            raise TimeoutError("Timeout waiting for Merlin future")
        time.sleep(0.01)
    return fut.value()


class TestIntrospectionAndPolicy:
    def test_offloads_by_default(self, remote_processor):
        B = 3
        layer = _make_layer(5, 2, 2, True)
        x = torch.rand(B, 2)

        # discover output size via local call
        dist_size = layer(x).shape[1]

        proc = MerlinProcessor(remote_processor)
        fut = proc.forward_async(layer, x, shots=2000)
        y = _wait(fut)

        assert y.shape == (B, dist_size)
        assert hasattr(fut, "job_ids") and len(fut.job_ids) >= 1

    def test_force_simulation_executes_locally(self, remote_processor, monkeypatch):
        B = 3
        layer = _make_layer(5, 2, 2, True)
        layer.force_simulation = True

        x = torch.rand(B, 2)
        dist_size = layer(x).shape[1]

        called = {"flag": False}
        orig_forward = layer.forward

        def spy_forward(*args, **kwargs):
            called["flag"] = True
            return orig_forward(*args, **kwargs)

        monkeypatch.setattr(layer, "forward", spy_forward, raising=True)

        proc = MerlinProcessor(remote_processor)
        fut = proc.forward_async(layer, x, shots=2000)
        y = _wait(fut)

        assert y.shape == (B, dist_size)
        assert len(fut.job_ids) == 0, "Should not offload when force_simulation=True"
        assert called["flag"], "Local layer.forward must be called"

    def test_mixed_sequential_one_offload_one_local(self, remote_processor):
        B = 2
        q1 = _make_layer(5, 2, 2, True).eval()   # offloadable
        q2 = _make_layer(6, 2, 3, True).eval()   # force local
        q2.force_simulation = True

        # Probe sizes
        dist1 = q1(torch.rand(B, 2)).shape[1]
        dist2 = q2(torch.rand(B, 3)).shape[1]

        adapter = nn.Linear(dist1, 3, bias=False)
        model = nn.Sequential(q1, adapter, q2).eval()

        proc = MerlinProcessor(remote_processor)
        fut = proc.forward_async(model, torch.rand(B, 2), shots=3000)
        y = _wait(fut)

        assert y.shape == (B, dist2)
        assert len(fut.job_ids) == 1, "Exactly one quantum layer should be offloaded"

    def test_context_override_temporarily(self, remote_processor):
        B = 2
        q = _make_layer(5, 2, 2, True).eval()
        x = torch.rand(B, 2)
        dist = q(x).shape[1]

        proc = MerlinProcessor(remote_processor)

        with q.as_simulation():
            fut_local = proc.forward_async(q, x, shots=2000)
            y_local = _wait(fut_local)
            assert y_local.shape == (B, dist)
            assert len(fut_local.job_ids) == 0

        fut_remote = proc.forward_async(q, x, shots=2000)
        y_remote = _wait(fut_remote)
        assert y_remote.shape == (B, dist)
        assert len(fut_remote.job_ids) >= 1
