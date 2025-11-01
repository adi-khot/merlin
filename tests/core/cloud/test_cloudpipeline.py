# tests/core/cloud/test_cloud_policy_pipeline.py
from __future__ import annotations

import torch
import torch.nn as nn
from _helpers import make_layer

from merlin.core.merlin_processor import MerlinProcessor


def _wait(fut, timeout_s: float = 30.0):
    import time

    end = time.time() + timeout_s
    while not fut.done():
        if time.time() >= end:
            raise TimeoutError("Timeout waiting for Merlin future")
        time.sleep(0.01)
    return fut.value()


class TestPolicyAndPipeline:
    def test_offloads_by_default(self, remote_processor):
        layer = make_layer(5, 2, 2, no_bunching=True)
        x = torch.rand(3, 2)
        dist = layer(x).shape[1]

        proc = MerlinProcessor(remote_processor)
        fut = proc.forward_async(layer, x, nsample=2000)
        y = _wait(fut)

        assert y.shape == (3, dist)
        assert hasattr(fut, "job_ids") and len(fut.job_ids) >= 1

    def test_force_simulation_executes_locally(self, remote_processor, monkeypatch):
        layer = make_layer(5, 2, 2, no_bunching=True)
        layer.force_local = True
        x = torch.rand(3, 2)
        dist = layer(x).shape[1]

        called = {"flag": False}
        orig_forward = layer.forward

        def spy_forward(*args, **kwargs):
            called["flag"] = True
            return orig_forward(*args, **kwargs)

        monkeypatch.setattr(layer, "forward", spy_forward, raising=True)

        proc = MerlinProcessor(remote_processor)
        fut = proc.forward_async(layer, x, nsample=2000)
        y = _wait(fut)

        assert y.shape == (3, dist)
        assert len(fut.job_ids) == 0  # local path
        assert called["flag"]

    def test_mixed_sequential_one_offload_one_local(self, remote_processor):
        q1 = make_layer(5, 2, 2, no_bunching=True).eval()
        q2 = make_layer(6, 2, 3, no_bunching=True).eval()
        q2.force_local = True  # force local

        dist1 = q1(torch.rand(2, 2)).shape[1]
        dist2 = q2(torch.rand(2, 3)).shape[1]

        adapter = nn.Linear(dist1, 3, bias=False)
        model = nn.Sequential(q1, adapter, q2).eval()

        proc = MerlinProcessor(remote_processor)
        fut = proc.forward_async(model, torch.rand(2, 2), nsample=3000)
        y = _wait(fut)
        assert y.shape == (2, dist2)
        assert len(fut.job_ids) == 1  # exactly one offloaded layer

    def test_local_vs_remote_shape_and_norm(self, remote_processor):
        q = make_layer(5, 2, 2, no_bunching=True)
        X = torch.randn(3, 2)

        y_local = q(X)  # exact probs
        proc = MerlinProcessor(remote_processor)
        y_remote = proc.forward(q, X, nsample=20_000)

        assert y_local.shape == y_remote.shape
        assert torch.allclose(y_local.sum(dim=1), torch.ones(3), atol=1e-5)
        assert torch.allclose(y_remote.sum(dim=1), torch.ones(3), atol=1e-3)
