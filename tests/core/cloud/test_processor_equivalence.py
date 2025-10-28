"""
Probability consistency & gradient-free tuning tests for MerlinProcessor.

Covers:
- Local vs remote probability correspondence (no_bunching and bunching)
- Mixed classical/quantum workflows preserve classical ops locally and
  offload quantum ops (and can be flipped to local with force_simulation)
- COBYLA optimization loop that mutates QuantumLayer params and evaluates
  the objective via MerlinProcessor.forward (remote if 'probs' allowed,
  else high-nsample sampling)

All tests that require cloud auto-skip via the `remote_processor` fixture.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.core.merlin_processor import MerlinProcessor
from merlin.sampling.strategies import OutputMappingStrategy

# -------------------------
# Utilities
# -------------------------


def _make_layer(m: int, n: int, input_size: int, no_bunching: bool) -> QuantumLayer:
    builder = CircuitBuilder(n_modes=m)
    # Keep the layer small-ish but non-trivial
    builder.add_rotations(trainable=True, name="theta")
    if m >= 3:
        builder.add_entangling_layer()
    builder.add_angle_encoding(modes=list(range(input_size)), name="px")

    return QuantumLayer(
        input_size=input_size,
        output_size=None,  # raw distribution
        builder=builder,
        n_photons=n,
        no_bunching=no_bunching,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    ).eval()


def _l1_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # shape: (B, D) -> (B,)
    return (a - b).abs().sum(dim=1)


def _spin_until(pred, timeout_s: float = 10.0, sleep_s: float = 0.02) -> bool:
    import time as _t
    start = _t.time()
    while not pred():
        if _t.time() - start > timeout_s:
            return False
        _t.sleep(sleep_s)
    return True


# -------------------------
# Probability correspondence (no_bunching / bunching)
# -------------------------

@pytest.mark.parametrize("no_bunching, m, n, input_size, tol_l1", [
    (True, 6, 2, 2, 0.12),   # C(6,2)=15
    (False, 5, 3, 3, 0.15),   # C(7,3)=35
])
def test_prob_consistency_local_vs_remote(remote_processor, no_bunching, m, n, input_size, tol_l1):
    """
    Compare local exact distribution to MerlinProcessor remote output.
    If the backend supports 'probs', use it directly; else use high nsample
    to approximate (looser tolerance).
    """
    layer = _make_layer(m, n, input_size, no_bunching)
    bsz = 3
    X = torch.rand(bsz, input_size)

    # Local "ground truth"
    y_local = layer(X)  # exact probs
    assert y_local.shape[0] == bsz
    assert torch.allclose(y_local.sum(dim=1), torch.ones(bsz), atol=1e-5)

    proc = MerlinProcessor(remote_processor)
    # Decide evaluation mode:
    use_probs = "probs" in getattr(proc, "available_commands", [])
    if use_probs:
        # We rely on exact probs (fast & precise)
        y_remote = proc.forward(layer, X, nsample=None)
    else:
        # Sampling: push nsample high to shrink variance
        # (these values are a trade-off for CI/runtime)
        y_remote = proc.forward(layer, X, nsample=200_000)

    assert y_remote.shape == y_local.shape
    # Normalization is looser if sampling
    atol_norm = 1e-5 if use_probs else 2e-2
    assert torch.allclose(y_remote.sum(dim=1), torch.ones(bsz), atol=atol_norm)

    # Distribution similarity (L1 per sample)
    l1 = _l1_dist(y_local, y_remote)
    # Allow looser tolerance for sampling backends
    assert torch.all(l1 <= tol_l1), f"L1 distances too high: {l1.tolist()}"

# -------------------------
# Workflow preservation (classical/quantum mix)
# -------------------------


def test_workflow_preserves_classical_layers_and_offloads_quantum(remote_processor):
    """
    Build a small pipeline: Linear -> ReLU -> Quantum -> Linear -> Softmax.
    Check that:
      - quantum layer is offloaded (>=1 job id)
      - classical transforms are applied locally around it
      - forcing local execution yields similar results (within tolerance)
    """
    # Quantum core
    q = _make_layer(m=5, n=2, input_size=2, no_bunching=True).eval()
    # Probe output size
    dist1 = q(torch.rand(2, 2)).shape[1]

    model_remote = nn.Sequential(
        nn.Linear(3, 2, bias=False),
        nn.ReLU(),
        q,
        nn.Linear(dist1, 4, bias=False),
        nn.Softmax(dim=-1),
    ).eval()

    X = torch.rand(4, 3)

    proc = MerlinProcessor(remote_processor)
    fut = proc.forward_async(model_remote, X, nsample=5000)
    _spin_until(lambda: len(fut.job_ids) > 0 or fut.done(), timeout_s=12.0)
    Y_remote = fut.wait()
    assert Y_remote.shape == (4, 4)
    assert len(fut.job_ids) >= 1  # quantum offload happened

    # Now force local quantum execution and compare
    q.force_simulation = True
    fut_local = proc.forward_async(model_remote, X, nsample=5000)
    Y_local = fut_local.wait()
    assert Y_local.shape == (4, 4)

    # Sampling noise -> lenient tolerance
    assert torch.allclose(Y_local.sum(dim=1), torch.ones(4), atol=2e-2)
    assert torch.allclose(Y_remote.sum(dim=1), torch.ones(4), atol=2e-2)

    # Values should be "reasonably" close despite sampling noise
    # Use per-row L1 with a loose threshold
    l1 = _l1_dist(Y_local, Y_remote)
    assert torch.all(l1 <= 0.35), f"Classical/quantum pipeline outputs diverged: {l1.tolist()}"
