# MIT License
#
# Copyright (c)
#
# Tests for FeedForwardBlock2 experimental API.

import math

import perceval as pcvl
import torch
from perceval.components import PERM

from merlin.algorithms.feed_forward import FeedForwardBlock2


def _build_balanced_feedforward_experiment():
    """Construct a small experiment with one detector and a feed-forward provider."""
    exp = pcvl.Experiment()
    root = pcvl.Circuit(3)
    root.add(0, pcvl.BS())
    exp.add(0, root)

    exp.add(0, pcvl.Detector.pnr())

    reflective = pcvl.Circuit(2)
    reflective.add(0, PERM([1, 0]))

    transmissive = pcvl.Circuit(2)
    transmissive.add(0, pcvl.BS())

    provider = pcvl.FFCircuitProvider(1, 0, reflective)
    provider.add_configuration([1], transmissive)
    exp.add(0, provider)
    return exp


def _build_two_stage_experiment():
    """Approximate the multi-level experiment from ff_perceval.py."""
    exp = pcvl.Experiment()
    root = pcvl.Circuit(4)
    root.add(0, pcvl.BS())
    exp.add(0, root)

    exp.add(0, pcvl.Detector.pnr())
    v0 = pcvl.Circuit(3) // pcvl.BS()
    v1 = pcvl.Circuit(3) // pcvl.BS()
    v2 = pcvl.Circuit(3) // pcvl.BS()
    provider1 = pcvl.FFCircuitProvider(1, 0, v0)
    provider1.add_configuration([1], v1)
    provider1.add_configuration([2], v2)
    exp.add(0, provider1)

    exp.add(3, pcvl.Detector.threshold())
    provider2 = pcvl.FFCircuitProvider(1, -1, pcvl.Circuit(2))
    provider2.add_configuration([1], pcvl.Unitary(pcvl.Matrix.random_unitary(2)))
    exp.add(3, provider2)

    for mode in (1, 2):
        exp.add(mode, pcvl.Detector.pnr())

    return exp


def test_feedforward_block2_balanced_split():
    exp = _build_balanced_feedforward_experiment()
    block = FeedForwardBlock2(
        exp,
        input_state=[2, 0, 0],
    )

    x = torch.zeros((1, 0))
    outputs = block(x)

    total_prob = torch.zeros(1)
    measurement_probs = {}
    for key, distribution in outputs.items():
        prob = distribution.sum(dim=-1)
        total_prob = total_prob + prob
        measured_value = next(v for v in key if v is not None)
        measurement_probs[measured_value] = prob.item()

    assert torch.allclose(total_prob, torch.ones_like(total_prob), atol=1e-5)
    assert math.isclose(sum(measurement_probs.values()), 1.0, rel_tol=1e-5)
    assert len(measurement_probs) == 3


def test_feedforward_block2_parses_multiple_stages():
    exp = _build_two_stage_experiment()
    block = FeedForwardBlock2(exp, input_state=[1, 1, 0, 0])

    assert len(block.stages) == 2
    assert block.stages[0].measured_modes == (0,)
    assert block.stages[1].measured_modes == (3,)
    desc = block.describe()
    assert "Stage 1" in desc and "Stage 2" in desc
