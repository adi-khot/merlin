from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_HELPERS_PATH = Path(__file__).resolve().parents[1] / "helpers.py"
_SPEC = importlib.util.spec_from_file_location("_merlin_test_helpers", _HELPERS_PATH)
_HELPERS_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("_merlin_test_helpers", _HELPERS_MODULE)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_HELPERS_MODULE)
load_merlin_module = _HELPERS_MODULE.load_merlin_module


@pytest.fixture(autouse=True)
def perceval_home(monkeypatch):
    home_dir = Path(__file__).resolve().parents[2] / ".pcvl_home"
    (home_dir / "Library" / "Application Support").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home_dir))


@pytest.fixture
def quantum_layer_api():
    layer_mod = load_merlin_module("merlin.algorithms.layer")
    strategies_mod = load_merlin_module("merlin.sampling.strategies")
    return layer_mod.QuantumLayer, strategies_mod.OutputMappingStrategy


def test_none_strategy_without_output_size(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
        output_mapping_strategy=OutputMappingStrategy.NONE,
        dtype=torch.float32,
    )

    x = torch.rand(4, 3)
    output = layer(x)

    assert output.shape == (4, layer.output_size)
    assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-5)


def test_none_strategy_with_matching_output_size(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    reference_layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    )
    dist_size = reference_layer.output_size

    layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
        output_size=dist_size,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    )

    x = torch.rand(2, 3)
    output = layer(x)
    assert output.shape == (2, dist_size)


def test_none_strategy_with_mismatched_output_size(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    with pytest.raises(ValueError):
        QuantumLayer.simple(
            input_size=3,
            n_params=60,
            output_size=10,
            output_mapping_strategy=OutputMappingStrategy.NONE,
        )


def test_linear_strategy_requires_output_size(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    with pytest.raises(ValueError):
        QuantumLayer.simple(
            input_size=3,
            n_params=60,
            output_mapping_strategy=OutputMappingStrategy.LINEAR,
        )


def test_linear_strategy_creates_linear_mapping(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
        output_size=5,
        output_mapping_strategy=OutputMappingStrategy.LINEAR,
    )

    assert isinstance(layer.output_mapping, nn.Linear)
    x = torch.rand(6, 3)
    output = layer(x)
    assert output.shape == (6, 5)


def test_default_strategy_is_none(quantum_layer_api):
    QuantumLayer, _ = quantum_layer_api
    sig = inspect.signature(QuantumLayer.simple)
    assert sig.parameters[
        "output_mapping_strategy"
    ].default.name.lower() == "none"


def test_trainable_parameter_budget_matches_request(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    requested_params = 37
    layer = QuantumLayer.simple(
        input_size=3,
        n_params=requested_params,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    )

    theta_param_count = sum(
        param.numel()
        for name, param in layer.named_parameters()
        if name.startswith("theta_layer")
    )
    assert theta_param_count == requested_params


def test_gradient_flow_for_strategies(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    layer_linear = QuantumLayer.simple(
        input_size=3,
        n_params=60,
        output_size=4,
        output_mapping_strategy=OutputMappingStrategy.LINEAR,
    )

    x = torch.rand(8, 3, requires_grad=True)
    loss = layer_linear(x).sum()
    loss.backward()
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in layer_linear.parameters()
    )

    layer_none = QuantumLayer.simple(
        input_size=3,
        n_params=60,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    )

    x = torch.rand(8, 3, requires_grad=True)
    loss = layer_none(x).sum()
    loss.backward()
    assert any(
        p.grad is not None and torch.any(p.grad != 0)
        for p in layer_none.parameters()
    )


def test_batch_shapes_and_probabilities(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=4,
        n_params=80,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    )

    for batch_size in [1, 5, 16]:
        x = torch.rand(batch_size, 4)
        output = layer(x)
        assert output.shape == (batch_size, layer.output_size)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.all(output >= 0)


def test_dtype_propagation(quantum_layer_api):
    QuantumLayer, OutputMappingStrategy = quantum_layer_api

    for dtype in (torch.float32, torch.float64):
        layer = QuantumLayer.simple(
            input_size=3,
            n_params=60,
            dtype=dtype,
            output_mapping_strategy=OutputMappingStrategy.NONE,
        )

        for param in layer.parameters():
            assert param.dtype == dtype

        x = torch.rand(2, 3, dtype=dtype)
        output = layer(x)
        assert output.dtype == dtype
