from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from merlin import Probabilities, QuantumLayer
from merlin.utils.grouping import ModGrouping

_PCVL_HOME = Path(__file__).resolve().parents[2] / ".pcvl_home"
(
    _PCVL_HOME / "Library" / "Application Support" / "perceval-quandela" / "job_group"
).mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(_PCVL_HOME)


@pytest.fixture(autouse=True)
def perceval_home(monkeypatch):
    monkeypatch.setenv("HOME", str(_PCVL_HOME))


@pytest.fixture
def quantum_layer_api():
    return QuantumLayer


def _unwrap(layer: nn.Module) -> QuantumLayer:
    if hasattr(layer, "quantum_layer"):
        return layer.quantum_layer  # type: ignore[attr-defined]
    if isinstance(layer, nn.Sequential) and len(layer) > 0:
        return _unwrap(layer[0])
    return layer  # type: ignore[return-value]


def test_none_strategy_without_output_size(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
        dtype=torch.float32,
    )
    base = _unwrap(layer)

    x = torch.rand(4, 3)
    output = layer(x)

    assert output.shape == (4, base.output_size)
    assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-5)


def test_none_strategy_with_matching_output_size(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    reference_layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
    )
    dist_size = _unwrap(reference_layer).output_size

    layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
        output_size=dist_size,
    )

    base = _unwrap(layer)
    x = torch.rand(2, 3)
    output = layer(x)
    assert output.shape == (2, dist_size)
    assert base.output_size == dist_size


def test_simple_groups_output_when_requested(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    target_size = 16
    layer = QuantumLayer.simple(
        input_size=3,
        n_params=40,
        output_size=target_size,
    )

    assert hasattr(layer, "post_processing")
    assert isinstance(layer.post_processing, ModGrouping)
    base = _unwrap(layer)
    x = torch.rand(4, 3)
    output = layer(x)
    assert base.output_size != target_size
    assert output.shape == (4, target_size)
    assert torch.allclose(output.sum(dim=1), torch.ones(4), atol=1e-5)


def test_linear_strategy_creates_linear_mapping(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
    )
    model = nn.Sequential(layer, nn.Linear(layer.output_size, 5))

    base = _unwrap(layer)
    assert isinstance(base.measurement_mapping, Probabilities)
    x = torch.rand(6, 3)
    output = model(x)
    assert output.shape == (6, 5)


def test_default_strategy_is_none(quantum_layer_api):
    QuantumLayer = quantum_layer_api
    sig = inspect.signature(QuantumLayer.simple)
    assert "measurement_strategy" not in sig.parameters


def test_simple_signature_does_not_include_reservoir_mode(quantum_layer_api):
    QuantumLayer = quantum_layer_api
    sig = inspect.signature(QuantumLayer.simple)
    assert "reservoir_mode" not in sig.parameters


def test_trainable_parameter_budget_matches_request(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    requested_params = 37
    with pytest.warns(RuntimeWarning):
        layer = QuantumLayer.simple(
            input_size=3,
            n_params=requested_params,
        )

    base = _unwrap(layer)
    mzi_param_count = sum(
        param.numel()
        for name, param in base.named_parameters()
        if name.startswith("mzi_extra")
    )
    interferometer_param_count = sum(
        param.numel()
        for name, param in base.named_parameters()
        if name.startswith("gi_simple")
    )

    total_trainable = mzi_param_count + interferometer_param_count
    expected_total = max(requested_params, interferometer_param_count)
    expected_mzi = max(requested_params - interferometer_param_count, 0)

    assert total_trainable == expected_total
    assert mzi_param_count == expected_mzi


def test_simple_rejects_odd_mzi_budget(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    with pytest.raises(ValueError, match="Additional parameter budget must be even"):
        QuantumLayer.simple(
            input_size=3,
            n_params=95,
        )


def test_simple_allocates_full_mzi_pairs(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    requested_params = 100  # 10 above the base GI budget (90)
    layer = QuantumLayer.simple(
        input_size=3,
        n_params=requested_params,
    )

    base = _unwrap(layer)
    gi_params = sum(
        param.numel()
        for name, param in base.named_parameters()
        if name.startswith("gi_simple")
    )
    mzi_params = sum(
        param.numel()
        for name, param in base.named_parameters()
        if name.startswith("mzi_extra")
    )

    assert gi_params == 90
    assert mzi_params == requested_params - gi_params == 10
    # Every MZI exposes both inner and outer phases
    mzi_roles: dict[str, set[str]] = {}
    for name, _ in base.named_parameters():
        if name.startswith("mzi_extra"):
            if "_li" in name:
                prefix = name.split("_li", 1)[0]
                role = "li"
            elif "_lo" in name:
                prefix = name.split("_lo", 1)[0]
                role = "lo"
            else:
                continue
            mzi_roles.setdefault(prefix, set()).add(role)
    assert all({"li", "lo"}.issubset(roles) for roles in mzi_roles.values())


def test_gradient_flow_for_strategies(quantum_layer_api):
    QuantumLayer = quantum_layer_api
    nb_params = 40

    layer = QuantumLayer.simple(
        input_size=3,
        n_params=nb_params,
    )
    model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 4))

    x = torch.rand(8, 3, requires_grad=True)
    loss = model(x).sum()
    loss.backward()
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()
    )

    layer_none = QuantumLayer.simple(
        input_size=3,
        n_params=nb_params,
    )

    x = torch.rand(8, 3, requires_grad=True)
    loss = layer_none(x).sum()
    loss.backward()
    base_none = _unwrap(layer_none)
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in base_none.parameters()
    )
    mzi_param_count = sum(
        param.numel()
        for name, param in base_none.named_parameters()
        if name.startswith("mzi_extra")
    )
    interferometer_param_count = sum(
        param.numel()
        for name, param in base_none.named_parameters()
        if name.startswith("gi_simple")
    )

    total_trainable = mzi_param_count + interferometer_param_count
    expected_total = max(nb_params, interferometer_param_count)
    expected_mzi = max(nb_params - interferometer_param_count, 0)

    assert total_trainable == expected_total
    assert mzi_param_count == expected_mzi


def test_quantum_layer_simple_raises_when_input_exceeds_modes(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    with pytest.raises(
        ValueError, match="You cannot encore more features than mode with Builder"
    ):
        QuantumLayer.simple(
            input_size=12,
            n_params=30,
        )


def test_batch_shapes_and_probabilities(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    layer = QuantumLayer.simple(
        input_size=4,
        n_params=80,
    )

    for batch_size in [1, 5, 16]:
        x = torch.rand(batch_size, 4)
        output = layer(x)
        assert output.shape == (batch_size, layer.output_size)
        assert torch.allclose(output.sum(dim=1), torch.ones(batch_size), atol=1e-5)
        assert torch.all(output >= 0)


def test_dtype_propagation(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    for dtype in (torch.float32, torch.float64):
        layer = QuantumLayer.simple(
            input_size=3,
            n_params=60,
            dtype=dtype,
        )

        base = _unwrap(layer)
        for param in base.parameters():
            assert param.dtype == dtype

        x = torch.rand(2, 3, dtype=dtype)
        output = layer(x)
        assert output.dtype == dtype


def test_circuit_and_output_size_access(quantum_layer_api):
    QuantumLayer = quantum_layer_api

    simple_layer = QuantumLayer.simple(
        input_size=3,
        n_params=60,
    )
    circuit = simple_layer.circuit
    output_size = simple_layer.output_size

    assert circuit == simple_layer.quantum_layer.circuit
    assert output_size == simple_layer.quantum_layer.output_size

    simple_layer = QuantumLayer.simple(input_size=3, n_params=60, output_size=3)
    output_size = simple_layer.output_size
    assert output_size == 3
    assert output_size != simple_layer.quantum_layer.output_size
    assert output_size == simple_layer.post_processing.output_size
