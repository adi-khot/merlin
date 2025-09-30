from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

_HELPERS_PATH = Path(__file__).resolve().parents[1] / "helpers.py"
_SPEC = importlib.util.spec_from_file_location("_merlin_test_helpers", _HELPERS_PATH)
_HELPERS_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("_merlin_test_helpers", _HELPERS_MODULE)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_HELPERS_MODULE)
load_merlin_module = _HELPERS_MODULE.load_merlin_module


def _ensure_perceval_home(monkeypatch):
    home_dir = Path(__file__).resolve().parents[2] / ".pcvl_home"
    (home_dir / "Library" / "Application Support").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home_dir))


@pytest.fixture(autouse=True)
def perceval_home(monkeypatch):
    _ensure_perceval_home(monkeypatch)


@pytest.fixture
def iris_batch():
    load_merlin_module("merlin.datasets")
    iris_mod = load_merlin_module("merlin.datasets.iris")
    features, labels, _ = iris_mod.get_data_train()
    x = torch.tensor(features[:16], dtype=torch.float32)
    y = torch.tensor(labels[:16], dtype=torch.long)
    return x, y


def _load_core_modules():
    layer_mod = load_merlin_module("merlin.algorithms.layer")
    strategies_mod = load_merlin_module("merlin.sampling.strategies")
    builder_mod = load_merlin_module("merlin.builder.circuit_builder")
    return (
        layer_mod.QuantumLayer,
        strategies_mod.OutputMappingStrategy,
        builder_mod.CircuitBuilder,
    )


def _check_training_step(layer, inputs, targets):
    layer.train()
    layer.zero_grad()
    logits = layer(inputs)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    grads = [p.grad for p in layer.parameters() if p.requires_grad]
    assert logits.shape == (inputs.shape[0], 3)
    assert torch.isfinite(loss)
    assert any(g is not None for g in grads)
    assert all(torch.all(torch.isfinite(g)) for g in grads if g is not None)


def test_builder_api_pipeline_on_iris(iris_batch):
    features, labels = iris_batch
    QuantumLayer, OutputMappingStrategy, CircuitBuilder = _load_core_modules()
    import perceval as pcvl

    builder = CircuitBuilder(n_modes=10, n_photons=5)
    builder.add_entangling_layer(depth=1)
    builder.add_input_encoding(modes=list(range(features.shape[1])), name="input")
    builder.add_trainable_layer(name="theta")
    builder.add_entangling_layer(depth=1)

    pcvl_circuit = builder.to_pcvl_circuit(pcvl)

    layer = QuantumLayer(
        input_size=features.shape[1],
        circuit=pcvl_circuit,
        n_photons=5,
        trainable_parameters=["theta"],
        input_parameters=["input"],
        output_size=3,
        output_mapping_strategy=OutputMappingStrategy.LINEAR,
        dtype=features.dtype,
    )

    _check_training_step(layer, features, labels)


def test_simple_api_pipeline_on_iris(iris_batch):
    features, labels = iris_batch
    QuantumLayer, OutputMappingStrategy, _ = _load_core_modules()

    layer = QuantumLayer.simple(
        input_size=features.shape[1],
        n_params=40,
        output_size=3,
        output_mapping_strategy=OutputMappingStrategy.LINEAR,
        dtype=features.dtype,
    )

    _check_training_step(layer, features, labels)


def test_manual_pcvl_circuit_pipeline_on_iris(iris_batch):
    features, labels = iris_batch
    QuantumLayer, OutputMappingStrategy, _ = _load_core_modules()
    import perceval as pcvl

    circuit = pcvl.Circuit(4)
    for mode in range(4):
        circuit.add(mode, pcvl.PS(pcvl.P(f"input{mode}")))
    circuit.add((0, 1), pcvl.BS(theta=pcvl.P("theta0")))
    circuit.add((2, 3), pcvl.BS(theta=pcvl.P("theta1")))
    circuit.add((1, 2), pcvl.BS(theta=pcvl.P("theta2")))

    layer = QuantumLayer(
        input_size=features.shape[1],
        circuit=circuit,
        n_photons=1,
        trainable_parameters=["theta"],
        input_parameters=["input"],
        output_size=3,
        output_mapping_strategy=OutputMappingStrategy.LINEAR,
        dtype=features.dtype,
    )

    _check_training_step(layer, features, labels)
