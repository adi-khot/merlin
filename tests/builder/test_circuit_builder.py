from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

# Load helper without importing the top-level merlin package
_HELPERS_PATH = Path(__file__).resolve().parents[1] / "helpers.py"
_SPEC = importlib.util.spec_from_file_location("_merlin_test_helpers", _HELPERS_PATH)
_HELPERS_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("_merlin_test_helpers", _HELPERS_MODULE)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_HELPERS_MODULE)
load_merlin_module = _HELPERS_MODULE.load_merlin_module

# Ensure Perceval persistent data is routed to a writable location
_PCVL_HOME = Path(__file__).resolve().parents[2] / ".pcvl_home"
(_PCVL_HOME / "Library" / "Application Support").mkdir(parents=True, exist_ok=True)
os.environ["HOME"] = str(_PCVL_HOME)

# Import perceval after HOME override so persistent data is created inside the project tree
import perceval as pcvl

circuit_mod = load_merlin_module("merlin.core.circuit")
components_mod = load_merlin_module("merlin.core.components")
circuit_builder_mod = load_merlin_module("merlin.builder.circuit_builder")

CircuitBuilder = circuit_builder_mod.CircuitBuilder
Rotation = components_mod.Rotation
BeamSplitter = components_mod.BeamSplitter
EntanglingBlock = components_mod.EntanglingBlock
ParameterRole = components_mod.ParameterRole

_PS_TYPE = type(pcvl.PS(0.0))
_BS_TYPE = type(pcvl.BS())


def test_rotation_layer_assigns_trainable_names_per_mode():
    builder = CircuitBuilder(n_modes=3)
    builder.add_rotation_layer(trainable=True)

    rotations = builder.circuit.components
    assert len(rotations) == 3

    for idx, rotation in enumerate(rotations):
        assert isinstance(rotation, Rotation)
        assert rotation.role == ParameterRole.TRAINABLE
        assert rotation.target == idx
        assert rotation.custom_name == f"theta_{idx}_{idx}"
        assert rotation.axis == "z"


def test_rotation_layer_input_custom_prefix_uses_global_counter():
    builder = CircuitBuilder(n_modes=4)
    builder.add_rotation_layer(modes=[1, 3], role=ParameterRole.INPUT, name="feature")

    rotations = builder.circuit.components
    assert [rotation.target for rotation in rotations] == [1, 3]
    assert [rotation.custom_name for rotation in rotations] == ["feature1", "feature2"]
    assert all(rotation.role == ParameterRole.INPUT for rotation in rotations)


def test_section_reference_copies_components_without_sharing_trainables():
    builder = CircuitBuilder(n_modes=2)
    builder.add_rotation_layer(trainable=True, name="theta")

    builder.begin_section("first")
    builder.add_beam_splitter(
        targets=(0, 1),
        theta=0.25,
        trainable_theta=True,
        name="bs",
    )
    builder.end_section()

    builder.begin_section("second", reference="first", share_trainable=False)

    original, cloned = builder.circuit.components[-2:]
    assert isinstance(original, BeamSplitter)
    assert isinstance(cloned, BeamSplitter)
    assert cloned is not original
    assert cloned.theta_role == ParameterRole.TRAINABLE
    assert cloned.theta_name == "bs_theta_copy0"
    assert cloned.theta_value == pytest.approx(original.theta_value)


def test_build_closes_open_sections_and_sets_metadata():
    builder = CircuitBuilder(n_modes=1, n_photons=2)
    builder.begin_section("encoder", compute_adjoint=True)
    builder.add_rotation(target=0)

    with pytest.warns(UserWarning):
        circuit = builder.build()

    sections = circuit.metadata["sections"]
    assert len(sections) == 1
    section = sections[0]
    assert section["name"] == "encoder"
    assert section["compute_adjoint"] is True
    assert section["start_idx"] == 0
    assert section["end_idx"] == len(circuit.components)

    assert circuit.metadata["n_photons"] == 2


def test_complex_builder_pipeline_exports_pcvl_circuit():
    builder = CircuitBuilder(n_modes=3, n_photons=1)
    builder.add_input_encoding(name="input")
    builder.add_entangling_layer(depth=1, name="ent")
    builder.add_rotation(target=1, angle=0.25)

    pcvl_circuit = builder.to_pcvl_circuit(pcvl)
    assert isinstance(pcvl_circuit, pcvl.Circuit)
    assert pcvl_circuit.m == 3

    ps_ops = [gate for _, gate in pcvl_circuit._components if isinstance(gate, _PS_TYPE)]
    assert ps_ops, "Input encoding should add phase shifters"
    assert any(param.name.startswith("input") for gate in ps_ops for param in gate.get_parameters())

    entangling_ops = [gate for _, gate in pcvl_circuit._components if isinstance(gate, _BS_TYPE)]
    assert entangling_ops, "Entangling layer should contribute beam splitters"


def test_to_pcvl_circuit_supports_gradient_backpropagation():
    builder = CircuitBuilder(n_modes=2, n_photons=1)
    builder.add_trainable_layer(name="theta")
    builder.add_beam_splitter(
        targets=(0, 1),
        trainable_theta=True,
        trainable_phi=True,
    )
    builder.add_entangling_layer(depth=1)

    pcvl_circuit = builder.to_pcvl_circuit(pcvl)

    converter_mod = load_merlin_module("merlin.pcvl_pytorch.locirc_to_tensor")
    CircuitConverter = converter_mod.CircuitConverter
    converter = CircuitConverter(pcvl_circuit, input_specs=["theta", "phi"])

    total_params = sum(len(params) for params in converter.spec_mappings.values())
    assert total_params == 4

    theta_params = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32, requires_grad=True)
    phi_params = torch.tensor([0.4], dtype=torch.float32, requires_grad=True)

    unitary = converter.to_tensor(theta_params, phi_params)
    loss = (unitary.real.pow(2) + unitary.imag.pow(2)).sum()
    loss.backward()

    assert theta_params.grad is not None
    assert phi_params.grad is not None
    assert torch.all(torch.isfinite(theta_params.grad))
    assert torch.all(torch.isfinite(phi_params.grad))
    assert theta_params.grad.norm() > 0
    assert phi_params.grad.norm() > 0
