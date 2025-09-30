from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_HELPERS_PATH = Path(__file__).resolve().parents[1] / "helpers.py"
_SPEC = importlib.util.spec_from_file_location("_merlin_test_helpers", _HELPERS_PATH)
_HELPERS_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("_merlin_test_helpers", _HELPERS_MODULE)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_HELPERS_MODULE)
load_merlin_module = _HELPERS_MODULE.load_merlin_module

components_mod = load_merlin_module("merlin.core.components")

Rotation = components_mod.Rotation
BeamSplitter = components_mod.BeamSplitter
EntanglingBlock = components_mod.EntanglingBlock
Measurement = components_mod.Measurement
ParameterRole = components_mod.ParameterRole
adapt_old_rotation = components_mod.adapt_old_rotation


def test_rotation_get_params_for_trainable_custom_name():
    rotation = Rotation(target=0, role=ParameterRole.TRAINABLE, custom_name="theta")
    assert rotation.get_params() == {"theta": None}


def test_rotation_get_params_for_fixed_returns_empty():
    rotation = Rotation(target=1, role=ParameterRole.FIXED, custom_name="phi", value=0.5)
    assert rotation.get_params() == {}


def test_beam_splitter_get_params_exposes_non_fixed_names():
    beam_splitter = BeamSplitter(
        targets=(0, 1),
        theta_role=ParameterRole.TRAINABLE,
        theta_name="theta",
        phi_role=ParameterRole.INPUT,
        phi_name="phi",
    )
    assert beam_splitter.get_params() == {"theta": None, "phi": None}


def test_adapt_old_rotation_translates_legacy_flags():
    trainable = adapt_old_rotation(target=2, angle="theta0", trainable=True)
    assert trainable.role == ParameterRole.TRAINABLE
    assert trainable.custom_name == "theta0"

    input_rot = adapt_old_rotation(target=3, value=0.2, as_input=True)
    assert input_rot.role == ParameterRole.INPUT
    assert input_rot.value == pytest.approx(0.2)


def test_measurement_defaults_are_preserved():
    measurement = Measurement(targets=[0, 1])
    assert measurement.targets == [0, 1]
    assert measurement.basis == "computational"
    assert measurement.get_params() == {}


def test_entangling_block_exposes_no_parameters():
    block = EntanglingBlock(targets=[0, 1], depth=2, trainable=True)
    assert block.get_params() == {}
