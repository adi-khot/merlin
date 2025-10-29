from types import MethodType
import warnings

import perceval as pcvl
import pytest
import torch

from merlin.algorithms.layer import QuantumLayer
from merlin.sampling.strategies import OutputMappingStrategy


@pytest.fixture
def make_layer():
    def _make(**overrides):
        circuit = pcvl.components.GenericInterferometer(
            3,
            pcvl.components.catalog["mzi phase last"].generate,
            shape=pcvl.InterferometerShape.RECTANGLE,
        )
        params = {
            "input_size": 0,
            "circuit": circuit,
            "n_photons": 1,
            "output_mapping_strategy": OutputMappingStrategy.NONE,
            "trainable_parameters": ["phi"],
            "input_parameters": [],
            "dtype": torch.float64,
            "amplitude_encoding": True,
            "computation_space": "no_bunching",
        }
        params.update(overrides)
        return QuantumLayer(**params)

    return _make


def test_amplitude_encoding_matches_superposition(make_layer):
    layer = make_layer()
    num_states = len(layer.computation_process.simulation_graph.mapped_keys)
    raw_amplitude = torch.arange(1, num_states + 1, dtype=torch.float64)

    prepared_state = layer._validate_amplitude_input(raw_amplitude)
    layer.set_input_state(prepared_state)
    params = layer.prepare_parameters([])
    expected = layer.computation_process.compute_superposition_state(params)

    _, amplitudes = layer(raw_amplitude, return_amplitudes=True)

    assert torch.allclose(amplitudes, expected, rtol=1e-12, atol=1e-14)


def test_amplitude_encoding_batches_use_vectorised_kernel(make_layer):
    layer = make_layer()
    process = layer.computation_process

    call_tracker = {"ebs": 0, "super": 0}
    original_ebs = process.compute_ebs_simultaneously
    original_super = process.compute_superposition_state

    def tracked_ebs(self, parameters, simultaneous_processes=1):
        call_tracker["ebs"] += 1
        return original_ebs(
            parameters, simultaneous_processes=simultaneous_processes
        )

    def tracked_super(self, parameters):
        call_tracker["super"] += 1
        return original_super(parameters)

    process.compute_ebs_simultaneously = MethodType(tracked_ebs, process)
    process.compute_superposition_state = MethodType(tracked_super, process)

    num_states = len(process.simulation_graph.mapped_keys)
    batched_state = torch.rand(3, num_states, dtype=torch.float64)

    layer(batched_state)

    assert call_tracker["ebs"] == 1
    assert call_tracker["super"] == 0


def test_amplitude_encoding_requires_first_argument(make_layer):
    layer = make_layer()
    with pytest.raises(ValueError, match="expects an amplitude tensor input"):
        layer()


def test_amplitude_encoding_validates_dimension(make_layer):
    layer = make_layer()
    num_states = len(layer.computation_process.simulation_graph.mapped_keys)
    invalid = torch.rand(num_states + 1, dtype=torch.float64)

    with pytest.raises(ValueError, match="Amplitude input expects"):
        layer(invalid)


def test_computation_space_selector(make_layer):
    layer_fock = make_layer(amplitude_encoding=False, computation_space="fock")
    assert layer_fock.computation_space == "fock"
    assert layer_fock.no_bunching is False

    layer_nb = make_layer(amplitude_encoding=False, computation_space="no_bunching")
    assert layer_nb.computation_space == "no_bunching"
    assert layer_nb.no_bunching is True

    with pytest.raises(ValueError):
        make_layer(amplitude_encoding=False, computation_space="invalid")

    with pytest.warns(
        UserWarning,
        match="Overriding 'no_bunching' to match the requested computation_space",
    ):
        layer_override = make_layer(
            amplitude_encoding=False,
            computation_space="fock",
            no_bunching=True,
        )
    assert layer_override.computation_space == "fock"
    assert layer_override.no_bunching is False


def test_computation_space_consistency_no_warning(make_layer):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        layer = make_layer(
            amplitude_encoding=False,
            computation_space="no_bunching",
            no_bunching=True,
        )

    assert layer.no_bunching is True
    assert layer.computation_space == "no_bunching"
    assert caught == []
