"""
Algorithms-level tests for amplitude-encoded QuantumLayer workflows.

These cases validate:
* Construction and execution of QuantumLayer when amplitude encoding is enabled.
* Measurement strategies applied to amplitude vectors (e.g. returning probabilities).
* The combinatorial integrity of `state_keys` for both `no_bunching` and full Fock spaces.

Keeping these checks here ensures the public algorithms facade keeps exposing
the right behaviour for amplitude-centric users without dipping into lower-level tests.
"""

import itertools
import math
import warnings
from types import MethodType

import perceval as pcvl
import pytest
import torch

from merlin.algorithms.layer import QuantumLayer
from merlin.measurement.strategies import MeasurementStrategy


@pytest.fixture
def make_layer():
    def _make(**overrides):
        circuit = pcvl.components.GenericInterferometer(
            3,
            pcvl.components.catalog["mzi phase last"].generate,
            shape=pcvl.InterferometerShape.RECTANGLE,
        )
        params = {
            "circuit": circuit,
            "n_photons": 1,
            "measurement_strategy": MeasurementStrategy.AMPLITUDES,
            "trainable_parameters": ["phi"],
            "input_parameters": [],
            "dtype": torch.float32,
            "amplitude_encoding": True,
            "computation_space": "no_bunching",
        }
        params.update(overrides)
        if not params.get("amplitude_encoding", False):
            params.setdefault("input_size", 0)
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

    amplitudes = layer(raw_amplitude)

    assert torch.allclose(amplitudes, expected, rtol=1e-6, atol=1e-8)


def test_amplitude_encoding_batches_use_vectorised_kernel(make_layer):
    layer = make_layer()
    process = layer.computation_process

    call_tracker = {"ebs": 0, "super": 0}
    original_ebs = process.compute_ebs_simultaneously
    original_super = process.compute_superposition_state

    def tracked_ebs(self, parameters, simultaneous_processes=1):
        call_tracker["ebs"] += 1
        return original_ebs(parameters, simultaneous_processes=simultaneous_processes)

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


def test_amplitude_encoding_probabilities_strategy(make_layer):
    layer = make_layer(measurement_strategy=MeasurementStrategy.PROBABILITIES)
    num_states = len(layer.computation_process.simulation_graph.mapped_keys)
    raw_amplitude = torch.arange(1, num_states + 1, dtype=torch.float32)

    prepared_state = layer._validate_amplitude_input(raw_amplitude)
    layer.set_input_state(prepared_state)
    params = layer.prepare_parameters([])
    expected_amplitudes = layer.computation_process.compute_superposition_state(params)
    expected_probabilities = expected_amplitudes.abs() ** 2

    probabilities = layer(raw_amplitude)

    assert torch.allclose(probabilities, expected_probabilities, rtol=1e-6, atol=1e-8)


def _normalised_state(n_states: int, dtype: torch.dtype) -> torch.Tensor:
    state = torch.rand(1, n_states, dtype=dtype)
    norm = state.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
    return state / norm


def _fock_keys(modes: int, n_photons: int) -> set[tuple[int, ...]]:
    keys: set[tuple[int, ...]] = set()

    def build(prefix: list[int], remaining: int, idx: int) -> None:
        if idx == modes - 1:
            keys.add(tuple(prefix + [remaining]))
            return
        for value in range(remaining + 1):
            build(prefix + [value], remaining - value, idx + 1)

    build([], n_photons, 0)
    return keys


def _no_bunching_keys(modes: int, n_photons: int) -> set[tuple[int, ...]]:
    return {
        tuple(1 if i in combo else 0 for i in range(modes))
        for combo in itertools.combinations(range(modes), n_photons)
    }


def test_mapped_keys_no_bunching_space():
    circuit = pcvl.components.GenericInterferometer(
        4,
        pcvl.components.catalog["mzi phase last"].generate,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    n_photons = 2
    expected_states = math.comb(circuit.m, n_photons)
    input_state = _normalised_state(expected_states, dtype=torch.float32)

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.PROBABILITIES,
        input_state=input_state,
        trainable_parameters=["phi"],
        input_parameters=[],
        dtype=torch.float32,
        amplitude_encoding=True,
        computation_space="no_bunching",
        no_bunching=True,
    )

    mapped_keys = layer.state_keys
    assert len(mapped_keys) == expected_states
    assert len(set(mapped_keys)) == expected_states
    assert set(mapped_keys) == _no_bunching_keys(circuit.m, n_photons)


def test_mapped_keys_fock_space():
    circuit = pcvl.components.GenericInterferometer(
        4,
        pcvl.components.catalog["mzi phase last"].generate,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    n_photons = 2
    expected_states = math.comb(circuit.m + n_photons - 1, n_photons)
    input_state = _normalised_state(expected_states, dtype=torch.float32)

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.PROBABILITIES,
        input_state=input_state,
        trainable_parameters=["phi"],
        input_parameters=[],
        dtype=torch.float32,
        amplitude_encoding=True,
        computation_space="fock",
        no_bunching=False,
    )

    mapped_keys = layer.state_keys
    assert len(mapped_keys) == expected_states
    assert len(set(mapped_keys)) == expected_states
    assert set(mapped_keys) == _fock_keys(circuit.m, n_photons)


@pytest.mark.parametrize("computation_space", ["fock", "no_bunching"])
def test_ebs_batches_group_fock_states(computation_space):
    circuit = pcvl.components.GenericInterferometer(
        4,
        pcvl.components.catalog["mzi phase last"].generate,
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    n_photons = 2

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.PROBABILITIES,
        input_state=None,
        trainable_parameters=["phi"],
        input_parameters=[],
        dtype=torch.float32,
        amplitude_encoding=True,
        computation_space=computation_space,
        no_bunching=(computation_space == "no_bunching"),
    )

    expected_states = len(layer.state_keys)
    amplitude = torch.rand(8, expected_states, dtype=torch.float32)

    process = layer.computation_process
    original_compute_batch = process.simulation_graph.compute_batch
    recorded_batches: list[list[tuple[int, ...]]] = []
    print(f"Initial recorded batches: {recorded_batches}")

    def tracked_compute_batch(unitary, batch_fock_states):
        recorded_batches.append([tuple(state) for state in batch_fock_states])
        return original_compute_batch(unitary, batch_fock_states)

    process.simulation_graph.compute_batch = tracked_compute_batch  # type: ignore[assignment]
    try:
        layer(amplitude, simultaneous_processes=8)
        print(f"Recorded batches: {recorded_batches}")
    finally:
        process.simulation_graph.compute_batch = original_compute_batch  # type: ignore[assignment]
        print(f"Finally Recorded batches: {recorded_batches}")
    expected_batches = [
        [tuple(state) for state in layer.state_keys[i : i + 8]]
        for i in range(0, expected_states, 8)
    ]
    assert recorded_batches == expected_batches


@pytest.mark.parametrize(
    ("space", "n_photons", "n_modes", "expected_size"),
    [
        ("fock", 4, 8, math.comb(4 + 8 - 1, 4)),
        ("no_bunching", 4, 8, math.comb(8, 4)),
        # TODO: support dual_rail
        # ("dual_rail", 4, 8, 2**4),
    ],
)
def test_amplitude_encoding_input_size(space, n_photons, n_modes, expected_size):
    """QuantumLayer computes the correct input size for amplitude encoding."""

    circuit = pcvl.Circuit(n_modes)

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=n_photons,
        amplitude_encoding=True,
        computation_space=space,
    )

    assert layer.input_size == expected_size
    assert len(layer.state_keys) == expected_size


def test_amplitude_encoding_requires_valid_configuration():
    """Amplitude encoding enforces required constructor constraints."""

    circuit = pcvl.Circuit(8)

    with pytest.raises(ValueError, match="n_photons must be provided"):
        QuantumLayer(
            circuit=circuit,
            n_photons=None,
            amplitude_encoding=True,
        )

    with pytest.raises(ValueError, match="Amplitude encoding cannot be combined"):
        QuantumLayer(
            circuit=circuit,
            n_photons=4,
            input_parameters=["theta"],
            amplitude_encoding=True,
        )


"""def test_amplitude_encoding_superposition_matches_basis_sum():
    #The amplitudes are a weighted sum over basis-state simulations.

    n_photons = 4
    n_modes = 8
    circuit = pcvl.Circuit(n_modes)
    for k in range(0, n_modes, 2):
        for mode in range(k%2, n_modes, 2):
            circuit.add(mode, pcvl.BS())

    layer = QuantumLayer(
        circuit=circuit,
        input_size = 0,
        n_photons=n_photons,
        measurement_strategy=MeasurementStrategy.AMPLITUDES,
        amplitude_encoding=True,
        computation_space="dual_rail",
    )

    basis_indices = [0, 1, 2]
    basis_vectors = torch.eye(layer.input_size, dtype=torch.complex64)[basis_indices]
    basis_outputs = torch.stack([layer(state) for state in basis_vectors])

    coefficients = torch.tensor(
        [0.6 + 0.2j, -0.3 + 0.5j, 0.1 - 0.4j], dtype=torch.complex64
    )
    coefficients = coefficients / torch.linalg.norm(coefficients)

    amplitude_input = torch.zeros(layer.input_size, dtype=torch.complex64)
    amplitude_input[basis_indices] = coefficients

    combined_output = layer(amplitude_input)
    expected_output = torch.sum(
        coefficients.unsqueeze(-1) * basis_outputs,
        dim=0,
    )

    assert torch.allclose(combined_output, expected_output, atol=1e-6)

    with pytest.raises(ValueError, match="Amplitude input dimension"):
        layer(torch.ones(layer.input_size + 1, dtype=torch.complex64))"""
