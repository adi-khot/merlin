from types import MethodType

import perceval as pcvl
import torch

from merlin.algorithms.layer import QuantumLayer
from merlin.sampling.strategies import OutputMappingStrategy


def classical_method(layer, input_state):
    output_classical = torch.zeros(1, layer.output_size)
    dtype = layer.computation_process.simulation_graph.prev_amplitudes.dtype
    output_classical = output_classical.to(dtype)

    for key, value in input_state.items():
        layer.computation_process.input_state = key
        _ = layer()

        output_classical += (
            value * layer.computation_process.simulation_graph.prev_amplitudes
        )

    output_probs = (
        layer.computation_process.simulation_graph.compute_probs_from_amplitudes(
            output_classical
        )
    )
    return output_probs[1]


class TestOutputSuperposedState:
    """Test cases for output mapping strategies in QuantumLayer.simple()."""

    def test_superposed_state(self, benchmark):
        """Test NONE strategy when output_size is not specified."""
        print("\n=== Testing Superposed input state method ===")

        # When using NONE strategy without specifying output_size,
        # the output size should equal the distribution size
        circuit = pcvl.components.GenericInterferometer(
            10,
            pcvl.components.catalog["mzi phase last"].generate,
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

        input_state = torch.rand(3, 10).to(torch.float64)

        sum_values = (input_state**2).sum(dim=-1, keepdim=True)

        input_state = input_state / sum_values

        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            n_photons=3,
            output_mapping_strategy=OutputMappingStrategy.NONE,
            input_state=input_state,
            trainable_parameters=["phi"],
            input_parameters=[],
            dtype=torch.float64,
            no_bunching=True,
        )

        input_state_superposed = {
            layer.computation_process.simulation_graph.mapped_keys[k]: input_state[1, k]
            for k in range(len(input_state[0]))
        }

        output_superposed = benchmark(layer)

        output_classical = classical_method(layer, input_state_superposed)

        assert torch.allclose(
            output_superposed[1], output_classical, rtol=3e-4, atol=1e-7
        )

    def test_classical_method(self, benchmark):
        """Test NONE strategy when output_size is not specified."""
        print("\n=== Testing Superposed input state method ===")

        # When using NONE strategy without specifying output_size,
        # the output size should equal the distribution size
        circuit = pcvl.components.GenericInterferometer(
            10,
            pcvl.components.catalog["mzi phase last"].generate,
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

        input_state = torch.rand(3, 10).to(torch.float64)

        sum_values = (input_state**2).sum(dim=-1, keepdim=True)

        input_state = input_state / sum_values

        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            n_photons=3,
            output_mapping_strategy=OutputMappingStrategy.NONE,
            input_state=input_state,
            trainable_parameters=["phi"],
            input_parameters=[],
            dtype=torch.float64,
            no_bunching=True,
        )

        input_state_superposed = {
            layer.computation_process.simulation_graph.mapped_keys[k]: input_state[0, k]
            for k in range(len(input_state[0]))
        }

        output_superposed = layer()

        output_classical = benchmark(
            lambda: classical_method(layer, input_state_superposed)
        )

        assert torch.allclose(
            output_superposed[0], output_classical, rtol=3e-4, atol=1e-7
        )

    def test_forward_infers_batch_for_superposed_state(self):
        circuit = pcvl.components.GenericInterferometer(
            10,
            pcvl.components.catalog["mzi phase last"].generate,
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

        input_state = torch.rand(2, 10).to(torch.float64)
        sum_values = (input_state**2).sum(dim=-1, keepdim=True)
        input_state = input_state / sum_values

        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            n_photons=3,
            output_mapping_strategy=OutputMappingStrategy.NONE,
            input_state=input_state,
            trainable_parameters=["phi"],
            input_parameters=[],
            dtype=torch.float64,
            no_bunching=True,
        )

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

        layer()

        assert call_tracker["ebs"] == 1
        assert call_tracker["super"] == 0

    def test_forward_infers_single_state_without_batch(self):
        circuit = pcvl.components.GenericInterferometer(
            10,
            pcvl.components.catalog["mzi phase last"].generate,
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

        input_state = torch.rand(1, 10).to(torch.float64)
        sum_values = (input_state**2).sum(dim=-1, keepdim=True)
        input_state = input_state / sum_values

        layer = QuantumLayer(
            input_size=0,
            circuit=circuit,
            n_photons=3,
            output_mapping_strategy=OutputMappingStrategy.NONE,
            input_state=input_state,
            trainable_parameters=["phi"],
            input_parameters=[],
            dtype=torch.float64,
            no_bunching=True,
        )

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

        layer()

        assert call_tracker["ebs"] == 0
        assert call_tracker["super"] == 1
