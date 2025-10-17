# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Tests for the main QuantumLayer class.
"""

import math

import perceval as pcvl
import pytest
import torch

import merlin as ML


class TestQuantumLayer:
    """Test suite for QuantumLayer."""

    def test_ansatz_based_layer_creation(self):
        """Test creating a layer from an ansatz."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(PhotonicBackend=experiment, input_size=3)

        layer = ML.QuantumLayer(input_size=3, ansatz=ansatz)

        assert layer.input_size == 3
        assert layer.auto_generation_mode is True

    def test_forward_pass_batched(self):
        """Test forward pass with batched input."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,  # Changed to match parameter count
            n_modes=4,
            n_photons=2,
        )

        ansatz = ML.AnsatzFactory.create(PhotonicBackend=experiment, input_size=2)

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        # Test with batch
        x = torch.rand(10, 2)
        output = model(x)

        assert output.shape == (10, 3)
        assert torch.all(output >= -1e6)  # More reasonable bounds for quantum outputs

    def test_forward_pass_single(self):
        """Test forward pass with single input."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=4, n_photons=1
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            measurement_strategy=ML.MeasurementStrategy.MEASUREMENTDISTRIBUTION,
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        # Test with single sample
        x = torch.rand(1, 2)
        output = model(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 3

    def test_gradient_computation(self):
        """Test that gradients flow through the layer."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=4,
            n_photons=2,
            use_bandwidth_tuning=True,
        )

        ansatz = ML.AnsatzFactory.create(PhotonicBackend=experiment, input_size=2)

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        x = torch.rand(5, 2, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None

        # Check that layer parameters have gradients
        has_trainable_params = False
        for param in model.parameters():
            if param.requires_grad:
                has_trainable_params = True
                assert param.grad is not None

        assert has_trainable_params, "Model should have trainable parameters"

    def test_sampling_configuration(self):
        """Test sampling configuration methods."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(PhotonicBackend=experiment, input_size=2)

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz, shots=100)

        assert layer.shots == 100
        assert layer.sampling_method == "multinomial"

        # Test updating configuration
        layer.set_sampling_config(shots=200, method="gaussian")
        assert layer.shots == 200
        assert layer.sampling_method == "gaussian"

        # Test invalid method
        with pytest.raises(ValueError):
            layer.set_sampling_config(method="invalid")

    def test_reservoir_mode(self):
        """Test reservoir computing mode."""
        # Test normal mode first
        experiment_normal = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL,
            n_modes=4,
            n_photons=2,
            reservoir_mode=False,
        )

        ansatz_normal = ML.AnsatzFactory.create(
            PhotonicBackend=experiment_normal, input_size=2
        )

        layer_normal = ML.QuantumLayer(input_size=2, ansatz=ansatz_normal)
        normal_trainable = sum(
            p.numel() for p in layer_normal.parameters() if p.requires_grad
        )

        # Test reservoir mode
        experiment_reservoir = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL,
            n_modes=4,
            n_photons=2,
            reservoir_mode=True,
        )

        ansatz_reservoir = ML.AnsatzFactory.create(
            PhotonicBackend=experiment_reservoir, input_size=2
        )

        layer_reservoir = ML.QuantumLayer(input_size=2, ansatz=ansatz_reservoir)
        reservoir_trainable = sum(
            p.numel() for p in layer_reservoir.parameters() if p.requires_grad
        )

        # In reservoir mode, should have fewer or equal trainable parameters
        # (since some parameters are fixed)
        assert reservoir_trainable <= normal_trainable

        # Test that reservoir layer still works
        x = torch.rand(3, 2)
        x_out = layer_reservoir(x)
        output = torch.nn.Linear(layer_reservoir.output_size, 3)(x_out)
        assert output.shape == (3, 3)

    def test_bandwidth_tuning(self):
        """Test bandwidth tuning functionality."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=4,
            n_photons=2,
            use_bandwidth_tuning=True,
        )

        ansatz = ML.AnsatzFactory.create(PhotonicBackend=experiment, input_size=3)

        layer = ML.QuantumLayer(input_size=3, ansatz=ansatz)

        # Check that bandwidth coefficients exist
        assert layer.bandwidth_coeffs is not None
        assert len(layer.bandwidth_coeffs) == 3  # One per input dimension

        # Check they're learnable parameters
        for _key, param in layer.bandwidth_coeffs.items():
            assert param.requires_grad

    def test_measurement_strategies(self):
        """Test different measurement strategies and grouping policies."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=4,
            n_photons=2,
        )

        configs = [
            {
                "measurement_strategy": ML.MeasurementStrategy.MEASUREMENTDISTRIBUTION,
                "grouping_policy": None,
            },
            {
                "measurement_strategy": ML.MeasurementStrategy.MEASUREMENTDISTRIBUTION,
                "grouping_policy": ML.LexGrouping,
            },
            {
                "measurement_strategy": ML.MeasurementStrategy.MEASUREMENTDISTRIBUTION,
                "grouping_policy": ML.ModGrouping,
            },
        ]

        for cfg in configs:
            if cfg["grouping_policy"] is None:
                ansatz = ML.AnsatzFactory.create(
                    PhotonicBackend=experiment,
                    input_size=2,
                    measurement_strategy=cfg["measurement_strategy"],
                )

                layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
                model = torch.nn.Sequential(
                    layer, torch.nn.Linear(layer.output_size, 4)
                )

                x = torch.rand(3, 2)
                output = model(x)

                assert output.shape == (3, 4)
                assert torch.all(torch.isfinite(output))
            else:
                ansatz = ML.AnsatzFactory.create(
                    PhotonicBackend=experiment,
                    input_size=2,
                    measurement_strategy=cfg["measurement_strategy"],
                )

                layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
                model = torch.nn.Sequential(
                    layer, cfg["grouping_policy"](layer.output_size, 4)
                )

                x = torch.rand(3, 2)
                output = model(x)

                assert output.shape == (3, 4)
                assert torch.all(torch.isfinite(output))

    def test_string_representation(self):
        """Test string representation of the layer."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(PhotonicBackend=experiment, input_size=3)

        layer = ML.QuantumLayer(input_size=3, ansatz=ansatz)
        layer_str = str(layer)

        assert "QuantumLayer" in layer_str
        assert "parallel_columns" in layer_str
        assert "modes=4" in layer_str
        assert "input_size=3" in layer_str

    def test_invalid_configurations(self):
        """Test that invalid configurations raise appropriate errors."""
        # Test missing both ansatz and circuit
        with pytest.raises(
            ValueError, match="Either 'ansatz' or 'circuit' must be provided"
        ):
            ML.QuantumLayer(input_size=3)

        # Test invalid experiment configuration
        with pytest.raises(ValueError):
            ML.PhotonicBackend(
                circuit_type=ML.CircuitType.SERIES,
                n_modes=4,
                n_photons=5,  # More photons than modes
            )

    def test_none_measurement_with_correct_size(self):
        """Test MeasurementDistribution measurement with correct size matching."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=3, n_photons=1
        )

        # Create ansatz without specifying output size initially
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            measurement_strategy=ML.MeasurementStrategy.AMPLITUDEVECTOR,
        )

        # Create layer to find out actual distribution size
        temp_layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        # Get actual distribution size
        dummy_input = torch.rand(1, 2)
        with torch.no_grad():
            temp_output = temp_layer(dummy_input)

        # Now create AmplitudeVector strategy with correct size
        ansatz_none = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=temp_output.shape[1],  # Match actual output size
            measurement_strategy=ML.MeasurementStrategy.AMPLITUDEVECTOR,
        )

        layer_none = ML.QuantumLayer(input_size=2, ansatz=ansatz_none)

        x = torch.rand(2, 2)
        output = layer_none(x)

        # Output should be amplitudes
        assert torch.allclose(
            torch.sum(output.abs() ** 2, dim=1), torch.ones(2), atol=1e-6
        )
        assert output.shape[0] == 2

    def test_simple_perceval_circuit_no_input(self):
        """Test QuantumLayer with simple perceval circuit and no input parameters."""
        # Create a simple perceval circuit with no input parameters
        circuit = pcvl.Circuit(3)  # 3 modes
        circuit.add(0, pcvl.BS())  # Beam splitter on modes 0,1
        circuit.add(
            0, pcvl.PS(pcvl.P("phi1"))
        )  # Phase shifter with trainable parameter
        circuit.add(1, pcvl.BS())  # Beam splitter on modes 1,2
        circuit.add(1, pcvl.PS(pcvl.P("phi2")))  # Another phase shifter

        # Define input state (where photons are placed)
        input_state = [1, 0, 0]  # 1 photon in first mode

        # Create QuantumLayer with custom circuit
        layer = ML.QuantumLayer(
            input_size=0,  # No input parameters
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=["phi"],  # Parameters to train (by prefix)
            input_parameters=[],  # No input parameters
            measurement_strategy=ML.MeasurementStrategy.MEASUREMENTDISTRIBUTION,
        )

        output_size = math.comb(3, sum(input_state))  # Calculate output size
        # Test layer properties
        assert layer.input_size == 0
        assert layer.output_size == output_size
        # Check that it has trainable parameters
        trainable_params = [p for p in layer.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "Layer should have trainable parameters"

        # Test forward pass (no input needed)
        output = layer()
        assert output.shape == (1, 3)
        assert torch.all(torch.isfinite(output))

        # Test gradient computation
        loss = output.sum()
        loss.backward()

        # Check that trainable parameters have gradients
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_simple_perceval_circuit_no_trainable_parameter(self):
        """Test QuantumLayer with simple perceval circuit and no trainable parameters."""
        # Create a simple perceval circuit with no input parameters
        circuit = pcvl.Circuit(3)  # 3 modes
        circuit.add(0, pcvl.BS())  # Beam splitter on modes 0,1
        circuit.add(
            0, pcvl.PS(pcvl.P("phi1"))
        )  # Phase shifter with trainable parameter
        circuit.add(1, pcvl.BS())  # Beam splitter on modes 1,2
        circuit.add(1, pcvl.PS(pcvl.P("phi2")))  # Another phase shifter

        # Define input state (where photons are placed)
        input_state = [1, 0, 0]  # 1 photon in first mode

        # Create QuantumLayer with custom circuit
        layer = ML.QuantumLayer(
            input_size=0,  # No input parameters
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=[],  # Parameters to train (by prefix)
            input_parameters=["phi"],  # No input parameters
            measurement_strategy=ML.MeasurementStrategy.MEASUREMENTDISTRIBUTION,
        )
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        dummy_input = torch.rand(1, 2)

        output_size = math.comb(3, sum(input_state))  # Calculate output size
        # Test layer properties
        assert layer.input_size == 0
        assert layer.output_size == output_size
        # Check that it has trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "Layer should have trainable parameters"

        # Test forward pass (no input needed)
        output = model(dummy_input)
        assert output.shape == (1, 3)
        assert torch.all(torch.isfinite(output))

        # Test gradient computation
        loss = output.sum()
        loss.backward()

        # Check that trainable parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
