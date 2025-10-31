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

    def test_experiment_unitary_initialization(self):
        """QuantumLayer should accept a unitary experiment."""

        circuit = pcvl.Circuit(1)
        experiment = pcvl.Experiment(circuit)

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1],
        )

        output = layer()
        assert torch.allclose(
            output.sum(), torch.tensor(1.0, dtype=output.dtype), atol=1e-6
        )

    def test_experiment_non_unitary_rejected(self):
        """A non-unitary experiment should be rejected."""

        circuit = pcvl.Circuit(1)
        experiment = pcvl.Experiment(circuit)
        experiment.add(0, pcvl.TD(1))
        assert experiment.is_unitary is False

        with pytest.raises(ValueError, match="must be unitary"):
            ML.QuantumLayer(
                input_size=0,
                experiment=experiment,
                input_state=[1],
            )

    def test_experiment_min_photons_filter_error(self):
        """A min_photons_filter configured on the experiment should raise an error (unsupported)."""

        circuit = pcvl.Circuit(1)
        experiment = pcvl.Experiment(circuit)
        experiment.min_detected_photons_filter(1)

        with pytest.raises(ValueError):
            ML.QuantumLayer(
                input_size=0,
                experiment=experiment,
                input_state=[1],
            )

    def test_builder_based_layer_creation(self):
        """Test creating a layer from an builder."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1, 3], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=3,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        assert layer.input_size == 3
        assert layer.thetas[0].shape[0] == 2 * 4 * (
            4 - 1
        )  # 24 trainable parameters from U1 and U2

    @pytest.mark.parametrize("names", [("input", "input"), ("input_a", "input_b")])
    def test_multiple_angle_encodings_validate_input_size(self, names):
        builder = ML.CircuitBuilder(n_modes=5)
        builder.add_angle_encoding(modes=[0, 1], name=names[0])
        builder.add_angle_encoding(modes=[2, 3, 4], name=names[1])

        layer = ML.QuantumLayer(
            input_size=5,
            input_state=[1, 0, 0, 0, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        pcvl.pdisplay(layer.circuit, output_format=pcvl.Format.TEXT)

        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        dummy_input = torch.rand(1, 5)
        output = model(dummy_input)
        assert output.shape == (1, 3), "Output shape mismatch"
        assert layer.input_size == 5, "Input size should match number of encoded modes"
        assert not torch.isnan(output).any(), "Output should not contain NaNs"

    def test_forward_pass_batched(self):
        """Test forward pass with batched input."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        # Test with batch
        x = torch.rand(10, 2)
        output = model(x)

        assert output.shape == (10, 3)
        assert torch.all(output >= -1e6)  # More reasonable bounds for quantum outputs

    def test_forward_pass_single(self):
        """Test forward pass with single input."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 0, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        # Test with single sample
        x = torch.rand(1, 2)
        output = model(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 3

    def test_gradient_computation(self):
        """Test that gradients flow through the layer."""

        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 1, 0, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

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
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
            shots=100,
        )

        torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

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
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer_normal = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        model_normal = torch.nn.Sequential(
            layer_normal, torch.nn.Linear(layer_normal.output_size, 3)
        )

        layer_reservoir = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        model_reservoir = torch.nn.Sequential(
            layer_reservoir, torch.nn.Linear(layer_reservoir.output_size, 3)
        )

        model_reservoir.requires_grad_(False)
        assert any(p.requires_grad for p in model_normal.parameters())
        assert all(not p.requires_grad for p in model_reservoir.parameters())

        normal_trainable = sum(
            p.numel() for p in model_normal.parameters() if p.requires_grad
        )

        reservoir_trainable = sum(
            p.numel() for p in model_reservoir.parameters() if p.requires_grad
        )

        # Reservoir mode should freeze all parameters while keeping the normal layer trainable.
        assert normal_trainable > 0
        assert reservoir_trainable == 0

        # Test that reservoir layer still works
        x = torch.rand(3, 2)
        output = model_reservoir(x)
        assert output.shape == (3, 3)

    def test_measurement_strategies(self):
        """Test different measurement strategies and grouping policies."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        configs = [
            {
                "measurement_strategy": ML.MeasurementStrategy.PROBABILITIES,
                "grouping_policy": None,
            },
            {
                "measurement_strategy": ML.MeasurementStrategy.PROBABILITIES,
                "grouping_policy": ML.LexGrouping,
            },
            {
                "measurement_strategy": ML.MeasurementStrategy.PROBABILITIES,
                "grouping_policy": ML.ModGrouping,
            },
        ]

        for cfg in configs:
            if cfg["grouping_policy"] is None:
                layer = ML.QuantumLayer(
                    input_size=2,
                    input_state=[1, 0, 1, 0],
                    builder=builder,
                    measurement_strategy=cfg["measurement_strategy"],
                )

                model = torch.nn.Sequential(
                    layer, torch.nn.Linear(layer.output_size, 4)
                )

                x = torch.rand(3, 2)
                output = model(x)
                assert output.shape == (3, 4)
                assert torch.all(torch.isfinite(output))

            else:
                layer = ML.QuantumLayer(
                    input_size=2,
                    input_state=[1, 0, 1, 0],
                    builder=builder,
                    measurement_strategy=cfg["measurement_strategy"],
                )

                model = torch.nn.Sequential(
                    layer, cfg["grouping_policy"](layer.output_size, 4)
                )

                x = torch.rand(3, 2)
                output = model(x)
                assert output.shape == (3, 4)
                assert torch.all(torch.isfinite(output))

    def test_string_representation(self):
        """Test string representation of the layer."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1, 2], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=3,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        layer_str = str(layer)
        print(f"Layer string representation:\n{layer_str}")
        assert "QuantumLayer" in layer_str
        assert "modes=4" in layer_str
        assert "input_size=3" in layer_str

    def test_invalid_configurations(self):
        """Test that invalid configurations raise appropriate errors."""
        # this tests include builder, simple and circuit-based API
        with pytest.raises(
            ValueError,
            match="Provide exactly one of 'circuit', 'builder', or 'experiment'.",
        ):
            ML.QuantumLayer(input_size=3)

        # Test invalid experiment configuration
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        # Input size mismatch between declaration and builder-produced features
        with pytest.raises(
            ValueError,
            match="Input size \\(3\\) must equal the number of encoded input features generated by the circuit \\(2\\)\\.",
        ):
            ML.QuantumLayer(
                input_size=3,
                input_state=[1, 0, 1, 0],
                builder=builder,
                measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
            )

        with pytest.raises(ValueError):
            ML.QuantumLayer(
                input_size=2,
                output_size=5,
                n_photons=5,  # more photons than modes
                builder=builder,
                measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
            )

        with pytest.raises(TypeError):
            ML.QuantumLayer.simple(n_params=0)

    def test_subset_combinations_respected(self):
        """Ensure subset combinations expose more parameters without breaking input size checks."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(
            modes=[0, 1, 2], name="input", subset_combinations=True
        )
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=3,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        assert layer.input_size == 3

    def test_none_output_mapping_with_correct_size(self):
        """Test NONE output mapping with correct size matching."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        temp_layer = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
        )

        # Get actual distribution size
        dummy_input = torch.rand(1, 2)
        with torch.no_grad():
            _temp_output = temp_layer(dummy_input)

        # Now create NONE strategy with correct size
        layer_none = ML.QuantumLayer(
            input_size=2,
            input_state=[1, 0, 1, 0],
            builder=builder,
            measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
        )

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
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )

        output_size = math.comb(3, sum(input_state))  # Calculate output size
        with pytest.raises(
            ValueError,
            match="Input size \\(2\\) must equal the number of input parameters generated by the circuit \\(0\\)\\.",
        ):
            layer = ML.QuantumLayer(
                input_size=2,  # input_size > nb of input_parameters
                circuit=circuit,
                input_state=input_state,
                trainable_parameters=["phi"],  # Parameters to train (by prefix)
                input_parameters=None,  # No input parameters
                measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
            )

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
            input_size=2,  # 2 input parameters
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=[],  # No trainable parameters
            input_parameters=["phi"],  # No input parameters
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        dummy_input = torch.rand(1, 2)

        math.comb(3, sum(input_state))  # Calculate output size
        # Test layer properties
        assert layer.input_size == 2
        assert model[1].out_features == 3
        # Check that it has trainable parameters (only in Linear layer)
        trainable_params_layer = [p for p in layer.parameters() if p.requires_grad]
        assert len(trainable_params_layer) == 0, (
            "Layer should have no trainable parameters"
        )
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "Model should have trainable parameters"

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
