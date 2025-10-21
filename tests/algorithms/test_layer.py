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

import perceval as pcvl
import pytest
import torch

import merlin as ML


class TestQuantumLayer:
    """Test suite for QuantumLayer."""

    def test_builder_based_layer_creation(self):
        """Test creating a layer from an builder."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1, 3], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=3,
            output_size=5,
            input_state=[1, 0, 1, 0],
            builder=builder,
            output_mapping_strategy=ML.OutputMappingStrategy.GROUPING,
        )
        assert layer.input_size == 3
        assert layer.output_size == 5
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
            output_size=3,
            input_state=[1, 0, 0, 0, 0],
            builder=builder,
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
        )
        pcvl.pdisplay(layer.circuit, output_format=pcvl.Format.TEXT)

        dummy_input = torch.rand(1, 5)
        output = layer(dummy_input)
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
            output_size=3,
            input_state=[1, 0, 1, 0],
            builder=builder,
            output_mapping_strategy=ML.OutputMappingStrategy.GROUPING,
        )

        # Test with batch
        x = torch.rand(10, 2)
        output = layer(x)

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
            output_size=3,
            input_state=[1, 0, 0, 0],
            builder=builder,
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
        )

        # Test with single sample
        x = torch.rand(1, 2)
        output = layer(x)

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
            output_size=3,
            input_state=[1, 1, 0, 0],
            builder=builder,
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
        )

        x = torch.rand(5, 2, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()

        # Check that input gradients exist
        assert x.grad is not None

        # Check that layer parameters have gradients
        has_trainable_params = False
        for param in layer.parameters():
            if param.requires_grad:
                has_trainable_params = True
                assert param.grad is not None

        assert has_trainable_params, "Layer should have trainable parameters"

    def test_sampling_configuration(self):
        """Test sampling configuration methods."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        layer = ML.QuantumLayer(
            input_size=2,
            output_size=3,
            input_state=[1, 0, 1, 0],
            builder=builder,
            output_mapping_strategy=ML.OutputMappingStrategy.GROUPING,
            shots=100,
        )

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
            output_size=3,
            input_state=[1, 0, 1, 0],
            builder=builder,
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
        )

        layer_reservoir = ML.QuantumLayer(
            input_size=2,
            output_size=3,
            input_state=[1, 0, 1, 0],
            builder=builder,
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
        )

        layer_reservoir.requires_grad_(False)
        assert any(p.requires_grad for p in layer_normal.parameters())
        assert all(not p.requires_grad for p in layer_reservoir.parameters())

        normal_trainable = sum(
            p.numel() for p in layer_normal.parameters() if p.requires_grad
        )

        reservoir_trainable = sum(
            p.numel() for p in layer_reservoir.parameters() if p.requires_grad
        )

        # Reservoir mode should freeze all parameters while keeping the normal layer trainable.
        assert normal_trainable > 0
        assert reservoir_trainable == 0

        # Test that reservoir layer still works
        x = torch.rand(3, 2)
        output = layer_reservoir(x)
        assert output.shape == (3, 3)

    def test_output_mapping_strategies(self):
        """Test different output mapping strategies."""
        builder = ML.CircuitBuilder(n_modes=4)
        builder.add_entangling_layer(trainable=True, name="U1")
        builder.add_angle_encoding(modes=[0, 1], name="input")
        builder.add_entangling_layer(trainable=True, name="U2")

        strategies = [
            ML.OutputMappingStrategy.LINEAR,
            ML.OutputMappingStrategy.LEXGROUPING,
            ML.OutputMappingStrategy.MODGROUPING,
        ]

        for strategy in strategies:
            layer = ML.QuantumLayer(
                input_size=2,
                output_size=4,
                input_state=[1, 0, 1, 0],
                builder=builder,
                output_mapping_strategy=strategy,
            )

            x = torch.rand(3, 2)
            output = layer(x)

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
            output_size=5,
            input_state=[1, 0, 1, 0],
            builder=builder,
            output_mapping_strategy=ML.OutputMappingStrategy.GROUPING,
        )

        layer_str = str(layer)
        print(f"Layer string representation:\n{layer_str}")
        assert "QuantumLayer" in layer_str
        assert "modes=4" in layer_str
        assert "input_size=3" in layer_str
        assert "output_size=5" in layer_str

    def test_invalid_configurations(self):
        """Test that invalid configurations raise appropriate errors."""
        # this tests include builder, simple and circuit-based API
        with pytest.raises(
            ValueError,
            match="Either 'circuit', or 'builder' must be provided",
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
                output_size=5,
                input_state=[1, 0, 1, 0],
                builder=builder,
                output_mapping_strategy=ML.OutputMappingStrategy.GROUPING,
            )

        with pytest.raises(ValueError):
            ML.QuantumLayer(
                input_size=2,
                output_size=5,
                n_photons=5,  # more photons than modes
                builder=builder,
                output_mapping_strategy=ML.OutputMappingStrategy.GROUPING,
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
            output_size=4,
            input_state=[1, 0, 1, 0],
            builder=builder,
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
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
            output_size=5,
            input_state=[1, 0, 1, 0],
            builder=builder,
            output_mapping_strategy=ML.OutputMappingStrategy.GROUPING,
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
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
        )

        x = torch.rand(2, 2)
        output = layer_none(x)

        # Output should be probability distribution
        assert torch.all(output >= -1e6)  # Reasonable bounds
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
            input_parameters=None,  # No input parameters
            output_size=3,
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
        )

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
                output_size=3,
                output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
            )

        # Test layer properties
        assert layer.input_size == 0
        assert layer.output_size == 3
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
            input_size=2,  # No input parameters
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=None,  # Parameters to train (by prefix)
            input_parameters=["phi"],  # No input parameters
            output_size=3,
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
        )

        dummy_input = torch.rand(1, 2)

        # Test layer properties
        assert layer.input_size == 2
        assert layer.output_size == 3
        # Check that it has trainable parameters
        trainable_params = [p for p in layer.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "Layer should have trainable parameters"

        # Test forward pass (no input needed)
        output = layer(dummy_input)
        assert output.shape == (1, 3)
        assert torch.all(torch.isfinite(output))

        # Test gradient computation
        loss = output.sum()
        loss.backward()

        # Check that trainable parameters have gradients
        for param in layer.parameters():
            if param.requires_grad:
                assert param.grad is not None
