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

ANSATZ_SKIP = pytest.mark.skip(
    reason="Legacy ansatz-based QuantumLayer API has been removed; test pending migration."
)


class TestQuantumLayer:
    """Test suite for QuantumLayer."""

    @staticmethod
    def _experiment_with_detectors(
        circuit: pcvl.Circuit, detectors: list[pcvl.Detector]
    ) -> pcvl.Experiment:
        experiment = pcvl.Experiment(circuit)
        for mode, detector in enumerate(detectors):
            experiment.detectors[mode] = detector
        return experiment

    def test_experiment_unitary_initialization(self):
        """QuantumLayer should accept a unitary experiment."""

        circuit = pcvl.Circuit(1)
        experiment = pcvl.Experiment(circuit)

        layer = ML.QuantumLayer(
            input_size=0,
            output_size=None,
            experiment=experiment,
            input_state=[1],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
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
                output_mapping_strategy=ML.OutputMappingStrategy.NONE,
            )

    def test_experiment_min_photons_filter_warning(self):
        """A min_photons_filter configured on the experiment should raise a warning (unsupported)."""

        circuit = pcvl.Circuit(1)
        experiment = pcvl.Experiment(circuit)
        experiment.min_detected_photons_filter(1)

        with pytest.warns(UserWarning, match="min_photons_filter"):
            ML.QuantumLayer(
                input_size=0,
                experiment=experiment,
                input_state=[1],
                output_mapping_strategy=ML.OutputMappingStrategy.NONE,
            )

    def test_experiment_threshold_detectors_applied(self):
        """QuantumLayer should honour threshold detectors defined on the experiment."""

        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())
        experiment = self._experiment_with_detectors(
            circuit,
            [pcvl.Detector.threshold(), pcvl.Detector.threshold()],
        )

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 0],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
        )

        output = layer()
        expected = torch.tensor([[0.5, 0.5]], dtype=output.dtype)
        assert torch.allclose(output, expected, atol=1e-6)
        keys = [tuple(key) for key in layer.get_output_keys()]
        assert keys == [(1, 0), (0, 1)]

    def test_threshold_detectors_single_mode_two_photons(self):
        circuit = pcvl.Circuit(1)
        experiment = self._experiment_with_detectors(
            circuit, [pcvl.Detector.threshold()]
        )

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[2],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
            no_bunching=False,
        )

        output = layer()
        keys = [tuple(key) for key in layer.get_output_keys()]
        target_index = keys.index((1,))
        assert torch.allclose(
            output[:, target_index], torch.ones_like(output[:, target_index])
        )

    def test_threshold_detectors_preserve_binary_outcomes(self):
        circuit = pcvl.Circuit(3)
        experiment = self._experiment_with_detectors(
            circuit,
            [
                pcvl.Detector.threshold(),
                pcvl.Detector.threshold(),
                pcvl.Detector.threshold(),
            ],
        )

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 0, 1],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
        )

        output = layer()
        keys = [tuple(key) for key in layer.get_output_keys()]
        # All keys must be binary tuples
        assert all(all(value in (0, 1) for value in key) for key in keys)
        # Probability mass should sit entirely on the observed detection pattern
        target_index = keys.index((1, 0, 1))
        assert torch.allclose(
            output[:, target_index], torch.ones_like(output[:, target_index])
        )

    def test_pnr_detectors_match_default_distribution(self):
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())

        default_layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 1],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
            no_bunching=False,
        )

        experiment = self._experiment_with_detectors(
            circuit, [pcvl.Detector.pnr(), pcvl.Detector.pnr()]
        )
        detector_layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
            no_bunching=False,
        )

        probs_default = default_layer()
        probs_detector = detector_layer()
        assert torch.allclose(probs_detector, probs_default, atol=1e-6)
        assert [tuple(key) for key in detector_layer.get_output_keys()] == [
            tuple(key) for key in default_layer.get_output_keys()
        ]

    def test_pnr_detectors_multi_photon_identity(self):
        circuit = pcvl.Circuit(3)
        experiment = self._experiment_with_detectors(
            circuit,
            [pcvl.Detector.pnr() for _ in range(3)],
        )

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[2, 1, 0],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
            no_bunching=False,
        )
        output = layer()
        keys = [tuple(key) for key in layer.get_output_keys()]
        assert (2, 1, 0) in keys
        idx = keys.index((2, 1, 0))
        assert torch.allclose(output[:, idx], torch.ones_like(output[:, idx]))

    def test_interleaved_detectors_single_mode(self):
        circuit = pcvl.Circuit(1)
        experiment = self._experiment_with_detectors(
            circuit, [pcvl.Detector.ppnr(n_wires=1)]
        )

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
        )

        output = layer()
        assert torch.allclose(output.sum(dim=1), torch.ones_like(output[:, 0]))
        assert len(layer.get_output_keys()) >= 1

    def test_interleaved_detectors_multi_mode_probabilities(self):
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())
        experiment = self._experiment_with_detectors(
            circuit,
            [
                pcvl.Detector.ppnr(n_wires=1, max_detections=2),
                pcvl.Detector.pnr(),
            ],
        )

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[4, 0],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
            no_bunching=False,
        )
        output = layer()
        assert torch.allclose(output.sum(dim=1), torch.ones_like(output[:, 0]))
        assert output.shape[-1] == len(layer.get_output_keys())
        assert torch.all(output >= 0)
        keys = [tuple(key) for key in layer.get_output_keys()]
        assert all(value in (0, 1, 2) for key in keys for value in key[:1])
        assert all(value in (0, 1, 2, 3, 4) for key in keys for value in key[1:])

    def test_mixed_detectors_identity_distribution(self):
        circuit = pcvl.Circuit(4)
        experiment = self._experiment_with_detectors(
            circuit,
            [
                pcvl.Detector.pnr(),
                pcvl.Detector.pnr(),
                pcvl.Detector.threshold(),
                pcvl.Detector.threshold(),
            ],
        )

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1, 0, 0],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
        )

        output = layer()
        keys = [tuple(key) for key in layer.get_output_keys()]
        assert len(keys) == output.shape[-1]
        expected_key = (1, 1, 0, 0)
        assert expected_key in keys
        idx = keys.index(expected_key)
        assert torch.allclose(output[:, idx], torch.ones_like(output[:, idx]))
        assert all(value in (0, 1, 2) for key in keys for value in key[:2])
        assert all(value in (0, 1) for key in keys for value in key[2:])

    def test_mixed_detectors_probabilistic_distribution(self):
        circuit = pcvl.Circuit(4)
        circuit.add((0, 1), pcvl.BS())
        circuit.add((1, 2), pcvl.BS())
        experiment = self._experiment_with_detectors(
            circuit,
            [
                pcvl.Detector.pnr(),
                pcvl.Detector.pnr(),
                pcvl.Detector.threshold(),
                pcvl.Detector.threshold(),
            ],
        )

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1, 1, 0],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
            no_bunching=False,
        )

        output = layer()
        keys = [tuple(key) for key in layer.get_output_keys()]
        assert output.shape[-1] == len(keys)
        assert torch.allclose(
            output.sum(dim=1), torch.ones_like(output[:, 0]), atol=1e-6
        )
        assert all(value in (0, 1) for key in keys for value in key[2:])
        assert any(key[2] == 1 for key in keys)
        assert any(key[2] == 0 for key in keys)
        assert all(0 <= key[0] + key[1] <= 3 for key in keys)
        assert len({key[:2] for key in keys}) >= 2

    def test_experiment_missing_detectors_default_pnr(self):
        circuit = pcvl.Circuit(2)
        circuit.add(1, pcvl.PS(torch.pi / 2))
        circuit.add((0, 1), pcvl.BS())
        experiment = pcvl.Experiment(circuit)

        layer_experiment = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
        )

        layer_direct = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 1],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
        )

        probs_exp = layer_experiment()
        probs_direct = layer_direct()
        assert torch.allclose(probs_exp, probs_direct, atol=1e-6)

    def test_partial_detector_assignment_defaults_remaining_to_pnr(self):
        circuit = pcvl.Circuit(3)
        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.threshold()

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[0, 0, 2],
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
            no_bunching=False,
        )

        output = layer()
        keys = [tuple(key) for key in layer.get_output_keys()]
        assert (0, 0, 2) in keys
        idx = keys.index((0, 0, 2))
        assert torch.allclose(output[:, idx], torch.ones_like(output[:, idx]))
        assert all(value in (0, 1) for key in keys for value in key[:2])
        assert any(key[2] == 2 for key in keys)

    def test_detector_choice_adjusts_output_size(self):
        circuit = pcvl.Circuit(3)
        circuit.add((0, 1), pcvl.BS())
        circuit.add((1, 2), pcvl.BS())
        input_state = [3, 0, 0]

        pnr_layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=input_state,
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
            no_bunching=False,
        )

        pnr_output_size = pnr_layer.output_size
        assert pnr_output_size == len(pnr_layer.get_output_keys())

        experiment = self._experiment_with_detectors(
            circuit, [pcvl.Detector.threshold() for _ in range(3)]
        )
        threshold_layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=input_state,
            output_mapping_strategy=ML.OutputMappingStrategy.NONE,
            no_bunching=False,
        )

        threshold_output_size = threshold_layer.output_size
        assert threshold_output_size == len(threshold_layer.get_output_keys())
        assert threshold_output_size < pnr_output_size

    @ANSATZ_SKIP
    def test_ansatz_based_layer_creation(self):
        """Test creating a layer from an ansatz."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=3, output_size=5
        )

        layer = ML.QuantumLayer(input_size=3, ansatz=ansatz)

        assert layer.input_size == 3
        assert layer.output_size == 5
        assert layer.auto_generation_mode is True

    @ANSATZ_SKIP
    def test_forward_pass_batched(self):
        """Test forward pass with batched input."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,  # Changed to match parameter count
            n_modes=4,
            n_photons=2,
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        # Test with batch
        x = torch.rand(10, 2)
        output = layer(x)

        assert output.shape == (10, 3)
        assert torch.all(output >= -1e6)  # More reasonable bounds for quantum outputs

    @ANSATZ_SKIP
    def test_forward_pass_single(self):
        """Test forward pass with single input."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=4, n_photons=1
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=3,  # Don't use NONE strategy to avoid size mismatch
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        # Test with single sample
        x = torch.rand(1, 2)
        output = layer(x)

        assert output.shape[0] == 1
        assert output.shape[1] == 3

    @ANSATZ_SKIP
    def test_gradient_computation(self):
        """Test that gradients flow through the layer."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=4,
            n_photons=2,
            use_bandwidth_tuning=True,
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

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

    @ANSATZ_SKIP
    def test_sampling_configuration(self):
        """Test sampling configuration methods."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=2, output_size=3
        )

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

    @ANSATZ_SKIP
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
            PhotonicBackend=experiment_normal, input_size=2, output_size=3
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
            PhotonicBackend=experiment_reservoir, input_size=2, output_size=3
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
        output = layer_reservoir(x)
        assert output.shape == (3, 3)

    @ANSATZ_SKIP
    def test_bandwidth_tuning(self):
        """Test bandwidth tuning functionality."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,
            n_modes=4,
            n_photons=2,
            use_bandwidth_tuning=True,
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=3, output_size=5
        )

        layer = ML.QuantumLayer(input_size=3, ansatz=ansatz)

        # Check that bandwidth coefficients exist
        assert layer.bandwidth_coeffs is not None
        assert len(layer.bandwidth_coeffs) == 3  # One per input dimension

        # Check they're learnable parameters
        for _key, param in layer.bandwidth_coeffs.items():
            assert param.requires_grad

    @ANSATZ_SKIP
    def test_output_mapping_strategies(self):
        """Test different output mapping strategies."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS,  # Use consistent circuit type
            n_modes=4,
            n_photons=2,
        )

        strategies = [
            ML.OutputMappingStrategy.LINEAR,
            ML.OutputMappingStrategy.LEXGROUPING,
            ML.OutputMappingStrategy.MODGROUPING,
        ]

        for strategy in strategies:
            ansatz = ML.AnsatzFactory.create(
                PhotonicBackend=experiment,
                input_size=2,
                output_size=4,
                output_mapping_strategy=strategy,
            )

            layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

            x = torch.rand(3, 2)
            output = layer(x)

            assert output.shape == (3, 4)
            assert torch.all(torch.isfinite(output))

    @ANSATZ_SKIP
    def test_string_representation(self):
        """Test string representation of the layer."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment, input_size=3, output_size=5
        )

        layer = ML.QuantumLayer(input_size=3, ansatz=ansatz)
        layer_str = str(layer)

        assert "QuantumLayer" in layer_str
        assert "parallel_columns" in layer_str
        assert "modes=4" in layer_str
        assert "input_size=3" in layer_str
        assert "output_size=5" in layer_str

    def test_invalid_configurations(self):
        """Test that invalid configurations raise appropriate errors."""
        # Test missing both ansatz and circuit
        with pytest.raises(ValueError, match="circuit"):
            ML.QuantumLayer(input_size=3)

        # Test invalid experiment configuration
        with pytest.raises(ValueError):
            ML.PhotonicBackend(
                circuit_type=ML.CircuitType.SERIES,
                n_modes=4,
                n_photons=5,  # More photons than modes
            )

    @ANSATZ_SKIP
    def test_none_output_mapping_with_correct_size(self):
        """Test NONE output mapping with correct size matching."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=3, n_photons=1
        )

        # Create ansatz without specifying output size initially
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=10,  # We'll override this
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
        )

        # Create layer to find out actual distribution size
        temp_layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

        # Get actual distribution size
        dummy_input = torch.rand(1, 2)
        with torch.no_grad():
            temp_output = temp_layer(dummy_input)

        # Now create NONE strategy with correct size
        ansatz_none = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=temp_output.shape[1],  # Match actual output size
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
        )

        layer_none = ML.QuantumLayer(input_size=2, ansatz=ansatz_none)

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
            input_size=0,  # No input parameters
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=None,  # Parameters to train (by prefix)
            input_parameters=["phi"],  # No input parameters
            output_size=3,
            output_mapping_strategy=ML.OutputMappingStrategy.LINEAR,
        )

        dummy_input = torch.rand(1, 2)

        # Test layer properties
        assert layer.input_size == 0
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
