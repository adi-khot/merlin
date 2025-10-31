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


class TestDetectorsWithQuantumLayer:
    """Test suite for Detectors integration with QuantumLayer."""

    def test_threshold_detectors_preserve_binary_outcomes(self):
        circuit = pcvl.Circuit(3)
        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.threshold()
        experiment.detectors[2] = pcvl.Detector.threshold()

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 0, 1],
        )

        output = layer()
        keys = [tuple(key) for key in layer.output_keys]
        # All keys must be binary tuples
        assert all(all(value in (0, 1) for value in key) for key in keys)
        # Probability mass should sit entirely on the observed detection pattern
        target_index = keys.index((1, 0, 1))
        assert torch.allclose(
            output[:, target_index], torch.ones_like(output[:, target_index])
        )

    def test_amplitudes_strategy_rejected_with_detectors(self):
        circuit = pcvl.Circuit(2)
        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()

        with pytest.raises(
            RuntimeError,
            match="cannot be used when Experiment contains at least one Detector",
        ):
            ML.QuantumLayer(
                input_size=0,
                experiment=experiment,
                input_state=[1, 0],
                measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
            )

        # No error with MeasurementStrategy PROBABILITIES or MODE_EXPECTATIONS
        ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 0],
            measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
        )
        ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 0],
            measurement_strategy=ML.MeasurementStrategy.MODE_EXPECTATIONS,
        )

    def test_pnr_detectors_match_default_distribution(self):
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())

        default_layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 1],
            no_bunching=False,
        )

        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.pnr()
        experiment.detectors[1] = pcvl.Detector.pnr()

        detector_layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1],
            no_bunching=False,
        )

        probs_default = default_layer()
        probs_detector = detector_layer()
        assert torch.allclose(probs_detector, probs_default, atol=1e-6)
        assert [tuple(key) for key in detector_layer.output_keys] == [
            tuple(key) for key in default_layer.output_keys
        ]

    def test_interleaved_n_wire_1_equals_threshold(self):
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())
        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.ppnr(n_wires=1)

        experiment_threshold = pcvl.Experiment(circuit)
        experiment_threshold.detectors[0] = pcvl.Detector.threshold()

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[4, 0],
            no_bunching=False,
        )

        layer_threshold = ML.QuantumLayer(
            input_size=0,
            experiment=experiment_threshold,
            input_state=[4, 0],
            no_bunching=False,
        )

        output = layer()
        output_threshold = layer_threshold()
        assert output.shape[-1] == output_threshold.shape[-1]
        assert torch.allclose(output, output_threshold)
        assert torch.allclose(output.sum(dim=1), torch.ones_like(output[:, 0]))
        assert torch.allclose(
            output_threshold.sum(dim=1), torch.ones_like(output_threshold[:, 0])
        )
        assert output.shape[-1] == len(layer.output_keys)
        assert torch.all(output >= 0)
        keys = [tuple(key) for key in layer.output_keys]
        keys_threshold = [tuple(key) for key in layer_threshold.output_keys]
        assert keys == keys_threshold
        assert all(value in (0, 1) for key in keys for value in key[:1])
        assert all(
            any(value == i for key in keys for value in key[1:]) for i in range(5)
        )

    def test_interleaved_detectors_more_wires_than_modes(self):
        experiment = pcvl.Experiment()
        experiment.set_circuit(pcvl.Unitary(pcvl.Matrix.random_unitary(4)))
        experiment.detectors[0] = pcvl.Detector.ppnr(n_wires=2)
        experiment.detectors[1] = pcvl.Detector.ppnr(n_wires=5)

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[3, 4, 1, 0],
            no_bunching=False,
        )

        output = layer()
        assert torch.allclose(torch.sum(output, dim=1), torch.ones_like(output[:, 0]))
        keys = [tuple(key) for key in layer.output_keys]
        assert all(key[0] in (0, 1, 2) for key in keys)
        assert all(key[1] in (0, 1, 2, 3, 4, 5) for key in keys)
        assert all(any(key[0] == i for key in keys) for i in range(3))
        assert all(any(key[1] == i for key in keys) for i in range(6))
        assert all(any(key[2] == i for key in keys) for i in range(8))
        assert all(any(key[3] == i for key in keys) for i in range(8))

    def test_mixed_detectors_probabilistic_distribution(self):
        circuit = pcvl.Circuit(4)
        circuit.add((0, 1), pcvl.BS())
        circuit.add((1, 2), pcvl.BS())
        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.pnr()
        experiment.detectors[1] = pcvl.Detector.pnr()
        experiment.detectors[2] = pcvl.Detector.threshold()
        experiment.detectors[3] = pcvl.Detector.threshold()

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1, 1, 0],
            no_bunching=False,
        )

        output = layer()
        keys = [tuple(key) for key in layer.output_keys]
        assert output.shape[-1] == len(keys)
        assert torch.allclose(
            output.sum(dim=1), torch.ones_like(output[:, 0]), atol=1e-6
        )
        assert all(value in (0, 1) for key in keys for value in key[2:])
        assert all(any(key[0] == i for key in keys) for i in range(4))
        assert all(any(key[1] == i for key in keys) for i in range(4))
        assert all(0 <= key[0] + key[1] <= 3 for key in keys)

    def test_experiment_missing_detectors_default_pnr(self):
        circuit = pcvl.Circuit(2)
        circuit.add(1, pcvl.PS(torch.pi / 2))
        circuit.add((0, 1), pcvl.BS())
        experiment = pcvl.Experiment(circuit)

        layer_experiment = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1],
        )

        layer_direct = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=[1, 1],
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
            no_bunching=False,
        )

        output = layer()
        keys = [tuple(key) for key in layer.output_keys]
        assert (0, 0, 2) in keys
        idx = keys.index((0, 0, 2))
        assert torch.allclose(output[:, idx], torch.ones_like(output[:, idx]))
        assert all(value in (0, 1) for key in keys for value in key[:2])
        assert any(key[2] == 2 for key in keys)

    def test_experiment_layer_matches_perceval_pnr(self):
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())
        circuit.add(0, pcvl.PS(torch.pi / 3))

        input_state = [1, 1]

        experiment = pcvl.Experiment(circuit)
        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=input_state,
            no_bunching=False,
        )

        output = layer().squeeze(0)
        keys = [tuple(key) for key in layer.output_keys]
        assert output.shape[0] == len(keys)

        experiment_reference = pcvl.Experiment(circuit)
        processor = pcvl.Processor("SLOS", experiment_reference)
        processor.with_input(pcvl.BasicState(input_state))

        raw_results = processor.probs()["results"]
        probability_map = {
            tuple(int(v) for v in state): float(prob)
            for state, prob in raw_results.items()
        }

        reference = torch.tensor(
            [probability_map.get(key, 0.0) for key in keys],
            dtype=output.dtype,
        )

        assert torch.allclose(output, reference, atol=1e-6)

    def test_experiment_layer_matches_perceval_threshold(self):
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())
        circuit.add(1, pcvl.PS(torch.pi / 4))

        input_state = [1, 1]

        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.threshold()

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=input_state,
            no_bunching=False,
        )

        output = layer().squeeze(0)
        keys = [tuple(key) for key in layer.output_keys]
        assert output.shape[0] == len(keys)

        processor = pcvl.Processor("SLOS", experiment)
        processor.with_input(pcvl.BasicState(input_state))
        processor.min_detected_photons_filter(1)

        raw_results = processor.probs()["results"]
        probability_map = {
            tuple(int(v) for v in state): float(prob)
            for state, prob in raw_results.items()
        }

        reference = torch.tensor(
            [probability_map.get(key, 0.0) for key in keys],
            dtype=output.dtype,
        )

        assert torch.allclose(output, reference, atol=1e-6)

    def test_detector_choice_adjusts_output_size(self):
        circuit = pcvl.Circuit(3)
        circuit.add((0, 1), pcvl.BS())
        circuit.add((1, 2), pcvl.BS())
        input_state = [3, 0, 0]

        pnr_layer = ML.QuantumLayer(
            input_size=0,
            circuit=circuit,
            input_state=input_state,
            no_bunching=False,
        )

        pnr_output_size = pnr_layer.output_size
        assert pnr_output_size == len(pnr_layer.output_keys)

        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.threshold()
        experiment.detectors[2] = pcvl.Detector.threshold()

        threshold_layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=input_state,
            no_bunching=False,
        )

        threshold_output_size = threshold_layer.output_size
        assert threshold_output_size == len(threshold_layer.output_keys)
        assert threshold_output_size < pnr_output_size

    def test_detector_autograd_compatibility(self):
        """Detector transforms must preserve autograd support."""

        circuit = pcvl.Circuit(2)
        circuit.add(0, pcvl.PS(pcvl.P("theta_1")))
        circuit.add(0, pcvl.BS())
        circuit.add(0, pcvl.PS(pcvl.P("phi")))
        circuit.add(0, pcvl.BS())
        circuit.add(0, pcvl.PS(pcvl.P("theta_2")))
        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.threshold()

        layer = ML.QuantumLayer(
            input_size=1,
            experiment=experiment,
            input_state=[1, 0],
            input_parameters=["phi"],
            trainable_parameters=["theta"],
        )
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 1))

        x = torch.linspace(0.0, 0.5, steps=4, requires_grad=True)
        output = model(x.unsqueeze(1))
        target = torch.ones_like(output)
        loss = torch.sum(target - output)
        loss.backward()

        assert model[1].weight.grad is not None
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_simple_experiment_layer_detectors_vs_perceval(self):
        circuit = pcvl.Circuit(2)
        circuit.add((0, 1), pcvl.BS())
        circuit.add(1, pcvl.PS(torch.pi / 4))
        circuit.add((0, 1), pcvl.BS())

        input_state = [1, 1]

        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.pnr()
        experiment.detectors[1] = pcvl.Detector.threshold()

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=input_state,
            no_bunching=False,
        )

        output = layer().squeeze(0)
        keys = layer.output_keys
        assert output.shape[0] == len(keys)

        processor = pcvl.Processor("SLOS", experiment)
        processor.with_input(pcvl.BasicState(input_state))
        processor.min_detected_photons_filter(1)

        raw_results = processor.probs()["results"]
        probability_map = {
            tuple(int(v) for v in state): float(prob)
            for state, prob in raw_results.items()
        }

        reference = torch.tensor(
            [probability_map.get(key, 0.0) for key in keys],
            dtype=output.dtype,
        )

        assert torch.allclose(output, reference, atol=1e-6)

    def test_complex_experiment_layer_detectors_vs_perceval(self):
        exp = pcvl.Experiment()
        exp.set_circuit(pcvl.Unitary(pcvl.Matrix.random_unitary(4)))
        exp.add(0, pcvl.Detector.threshold())
        exp.add(1, pcvl.Detector.pnr())
        exp.add(2, pcvl.Detector.ppnr(2))
        exp.add(3, pcvl.Detector.ppnr(3))

        layer = ML.QuantumLayer(
            input_size=0,
            experiment=exp,
            input_state=[1, 1, 1, 1],
            no_bunching=False,
        )

        layer_probs = layer()
        layer_keys = layer.output_keys

        p = pcvl.Processor("SLOS", exp)
        p.with_input(pcvl.BasicState([1, 1, 1, 1]))
        p.min_detected_photons_filter(1)

        raw_results = p.probs()["results"]
        probability_map = {
            tuple(int(v) for v in state): float(prob)
            for state, prob in raw_results.items()
        }

        reference = torch.tensor(
            [probability_map.get(key, 0.0) for key in layer_keys],
            dtype=layer_probs.dtype,
        )

        assert torch.allclose(layer_probs, reference, atol=1e-6)

    def test_detector_plus_no_bunching_error(self):
        # Define Experiment without Detector
        experiment = pcvl.Experiment()
        experiment.set_circuit(pcvl.Unitary(pcvl.Matrix.random_unitary(4)))
        # No detector

        # Define Experiement with pnr Detector
        experiment_pnr_detector = pcvl.Experiment()
        experiment_pnr_detector.set_circuit(pcvl.Unitary(pcvl.Matrix.random_unitary(4)))
        experiment_pnr_detector.detectors[1] = pcvl.Detector.pnr()

        # Define Experiement with threshold Detector
        experiment_threshold_detector = pcvl.Experiment()
        experiment_threshold_detector.set_circuit(
            pcvl.Unitary(pcvl.Matrix.random_unitary(4))
        )
        experiment_threshold_detector.detectors[3] = pcvl.Detector.threshold()

        # Define Experiement with ppnr Detertors
        experiment_ppnr_detector = pcvl.Experiment()
        experiment_ppnr_detector.set_circuit(
            pcvl.Unitary(pcvl.Matrix.random_unitary(4))
        )
        experiment_ppnr_detector.detectors[0] = pcvl.Detector.ppnr(n_wires=2)
        experiment_ppnr_detector.detectors[2] = pcvl.Detector.ppnr(n_wires=2)

        # Define layers
        ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1, 1, 1],
            no_bunching=False,
        )
        ML.QuantumLayer(
            input_size=0,
            experiment=experiment,
            input_state=[1, 1, 1, 1],
            no_bunching=True,
        )
        ML.QuantumLayer(
            input_size=0,
            experiment=experiment_pnr_detector,
            input_state=[1, 1, 1, 1],
            no_bunching=False,
        )

        with pytest.raises(RuntimeError):
            ML.QuantumLayer(
                input_size=0,
                experiment=experiment_pnr_detector,
                input_state=[1, 1, 1, 1],
                no_bunching=True,
            )
        ML.QuantumLayer(
            input_size=0,
            experiment=experiment_threshold_detector,
            input_state=[1, 1, 1, 1],
            no_bunching=False,
        )
        with pytest.raises(RuntimeError):
            ML.QuantumLayer(
                input_size=0,
                experiment=experiment_threshold_detector,
                input_state=[1, 1, 1, 1],
                no_bunching=True,
            )
        ML.QuantumLayer(
            input_size=0,
            experiment=experiment_ppnr_detector,
            input_state=[1, 1, 1, 1],
            no_bunching=False,
        )
        with pytest.raises(RuntimeError):
            ML.QuantumLayer(
                input_size=0,
                experiment=experiment_ppnr_detector,
                input_state=[1, 1, 1, 1],
                no_bunching=True,
            )


class TestDetectorsWithKernels:
    """Test suite for Detectors integration with Kernels."""

    def test_fidelity_kernel_with_threshold_detectors(self):
        """FidelityKernel should respect detector configuration from the experiment."""

        feature_map = ML.FeatureMap.simple(
            input_size=1,
            n_modes=2,
            n_photons=1,
            trainable=False,
        )

        experiment = feature_map.experiment
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.threshold()

        kernel = ML.FidelityKernel(
            feature_map=feature_map,
            input_state=[1, 0],
        )

        data = torch.tensor([[0.0], [1.0]], dtype=kernel.dtype)
        K = kernel(data, data)

        assert K.shape == (2, 2)
        assert torch.allclose(
            torch.diag(K), torch.ones(2, dtype=kernel.dtype), atol=1e-6
        )

    def test_fidelity_kernel_with_mixed_detectors(self):
        """FidelityKernel can combine PNR and threshold detectors seamlessly."""

        feature_map = ML.FeatureMap.simple(
            input_size=1,
            n_modes=3,
            n_photons=1,
            trainable=False,
        )

        experiment = feature_map.experiment
        experiment.detectors[0] = pcvl.Detector.pnr()
        experiment.detectors[1] = pcvl.Detector.threshold()
        experiment.detectors[2] = pcvl.Detector.threshold()

        kernel = ML.FidelityKernel(
            feature_map=feature_map,
            input_state=[1, 0, 0],
        )

        data = torch.tensor([[0.0], [0.5], [1.0]], dtype=kernel.dtype)
        K = kernel(data, data)

        assert K.shape == (3, 3)
        diag = torch.diag(K)
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)
        assert torch.allclose(K, K.T.conj(), atol=1e-6)

    def test_fidelity_kernel_with_ppnr_detectors(self):
        """FidelityKernel supports partially projected number-resolving detectors."""

        feature_map = ML.FeatureMap.simple(
            input_size=1,
            n_modes=3,
            n_photons=1,
            trainable=False,
        )

        experiment = feature_map.experiment
        experiment.detectors[0] = pcvl.Detector.ppnr(n_wires=2, max_detections=1)
        experiment.detectors[1] = pcvl.Detector.pnr()
        experiment.detectors[2] = pcvl.Detector.pnr()

        kernel = ML.FidelityKernel(
            feature_map=feature_map,
            input_state=[1, 0, 0],
        )

        data = torch.tensor([[0.0], [0.6]], dtype=kernel.dtype)
        K = kernel(data, data)

        assert K.shape == (2, 2)
        diag = torch.diag(K)
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)
        assert torch.allclose(K, K.T.conj(), atol=1e-6)
        assert torch.all(K >= 0)

    def test_fidelity_kernel_detector_choice_adjusts_key_space(self):
        """Detector selection should change the size of the kernel detection basis."""

        feature_map = ML.FeatureMap.simple(
            input_size=1,
            n_modes=3,
            n_photons=3,
            trainable=False,
        )
        input_state = [1, 1, 1]

        kernel_pnr = ML.FidelityKernel(
            feature_map=feature_map,
            input_state=input_state,
        )

        feature_map_threshold = ML.FeatureMap.simple(
            input_size=1,
            n_modes=3,
            n_photons=3,
            trainable=False,
        )
        experiment_threshold = feature_map_threshold.experiment
        experiment_threshold.detectors[0] = pcvl.Detector.threshold()
        experiment_threshold.detectors[1] = pcvl.Detector.threshold()
        experiment_threshold.detectors[2] = pcvl.Detector.threshold()

        kernel_threshold = ML.FidelityKernel(
            feature_map=feature_map_threshold,
            input_state=input_state,
        )

        keys_pnr = kernel_pnr._detector_transform.output_keys
        keys_threshold = kernel_threshold._detector_transform.output_keys

        assert kernel_pnr._detector_transform.output_size == len(keys_pnr)
        assert kernel_threshold._detector_transform.output_size == len(keys_threshold)
        assert len(keys_pnr) > len(keys_threshold)
        assert all(sum(key) == sum(input_state) for key in keys_pnr)
        assert any(sum(key) < sum(input_state) for key in keys_threshold)
        assert all(value in (0, 1) for key in keys_threshold for value in key)

    def test_fidelity_kernel_respects_feature_map_experiment(self):
        """FidelityKernel should inherit detector configuration provided via FeatureMap."""

        circuit = pcvl.Circuit(2)
        circuit.add(0, pcvl.PS(pcvl.P("px")))
        circuit.add((0, 1), pcvl.BS())

        experiment = pcvl.Experiment(circuit)
        experiment.detectors[0] = pcvl.Detector.threshold()
        experiment.detectors[1] = pcvl.Detector.pnr()

        feature_map = ML.FeatureMap(
            input_size=1,
            experiment=experiment,
            input_parameters=["px"],
        )

        kernel = ML.FidelityKernel(
            feature_map=feature_map,
            input_state=[1, 1],
        )

        x_train = torch.tensor([[0.0], [0.5], [1.0]], dtype=kernel.dtype)
        x_test = torch.tensor([[0.2], [0.8]], dtype=kernel.dtype)

        k_train = kernel(x_train)
        k_test = kernel(x_test, x_train)

        assert k_train.shape == (3, 3)
        assert k_test.shape == (2, 3)
        diag = torch.diag(k_train)
        assert torch.allclose(diag, torch.ones_like(diag), atol=1e-6)
        assert torch.all(k_train >= 0)
        assert torch.all(k_test >= 0)
