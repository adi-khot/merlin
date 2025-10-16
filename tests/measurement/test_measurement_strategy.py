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

import math

import perceval as pcvl
import pytest
import torch

import merlin as ML
from merlin.measurement.strategies import MeasurementStrategy


class TestQuantumLayerMeasurementStrategy:
    def test_fock_distribution(self):
        # FockDistribution is equivalent to OutputMappingStrategy.NONE
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=3, n_photons=1
        )
        output_size = math.comb(3, 1)  # = 3
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=output_size,
            measurement_strategy=MeasurementStrategy.FOCKDISTRIBUTION,
        )
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)

        # Good output shape
        assert output.shape == (2, 3)

        # Ensure positive values (that represent probabilities)
        assert torch.all(output >= -1e6)

        # Ensure values sum to 1 (within numerical precision)
        assert torch.allclose(output.sum(dim=-1), torch.tensor(1.0), atol=1e-6)

        # Backprop compatibility
        output.sum().backward()
        assert x.grad is not None

        # Ensure that QuantumLayer with FockDistribution strategy can be initialized without specifying output_size and that its output_size is accessible
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=3, n_photons=1
        )
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            measurement_strategy=MeasurementStrategy.FOCKDISTRIBUTION,
        )
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)
        assert output.shape[-1] == layer.output_size

        # Works with full QuantumLayer API
        circuit = pcvl.Circuit(3)
        circuit.add(0, pcvl.BS(pcvl.P("px_0")))
        circuit.add(1, pcvl.BS(pcvl.P("px_1")))
        input_state = [1, 1, 0]
        layer = ML.QuantumLayer(
            input_size=2,
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=[],
            input_parameters=["px"],
            measurement_strategy=MeasurementStrategy.FOCKDISTRIBUTION,
            no_bunching=True,
        )
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)
        assert output.shape[-1] == layer.output_size
        output.sum().backward()
        assert x.grad is not None

        # Error: wrong output size specification (output_size must be None or equal to the number of possible Fock states)
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=3, n_photons=1
        )
        output_size = 20  # Wrong output size
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=output_size,
            measurement_strategy=MeasurementStrategy.FOCKDISTRIBUTION,
        )
        with pytest.raises(ValueError):
            layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

    def test_linear_equivalent(self):
        # OutputMappingStrategy.LINEAR is equivalent to FockDistribution + torch.nn.Linear
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=3, n_photons=1
        )
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            measurement_strategy=MeasurementStrategy.FOCKDISTRIBUTION,
        )
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        linear = torch.nn.Linear(layer.output_size, 2)
        x = torch.rand(5, 2, requires_grad=True)
        output = linear(layer(x))

        # Good output shape
        assert output.shape == (5, 2)

        # Ensure finite values
        assert torch.all(torch.isfinite(output))

        # Backprop compatibility
        output.sum().backward()
        assert x.grad is not None

        # Works with full QuantumLayer API
        circuit = pcvl.Circuit(3)
        circuit.add(0, pcvl.BS(pcvl.P("px_0")))
        circuit.add(1, pcvl.BS(pcvl.P("px_1")))
        input_state = [1, 1, 0]
        layer = ML.QuantumLayer(
            input_size=2,
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=[],
            input_parameters=["px"],
            measurement_strategy=MeasurementStrategy.FOCKDISTRIBUTION,
            no_bunching=True,
        )
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 2))
        x = torch.rand(2, 2, requires_grad=True)
        output = model(x)
        assert output.shape[-1] == 2
        output.sum().backward()
        assert x.grad is not None

    def test_mode_expectation(self):
        # ModeExpectation is a new strategy unlike any previous OutputMappingStrategy
        n_modes = 3
        n_photons = 2
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=n_modes, n_photons=n_photons
        )
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=n_modes,
            measurement_strategy=MeasurementStrategy.MODEEXPECTATION,
        )
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)

        # Good output shape
        assert output.shape == (2, n_modes)

        # Ensure finite values
        assert torch.all(torch.isfinite(output))

        # Backprop compatibility
        output.sum().backward()
        assert x.grad is not None

        # By default, no_bunching=True so output values cannot surpass 1
        assert torch.all(output <= 1.0 + 1e-6)
        assert torch.all(output >= -1e-6)  # No negative values

        # ModeExpectation with explicit no_bunching=True
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz, no_bunching=True)
        output = layer(x)
        assert output.shape == (2, n_modes)
        assert torch.all(torch.isfinite(output))
        output.sum().backward()
        assert x.grad is not None
        assert torch.all(output <= 1.0 + 1e-6)
        assert torch.all(output >= -1e-6)  # No negative values

        # ModeExpectation with no_bunching=False (allows output values > 1)
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz, no_bunching=False)
        output = layer(x)
        assert output.shape == (2, n_modes)
        assert torch.all(torch.isfinite(output))
        output.sum().backward()
        assert x.grad is not None
        assert torch.all(output >= -1e-6)  # No negative values
        assert torch.all(
            output <= n_photons + 1e-6
        )  # Some values can surpass 1 but cannot surpass the number of photons

        # ModeExpectation can be initialized without specifying output_size. Its output_size is accessible and equal to n_modes
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            measurement_strategy=MeasurementStrategy.MODEEXPECTATION,
        )
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        assert layer.output_size == n_modes
        output = layer(x)
        assert output.shape == (2, n_modes)
        assert torch.all(torch.isfinite(output))
        output.sum().backward()
        assert x.grad is not None

        # Works with full QuantumLayer API
        circuit = pcvl.Circuit(3)
        circuit.add(0, pcvl.BS(pcvl.P("px_0")))
        circuit.add(1, pcvl.BS(pcvl.P("px_1")))
        input_state = [1, 1, 0]
        layer = ML.QuantumLayer(
            input_size=2,
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=[],
            input_parameters=["px"],
            measurement_strategy=MeasurementStrategy.MODEEXPECTATION,
            no_bunching=True,
        )
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)
        assert output.shape[-1] == 3
        output.sum().backward()
        assert x.grad is not None

        """# Error: ModeExpectation with wrong output_size (not None or n_modes)
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=10,  # Wrong output size
            measurement_strategy=MeasurementStrategy.MODEEXPECTATION,
        )
        with pytest.raises(ValueError):
            layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)"""

    def test_state_vector(self):
        # StateVector is equivalent to return_amplitudes=True
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=3, n_photons=1
        )
        output_size = math.comb(3, 1)  # = 3
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=output_size,
            measurement_strategy=MeasurementStrategy.STATEVECTOR,
        )
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)

        # Good output shape
        assert output.shape == (2, layer.output_size)

        # Ensure finite values
        assert torch.all(torch.isfinite(output))

        # Backprop compatibility
        output.sum().backward()
        assert x.grad is not None

        # Ensure that it is normalized
        assert torch.allclose(
            torch.sum(output.abs() ** 2, dim=-1), torch.ones(output.shape[0]), atol=1e-6
        )

        # Ensure that QuantumLayer with StateVector strategy can be initialized without specifying output_size and that its output_size is accessible and equal to the number of possible Fock states
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            measurement_strategy=MeasurementStrategy.STATEVECTOR,
        )
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)
        assert output.shape[-1] == layer.output_size
        assert torch.allclose(
            torch.sum(output.abs() ** 2, dim=-1), torch.ones(output.shape[0]), atol=1e-6
        )

        # Works with full QuantumLayer API
        circuit = pcvl.Circuit(3)
        circuit.add(0, pcvl.BS(pcvl.P("px_0")))
        circuit.add(1, pcvl.BS(pcvl.P("px_1")))
        input_state = [1, 1, 0]
        layer = ML.QuantumLayer(
            input_size=2,
            circuit=circuit,
            input_state=input_state,
            trainable_parameters=[],
            input_parameters=["px"],
            measurement_strategy=MeasurementStrategy.STATEVECTOR,
            no_bunching=True,
        )
        x = torch.rand(2, 2, requires_grad=True)
        output = layer(x)
        assert output.shape[-1] == layer.output_size
        output.sum().backward()
        assert x.grad is not None
        assert torch.allclose(
            torch.sum(output.abs() ** 2, dim=-1), torch.ones(output.shape[0]), atol=1e-6
        )

        # Error: wrong output size specification (output_size must be None or equal to the number of possible Fock states)
        output_size = 20  # Wrong output size
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=output_size,
            measurement_strategy=MeasurementStrategy.STATEVECTOR,
        )
        with pytest.raises(ValueError):
            layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)

    def test_custom_observable(self):
        # CustomObservable strategy (should be implemented, but test placeholder)
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=2, n_photons=1
        )
        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            output_size=2,
            measurement_strategy=MeasurementStrategy.CUSTOMOBSERVABLE,
        )
        ML.QuantumLayer(input_size=2, ansatz=ansatz)
        torch.rand(2, 2, requires_grad=True)
        # TODO: implement CustomObservable strategy
