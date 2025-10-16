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

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import merlin as ML


class TestOutputMapper:
    def test_linear_mapping_creation(self):
        """Test creation of linear output mapping."""
        fock_distribution = ML.OutputMapper.create_mapping(
            ML.MeasurementStrategy.FOCKDISTRIBUTION
        )
        mapping = torch.nn.Sequential(fock_distribution, nn.Linear(6, 3))
        assert isinstance(mapping[0], ML.FockDistribution)
        assert isinstance(mapping[-1], nn.Linear)
        assert mapping[-1].in_features == 6
        assert mapping[-1].out_features == 3

    def test_fock_distribution_mapping_creation(self):
        fock_distribution = ML.OutputMapper.create_mapping(
            ML.MeasurementStrategy.FOCKDISTRIBUTION
        )
        assert isinstance(fock_distribution, ML.FockDistribution)

    def test_state_vector_mapping_creation_valid(self):
        """Test creation of state vector mapping with matching sizes."""
        mapping = ML.OutputMapper.create_mapping(ML.MeasurementStrategy.STATEVECTOR)
        batch_size = 4
        input_amps = torch.rand(batch_size, 5)
        output_amps = mapping(input_amps)
        assert isinstance(mapping, ML.StateVector)
        assert torch.allclose(input_amps, output_amps, atol=1e-6)

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""

        class FakeStrategy:
            pass

        with pytest.raises(ValueError, match="Unknown measurement strategy"):
            ML.OutputMapper.create_mapping(FakeStrategy())


class TestOutputMappingIntegration:
    """Integration tests for output mapping with QuantumLayer."""

    def test_linear_mapping_integration(self):
        """Test linear mapping integration with QuantumLayer."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            measurement_strategy=ML.MeasurementStrategy.FOCKDISTRIBUTION,
        )

        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))

        x = torch.rand(5, 2)
        output = model(x)

        assert output.shape == (5, 3)
        assert torch.all(torch.isfinite(output))

    def test_mapping_gradient_flow(self):
        """Test gradient flow through different mapping strategies."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=6, n_photons=2
        )

        strategies = [
            ML.MeasurementStrategy.FOCKDISTRIBUTION,
            ML.MeasurementStrategy.MODEEXPECTATION,
            ML.MeasurementStrategy.STATEVECTOR,
        ]

        for strategy in strategies:
            ansatz = ML.AnsatzFactory.create(
                PhotonicBackend=experiment,
                input_size=2,
                measurement_strategy=strategy,
            )

            layer = ML.QuantumLayer(input_size=2, ansatz=ansatz)
            model = (
                torch.nn.Sequential(
                    layer, torch.nn.Linear(layer.output_size, 3, dtype=torch.float32)
                )
                if strategy is not strategies[-1]
                else torch.nn.Sequential(
                    layer, torch.nn.Linear(layer.output_size, 3, dtype=torch.complex64)
                )
            )

            x = torch.rand(2, 2, requires_grad=True)
            output = model(x)

            if output.dtype == torch.complex64:
                output = output.to(torch.float32)

            # Use MSE loss instead of sum for better gradient flow
            target = torch.ones_like(output)
            loss = F.mse_loss(output, target)
            loss.backward()

            # Input should have gradients
            assert x.grad is not None, f"No gradients for strategy {strategy}"
            assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), (
                f"Zero gradients for strategy {strategy}"
            )

    def test_mapping_output_bounds(self):
        """Test that different mappings produce reasonable output bounds."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        x = torch.rand(5, 2)

        # LINEAR mapping - can have any range
        ansatz_linear = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=2,
            measurement_strategy=ML.MeasurementStrategy.FOCKDISTRIBUTION,
        )
        layer = ML.QuantumLayer(input_size=2, ansatz=ansatz_linear)
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 3))
        output_linear = model(x)
        assert torch.all(torch.isfinite(output_linear))

    def test_dtype_consistency_in_mappings(self):
        """Test that output mappings respect input dtypes."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL_COLUMNS, n_modes=4, n_photons=2
        )

        for dtype in [torch.float32, torch.float64]:
            ansatz = ML.AnsatzFactory.create(
                PhotonicBackend=experiment,
                input_size=2,
                measurement_strategy=ML.MeasurementStrategy.FOCKDISTRIBUTION,
                dtype=dtype,
            )

            layer = ML.QuantumLayer(input_size=2, ansatz=ansatz, dtype=dtype)
            model = torch.nn.Sequential(
                layer, torch.nn.Linear(layer.output_size, 3, dtype=dtype)
            )

            x = torch.rand(3, 2, dtype=dtype)
            output = model(x)

            # Output should be finite and have reasonable values
            assert torch.all(torch.isfinite(output))
            assert output.shape == (3, 3)

    def test_edge_case_single_dimension(self):
        """Test edge case with single input/output dimensions."""
        experiment = ML.PhotonicBackend(
            circuit_type=ML.CircuitType.PARALLEL, n_modes=3, n_photons=1
        )

        ansatz = ML.AnsatzFactory.create(
            PhotonicBackend=experiment,
            input_size=1,
            measurement_strategy=ML.MeasurementStrategy.FOCKDISTRIBUTION,
        )

        layer = ML.QuantumLayer(input_size=1, ansatz=ansatz)
        model = torch.nn.Sequential(layer, torch.nn.Linear(layer.output_size, 1))

        x = torch.rand(5, 1)
        output = model(x)

        assert output.shape == (5, 1)
        assert torch.all(torch.isfinite(output))
