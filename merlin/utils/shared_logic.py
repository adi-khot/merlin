# Utility module to decouple shared logic and resolve circular dependencies

# Add shared logic here that is used by both builder/ansatz.py and core/__init__.py

import torch
from merlin.pcvl_pytorch.locirc_to_tensor import CircuitConverter
from merlin.core.generators import CircuitGenerator

class Ansatz:
    """Complete configuration for a quantum neural network layer."""
    def __init__(
        self,
        PhotonicBackend,
        input_size,
        output_size=None,
        output_mapping_strategy=None,
        dtype=None,
    ):
        self.experiment = PhotonicBackend
        self.input_size = input_size
        self.output_size = output_size
        self.output_mapping_strategy = output_mapping_strategy
        self.dtype = dtype or torch.float32
        self.device = None

        # Create feature encoder
        self.feature_encoder = None  # Placeholder for FeatureEncoder

        # Generate circuit and state
        self.circuit = CircuitGenerator.generate(self.experiment, self.input_size)
        self.input_state = None  # Placeholder for StateGenerator

    def _build_computation_process(self):
        self.computation_process = ComputationProcessFactory.create(
            circuit=self.circuit,
            input_state=self.input_state,
            trainable_parameters=[],
            input_parameters=[],
            reservoir_mode=False,
            dtype=self.dtype,
            device=self.device,
        )
        return self.computation_process

class AnsatzFactory:
    """Factory for creating quantum layer ansatzes."""
    @staticmethod
    def create(
        PhotonicBackend,
        input_size,
        output_size=None,
        output_mapping_strategy=None,
        dtype=None,
    ):
        return Ansatz(
            PhotonicBackend=PhotonicBackend,
            input_size=input_size,
            output_size=output_size,
            output_mapping_strategy=output_mapping_strategy,
            dtype=dtype,
        )

class ComputationProcessFactory:
    """Factory for creating computation processes."""

    @staticmethod
    def create(circuit, input_state, trainable_parameters, input_parameters, reservoir_mode, dtype, device, **kwargs):
        """Create and return a computation process based on the provided parameters."""
        # Example implementation (replace with actual logic as needed)
        if reservoir_mode:
            return ReservoirComputationProcess(
                circuit=circuit,
                input_state=input_state,
                trainable_parameters=trainable_parameters,
                input_parameters=input_parameters,
                dtype=dtype,
                device=device,
                **kwargs
            )
        else:
            return StandardComputationProcess(
                circuit=circuit,
                input_state=input_state,
                trainable_parameters=trainable_parameters,
                input_parameters=input_parameters,
                dtype=dtype,
                device=device,
                **kwargs
            )

class ReservoirComputationProcess:
    """Implementation for reservoir computation process."""
    def __init__(self, circuit, input_state, trainable_parameters, input_parameters, dtype, device, **kwargs):
        self.circuit = circuit
        self.input_state = input_state
        self.trainable_parameters = trainable_parameters
        self.input_parameters = input_parameters
        self.dtype = dtype
        self.device = device
        # Initialize the CircuitConverter
        self.converter = CircuitConverter(
            circuit=circuit,
            input_specs=trainable_parameters,
            device=device,
        )


class StandardComputationProcess:
    """Implementation for standard computation process."""
    def __init__(self, circuit, input_state, trainable_parameters, input_parameters, dtype, device, **kwargs):
        self.circuit = circuit
        self.input_state = input_state
        self.trainable_parameters = trainable_parameters
        self.input_parameters = input_parameters
        self.dtype = dtype
        self.device = device
        # Initialize the CircuitConverter
        self.converter = CircuitConverter(
            circuit=circuit,
            input_specs=trainable_parameters,
            device=device,
        )