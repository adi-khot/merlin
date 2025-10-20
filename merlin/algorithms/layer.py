"""
Main QuantumLayer implementation with bug fixes and index_photons support.
Modified for MerlinProcessor compatibility - removed direct cloud processor coupling.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn

from ..builder.ansatz import Ansatz
from ..builder.circuit_builder import (
    ANGLE_ENCODING_MODE_ERROR,
    CircuitBuilder,
)
from ..core.components import GenericInterferometer
from ..core.process import ComputationProcessFactory
from ..sampling.autodiff import AutoDiffProcess
from ..sampling.strategies import OutputMappingStrategy
from ..torch_utils.torch_codes import OutputMapper


class QuantumLayer(nn.Module):
    """
    Enhanced Quantum Neural Network Layer.

    This layer can be created either from:
    1. An Ansatz object (from AnsatzFactory) - for auto-generated circuits
    2. Direct parameters - for custom circuits (backward compatible)

    Supports local PyTorch SLOS execution. Remote execution handled by MerlinProcessor.
    """

    def __init__(
            self,
            input_size: int,
            output_size: int | None = None,
            # Ansatz-based construction
            ansatz: Ansatz | None = None,
            # Custom circuit construction (backward compatible)
            circuit: pcvl.Circuit | pcvl.Experiment | CircuitBuilder | None = None,
            input_state: list[int] | None = None,
            n_photons: int | None = None,
            trainable_parameters: list[str] = None,
            input_parameters: list[str] = None,
            # Common parameters
            output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.LINEAR,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            shots: int = 0,
            sampling_method: str = "multinomial",
            no_bunching: bool = True,
            # New parameter for constrained photon placement
            index_photons: list[tuple[int, int]] | None = None,
    ):
        super().__init__()

        self.device = device
        self.dtype = dtype or torch.float32
        self.input_size = input_size
        self.no_bunching = no_bunching
        self.index_photons = index_photons
        self.shots = shots
        self.sampling_method = sampling_method

        # Store current parameter values for export
        self._current_params = {}

        # Track trainable parameters for optimization
        self.trainable_parameters = trainable_parameters or []
        self.input_parameters = input_parameters or []

        builder_trainable: list[str] = []
        builder_input: list[str] = []

        self.angle_encoding_specs: dict[str, dict[str, Any]] = {}

        # Handle Experiment vs Circuit
        self.experiment = None
        self.noise_model = None

        if isinstance(circuit, pcvl.Experiment):
            self.experiment = circuit
            actual_circuit = circuit.unitary_circuit(flatten=False)
            self.noise_model = circuit.noise
        elif isinstance(circuit, CircuitBuilder):
            builder_trainable = circuit.trainable_parameter_prefixes
            builder_input = circuit.input_parameter_prefixes
            self.angle_encoding_specs = circuit.angle_encoding_specs
            actual_circuit = circuit.to_pcvl_circuit(pcvl)
        else:
            actual_circuit = circuit

        # Fix trainable and input parameters from builder or circuit
        if trainable_parameters is None:
            trainable_parameters = list(builder_trainable)
        else:
            trainable_parameters = list(trainable_parameters)

        if input_parameters is None:
            input_parameters = list(builder_input)
        else:
            input_parameters = list(input_parameters)

        # Store for later use
        self.trainable_parameters = trainable_parameters
        self.input_parameters = input_parameters

        # Determine construction mode
        if ansatz is not None:
            self._init_from_ansatz(ansatz, output_size, output_mapping_strategy)
        elif actual_circuit is not None:
            self.circuit = actual_circuit
            self._init_from_custom_circuit(
                actual_circuit,
                input_state,
                n_photons,
                trainable_parameters,
                input_parameters,
                output_size,
                output_mapping_strategy,
            )
        else:
            raise ValueError("Either 'ansatz' or 'circuit' must be provided")

        # Setup sampling
        self.autodiff_process = AutoDiffProcess(sampling_method)

        # Check if this is a fixed layer (no trainable parameters)
        self._is_fixed = len(self.trainable_parameters) == 0

    def _init_from_ansatz(
            self,
            ansatz: Ansatz,
            output_size: int | None,
            output_mapping_strategy: OutputMappingStrategy,
    ):
        """Initialize from ansatz (auto-generated mode)."""
        self.ansatz = ansatz
        self.auto_generation_mode = True

        # For ansatz mode, we need to create a new computation process with correct device
        if self.index_photons is not None:
            self.computation_process = ComputationProcessFactory.create(
                circuit=ansatz.circuit,
                input_state=ansatz.input_state,
                trainable_parameters=ansatz.trainable_parameters,
                input_parameters=ansatz.input_parameters,
                reservoir_mode=ansatz.experiment.reservoir_mode,
                device=self.device,
                dtype=self.dtype,
                no_bunching=self.no_bunching,
                index_photons=self.index_photons,
            )
        else:
            if self.device is not None:
                ansatz.device = self.device
            self.computation_process = ansatz._build_computation_process()

        self.feature_encoder = ansatz.feature_encoder
        actual_strategy = ansatz.output_mapping_strategy
        actual_output_size = output_size or ansatz.output_size

        # Setup bandwidth tuning if enabled
        if ansatz.experiment.use_bandwidth_tuning:
            self.bandwidth_coeffs = nn.ParameterDict()
            for d in range(self.input_size):
                init = torch.linspace(
                    0.0,
                    2.0,
                    steps=ansatz.experiment.n_modes,
                    dtype=self.dtype,
                    device=self.device,
                )
                self.bandwidth_coeffs[f"dim_{d}"] = nn.Parameter(
                    init.clone(), requires_grad=True
                )
        else:
            self.bandwidth_coeffs = None

        self._setup_parameters_from_ansatz(ansatz)
        self._setup_output_mapping(ansatz, actual_output_size, actual_strategy)

    def _init_from_custom_circuit(
            self,
            circuit: pcvl.Circuit,
            input_state: list[int] | None,
            n_photons: int | None,
            trainable_parameters: list[str],
            input_parameters: list[str],
            output_size: int | None,
            output_mapping_strategy: OutputMappingStrategy,
    ):
        """Initialize from custom circuit (backward compatible mode)."""
        self.auto_generation_mode = False
        self.bandwidth_coeffs = None

        # Handle state - with index_photons consideration
        if input_state is not None:
            self.input_state = input_state
            if self.index_photons is not None:
                self._validate_input_state_with_index_photons(input_state)
        elif n_photons is not None:
            if self.index_photons is not None:
                self.input_state = self._create_input_state_from_index_photons(
                    n_photons, circuit.m
                )
            else:
                self.input_state = [1] * n_photons + [0] * (circuit.m - n_photons)
        else:
            raise ValueError("Either input_state or n_photons must be provided")

        self.computation_process = ComputationProcessFactory.create(
            circuit=circuit,
            input_state=self.input_state,
            trainable_parameters=trainable_parameters,
            input_parameters=input_parameters,
            n_photons=n_photons,
            device=self.device,
            dtype=self.dtype,
            no_bunching=self.no_bunching,
            index_photons=self.index_photons,
        )

        self._setup_parameters_from_custom(trainable_parameters)
        self._setup_output_mapping_from_custom(output_size, output_mapping_strategy)

    def _validate_input_state_with_index_photons(self, input_state: list[int]):
        """Validate that input_state respects index_photons constraints."""
        if self.index_photons is None:
            return

        photon_idx = 0
        for mode_idx, photon_count in enumerate(input_state):
            for _ in range(photon_count):
                if photon_idx >= len(self.index_photons):
                    raise ValueError(
                        f"Input state has more photons than index_photons constraints."
                    )
                min_mode, max_mode = self.index_photons[photon_idx]
                if not (min_mode <= mode_idx <= max_mode):
                    raise ValueError(
                        f"Photon {photon_idx} is in mode {mode_idx} but constrained to [{min_mode}, {max_mode}]"
                    )
                photon_idx += 1

    def _create_input_state_from_index_photons(
            self, n_photons: int, n_modes: int
    ) -> list[int]:
        """Create input state respecting index_photons constraints."""
        if self.index_photons is None or len(self.index_photons) != n_photons:
            raise ValueError(
                f"index_photons must specify constraints for exactly {n_photons} photons."
            )

        input_state = [0] * n_modes
        for photon_idx, (min_mode, max_mode) in enumerate(self.index_photons):
            if not (0 <= min_mode <= max_mode < n_modes):
                raise ValueError(
                    f"Invalid index_photons constraint for photon {photon_idx}"
                )
            input_state[min_mode] += 1

        return input_state

    def _setup_parameters_from_ansatz(self, ansatz: Ansatz):
        """Setup parameters from ansatz configuration."""
        spec_mappings = self.computation_process.converter.spec_mappings
        self.thetas = []
        self.theta_names = []

        if not ansatz.experiment.reservoir_mode:
            for tp in ansatz.trainable_parameters:
                if tp in spec_mappings:
                    theta_list = spec_mappings[tp]
                    self.theta_names += theta_list
                    parameter = nn.Parameter(
                        torch.randn(
                            (len(theta_list),), dtype=self.dtype, device=self.device
                        )
                        * torch.pi
                    )
                    self.register_parameter(tp, parameter)
                    self.thetas.append(parameter)

        if ansatz.experiment.reservoir_mode and "phi_" in spec_mappings:
            phi_list = spec_mappings["phi_"]
            if phi_list:
                phi_values = []
                for _param_name in phi_list:
                    phi_values.append(2 * math.pi * np.random.rand())
                phi_tensor = torch.tensor(
                    phi_values, dtype=self.dtype, device=self.device
                )
                self.register_buffer("phi_static", phi_tensor)

    def _setup_parameters_from_custom(self, trainable_parameters: list[str] | None):
        """Setup parameters from custom circuit configuration."""
        spec_mappings = self.computation_process.converter.spec_mappings
        self.thetas = []
        self.theta_names = []

        if trainable_parameters is None:
            return

        for tp in trainable_parameters:
            if tp in spec_mappings:
                theta_list = spec_mappings[tp]
                self.theta_names += theta_list
                parameter = nn.Parameter(
                    torch.randn(
                        (len(theta_list),), dtype=self.dtype, device=self.device
                    )
                    * torch.pi
                )
                self.register_parameter(tp, parameter)
                self.thetas.append(parameter)

    def _setup_output_mapping(
            self,
            ansatz: Ansatz,
            output_size: int | None,
            output_mapping_strategy: OutputMappingStrategy,
    ):
        """Setup output mapping for ansatz-based construction."""
        dummy_params = self._create_dummy_parameters()
        distribution = self.computation_process.compute(dummy_params)
        dist_size = distribution.shape[-1]

        if output_size is None:
            if output_mapping_strategy == OutputMappingStrategy.NONE:
                self.output_size = dist_size
            else:
                raise ValueError(
                    "output_size must be specified for non-NONE strategies"
                )
        else:
            self.output_size = output_size

        if (
                output_mapping_strategy == OutputMappingStrategy.NONE
                and self.output_size != dist_size
        ):
            raise ValueError(
                f"Distribution size ({dist_size}) must equal output size ({self.output_size})"
            )

        self.output_mapping = OutputMapper.create_mapping(
            output_mapping_strategy, dist_size, self.output_size
        )

        if hasattr(self.output_mapping, "weight"):
            self.output_mapping = self.output_mapping.to(
                dtype=self.dtype, device=self.device
            )

    def _setup_output_mapping_from_custom(
            self, output_size: int | None, output_mapping_strategy: OutputMappingStrategy
    ):
        """Setup output mapping for custom circuit construction."""
        dummy_params = self._create_dummy_parameters()
        distribution = self.computation_process.compute(dummy_params)
        dist_size = distribution.shape[-1]

        if output_size is None:
            if output_mapping_strategy == OutputMappingStrategy.NONE:
                self.output_size = dist_size
            else:
                raise ValueError(
                    "output_size must be specified for non-NONE strategies"
                )
        else:
            self.output_size = output_size

        self.output_mapping = OutputMapper.create_mapping(
            output_mapping_strategy, dist_size, self.output_size
        )

        if hasattr(self.output_mapping, "weight"):
            self.output_mapping = self.output_mapping.to(
                dtype=self.dtype, device=self.device
            )

    def _create_dummy_parameters(self) -> list[torch.Tensor]:
        """Create dummy parameters for initialization."""
        params = list(self.thetas)

        if self.auto_generation_mode:
            dummy_input = torch.zeros(
                self.ansatz.total_shifters, dtype=self.dtype, device=self.device
            )
            params.append(dummy_input)
        else:
            spec_mappings = self.computation_process.converter.spec_mappings
            input_params = self.computation_process.input_parameters
            for ip in input_params:
                if ip in spec_mappings:
                    param_count = len(spec_mappings[ip])
                    dummy_input = torch.zeros(
                        param_count, dtype=self.dtype, device=self.device
                    )
                    params.append(dummy_input)

        if hasattr(self, "phi_static"):
            params.append(self.phi_static)

        return params

    def _update_current_params(self):
        """Update current parameter values for export."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                self._current_params[name] = param.detach().cpu().numpy()

    def export_config(self) -> dict:
        """Export configuration for remote deployment by MerlinProcessor."""
        self._update_current_params()

        # Get the circuit with current trained values
        if self.experiment:
            exported_circuit = self.experiment.unitary_circuit(flatten=False)
        else:
            exported_circuit = self.circuit.copy()

        # Set trainable parameter values in the circuit
        for p in exported_circuit.get_parameters():
            # Check if this is a trainable parameter
            for tp_prefix in self.trainable_parameters:
                if p.name.startswith(tp_prefix):
                    # Find the corresponding PyTorch parameter
                    for name, param in self.named_parameters():
                        if param.requires_grad:
                            # Match parameter names - handle different naming conventions
                            if tp_prefix in name:
                                # Find the index in the tensor
                                if hasattr(self.computation_process.converter, 'spec_mappings'):
                                    spec_mappings = self.computation_process.converter.spec_mappings
                                    if tp_prefix in spec_mappings:
                                        param_list = spec_mappings[tp_prefix]
                                        if p.name in param_list:
                                            idx = param_list.index(p.name)
                                            # Get value from the tensor
                                            value = param[idx].item()
                                            p.set_value(float(value))
                                            break

        config = {
            'circuit': exported_circuit,
            'experiment': self.experiment,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'input_state': self.input_state if hasattr(self, 'input_state') else None,
            'n_modes': exported_circuit.m,
            'n_photons': sum(self.input_state) if hasattr(self, 'input_state') else None,
            'trainable_parameters': self.trainable_parameters,
            'input_parameters': self.input_parameters,
            'no_bunching': self.no_bunching,
            'shots': self.shots,
            'noise_model': self.noise_model,
        }

        return config

    def get_experiment(self) -> Optional[pcvl.Experiment]:
        """Return Experiment object if available (for future MerlinProcessor use)."""
        return self.experiment

    def forward(
            self,
            *input_parameters: torch.Tensor,
            apply_sampling: bool | None = None,
            shots: int | None = None,
            return_amplitudes: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass through the quantum layer using local PyTorch SLOS.

        Remote execution should be handled externally by MerlinProcessor.
        """
        # Update current parameter values for potential export
        self._update_current_params()

        # PyTorch SLOS execution path
        params = self.prepare_parameters(list(input_parameters))

        if type(self.computation_process.input_state) is torch.Tensor:
            amplitudes = self.computation_process.compute_superposition_state(params)
        else:
            amplitudes = self.computation_process.compute(params)

        needs_gradient = (
                self.training
                and torch.is_grad_enabled()
                and any(p.requires_grad for p in self.parameters())
        )

        apply_sampling, shots = self.autodiff_process.autodiff_backend(
            needs_gradient, apply_sampling or False, shots or self.shots
        )

        distribution = amplitudes.real ** 2 + amplitudes.imag ** 2

        if self.no_bunching:
            sum_probs = distribution.sum(dim=1, keepdim=True)
            valid_entries = sum_probs > 0
            if valid_entries.any():
                distribution = torch.where(
                    valid_entries,
                    distribution / torch.where(valid_entries, sum_probs, torch.ones_like(sum_probs)),
                    distribution,
                )
                amplitudes = torch.where(
                    valid_entries,
                    amplitudes / torch.where(valid_entries, sum_probs.sqrt(), torch.ones_like(sum_probs)),
                    amplitudes,
                )

        if apply_sampling and shots > 0:
            distribution = self.autodiff_process.sampling_noise.pcvl_sampler(
                distribution, shots
            )

        if return_amplitudes:
            return self.output_mapping(distribution), amplitudes

        return self.output_mapping(distribution)

    def _prepare_input_encoding(
            self, x: torch.Tensor, prefix: str | None = None
    ) -> torch.Tensor:
        """Prepare input encoding based on mode."""
        if self.auto_generation_mode:
            x_norm = torch.clamp(x, 0, 1)
            return self.feature_encoder.encode(
                x_norm,
                self.ansatz.experiment.circuit_type,
                self.ansatz.experiment.n_modes,
                self.bandwidth_coeffs,
            )

        spec = None
        if prefix is not None:
            spec = self.angle_encoding_specs.get(prefix)
        elif len(self.angle_encoding_specs) == 1:
            spec = next(iter(self.angle_encoding_specs.values()))

        if spec:
            return self._apply_angle_encoding(x, spec)

        return x * torch.pi

    def _apply_angle_encoding(
            self, x: torch.Tensor, spec: dict[str, Any]
    ) -> torch.Tensor:
        """Apply custom angle encoding using stored metadata."""
        combos: list[tuple[int, ...]] = spec.get("combinations", [])
        scale_map: dict[int, float] = spec.get("scales", {})

        if x.dim() == 1:
            x_batch = x.unsqueeze(0)
            squeeze = True
        elif x.dim() == 2:
            x_batch = x
            squeeze = False
        else:
            raise ValueError(f"Angle encoding expects 1D or 2D tensors")

        if not combos:
            encoded = x_batch * torch.pi
            return encoded.squeeze(0) if squeeze else encoded

        encoded_cols: list[torch.Tensor] = []
        feature_dim = x_batch.shape[-1]

        for combo in combos:
            indices = list(combo)
            if any(idx >= feature_dim for idx in indices):
                raise ValueError(f"Input dimension insufficient for encoding")
            selected = x_batch[:, indices]
            scales = [scale_map.get(idx, 1.0) for idx in indices]
            scale_tensor = x_batch.new_tensor(scales)
            value = (selected * scale_tensor).sum(dim=1, keepdim=True)
            encoded_cols.append(value)

        encoded = torch.cat(encoded_cols, dim=1) if encoded_cols else x_batch.new_zeros((x_batch.shape[0], 0))
        return encoded.squeeze(0) if squeeze else encoded

    def prepare_parameters(
            self, input_parameters: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Prepare parameter list for circuit evaluation."""
        if input_parameters and input_parameters[0].dim() > 1:
            batch_size = input_parameters[0].shape[0]
            params = [theta.expand(batch_size, -1) for theta in self.thetas]
        else:
            params = list(self.thetas)

        prefixes = getattr(self.computation_process, "input_parameters", [])

        if self.auto_generation_mode and len(input_parameters) == 1:
            prefix = prefixes[0] if prefixes else None
            encoded = self._prepare_input_encoding(input_parameters[0], prefix)
            params.append(encoded)
        else:
            for idx, x in enumerate(input_parameters):
                prefix = None
                if prefixes:
                    prefix = prefixes[idx] if idx < len(prefixes) else prefixes[-1]
                encoded = self._prepare_input_encoding(x, prefix)
                params.append(encoded)

        if hasattr(self, "phi_static"):
            if input_parameters and input_parameters[0].dim() > 1:
                batch_size = input_parameters[0].shape[0]
                params.append(self.phi_static.expand(batch_size, -1))
            else:
                params.append(self.phi_static)

        return params

    def set_sampling_config(self, shots: int | None = None, method: str | None = None):
        """Update sampling configuration."""
        if shots is not None:
            if not isinstance(shots, int) or shots < 0:
                raise ValueError(f"shots must be a non-negative integer")
            self.shots = shots
        if method is not None:
            valid_methods = ["multinomial", "binomial", "gaussian"]
            if method not in valid_methods:
                raise ValueError(f"Invalid sampling method: {method}")
            self.sampling_method = method

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            device = args[0]
        if device is not None:
            self.device = device
            self.computation_process.simulation_graph = (
                self.computation_process.simulation_graph.to(device)
            )
            self.computation_process.converter = self.computation_process.converter.to(
                self.dtype, device
            )
            if hasattr(self.output_mapping, "weight"):
                self.output_mapping = self.output_mapping.to(
                    dtype=self.dtype, device=self.device
                )
        return self

    @classmethod
    def simple(
            cls,
            input_size: int,
            n_params: int = 100,
            shots: int = 0,
            reservoir_mode: bool = False,
            output_size: int | None = None,
            output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.NONE,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
            no_bunching: bool = True,
    ):
        """Create a ready-to-train layer with a 10-mode, 5-photon architecture."""
        _ = reservoir_mode

        n_modes = 10
        n_photons = 5

        builder = CircuitBuilder(n_modes=n_modes)
        builder.add_generic_interferometer(trainable=True, name="gi_simple")

        generic_component = builder.circuit.components[-1]
        generic_params = 0
        if (
                isinstance(generic_component, GenericInterferometer)
                and generic_component.trainable
                and generic_component.span >= 2
        ):
            mzi_count = generic_component.span * (generic_component.span - 1) // 2
            generic_params = 2 * mzi_count

        requested_params = max(int(n_params), 0)
        if generic_params > requested_params:
            warnings.warn(
                f"Generic interferometer introduces {generic_params} trainable parameters, "
                f"exceeding the requested budget of {requested_params}.",
                RuntimeWarning,
                stacklevel=2,
            )

        if input_size > n_modes:
            raise ValueError(ANGLE_ENCODING_MODE_ERROR)

        input_modes = list(range(input_size))
        builder.add_angle_encoding(
            modes=input_modes,
            name="input",
            subset_combinations=False,
        )

        builder.add_entangling_layer(depth=1)

        remaining = max(requested_params - generic_params, 0)
        layer_idx = 0
        added_rotation_params = 0

        while remaining > 0:
            prefix = f"theta_layer{layer_idx}"
            if remaining >= n_modes:
                builder.add_rotation_layer(trainable=True, name=prefix)
                remaining -= n_modes
                added_rotation_params += n_modes
            else:
                modes = list(range(remaining))
                builder.add_rotation_layer(modes=modes, trainable=True, name=prefix)
                added_rotation_params += remaining
                remaining = 0
            layer_idx += 1

        builder.add_entangling_layer(depth=1)

        total_trainable = generic_params + added_rotation_params
        expected_trainable = max(requested_params, generic_params)
        if total_trainable != expected_trainable:
            raise ValueError(f"Circuit exposes {total_trainable} trainable parameters")

        return cls(
            input_size=input_size,
            output_size=output_size,
            circuit=builder,
            n_photons=n_photons,
            output_mapping_strategy=output_mapping_strategy,
            shots=shots,
            no_bunching=no_bunching,
            device=device,
            dtype=dtype,
        )

    def __str__(self) -> str:
        """String representation of the quantum layer."""
        base_str = ""
        if self.auto_generation_mode:
            base_str = (
                f"QuantumLayer(ansatz={self.ansatz.experiment.circuit_type.value}, "
                f"modes={self.ansatz.experiment.n_modes}, "
                f"input_size={self.input_size}, output_size={self.output_size}"
            )
        else:
            base_str = (
                f"QuantumLayer(custom_circuit, input_size={self.input_size}, "
                f"output_size={self.output_size}"
            )

        if self.index_photons is not None:
            base_str += f", index_photons={self.index_photons}"
        if self._is_fixed:
            base_str += f", fixed_layer=True"

        return base_str + ")"