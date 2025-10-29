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
Main QuantumLayer implementation
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import perceval as pcvl
import torch
import torch.nn as nn

from ..builder.circuit_builder import (
    ANGLE_ENCODING_MODE_ERROR,
    CircuitBuilder,
)
from ..core.process import ComputationProcessFactory
from ..measurement import OutputMapper
from ..measurement.autodiff import AutoDiffProcess
from ..measurement.strategies import (
    MeasurementStrategy,
)
from ..utils.grouping import ModGrouping


class QuantumLayer(nn.Module):
    """
    Enhanced Quantum Neural Network Layer with factory-based architecture.

    This layer can be created either from a :class:`CircuitBuilder` instance or a pre-compiled :class:`pcvl.Circuit`.
    """

    _deprecated_params: dict[str, str] = {
        "__init__.ansatz": "Use 'circuit' or 'CircuitBuilder' to define the quantum circuit.",
        "simple.reservoir_mode": "The 'reservoir_mode' argument is no longer supported in the 'simple' method.",
    }

    @classmethod
    def _validate_kwargs(cls, method_name: str, kwargs: dict[str, Any]) -> None:
        if not kwargs:
            return
        deprecated: list[str] = []
        unknown: list[str] = []
        for key in sorted(kwargs):
            full_name = f"{method_name}.{key}"
            if full_name in cls._deprecated_params:
                deprecated.append(
                    f"Parameter '{key}' is deprecated. {cls._deprecated_params[full_name]}"
                )
            else:
                unknown.append(key)
        if deprecated:
            raise ValueError(" ".join(deprecated))
        if unknown:
            unknown_list = ", ".join(unknown)
            raise ValueError(
                f"Unexpected keyword argument(s): {unknown_list}. "
                "Check the QuantumLayer signature for supported parameters."
            )

    def __init__(
        self,
        input_size: int | None = None,
        # Builder-based construction
        builder: CircuitBuilder | None = None,
        # Custom circuit construction
        circuit: pcvl.Circuit | None = None,
        # Custom experiment construction
        experiment: pcvl.Experiment | None = None,
        # For both custom circuits and builder
        input_state: list[int] | None = None,
        n_photons: int | None = None,
        # Amplitude encoding
        amplitude_encoding: bool = False,
        #TODO: add enum for computation_space
        # ComputationSpace.FOCK, ComputationSpace.UNBUNCHED, ComputationSpace.DUAL_RAIL
        computation_space: str | None = None,
        # only for custom circuits and experiments
        trainable_parameters: list[str] | None = None,
        input_parameters: list[str] | None = None,
        # Common parameters
        measurement_strategy: MeasurementStrategy = MeasurementStrategy.PROBABILITIES,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        shots: int = 0,
        sampling_method: str = "multinomial",
        no_bunching: bool | None = None,
        **kwargs,
    ):
        super().__init__()

        self._validate_kwargs("__init__", kwargs)

        self.device = device
        self.dtype = dtype or torch.float32
        self.input_state = input_state
        self.amplitude_encoding = amplitude_encoding

        # input_size management: input_size can be given only if amplitude_encoding is False
        # otherwise, it is determined by the computation space and n_photons
        if self.amplitude_encoding:
            if input_size is not None:
                raise ValueError(
                    "When amplitude_encoding is enabled, do not specify input_size; it "
                    "is inferred from the computation space."
                )
            self.input_size = 0  # temporary value, revisited after setup
            if n_photons is None:
                raise ValueError(
                    "n_photons must be provided when amplitude_encoding=True."
                )
            if input_parameters:
                raise ValueError(
                    "Amplitude encoding cannot be combined with classical input parameters."
                )
        else:
            self.input_size = int(input_size)

        # computation_space and no_bunching arguments management
        allowed_spaces = {"fock", "no_bunching", "dual_rail"}
        if computation_space is not None and computation_space not in allowed_spaces:
            raise ValueError(
                f"Invalid computation_space '{computation_space}'. "
                "Supported values are 'fock', 'no_bunching', and 'dual_rail'."
            )
        # no_bunching cannot be True by default, to fit with "fock" computation_space
        # to avoid situation such that no_bunching=True and computation_space="fock"
        #TODO: warning for no_bunching deprecation -> map it to computation_space
        if no_bunching is None:
            no_bunching_value = True
        else:
            no_bunching_value = bool(no_bunching)
        if computation_space is None:
            # if computation_space is not provided, derive it from no_bunching
            computation_space_value = "no_bunching" if no_bunching_value else "fock"
        else:
            # if it is provided, check consistency with no_bunching
            computation_space_value = computation_space
            derived_no_bunching = computation_space_value in {"no_bunching", "dual_rail"}
            if no_bunching is not None and no_bunching_value != derived_no_bunching:
                warnings.warn(
                    "Overriding 'no_bunching' to match the requested computation_space. "
                    f"Expected {derived_no_bunching} for computation_space='{computation_space_value}', "
                    f"received {no_bunching_value}.",
                    UserWarning,
                    stacklevel=2,
                )
            no_bunching_value = derived_no_bunching

        self.computation_space = computation_space_value
        self.no_bunching = no_bunching_value

        # ensure exclusivity of circuit/builder/experiment
        if sum(x is not None for x in (circuit, builder, experiment)) != 1:
            raise ValueError(
                "Provide exactly one of 'circuit', 'builder', or 'experiment'."
            )

        if builder is not None and (
            trainable_parameters is not None or input_parameters is not None
        ):
            raise ValueError(
                "When providing a builder, do not also specify 'trainable_parameters' "
                "or 'input_parameters'. Those prefixes are derived from the builder."
            )

        if experiment is not None:
            if (
                not experiment.is_unitary
                or experiment.post_select_fn is not None
                or experiment.heralds
                or experiment.in_heralds
            ):
                raise ValueError(
                    "The provided experiment must be unitary, and must not have post-selection or heralding."
                )

            # TODO: handle "min_detected_photons" from experiment, currently ignored => will come with post_selection_scheme introduction
            if experiment.min_photons_filter:
                warnings.warn(
                    "The 'min_photons_filter' from the experiment is currently ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            # TODO: handle "detectors" from experiment, currently ignored
            if experiment.detectors:
                warnings.warn(
                    "The 'detectors' from the experiment are currently ignored.",
                    UserWarning,
                    stacklevel=2,
                )

        self.angle_encoding_specs: dict[str, dict[str, Any]] = {}

        resolved_circuit: pcvl.Circuit | None = None
        trainable_parameters = (
            list(trainable_parameters) if trainable_parameters else []
        )
        input_parameters = list(input_parameters) if input_parameters else []

        if builder is not None:
            if circuit is not None:
                raise ValueError("Provide either 'circuit' or 'builder', not both")
            trainable_parameters = list(builder.trainable_parameter_prefixes)
            input_parameters = list(builder.input_parameter_prefixes)
            self.angle_encoding_specs = builder.angle_encoding_specs
            resolved_circuit = builder.to_pcvl_circuit(pcvl)
        elif circuit is not None:
            resolved_circuit = circuit
        elif experiment is not None:
            resolved_circuit = experiment.unitary_circuit()

        self.circuit = resolved_circuit
        # dual-rail needs n photons and 2n modes
        if self.computation_space == "dual_rail" and self.circuit.m != int(2*n_photons):
            raise ValueError("Dual-rail encoding requires the number of modes to be be two times the number of modes. Here "
                             f"{self.circuit.m} modes and {n_photons} photons were provided.")
        self._init_from_custom_circuit(
            resolved_circuit,
            input_state,
            n_photons,
            trainable_parameters,
            input_parameters,
            measurement_strategy,
        )

        # Setup sampling
        self.autodiff_process = AutoDiffProcess(sampling_method)
        self.shots = shots
        self.sampling_method = sampling_method

    def _init_from_custom_circuit(
        self,
        circuit: pcvl.Circuit,
        input_state: list[int] | None,
        n_photons: int | None,
        trainable_parameters: list[str],
        input_parameters: list[str],
        measurement_strategy: MeasurementStrategy,
    ):
        """Initialize from custom circuit (backward compatible mode)."""
        if input_state is not None:
            self.input_state = input_state
        elif n_photons is not None:
            # Default behavior: place photons in first n_photons modes
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
        )

        # Pick the effective state space after the factory creates the process so
        # dual-rail can shrink the logical basis without extra factory plumbing.
        self.computation_process.configure_computation_space(self.computation_space)

        # Validate that the declared input size matches the builder-provided parameters
        spec_mappings = self.computation_process.converter.spec_mappings
        total_input_params = 0
        if input_parameters is not None:
            total_input_params = sum(
                len(spec_mappings.get(prefix, [])) for prefix in input_parameters
            )

        # Prefer metadata from angle encoding specs when available to deduce feature count
        expected_features: int | None = None
        if self.angle_encoding_specs:
            expected_features = 0
            specs_provided = False
            for metadata in self.angle_encoding_specs.values():
                # Each prefix maintains its own logical feature indices; count them separately
                # so distinct encoders do not collide when they reuse low-order indices.
                combos = metadata.get("combinations", [])
                prefix_indices = {idx for combo in combos for idx in combo}
                if not prefix_indices:
                    continue
                specs_provided = True
                expected_features += len(prefix_indices)
            if not specs_provided:
                expected_features = None

        if not self.amplitude_encoding:
            if expected_features is not None:
                if expected_features != self.input_size:
                    raise ValueError(
                        f"Input size ({self.input_size}) must equal the number of encoded input features "
                        f"generated by the circuit ({expected_features})."
                    )
            elif total_input_params != self.input_size:
                raise ValueError(
                    f"Input size ({self.input_size}) must equal the number of input parameters "
                    f"generated by the circuit ({total_input_params})."
                )

        # Setup parameters
        self._setup_parameters_from_custom(trainable_parameters)

        # Setup measurement strategy
        self._setup_measurement_strategy_from_custom(measurement_strategy)

        if self.amplitude_encoding:
            self._init_amplitude_metadata()

        # set input_size for amplitude encoding
        if self.amplitude_encoding:
            self.input_size = len(self.state_keys)

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

    def _setup_measurement_strategy_from_custom(
        self, measurement_strategy: MeasurementStrategy
    ):
        """Setup output mapping for custom circuit construction."""
        # Get distribution size
        dummy_params = self._create_dummy_parameters()
        if self.input_state is not None and type(self.input_state) is torch.Tensor:
            keys, distribution = self.computation_process.compute_superposition_state(
                dummy_params, return_keys=True
            )
        else:
            keys, distribution = self.computation_process.compute_with_keys(
                dummy_params
            )
        dist_size = distribution.shape[-1]

        # Determine output size
        if measurement_strategy == MeasurementStrategy.PROBABILITIES:
            self._output_size = dist_size
        elif measurement_strategy == MeasurementStrategy.MODE_EXPECTATIONS:
            if type(self.circuit) is pcvl.Circuit:
                self._output_size = self.circuit.m
            elif type(self.circuit) is CircuitBuilder:
                self._output_size = self.circuit.n_modes
            else:
                raise TypeError(f"Unknown circuit type: {type(self.circuit)}")
        elif measurement_strategy == MeasurementStrategy.AMPLITUDES:
            self._output_size = dist_size
        else:
            raise TypeError(f"Unknown measurement_strategy: {measurement_strategy}")

        # Create output mapping
        self.measurement_mapping = OutputMapper.create_mapping(
            measurement_strategy,
            self.computation_process.no_bunching,
            keys,
        )

    def _init_amplitude_metadata(self) -> None:
        logical_keys = getattr(
            self.computation_process,
            "logical_keys",
            list(self.computation_process.simulation_graph.mapped_keys),
        )
        self.input_size = len(logical_keys)

    def _create_dummy_parameters(self) -> list[torch.Tensor]:
        """Create dummy parameters for initialization."""
        spec_mappings = self.computation_process.converter.spec_mappings
        trainable_prefixes = list(
            getattr(self.computation_process, "trainable_parameters", [])
        )
        input_prefixes = list(self.computation_process.input_parameters)

        params: list[torch.Tensor] = []

        def _zeros(count: int) -> torch.Tensor:
            return torch.zeros(count, dtype=self.dtype, device=self.device)

        # Feed the true trainable parameters first, preserving converter order.
        theta_iter = iter(self.thetas)
        for prefix in trainable_prefixes:
            param = next(theta_iter, None)
            if param is not None:
                params.append(param)
                continue

            # Fall back to zero tensors only if no nn.Parameter exists yet.
            param_count = len(spec_mappings.get(prefix, []))
            params.append(_zeros(param_count))

        # Append any additional trainable parameters not covered by prefixes (defensive guard).
        params.extend(list(theta_iter))

        # Generate placeholder tensors for every declared input prefix in order. Encoders
        # sometimes omit converter specs ->  we fall
        # back to their stored combination metadata to deduce tensor length.
        for prefix in input_prefixes:
            # Counting parameters using their prefix
            param_count = self._feature_count_for_prefix(prefix) or 0
            if prefix in self.angle_encoding_specs:
                combos = self.angle_encoding_specs[prefix].get("combinations", [])
                if combos:
                    param_count = max(param_count, len(combos))
            params.append(_zeros(param_count))

        return params  # type: ignore[return-value]

    def _feature_count_for_prefix(self, prefix: str) -> int | None:
        """Infer the number of raw features associated with an encoding prefix."""
        spec = self.angle_encoding_specs.get(prefix)
        if spec:
            combos = spec.get("combinations", [])
            feature_indices = {idx for combo in combos for idx in combo}
            if feature_indices:
                return len(feature_indices)

        spec_mappings = getattr(self.computation_process.converter, "spec_mappings", {})
        mapping = spec_mappings.get(prefix, [])
        if mapping:
            return len(mapping)

        return None

    def _split_inputs_by_prefix(
        self, prefixes: list[str], tensor: torch.Tensor
    ) -> list[torch.Tensor] | None:
        """Split a single logical input tensor into per-prefix chunks when possible."""
        counts: list[int] = []
        for prefix in prefixes:
            count = self._feature_count_for_prefix(prefix)
            if count is None:
                return None
            counts.append(count)

        total_required = sum(counts)
        feature_dim = tensor.shape[-1] if tensor.dim() > 1 else tensor.shape[0]
        if total_required != feature_dim:
            return None

        slices: list[torch.Tensor] = []
        offset = 0
        for count in counts:
            end = offset + count
            if tensor.dim() == 1:
                slices.append(tensor[offset:end])
            else:
                slices.append(tensor[..., offset:end])
            offset = end
        return slices

    def _prepare_input_encoding(
        self, x: torch.Tensor, prefix: str | None = None
    ) -> torch.Tensor:
        """Prepare input encoding based on mode."""
        spec = None
        if prefix is not None:
            spec = self.angle_encoding_specs.get(prefix)
        elif len(self.angle_encoding_specs) == 1:
            spec = next(iter(self.angle_encoding_specs.values()))

        if spec:
            return self._apply_angle_encoding(x, spec)

        # For custom circuits without explicit encoding metadata, apply π scaling
        return x * torch.pi

    def _complex_dtype(self) -> torch.dtype:
        """Return the complex dtype matching the layer precision."""
        if self.dtype == torch.float16 and hasattr(torch, "complex32"):
            return torch.complex32
        if self.dtype == torch.float32:
            return torch.complex64
        if self.dtype == torch.float64:
            return torch.complex128
        raise ValueError(f"Unsupported dtype for amplitude encoding: {self.dtype}")

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
            raise ValueError(
                f"Angle encoding expects 1D or 2D tensors, got shape {tuple(x.shape)}"
            )

        if not combos:
            encoded = x_batch * torch.pi
            return encoded.squeeze(0) if squeeze else encoded

        encoded_cols: list[torch.Tensor] = []
        feature_dim = x_batch.shape[-1]

        for combo in combos:
            indices = list(combo)
            if any(idx >= feature_dim for idx in indices):
                raise ValueError(
                    f"Input feature dimension {feature_dim} insufficient for angle encoding combination {combo}"
                )

            selected = x_batch[:, indices]
            scales = [scale_map.get(idx, 1.0) for idx in indices]
            scale_tensor = x_batch.new_tensor(scales)
            value = (selected * scale_tensor).sum(dim=1, keepdim=True)
            encoded_cols.append(value)

        encoded = (
            torch.cat(encoded_cols, dim=1)
            if encoded_cols
            else x_batch.new_zeros((x_batch.shape[0], 0))
        )

        if squeeze:
            return encoded.squeeze(0)
        return encoded

    def _validate_amplitude_input(self, amplitude: torch.Tensor) -> torch.Tensor:
        if not isinstance(amplitude, torch.Tensor):
            raise TypeError(
                "Amplitude-encoded inputs must be provided as torch.Tensor instances"
            )

        if amplitude.dim() not in (1, 2):
            raise ValueError(
                "Amplitude-encoded inputs must be 1D (single state) or 2D (batch of states) tensors"
            )

        expected_dim = len(self.state_keys)
        feature_dim = amplitude.shape[-1]
        if feature_dim != expected_dim:
            raise ValueError(
                f"Amplitude input expects {expected_dim} components, received {feature_dim}."
            )
            # TODO: suggest/implement zero-padding or sparsity tensor format

        if amplitude.dtype not in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ):
            raise TypeError(
                "Amplitude-encoded inputs must use float32/float64 or complex64/complex128 dtype"
            )

        if self.device is not None and amplitude.device != self.device:
            amplitude = amplitude.to(self.device)

        if amplitude.is_complex():
            amplitude = amplitude.to(self._complex_dtype())
        else:
            amplitude = amplitude.to(self.dtype)

        return amplitude

    def _infer_batch_mode(
        self,
        amplitude_input: torch.Tensor | None,
        classical_inputs: list[torch.Tensor],
        fallback_state: torch.Tensor | None,
    ) -> bool:
        """Derive batch behaviour from provided inputs."""
        if amplitude_input is not None and amplitude_input.dim() > 1:
            return amplitude_input.size(0) > 1

        for tensor in classical_inputs:
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 1:
                return tensor.size(0) > 1

        if isinstance(fallback_state, torch.Tensor) and fallback_state.dim() > 1:
            return fallback_state.size(0) > 1

        return False

    def set_input_state(self, input_state):
        self.input_state = input_state
        self.computation_process.input_state = input_state

    def prepare_parameters(
        self, input_parameters: list[torch.Tensor]
    ) -> list[torch.Tensor]:
        """Prepare parameter list for circuit evaluation."""
        # Handle batching
        if input_parameters and input_parameters[0].dim() > 1:
            batch_size = input_parameters[0].shape[0]
            params = [theta.expand(batch_size, -1) for theta in self.thetas]
        else:
            params = list(self.thetas)

        # Apply input encoding
        prefixes = getattr(self.computation_process, "input_parameters", [])

        # Automatically split a single logical input across multiple prefixes when possible.
        # Builder circuits that define several encoders typically expose one logical tensor
        # to the user, while the converter expects separate tensors per prefix.
        if len(prefixes) > 1 and len(input_parameters) == 1:
            split_inputs = self._split_inputs_by_prefix(prefixes, input_parameters[0])
            if split_inputs is not None:
                input_parameters = split_inputs

        # Custom mode or multiple parameters
        for idx, x in enumerate(input_parameters):
            prefix = None
            if prefixes:
                prefix = prefixes[idx] if idx < len(prefixes) else prefixes[-1]
            encoded = self._prepare_input_encoding(x, prefix)
            params.append(encoded)

        return params

    def forward(
        self,
        *input_parameters: torch.Tensor,
        apply_sampling: bool | None = None,
        shots: int | None = None,
        simultaneous_processes: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Forward pass through the quantum layer.

        When ``self.amplitude_encoding`` is ``True`` the first positional argument
        must contain the amplitude-encoded input state (either ``[num_states]`` or
        ``[batch_size, num_states]``). Remaining positional arguments are treated
        as classical inputs and processed via the standard encoding pipeline.
        """

        inputs = list(input_parameters)
        amplitude_input: torch.Tensor | None = None
        original_input_state = None

        if self.amplitude_encoding:
            if not inputs:
                raise ValueError(
                    "QuantumLayer configured with amplitude_encoding=True expects an amplitude tensor input."
                )
            # verify that inputs is of the shape of layer.compute_graph.mapped_keys
            amplitude_input = self._validate_amplitude_input(inputs.pop(0))
            original_input_state = getattr(
                self.computation_process, "input_state", None
            )
            # amplitude_input becomes the new input_state
            self.set_input_state(amplitude_input)

        classical_inputs = [
            tensor for tensor in inputs if isinstance(tensor, torch.Tensor)
        ]

        # Prepare circuit parameters and any remaining classical inputs
        params = self.prepare_parameters(inputs)

        # Determine batch mode directly from provided inputs
        inferred_state = getattr(self.computation_process, "input_state", None)
        batch_mode = self._infer_batch_mode(
            amplitude_input, classical_inputs, inferred_state
        )

        amplitudes: torch.Tensor

        try:
            if isinstance(inferred_state, torch.Tensor):
                if batch_mode:
                    amplitudes = self.computation_process.compute_ebs_simultaneously(
                        params, simultaneous_processes=simultaneous_processes
                    )
                else:
                    amplitudes = cast(
                        torch.Tensor,
                        self.computation_process.compute_superposition_state(params),
                    )
            else:
                amplitudes = self.computation_process.compute(params)
        finally:
            if amplitude_input is not None and original_input_state is not None:
                self.set_input_state(original_input_state)

        # Handle sampling
        needs_gradient = (
            self.training
            and torch.is_grad_enabled()
            and any(p.requires_grad for p in self.parameters())
        )
        # TODO/CAUTION: if needs_gradient is True and shots>0, the code raises a warning and casts apply_sampling = False and shots = 0
        apply_sampling, shots = self.autodiff_process.autodiff_backend(
            needs_gradient, apply_sampling or False, shots or self.shots
        )
        distribution = amplitudes.real**2 + amplitudes.imag**2
        if self.no_bunching:
            sum_probs = distribution.sum(dim=1, keepdim=True)

            # Only normalize when sum > 0 to avoid division by zero
            valid_entries = sum_probs > 0
            if valid_entries.any():
                distribution = torch.where(
                    valid_entries,
                    distribution
                    / torch.where(valid_entries, sum_probs, torch.ones_like(sum_probs)),
                    distribution,
                )
                amplitudes = torch.where(
                    valid_entries,
                    amplitudes
                    / torch.where(
                        valid_entries, sum_probs.sqrt(), torch.ones_like(sum_probs)
                    ),
                    amplitudes,
                )
        if apply_sampling and shots > 0:
            counts = self.autodiff_process.sampling_noise.pcvl_sampler(
                distribution, shots
            )
            results = counts
        else:
            results = amplitudes

        # Apply measurements mapping
        return self.measurement_mapping(results)

    def set_sampling_config(self, shots: int | None = None, method: str | None = None):
        """Update sampling configuration."""
        if shots is not None:
            if not isinstance(shots, int) or shots < 0:
                raise ValueError(f"shots must be a non-negative integer, got {shots}")
            self.shots = shots
        if method is not None:
            valid_methods = ["multinomial", "binomial", "gaussian"]
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid sampling method: {method}. Valid options are: {valid_methods}"
                )
            self.sampling_method = method

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Manually move any additional tensors
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
        return self

    @property
    def state_keys(self):
        return getattr(self.computation_process, "logical_keys", None) or list(
            self.computation_process.simulation_graph.mapped_keys
        )

    @property
    def output_size(self):
        return self._output_size

    @classmethod
    def simple(
        cls,
        input_size: int,
        n_params: int = 100,
        shots: int = 0,
        output_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        no_bunching: bool = True,
        **kwargs,
    ):
        """Create a ready-to-train layer with a 10-mode, 5-photon architecture.

        The circuit is assembled via :class:`CircuitBuilder` with the following layout:

        1. A fully trainable entangling layer acting on all modes;
        2. A full input encoding layer spanning all encoded features;
        3. A non-trainable entangling layer that redistributes encoded information;
        4. Optional trainable Mach-Zehnder blocks (two parameters each) to reach the requested ``n_params`` budget;
        5. A final entangling layer prior to measurement.

        Args:
            input_size: Size of the classical input vector.
            n_params: Number of trainable parameters to allocate across the additional MZI blocks. Values
                below the default entangling budget trigger a warning; values above it must differ by an even
                amount because each added MZI exposes two parameters.
            shots: Number of sampling shots for stochastic evaluation.
            output_size: Optional classical output width.
            device: Optional target device for tensors.
            dtype: Optional tensor dtype.
            no_bunching: Whether to restrict to states without photon bunching.

        Returns:
            QuantumLayer configured with the described architecture.
        """
        cls._validate_kwargs("simple", kwargs)

        n_modes = 10
        n_photons = 5

        builder = CircuitBuilder(n_modes=n_modes)

        # Trainable entangling layer before encoding
        builder.add_entangling_layer(trainable=True, name="gi_simple")
        entangling_params = n_modes * (n_modes - 1)

        requested_params = max(int(n_params), 0)
        if entangling_params > requested_params:
            warnings.warn(
                "Entangling layer introduces "
                f"{entangling_params} trainable parameters, exceeding the requested "
                f"budget of {requested_params}. The simple layer will expose "
                f"{entangling_params} trainable parameters.",
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

        # Allocate additional trainable MZIs only if the budget exceeds the entangling layer
        remaining = max(requested_params - entangling_params, 0)
        if remaining % 2 != 0:
            raise ValueError(
                "Additional parameter budget must be even: each extra MZI exposes "
                "two trainable parameters."
            )

        mzi_idx = 0
        added_mzi_params = 0

        while remaining > 0:
            if n_modes < 2:
                raise ValueError("At least two modes are required to place an MZI.")

            start_mode = mzi_idx % (n_modes - 1)
            span_modes = [start_mode, start_mode + 1]
            prefix = f"mzi_extra{mzi_idx}"

            builder.add_entangling_layer(
                modes=span_modes,
                trainable=True,
                name=prefix,
            )

            remaining -= 2
            added_mzi_params += 2
            mzi_idx += 1

        # Post-MZI entanglement
        builder.add_superpositions()

        total_trainable = entangling_params + added_mzi_params
        expected_trainable = max(requested_params, entangling_params)
        if total_trainable != expected_trainable:
            raise ValueError(
                "Constructed circuit exposes "
                f"{total_trainable} trainable parameters but {expected_trainable} were expected."
            )

        quantum_layer = cls(
            input_size=input_size,
            builder=builder,
            n_photons=n_photons,
            measurement_strategy=MeasurementStrategy.PROBABILITIES,
            shots=shots,
            no_bunching=no_bunching,
            device=device,
            dtype=dtype,
        )

        class SimpleSequential(nn.Module):
            """Simple Sequential Module that contains the quantum layer as well as the post processing"""

            def __init__(self, quantum_layer: QuantumLayer, post_processing: nn.Module):
                super().__init__()
                self.quantum_layer = quantum_layer
                self.post_processing = post_processing
                self.add_module("quantum_layer", quantum_layer)
                self.add_module("post_processing", post_processing)
                self.circuit = quantum_layer.circuit
                if hasattr(post_processing, "output_size"):
                    self._output_size = post_processing.output_size  # type: ignore[attr-defined]
                else:
                    self._output_size = quantum_layer.output_size

            @property
            def output_size(self):
                return self._output_size

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.post_processing(self.quantum_layer(x))

        if output_size is not None:
            if not isinstance(output_size, int):
                raise TypeError("output_size must be an integer.")
            if output_size <= 0:
                raise ValueError("output_size must be a positive integer.")
            if output_size != quantum_layer.output_size:
                model = SimpleSequential(
                    quantum_layer, ModGrouping(quantum_layer.output_size, output_size)
                )
            else:
                model = SimpleSequential(quantum_layer, nn.Identity())
        else:
            model = SimpleSequential(quantum_layer, nn.Identity())

        return model

    def __str__(self) -> str:
        """String representation of the quantum layer."""
        n_modes = None
        if hasattr(self, "circuit") and getattr(self.circuit, "m", None) is not None:
            n_modes = self.circuit.m

        modes_fragment = f", modes={n_modes}" if n_modes is not None else ""
        base_str = (
            f"QuantumLayer(custom_circuit{modes_fragment}, input_size={self.input_size}, "
            f"output_size={self.output_size}"
        )

        return base_str + ")"
