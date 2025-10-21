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
from typing import Any

import perceval as pcvl
import torch
import torch.nn as nn

from ..builder.circuit_builder import (
    ANGLE_ENCODING_MODE_ERROR,
    CircuitBuilder,
)
from ..core.process import ComputationProcessFactory
from ..sampling.autodiff import AutoDiffProcess
from ..sampling.detectors import DetectorTransform, resolve_detectors
from ..sampling.strategies import OutputMappingStrategy
from ..torch_utils.torch_codes import OutputMapper


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
        input_size: int,
        output_size: int | None = None,
        # Builder-based construction
        builder: CircuitBuilder | None = None,
        # Custom circuit construction
        circuit: pcvl.Circuit | None = None,
        # Custom experiment construction
        experiment: pcvl.Experiment | None = None,
        # For both custom circuits and builder
        input_state: list[int] | None = None,
        n_photons: int | None = None,
        # only for custom circuits and experiments
        trainable_parameters: list[str] = None,
        input_parameters: list[str] = None,
        # Common parameters
        output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.LINEAR,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        shots: int = 0,
        sampling_method: str = "multinomial",
        no_bunching: bool = True,
        **kwargs,
    ):
        super().__init__()

        self._validate_kwargs("__init__", kwargs)

        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype or torch.float32
        self.input_size = input_size
        self.no_bunching = no_bunching
        self.experiment: pcvl.Experiment | None = None

        self._detector_transform: DetectorTransform | None = None
        self._detector_keys: list[tuple[int, ...]] = []
        self._raw_output_keys: list[tuple[int, ...]] = []
        self._detector_is_identity: bool = True
        self._output_size: int = 0

        # ensure exclusivity of circuit/builder/experiment
        if (
            (circuit is not None and (builder is not None or experiment is not None))
            or (builder is not None and experiment is not None)
            or (circuit is None and builder is None and experiment is None)
        ):
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
            self.experiment = experiment

        self.angle_encoding_specs: dict[str, dict[str, Any]] = {}

        resolved_circuit: pcvl.Circuit | None = None
        trainable_parameter_list = (
            list(trainable_parameters) if trainable_parameters else []
        )
        input_parameter_list = list(input_parameters) if input_parameters else []

        if builder is not None:
            if circuit is not None:
                raise ValueError("Provide either 'circuit' or 'builder', not both")
            trainable_parameter_list = list(builder.trainable_parameter_prefixes)
            input_parameter_list = list(builder.input_parameter_prefixes)
            self.angle_encoding_specs = builder.angle_encoding_specs
            resolved_circuit = builder.to_pcvl_circuit(pcvl)
            self.experiment = pcvl.Experiment(resolved_circuit)
        elif circuit is not None:
            resolved_circuit = circuit
            self.experiment = pcvl.Experiment(resolved_circuit)
        elif experiment is not None:
            resolved_circuit = experiment.unitary_circuit()
        else:
            raise RuntimeError("Resolved circuit could not be determined.")

        if self.experiment is None:
            raise RuntimeError("Experiment must be initialised.")

        self.circuit = resolved_circuit
        self._detectors = resolve_detectors(self.experiment, resolved_circuit.m)
        self.detectors = self._detectors  # Backward compatibility alias
        self._init_from_custom_circuit(
            resolved_circuit,
            input_state,
            n_photons,
            trainable_parameter_list,
            input_parameter_list,
            output_size,
            output_mapping_strategy,
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
        output_size: int | None,
        output_mapping_strategy: OutputMappingStrategy,
    ):
        """Initialize from custom circuit (backward compatible mode)."""
        if input_state is not None:
            self.input_state = input_state
        elif n_photons is not None:
            # Default behavior: place photons in first n_photons modes
            self.input_state = [1] * n_photons + [0] * (circuit.m - n_photons)
        else:
            raise ValueError("Either input_state or n_photons must be provided")

        resolved_n_photons = (
            n_photons if n_photons is not None else sum(self.input_state)
        )

        self.computation_process = ComputationProcessFactory.create(
            circuit=circuit,
            input_state=self.input_state,
            trainable_parameters=trainable_parameters,
            input_parameters=input_parameters,
            n_photons=resolved_n_photons,
            device=self.device,
            dtype=self.dtype,
            no_bunching=self.no_bunching,
        )

        self.n_photons = self.computation_process.n_photons
        raw_keys = list(self.computation_process.simulation_graph.mapped_keys)
        self._raw_output_keys = []
        for key in raw_keys:
            if isinstance(key, torch.Tensor):
                iterable = key.tolist()
            else:
                iterable = key
            self._raw_output_keys.append(tuple(int(v) for v in iterable))
        self._detector_transform = DetectorTransform(
            self._raw_output_keys,
            self._detectors,
            dtype=self.dtype,
            device=self.device,
        )
        self._detector_keys = self._detector_transform.output_keys
        self._detector_is_identity = self._detector_transform.is_identity

        # Setup parameters
        self._setup_parameters_from_custom(trainable_parameters)

        # Setup output mapping
        self._setup_output_mapping_from_custom(output_size, output_mapping_strategy)

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

    def _setup_output_mapping_from_custom(
        self, output_size: int | None, output_mapping_strategy: OutputMappingStrategy
    ):
        """Setup output mapping for custom circuit construction."""
        if self._detector_transform is None:
            raise RuntimeError("Detector transform must be initialised before sizing.")

        dist_size = self._detector_transform.output_size

        # Determine output size
        if output_size is None:
            if output_mapping_strategy == OutputMappingStrategy.NONE:
                resolved_output_size = dist_size
            else:
                raise ValueError(
                    "output_size must be specified for non-NONE strategies"
                )
        else:
            resolved_output_size = output_size

        self._output_size = resolved_output_size

        # Create output mapping
        self.output_mapping = OutputMapper.create_mapping(
            output_mapping_strategy, dist_size, self._output_size
        )

        # Ensure output mapping has correct dtype and device
        if hasattr(self.output_mapping, "weight"):
            self.output_mapping = self.output_mapping.to(
                dtype=self.dtype, device=self.device
            )

    def _create_dummy_parameters(self) -> list[torch.Tensor]:
        """Create dummy parameters for initialization."""
        params = list(self.thetas)

        # For custom circuits, create dummy based on input parameter count
        spec_mappings = self.computation_process.converter.spec_mappings
        input_params = self.computation_process.input_parameters
        for ip in input_params:
            if ip in spec_mappings:
                param_count = len(spec_mappings[ip])
                dummy_input = torch.zeros(
                    param_count, dtype=self.dtype, device=self.device
                )
                params.append(dummy_input)  # type: ignore[arg-type]

        return params  # type: ignore[return-value]

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
        return_amplitudes: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Forward pass through the quantum layer.

        When ``return_amplitudes`` is ``True`` the second element of the returned
        tuple contains the complex amplitudes **before** detector application.
        """
        # Prepare parameters
        params = self.prepare_parameters(list(input_parameters))
        # Get quantum output
        if type(self.computation_process.input_state) is torch.Tensor:
            amplitudes = self.computation_process.compute_superposition_state(params)
        else:
            amplitudes = self.computation_process.compute(params)

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
        if self._detector_transform is not None:
            distribution = self._detector_transform(distribution)

        if apply_sampling and shots > 0:
            distribution = self.autodiff_process.sampling_noise.pcvl_sampler(
                distribution, shots
            )
        if return_amplitudes:
            return self.output_mapping(distribution), amplitudes
        # Apply output mapping

        return self.output_mapping(distribution)

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

        dtype_arg = kwargs.get("dtype")
        device_arg = kwargs.get("device")

        if dtype_arg is None and len(args) > 0 and isinstance(args[0], torch.dtype):
            dtype_arg = args[0]
        if device_arg is None and len(args) > 0:
            first = args[0]
            if isinstance(first, torch.device):
                device_arg = first
            elif isinstance(first, str):
                device_arg = torch.device(first)

        if dtype_arg is not None:
            self.dtype = dtype_arg
        if device_arg is not None:
            self.device = (
                device_arg
                if isinstance(device_arg, torch.device)
                else torch.device(device_arg)
            )
            self.computation_process.simulation_graph = (
                self.computation_process.simulation_graph.to(self.device)
            )

        target_device = self.device if self.device is not None else torch.device("cpu")
        self.computation_process.converter = self.computation_process.converter.to(
            self.dtype, target_device
        )

        if hasattr(self.output_mapping, "weight"):
            self.output_mapping = self.output_mapping.to(
                dtype=self.dtype, device=target_device
            )

        if self._detector_transform is not None:
            self._detector_transform = self._detector_transform.to(*args, **kwargs)

        return self

    def get_output_keys(self):
        if getattr(self, "_detector_transform", None) is None:
            return self.computation_process.simulation_graph.mapped_keys
        return (
            self._raw_output_keys
            if getattr(self, "_detector_is_identity", False)
            else self._detector_keys
        )

    @property
    def output_size(self) -> int:
        return self._output_size

    @classmethod
    def simple(
        cls,
        input_size: int,
        n_params: int = 100,
        shots: int = 0,
        output_size: int | None = None,
        output_mapping_strategy: OutputMappingStrategy = OutputMappingStrategy.NONE,
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
            output_mapping_strategy: Strategy used to post-process the quantum distribution.
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

        return cls(
            input_size=input_size,
            output_size=output_size,
            builder=builder,
            n_photons=n_photons,
            output_mapping_strategy=output_mapping_strategy,
            shots=shots,
            no_bunching=no_bunching,
            device=device,
            dtype=dtype,
        )

    def __str__(self) -> str:
        """String representation of the quantum layer."""
        return (
            "QuantumLayer(custom_circuit, "
            f"input_size={self.input_size}, output_size={self.output_size})"
        )
