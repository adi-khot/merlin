import numpy as np
import perceval as pcvl
import torch
from torch import Tensor
import itertools
from typing import Optional, Union, Callable, List  # NEW

from ..core.ansatz import AnsatzFactory
from ..core.generators import CircuitType, StatePattern
from ..core.photonicbackend import PhotonicBackend
from ..pcvl_pytorch.locirc_to_tensor import CircuitConverter
from ..pcvl_pytorch.slos_torchscript import (
    build_slos_distribution_computegraph as build_slos_graph,
)
from ..sampling.autodiff import AutoDiffProcess

dtype_to_torch = {
    "float": torch.float64,
    "complex": torch.complex128,
    "float64": torch.float64,
    "float32": torch.float32,
    "complex128": torch.complex128,
    "complex64": torch.complex64,
    torch.float64: torch.float64,
    torch.float32: torch.float32,
    torch.complex128: torch.complex128,
    torch.complex64: torch.complex64,
    np.float64: torch.float64,
    np.float32: torch.float32,
    np.complex128: torch.complex128,
    np.complex64: torch.complex64,
}


class FeatureMap:
    """
    Quantum Feature Map

    FeatureMap embeds a datapoint within a quantum circuit and
    computes the associated unitary for quantum kernel methods.

    :param circuit: Circuit with data-embedding parameters.
    :param input_parameters: Parameters which encode each datapoint.
    :param dtype: Data type for generated unitary.
    :param device: Device on which to calculate the unitary.
    """

    def __init__(
        self,
        circuit: pcvl.Circuit,
        input_size: int,
        input_parameters: Union[str, List[str]],
        *,
        trainable_parameters: Optional[List[str]] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Optional[torch.device] = None,
        encoder: Optional[Callable[[Tensor], Tensor]] = None,  # was: callable | None
    ):
        self.circuit = circuit
        self.input_size = input_size
        self.trainable_parameters = trainable_parameters or []
        self.dtype = dtype_to_torch.get(dtype, torch.float32)
        self.device = device or torch.device("cpu")
        self.is_trainable = bool(trainable_parameters)
        self._encoder = encoder  # NEW

        if isinstance(input_parameters, list):
            if len(input_parameters) > 1:
                raise ValueError("Only a single input parameter is allowed.")

            self.input_parameters = input_parameters[0]
        else:
            self.input_parameters = input_parameters

        self._circuit_graph = CircuitConverter(
            circuit,
            [self.input_parameters] + self.trainable_parameters,
            dtype=self.dtype,
            device=device,
        )
        # Set training parameters as torch parameters
        self._training_dict = {}
        for param_name in self.trainable_parameters:
            param_length = len(self._circuit_graph.spec_mappings[param_name])

            p = torch.rand(param_length, requires_grad=True)
            self._training_dict[param_name] = torch.nn.Parameter(p)

    def _px_len(self) -> int:
        """Number of circuit input-parameter slots (e.g. 'px') required by the circuit."""
        return len(self._circuit_graph.spec_mappings.get(self.input_parameters, []))

    def _subset_sum_expand(self, x: Tensor, k: int) -> Tensor:
        """
        Deterministic series-style expansion: non-empty subset sums of x in
        increasing subset-size order, truncated/padded to length k.
        """
        x = x.to(dtype=self.dtype, device=self.device).reshape(-1)
        d = x.shape[0]
        vals: list[Tensor] = []
        # generate sums for subset sizes 1..d
        for r in range(1, d + 1):
            for idxs in itertools.combinations(range(d), r):
                vals.append(x[list(idxs)].sum())
                if len(vals) == k:
                    return torch.stack(vals, dim=0)
        # if fewer than k (shouldn't happen for k <= 2^d-1), pad with zeros
        if len(vals) == 0:
            return torch.zeros(k, dtype=self.dtype, device=self.device)
        pad = k - len(vals)
        return torch.cat([torch.stack(vals, dim=0), torch.zeros(pad, dtype=self.dtype, device=self.device)], dim=0)

    def _encode_x(self, x: Tensor) -> Tensor:
        """
        Encode raw features x (length = input_size) to match px_len.
        Tries ansatz-like encoder first; falls back to subset-sum expansion.
        """
        x = x.to(dtype=self.dtype, device=self.device).reshape(-1)
        px_len = self._px_len()

        if x.numel() == px_len:
            return x
        if x.numel() < px_len:
            # Try provided encoder (e.g., from Ansatz)
            if callable(self._encoder):
                try:
                    encoded = self._encoder(x)
                    # Allow numpy/torch outputs and ensure correct shape/device/dtype
                    if isinstance(encoded, np.ndarray):
                        encoded = torch.from_numpy(encoded)
                    encoded = torch.as_tensor(encoded, dtype=self.dtype, device=self.device).reshape(-1)
                    if encoded.numel() != px_len:
                        # Fall back if encoder does not match spec
                        return self._subset_sum_expand(x, px_len)
                    return encoded
                except Exception:
                    # Encoder failed; use deterministic subset-sum expansion
                    return self._subset_sum_expand(x, px_len)
            # No encoder provided; series-style expansion
            return self._subset_sum_expand(x, px_len)
        # x longer than needed; truncate
        return x[:px_len]

    def compute_unitary(
        self, x: Tensor | np.ndarray | float, *training_parameters: Tensor
    ) -> Tensor:
        """
        Computes the unitary associated with the feature map and given datapoint.
        """
        # Normalize input to tensor on correct device/dtype
        if not isinstance(x, torch.Tensor):
            x = [x] if isinstance(x, (float, int)) else x
            x = torch.tensor(x, dtype=self.dtype, device=self.device)
        else:
            x = x.to(dtype=self.dtype, device=self.device)

        # Encode x to match the circuit's input parameter spec
        x_encoded = self._encode_x(x)

        if not self.is_trainable:
            return self._circuit_graph.to_tensor(x_encoded)

        if not training_parameters:
            training_parameters = self._training_dict.values()

        return self._circuit_graph.to_tensor(x_encoded, *training_parameters)

    def is_datapoint(self, x: Tensor | np.ndarray | float | int) -> bool:
        """Checks whether an input data is a singular datapoint or dataset."""
        if self.input_size == 1 and (isinstance(x, (float, int)) or x.ndim == 0):
            return True

        error_msg = f"Given value shape {tuple(x.shape)} does not match data shape {self.input_size}."
        num_elements = x.numel() if isinstance(x, Tensor) else x.size

        if num_elements % self.input_size or x.ndim > 2:
            raise ValueError(error_msg)

        if self.input_size == 1:
            if num_elements == 1:
                return True
            elif x.ndim == 1:
                return False
            elif x.ndim == 2 and 1 in x.shape:
                return False
        else:
            if x.ndim == 1 and x.shape[0] == self.input_size:
                return True
            elif x.ndim == 2:
                return 1 in x.shape and self.input_size in x.shape

        raise ValueError(error_msg)

    @classmethod
    def from_photonic_backend(
        cls,
        input_size: int,
        photonic_backend: PhotonicBackend,
        *,
        trainable_parameters: Optional[List[str]] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Optional[torch.device] = None,
    ) -> "FeatureMap":
        """
        Create a FeatureMap from a PhotonicBackend configuration.

        This factory method uses the PhotonicBackend to automatically generate
        a circuit and appropriate parameters for quantum kernel applications.

        :param input_size: Dimensionality of input data
        :param photonic_backend: PhotonicBackend configuration
        :param trainable_parameters: Optional trainable parameters.
            If None, automatically determined from backend configuration
        :param dtype: Data type for computations
        :param device: Device for computations
        :return: Configured FeatureMap instance

        Examples
        --------
        >>> backend = PhotonicBackend(
        ...     circuit_type=CircuitType.SERIES,
        ...     n_modes=4,
        ...     n_photons=2
        ... )
        >>> feature_map = FeatureMap.from_photonic_backend(
        ...     input_size=2,
        ...     photonic_backend=backend
        ... )
        """
        ansatz = AnsatzFactory.create(
            PhotonicBackend=photonic_backend,
            input_size=input_size,
            dtype=dtype if isinstance(dtype, torch.dtype) else None,
        )

        # Override trainable parameters if specified
        if trainable_parameters is not None:
            ansatz.trainable_parameters = trainable_parameters

        # Try to grab the same encoder QuantumLayer uses
        encoder = None
        for name in ("encode_input", "encode_features", "feature_encoder", "encode"):
            if hasattr(ansatz, name) and callable(getattr(ansatz, name)):
                encoder = getattr(ansatz, name)
                break

        return cls(
            circuit=ansatz.circuit,
            input_size=input_size,
            input_parameters=ansatz.input_parameters,
            trainable_parameters=ansatz.trainable_parameters,
            dtype=dtype,
            device=device,
            encoder=encoder,  # pass encoder through
        )

    @classmethod
    def simple(
        cls,
        input_size: int,
        n_modes: int,
        n_photons: Optional[int] = None,
        *,
        circuit_type: Union[CircuitType, str] = CircuitType.SERIES,
        state_pattern: Union[StatePattern, str] = StatePattern.PERIODIC,
        reservoir_mode: bool = False,
        trainable_parameters: Optional[List[str]] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Optional[torch.device] = None,
    ) -> "FeatureMap":
        """
        Simple factory method to create a FeatureMap with minimal configuration.

        :param input_size: Dimensionality of input data
        :param n_modes: Number of modes in the photonic circuit
        :param n_photons: Number of photons. If None, defaults to input_size
        :param circuit_type: Type of circuit topology
        :param state_pattern: Pattern for initial state generation
        :param reservoir_mode: Whether to use reservoir computing mode
        :param trainable_parameters: Optional trainable parameters
        :param dtype: Data type for computations
        :param device: Device for computations
        :return: Configured FeatureMap instance

        Examples
        --------
        >>> # Create a simple feature map for 2D data
        >>> feature_map = FeatureMap.simple(
        ...     input_size=2,
        ...     n_modes=4,
        ...     circuit_type="series"
        ... )
        """
        if n_photons is None:
            n_photons = input_size

        backend = PhotonicBackend(
            circuit_type=circuit_type,
            n_modes=n_modes,
            n_photons=n_photons,
            state_pattern=state_pattern,
            reservoir_mode=reservoir_mode,
        )

        return cls.from_photonic_backend(
            input_size=input_size,
            photonic_backend=backend,
            trainable_parameters=trainable_parameters,
            dtype=dtype,
            device=device,
        )


class KernelCircuitBuilder:
    """
    Builder class for creating quantum kernel circuits with photonic backends.

    This class provides a fluent interface for building quantum kernel circuits
    with various configurations, inspired by the core.layer architecture.
    """

    def __init__(self):
        self._input_size: int | None = None
        self._circuit_type: CircuitType = CircuitType.SERIES
        self._n_modes: int | None = None
        self._n_photons: int | None = None
        self._state_pattern: StatePattern = StatePattern.PERIODIC
        self._reservoir_mode: bool = False
        self._trainable_parameters: Optional[List[str]] = None
        self._dtype: Union[str, torch.dtype] = torch.float32
        self._device: Optional[torch.device] = None
        self._use_bandwidth_tuning: bool = False

    def input_size(self, size: int) -> "KernelCircuitBuilder":
        """Set the input dimensionality."""
        self._input_size = size
        return self

    def circuit_type(self, circuit_type: Union[CircuitType, str]) -> "KernelCircuitBuilder":
        """Set the circuit topology type."""
        if isinstance(circuit_type, str):
            circuit_type = CircuitType(circuit_type.lower())
        self._circuit_type = circuit_type
        return self

    def n_modes(self, modes: int) -> "KernelCircuitBuilder":
        """Set the number of modes in the circuit."""
        self._n_modes = modes
        return self

    def n_photons(self, photons: int) -> "KernelCircuitBuilder":
        """Set the number of photons."""
        self._n_photons = photons
        return self

    def state_pattern(self, pattern: Union[StatePattern, str]) -> "KernelCircuitBuilder":
        """Set the state initialization pattern."""
        if isinstance(pattern, str):
            pattern = StatePattern(pattern.lower())
        self._state_pattern = pattern
        return self

    def reservoir_mode(self, enabled: bool = True) -> "KernelCircuitBuilder":
        """Enable or disable reservoir computing mode."""
        self._reservoir_mode = enabled
        return self

    def trainable_parameters(self, params: List[str]) -> "KernelCircuitBuilder":
        """Set custom trainable parameters."""
        self._trainable_parameters = params
        return self

    def dtype(self, dtype: Union[str, torch.dtype]) -> "KernelCircuitBuilder":
        """Set the data type for computations."""
        self._dtype = dtype
        return self

    def device(self, device: torch.device) -> "KernelCircuitBuilder":
        """Set the computation device."""
        self._device = device
        return self

    def bandwidth_tuning(self, enabled: bool = True) -> "KernelCircuitBuilder":
        """Enable or disable bandwidth tuning."""
        self._use_bandwidth_tuning = enabled
        return self

    def build_feature_map(self) -> FeatureMap:
        """
        Build and return a FeatureMap instance.

        :return: Configured FeatureMap
        :raises ValueError: If required parameters are missing
        """
        if self._input_size is None:
            raise ValueError("Input size must be specified")

        # Set defaults
        n_modes = self._n_modes or max(self._input_size + 1, 4)
        n_photons = self._n_photons or self._input_size

        backend = PhotonicBackend(
            circuit_type=self._circuit_type,
            n_modes=n_modes,
            n_photons=n_photons,
            state_pattern=self._state_pattern,
            reservoir_mode=self._reservoir_mode,
            use_bandwidth_tuning=self._use_bandwidth_tuning,
        )

        return FeatureMap.from_photonic_backend(
            input_size=self._input_size,
            photonic_backend=backend,
            trainable_parameters=self._trainable_parameters,
            dtype=self._dtype,
            device=self._device,
        )

    def build_fidelity_kernel(
        self,
        input_state: Optional[List[int]] = None,
        *,
        shots: int = 0,
        sampling_method: str = "multinomial",
        no_bunching: bool = False,
        force_psd: bool = True,
    ) -> "FidelityKernel":
        """
        Build and return a FidelityKernel instance.

        :param input_state: Input Fock state. If None, automatically generated
        :param shots: Number of sampling shots
        :param sampling_method: Sampling method for shots
        :param no_bunching: Whether to exclude bunched states
        :param force_psd: Whether to project to positive semi-definite
        :return: Configured FidelityKernel
        """
        feature_map = self.build_feature_map()

        # Generate default input state if not provided
        if input_state is None:
            n_modes = self._n_modes or max(self._input_size or 2, 4)
            n_photons = self._n_photons or (self._input_size or 2)

            # Create input state based on state pattern
            if self._state_pattern == StatePattern.PERIODIC:
                input_state = [1] * n_photons + [0] * (n_modes - n_photons)
            else:  # FIRST_MODES or other patterns
                input_state = [1] * n_photons + [0] * (n_modes - n_photons)

        return FidelityKernel(
            feature_map=feature_map,
            input_state=input_state,
            shots=shots,
            sampling_method=sampling_method,
            no_bunching=no_bunching,
            force_psd=force_psd,
            device=self._device,
            dtype=self._dtype,
        )


class FidelityKernel(torch.nn.Module):
    r"""
    Fidelity Quantum Kernel

    For a given input Fock state, :math:`|s \rangle` and feature map,
    :math:`U`, the fidelity quantum kernel estimates the following inner
    product using SLOS:
    .. math::
        |\langle s | U^{\dagger}(x_2) U(x_1) | s \rangle|^{2}

    Transition probabilities are computed in parallel for each pair of
    datapoints in the input datasets.

    :param feature_map: Feature map object that encodes a given
        datapoint within its circuit
    :param input_state: Input state into circuit.
    :param shots: Number of circuit shots. If `None`, the exact
        transition probabilities are returned. Default: `None`.
    :param sampling_method: Probability distributions are post-
        processed with some pseudo-sampling method: 'multinomial',
        'binomial' or 'gaussian'.
    :param no_bunching: Whether or not to post-select out results with
        bunching. Default: `False`.
    :param force_psd: Projects training kernel matrix to closest
        positive semi-definite. Default: `True`.
    :param device: Device on which to perform SLOS
    :param dtype: Datatype with which to perform SLOS

    Examples
    --------
    For a given training and test datasets, one can construct the
    training and test kernel matrices in the following structure:
    .. code-block:: python
        >>> circuit = Circuit(2) // PS(P("X0") // BS() // PS(P("X1") // BS()
        >>> feature_map = FeatureMap(circuit, ["X"])
        >>>
        >>> quantum_kernel = FidelityKernel(
        >>>     feature_map,
        >>>     input_state=[0, 4],
        >>>     no_bunching=False,
        >>> )
        >>> # Construct the training & test kernel matrices
        >>> K_train = quantum_kernel(X_train)
        >>> K_test = quantum_kernel(X_test, X_train)

    Use with scikit-learn for kernel-based machine learning:.
    .. code-block:: python
        >>> from sklearn import SVC
        >>> # For a support vector classification problem
        >>> svc = SVC(kernel='precomputed')
        >>> svc.fit(K_train, y_train)
        >>> y_pred = svc.predict(K_test)
    """

    def __init__(
        self,
        feature_map: Union[FeatureMap, pcvl.Circuit],
        input_state: List[int],
        *,
        shots: Optional[int] = None,
        sampling_method: str = "multinomial",
        no_bunching: bool = False,
        force_psd: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
    ):
        super().__init__()
        self.feature_map = feature_map
        self.input_state = input_state
        self.shots = shots or 0
        self.sampling_method = sampling_method
        self.no_bunching = no_bunching
        self.force_psd = force_psd
        self.device = device or feature_map.device
        self.dtype = dtype or feature_map.dtype
        self.input_size = self.feature_map.input_size

        if self.feature_map.circuit.m != len(input_state):
            raise ValueError("Input state length does not match circuit size.")

        self.is_trainable = feature_map.is_trainable
        if self.is_trainable:
            for param_name, param in feature_map._training_dict.items():
                self.register_parameter(param_name, param)

        if max(input_state) > 1 and no_bunching:
            raise ValueError(
                f"Bunching must be enabled for an input state with"
                f"{max(input_state)} in one mode."
            )
        elif all(x == 1 for x in input_state) and no_bunching:
            raise ValueError(
                "For `no_bunching = True`, the kernel value will always be 1"
                " for an input state with a photon in all modes."
            )

        m, n = len(input_state), sum(input_state)

        self._slos_graph = build_slos_graph(
            m=m,
            n_photons=n,
            no_bunching=no_bunching,
            keep_keys=True,
            device=device,
            dtype=self.dtype,
        )
        # Find index of input state in output distribution
        complex_dtype = (
            torch.complex128 if self.dtype == torch.float64 else torch.complex64
        )
        keys, _ = self._slos_graph.compute(
            torch.eye(m, dtype=complex_dtype), input_state
        )
        self._input_state_index = keys.index(tuple(input_state))
        # For sampling
        self._autodiff_process = AutoDiffProcess()

    def forward(self, x1: float | np.ndarray | Tensor, x2=None):
        """
        Calculate the quantum kernel for input data `x1` and `x2.` If
        `x1` and `x2` are datapoints, a scalar value is returned. For
        input datasets the kernel matrix is computed.

        :param x1: Input datapoint or dataset.
        :param x2: Input datapoint or dataset. If `None`, the kernel
            matrix is assumed to be symmetric with input datasets, x1,
            x1 and only the upper triangular is calculated. Default:
            `None`.

        If you would like the diagonal and lower triangular to be
        explicitly calculated for identical inputs, please specify an
        argument `x2`.
        """
        # Convert inputs to tensors and ensure they are on the correct device
        if isinstance(x1, np.ndarray):
            x1 = torch.from_numpy(x1).to(device=x1.device, dtype=self.dtype)

        if x2 is not None:
            if isinstance(x2, np.ndarray):
                x2 = torch.from_numpy(x2).to(device=x1.device, dtype=self.dtype)
            elif isinstance(x2, torch.Tensor):
                x2 = x2.to(device=x1.device, dtype=self.dtype)

        # Return scalar value for input datapoints
        if self.feature_map.is_datapoint(x1):
            if x2 is None:
                raise ValueError("For input datapoints, please specify an x2 argument.")
            return self._return_kernel_scalar(x1, x2)

        x1 = x1.reshape(-1, self.input_size)
        x2 = x2.reshape(-1, self.input_size) if x2 is not None else None

        # Check if we are constructing training matrix
        equal_inputs = self._check_equal_inputs(x1, x2)
        U_forward = torch.stack([
            self.feature_map.compute_unitary(x).to(x1.device) for x in x1
        ])

        len_x1 = len(x1)
        if x2 is not None:
            U_adjoint = torch.stack([
                self.feature_map.compute_unitary(x).transpose(0, 1).conj().to(x1.device)
                for x in x2
            ])

            # Calculate circuit unitary for every pair of datapoints
            all_circuits = U_forward.unsqueeze(1) @ U_adjoint.unsqueeze(0)
            all_circuits = all_circuits.view(-1, *all_circuits.shape[2:])
        else:
            U_adjoint = U_forward.conj().transpose(1, 2)

            # Take circuit unitaries for upper diagonal of kernel matrix only
            upper_idx = torch.triu_indices(
                len_x1,
                len_x1,
                offset=1,
                device=x1.device,
            )
            all_circuits = U_forward[upper_idx[0]] @ U_adjoint[upper_idx[1]]

        # Distribution for every evaluated circuit
        all_probs = self._slos_graph.compute(all_circuits, self.input_state)[1]

        if self.shots > 0:
            # Convert complex amplitudes to real probabilities for multinomial sampling
            real_probs = torch.abs(all_probs)
            all_probs = self._autodiff_process.sampling_noise.pcvl_sampler(
                real_probs, self.shots, self.sampling_method
            )

        transition_probs = torch.abs(all_probs[:, self._input_state_index])

        if x2 is None:
            # Copy transition probs to upper & lower diagonal
            kernel_matrix = torch.zeros(
                len_x1, len_x1, dtype=self.dtype, device=x1.device
            )

            upper_idx = upper_idx.to(x1.device)
            transition_probs = transition_probs.to(dtype=self.dtype, device=x1.device)
            kernel_matrix[upper_idx[0], upper_idx[1]] = transition_probs
            kernel_matrix[upper_idx[1], upper_idx[0]] = transition_probs
            kernel_matrix.fill_diagonal_(1)

            if self.force_psd:
                kernel_matrix = self._project_psd(kernel_matrix)

        else:
            transition_probs = transition_probs.to(dtype=self.dtype, device=x1.device)
            kernel_matrix = transition_probs.reshape(len_x1, len(x2))

            if self.force_psd and equal_inputs:
                # Symmetrize the matrix
                kernel_matrix = 0.5 * (kernel_matrix + kernel_matrix.T)
                kernel_matrix = self._project_psd(kernel_matrix)

        return kernel_matrix

    def _return_kernel_scalar(self, x1, x2):
        """Returns scalar kernel value for input datapoints"""
        if isinstance(x1, float):
            x1, x2 = np.array(x1), np.array(x2)

        x1, x2 = x1.reshape(self.input_size), x2.reshape(self.input_size)

        U = self.feature_map.compute_unitary(x1)
        U_adjoint = self.feature_map.compute_unitary(x2)
        U_adjoint = U_adjoint.conj().T

        probs = self._slos_graph.compute(U @ U_adjoint, self.input_state)[1]

        if self.shots > 0:
            # Convert complex amplitudes to real probabilities for multinomial sampling
            real_probs = torch.abs(probs)
            probs = self._autodiff_process.sampling_noise.pcvl_sampler(
                real_probs, self.shots, self.sampling_method
            )
        return torch.abs(probs[0, self._input_state_index]).item()

    @classmethod
    def from_photonic_backend(
        cls,
        input_size: int,
        photonic_backend: PhotonicBackend,
        input_state: Optional[List[int]] = None,
        *,
        shots: int = 0,
        sampling_method: str = "multinomial",
        no_bunching: bool = False,
        force_psd: bool = True,
        trainable_parameters: Optional[List[str]] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Optional[torch.device] = None,
    ) -> "FidelityKernel":
        """
        Create a FidelityKernel from a PhotonicBackend configuration.

        :param input_size: Dimensionality of input data
        :param photonic_backend: PhotonicBackend configuration
        :param input_state: Input Fock state. If None, auto-generated
        :param shots: Number of sampling shots
        :param sampling_method: Sampling method for shots
        :param no_bunching: Whether to exclude bunched states
        :param force_psd: Whether to project to positive semi-definite
        :param trainable_parameters: Optional trainable parameters
        :param dtype: Data type for computations
        :param device: Device for computations
        :return: Configured FidelityKernel
        """
        # Create feature map
        feature_map = FeatureMap.from_photonic_backend(
            input_size=input_size,
            photonic_backend=photonic_backend,
            trainable_parameters=trainable_parameters,
            dtype=dtype,
            device=device,
        )

        # Generate default input state if not provided
        if input_state is None:
            ansatz = AnsatzFactory.create(
                PhotonicBackend=photonic_backend,
                input_size=input_size,
                dtype=dtype if isinstance(dtype, torch.dtype) else None,
            )
            input_state = ansatz.input_state

        return cls(
            feature_map=feature_map,
            input_state=input_state,
            shots=shots,
            sampling_method=sampling_method,
            no_bunching=no_bunching,
            force_psd=force_psd,
            device=device,
            dtype=dtype,
        )

    @classmethod
    def simple(
        cls,
        input_size: int,
        n_modes: int,
        n_photons: Optional[int] = None,
        input_state: Optional[List[int]] = None,
        *,
        circuit_type: Union[CircuitType, str] = CircuitType.SERIES,
        state_pattern: Union[StatePattern, str] = StatePattern.PERIODIC,
        reservoir_mode: bool = False,
        shots: int = 0,
        sampling_method: str = "multinomial",
        no_bunching: bool = False,
        force_psd: bool = True,
        trainable_parameters: Optional[List[str]] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Optional[torch.device] = None,
    ) -> "FidelityKernel":
        """
        Simple factory method to create a FidelityKernel with minimal configuration.

        :param input_size: Dimensionality of input data
        :param n_modes: Number of modes in the photonic circuit
        :param n_photons: Number of photons. If None, defaults to input_size
        :param input_state: Input Fock state. If None, auto-generated
        :param circuit_type: Type of circuit topology
        :param state_pattern: Pattern for initial state generation
        :param reservoir_mode: Whether to use reservoir computing mode
        :param shots: Number of sampling shots
        :param sampling_method: Sampling method for shots
        :param no_bunching: Whether to exclude bunched states
        :param force_psd: Whether to project to positive semi-definite
        :param trainable_parameters: Optional trainable parameters
        :param dtype: Data type for computations
        :param device: Device for computations
        :return: Configured FidelityKernel
        """
        if n_photons is None:
            n_photons = input_size

        backend = PhotonicBackend(
            circuit_type=circuit_type,
            n_modes=n_modes,
            n_photons=n_photons,
            state_pattern=state_pattern,
            reservoir_mode=reservoir_mode,
        )

        return cls.from_photonic_backend(
            input_size=input_size,
            photonic_backend=backend,
            input_state=input_state,
            shots=shots,
            sampling_method=sampling_method,
            no_bunching=no_bunching,
            force_psd=force_psd,
            trainable_parameters=trainable_parameters,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def _project_psd(matrix: Tensor) -> Tensor:
        """Projects a symmetric matrix to closest positive semi-definite"""
        # Perform spectral decomposition and set negative eigenvalues to 0
        eigenvals, eigenvecs = torch.linalg.eigh(matrix)
        eigenvals = torch.diag(torch.where(eigenvals > 0, eigenvals, 0))

        matrix_psd = eigenvecs @ eigenvals @ eigenvecs.T

        return matrix_psd

    @staticmethod
    def _check_equal_inputs(x1, x2) -> bool:
        """Checks whether x1 and x2 are equal."""
        if x2 is None:
            return True
        elif x1.shape != x2.shape:
            return False
        elif isinstance(x1, Tensor):
            return torch.allclose(x1, x2)
        elif isinstance(x1, np.ndarray):
            return np.allclose(x1, x2)
        return False
