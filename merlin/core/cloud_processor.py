"""
Cloud processor for Merlin QuantumLayer.
Handles connection to Perceval remote processors, batch execution, and job management.
Supports GPU tensors in hybrid quantum-classical workflows.
"""

import time
import warnings
from typing import Any

import numpy as np
import perceval as pcvl
import torch
from perceval.algorithm import Sampler
from perceval.runtime import JobGroup, RemoteConfig, RemoteJob, RemoteProcessor


class CloudProcessor:
    """
    Manages cloud execution for QuantumLayers.

    Handles authentication, processor setup, batch jobs, and result aggregation.
    Uses Perceval's batch job system for efficient cloud execution.
    Supports GPU tensors for hybrid quantum-classical workflows.
    """

    # Cloud platform constraints
    MAX_BATCH_SIZE: int = 32  # Maximum batch size for cloud execution
    DEFAULT_MAX_SHOTS: int = 100000  # Default max shots per call for cloud

    def __init__(
        self,
        platform: str,
        token: str | None = None,
        url: str = "https://api.cloud.quandela.com",
        proxies: dict[str, str] | None = None,
        use_job_group: bool = True,
        job_group_name: str | None = None,
        max_batch_size: int = 32,
        max_shots_per_call: int = 100000,
        wait_timeout: int = 60,  # Timeout for waiting on jobs
    ):
        """
        Initialize cloud processor.

        Args:
            platform: Platform name (e.g., "qpu:ascella", "sim:clifford")
            token: Authentication token. If None, searches environment/config
            url: Cloud API URL
            proxies: Proxy configuration dict
            use_job_group: Whether to use JobGroup for batch management
            job_group_name: Name for job group (auto-generated if None)
            max_batch_size: Maximum batch size for cloud execution (default: 32)
            max_shots_per_call: Maximum shots per cloud call (default: 100000)
            wait_timeout: Maximum seconds to wait for job completion (default: 60)
        """
        self.platform = platform
        self.url = url
        self.proxies = proxies
        self.use_job_group = use_job_group
        self.max_batch_size = min(max_batch_size, self.MAX_BATCH_SIZE)
        self.max_shots_per_call = max_shots_per_call
        self.wait_timeout = wait_timeout

        if max_batch_size > self.MAX_BATCH_SIZE:
            warnings.warn(
                f"Requested batch size {max_batch_size} exceeds cloud limit {self.MAX_BATCH_SIZE}. "
                f"Using maximum allowed: {self.MAX_BATCH_SIZE}"
            )

        # Handle authentication
        if token:
            RemoteConfig.set_token(token)
        else:
            remote_config = RemoteConfig()
            stored_token = remote_config.get_token()
            if not stored_token:
                raise ValueError(
                    "No authentication token provided. Either:\n"
                    "1. Pass token parameter\n"
                    "2. Set PCVL_CLOUD_TOKEN environment variable\n"
                    "3. Use RemoteConfig.set_token() and save()"
                )

        # Set proxies if provided
        if proxies:
            RemoteConfig.set_proxies(proxies)

        # Initialize processor
        self._processor: RemoteProcessor | None = None
        self._circuit: pcvl.Circuit | None = None
        self._attached_layers: list[Any] = []
        self._sampler: Sampler | None = None
        self._min_detected_photons: int | None = None
        self._no_bunching: bool = False  # Track no_bunching setting

        # Job management
        if use_job_group:
            self.job_group = JobGroup(
                job_group_name or f"merlin_{platform}_{int(time.time())}"
            )
        else:
            self.job_group = None

    def attach_layer(self, quantum_layer: Any) -> None:
        """
        Attach a QuantumLayer for cloud execution.

        Args:
            quantum_layer: The QuantumLayer to deploy
        """
        # Set cloud processor in layer
        quantum_layer._cloud_processor = self
        self._attached_layers.append(quantum_layer)

        # Build processor from layer if not already done
        if self._processor is None:
            self._build_processor(quantum_layer)

    def detach_layer(self, quantum_layer: Any) -> None:
        """Remove cloud processor from layer."""
        if quantum_layer in self._attached_layers:
            quantum_layer._cloud_processor = None
            self._attached_layers.remove(quantum_layer)

    def _build_processor(self, quantum_layer: Any) -> None:
        """Build Perceval processor from QuantumLayer."""
        config = quantum_layer.export_config()

        # Get the circuit with trained parameters already set
        self._circuit = config["circuit"]

        # Store no_bunching setting
        self._no_bunching = config.get("no_bunching", False)

        # Initialize remote processor
        self._processor = RemoteProcessor(
            name=self.platform, url=self.url, proxies=self.proxies
        )

        # Set circuit on processor
        self._processor.set_circuit(self._circuit)

        # Set input state if provided
        if config["input_state"]:
            input_state = pcvl.BasicState(config["input_state"])
            self._processor.with_input(input_state)
            # Default min detected photons to number of input photons
            self._min_detected_photons = sum(config["input_state"])

        # Set min detected photons filter
        if config.get("min_detected_photons_filter") is not None:
            self._min_detected_photons = config["min_detected_photons_filter"]

        if self._min_detected_photons is not None:
            self._processor.min_detected_photons_filter(self._min_detected_photons)

        # Create sampler with max_shots_per_call
        self._sampler = Sampler(
            self._processor, max_shots_per_call=self.max_shots_per_call
        )

    def execute(
        self,
        quantum_layer: Any,
        input_data: torch.Tensor,
        shots: int = 1000,
        return_probs: bool = True,
    ) -> torch.Tensor:
        """
        Execute quantum layer on cloud backend using batch jobs.
        Handles GPU tensors by transferring to CPU for cloud execution
        and returning results on the same device as input.

        Args:
            quantum_layer: The QuantumLayer being executed
            input_data: Input tensor of shape (batch_size, input_size)
            shots: Number of shots per sample
            return_probs: If True, returns probability distributions

        Returns:
            Output tensor of shape (batch_size, output_size) containing probabilities
            on the same device as input_data
        """
        if quantum_layer.training:
            raise RuntimeError(
                "Cannot compute gradients through cloud backend. "
                "Use model.eval() for inference or train with PyTorch backend."
            )

        # Store original device and dtype
        original_device = input_data.device
        original_dtype = input_data.dtype

        # Transfer to CPU if on GPU
        if input_data.is_cuda:
            input_data_cpu = input_data.cpu()
        else:
            input_data_cpu = input_data

        batch_size = input_data_cpu.shape[0]

        # Check if batch size exceeds limit
        if batch_size > self.max_batch_size:
            # Split into sub-batches
            output_cpu = self._execute_split_batch(
                quantum_layer, input_data_cpu, shots, return_probs
            )
        else:
            # Execute single batch (within size limit)
            output_cpu = self._execute_single_batch(
                quantum_layer, input_data_cpu, shots, return_probs
            )

        # Return on same device and dtype as input
        output = output_cpu.to(device=original_device, dtype=original_dtype)

        # Ensure no gradients flow through cloud execution
        output = output.detach()

        return output

    def _execute_split_batch(
        self,
        quantum_layer: Any,
        input_data: torch.Tensor,
        shots: int,
        return_probs: bool,
    ) -> torch.Tensor:
        """
        Execute large batch by splitting into smaller sub-batches.
        """
        batch_size = input_data.shape[0]
        n_splits = (batch_size + self.max_batch_size - 1) // self.max_batch_size

        warnings.warn(
            f"Batch size {batch_size} exceeds cloud limit {self.max_batch_size}. "
            f"Splitting into {n_splits} sub-batches."
        )

        results = []

        for i in range(0, batch_size, self.max_batch_size):
            end_idx = min(i + self.max_batch_size, batch_size)
            sub_batch = input_data[i:end_idx]

            # Execute sub-batch
            sub_result = self._execute_single_batch(
                quantum_layer, sub_batch, shots, return_probs
            )
            results.append(sub_result)

        # Concatenate results
        return torch.cat(results, dim=0)

    def _wait_for_job(self, job: RemoteJob) -> dict:
        """
        Wait for a job to complete and return results.

        Args:
            job: RemoteJob to wait for

        Returns:
            Results dictionary from the job

        Raises:
            RuntimeError: If job fails or times out
        """
        start_time = time.time()

        while not job.is_complete:
            if time.time() - start_time > self.wait_timeout:
                raise RuntimeError(f"Job timed out after {self.wait_timeout} seconds")

            # Wait a bit before checking again
            time.sleep(0.5)

        if job.is_failed:
            raise RuntimeError(f"Job failed: {job.status.stop_message}")

        return job.get_results()

    def _execute_single_batch(
        self,
        quantum_layer: Any,
        input_data: torch.Tensor,
        shots: int,
        return_probs: bool,
    ) -> torch.Tensor:
        """
        Execute a single batch within size limits.
        """
        batch_size = input_data.shape[0]

        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Single batch size {batch_size} exceeds limit {self.max_batch_size}"
            )

        config = quantum_layer.export_config()

        # Clear previous iterations
        self._sampler.clear_iterations()

        # Get all parameters from circuit
        all_circuit_params = [p.name for p in self._circuit.get_parameters()]

        # Identify which are input parameters
        input_param_names = []
        for input_spec in config["input_parameters"]:
            if input_spec == "px":
                # Handle px pattern - find all px1, px2, etc.
                for p_name in all_circuit_params:
                    if p_name.startswith("px") and p_name[2:].isdigit():
                        input_param_names.append(p_name)
            else:
                # Handle other patterns
                for p_name in all_circuit_params:
                    if p_name.startswith(input_spec):
                        input_param_names.append(p_name)

        # Sort input param names to ensure correct ordering
        input_param_names.sort()

        # Check if we have the right number of input parameters
        expected_input_params = len(input_param_names)
        provided_inputs = input_data.shape[1]

        if provided_inputs < expected_input_params:
            warnings.warn(
                f"Circuit expects {expected_input_params} input parameters {input_param_names} "
                f"but only {provided_inputs} inputs provided. "
                f"Remaining parameters will be set to 0."
            )
        elif provided_inputs > expected_input_params:
            warnings.warn(
                f"Circuit expects {expected_input_params} input parameters {input_param_names} "
                f"but {provided_inputs} inputs provided. "
                f"Extra inputs will be ignored."
            )

        # Prepare batch iterations
        input_data_np = input_data.detach().cpu().numpy()

        for i in range(batch_size):
            # Prepare circuit parameters for this sample
            circuit_params = {}

            # Map input data to input parameters
            for j, param_name in enumerate(input_param_names):
                if j < input_data.shape[1]:
                    # Use the provided input value
                    circuit_params[param_name] = float(input_data_np[i, j] * np.pi)
                else:
                    # No input provided for this parameter, set to 0
                    circuit_params[param_name] = 0.0

            # Prepare iteration parameters
            iteration_params = {
                "circuit_params": circuit_params,
                "input_state": pcvl.BasicState(config["input_state"])
                if config["input_state"]
                else None,
            }

            # Only add min_detected_photons if it's not None
            min_photons = config.get("min_detected_photons_filter")
            if min_photons is not None:
                iteration_params["min_detected_photons"] = min_photons

            # Add iteration to batch
            self._sampler.add_iteration(**iteration_params)

        # Execute batch job
        actual_shots = min(shots, self.max_shots_per_call)

        if return_probs and "probs" in self._processor.available_commands:
            # Use probs if available
            job = self._sampler.probs.execute_async()
        else:
            # Use sample_count and convert to probabilities
            job = self._sampler.sample_count.execute_async(max_samples=actual_shots)

        # Track job if using job group
        if self.job_group:
            self.job_group.add(job)

        # Wait for job completion
        try:
            results = self._wait_for_job(job)
        except Exception as e:
            raise RuntimeError(f"Cloud execution failed: {e}")

        # Process results into tensor
        return self._process_batch_results(
            results,
            batch_size,
            quantum_layer,
            actual_shots if not return_probs else None,
        )

    def _process_batch_results(
        self,
        raw_results: dict,
        batch_size: int,
        quantum_layer: Any,
        shots: int | None = None,
    ) -> torch.Tensor:
        """
        Process batch results from Sampler matching SLOS format.
        Properly handles no_bunching constraint.
        """
        # Determine actual distribution size and valid states from quantum layer
        if hasattr(quantum_layer.computation_process, "simulation_graph"):
            graph = quantum_layer.computation_process.simulation_graph
            if graph.final_keys:
                dist_size = len(graph.final_keys)
                state_to_index = {
                    state: idx for idx, state in enumerate(graph.final_keys)
                }
                valid_states = (
                    set(graph.final_keys) if quantum_layer.no_bunching else None
                )
            else:
                # Calculate based on combinatorics
                n_modes = (
                    quantum_layer.circuit.m
                    if hasattr(quantum_layer, "circuit")
                    else graph.m
                )
                n_photons = (
                    sum(quantum_layer.input_state)
                    if hasattr(quantum_layer, "input_state")
                    else graph.n_photons
                )

                if quantum_layer.no_bunching:
                    from math import comb

                    dist_size = comb(n_modes, n_photons)
                    # Generate valid no-bunching states
                    valid_states = set(
                        self._generate_no_bunching_states(n_modes, n_photons)
                    )
                    state_to_index = {
                        state: idx for idx, state in enumerate(sorted(valid_states))
                    }
                else:
                    from math import comb

                    dist_size = comb(n_modes + n_photons - 1, n_photons)
                    state_to_index = None
                    valid_states = None
        else:
            dist_size = quantum_layer.output_size
            state_to_index = None
            valid_states = None

        output_tensors = []

        # Check if we have batch results
        if "results_list" in raw_results:
            results_list = raw_results["results_list"]

            # Process each iteration result
            for i, result_item in enumerate(results_list):
                if i >= batch_size:
                    break

                if "results" in result_item:
                    state_counts = result_item["results"]
                    probs = torch.zeros(dist_size)

                    if state_counts:
                        # Filter states if no_bunching is True
                        if quantum_layer.no_bunching and valid_states is not None:
                            # Only include states that are in the valid set
                            filtered_counts = {}
                            for state_str, count in state_counts.items():
                                state_tuple = self._parse_perceval_state(state_str)
                                # Check if state is valid (no bunching)
                                if state_tuple in valid_states:
                                    filtered_counts[state_str] = count
                            state_counts = filtered_counts

                        # If all states were filtered out, return zeros (matching local behavior)
                        if not state_counts:
                            output_tensors.append(torch.zeros(dist_size))
                            continue

                        # Process remaining states
                        first_value = next(iter(state_counts.values()))
                        is_probability = (
                            isinstance(first_value, float) and first_value <= 1.0
                        )

                        if not is_probability:
                            total = sum(state_counts.values())
                        else:
                            total = 1.0

                        for state_str, value in state_counts.items():
                            # Parse state string
                            state_tuple = self._parse_perceval_state(state_str)

                            # Map to correct index
                            if state_to_index and state_tuple in state_to_index:
                                idx = state_to_index[state_tuple]
                                if idx < dist_size:
                                    if is_probability:
                                        probs[idx] = value
                                    else:
                                        probs[idx] = value / total if total > 0 else 0

                        # Normalize if needed
                        prob_sum = probs.sum()
                        if prob_sum > 0 and abs(prob_sum - 1.0) > 1e-6:
                            probs = probs / prob_sum

                        output_tensors.append(probs)
                else:
                    # No results for this iteration
                    output_tensors.append(torch.zeros(dist_size))
        else:
            # Shouldn't happen with batch job but handle single result case
            output_tensors.append(torch.zeros(dist_size))

        # Pad with zeros if needed
        while len(output_tensors) < batch_size:
            output_tensors.append(torch.zeros(dist_size))

        return torch.stack(output_tensors[:batch_size])

    def _generate_no_bunching_states(
        self, n_modes: int, n_photons: int
    ) -> list[tuple[int, ...]]:
        """Generate all valid no-bunching states in lexicographic order."""
        valid_states = []

        def generate_states(current: list[int], remaining: int, start: int) -> None:
            if remaining == 0:
                valid_states.append(tuple(current))
                return
            for i in range(start, n_modes):
                if current[i] == 0:  # Can only place photon if mode is empty
                    current[i] = 1
                    generate_states(current, remaining - 1, i + 1)
                    current[i] = 0

        generate_states([0] * n_modes, n_photons, 0)
        return sorted(valid_states)  # Return in lexicographic order

    def _parse_perceval_state(self, state_str: Any) -> tuple:
        """Parse Perceval state string like '|1,0,1,0>' to tuple."""
        if isinstance(state_str, str):
            # Handle Perceval BasicState string format
            if "|" in state_str and ">" in state_str:
                state_str = state_str.strip("|>")
                try:
                    return tuple(int(v) for v in state_str.split(","))
                except:
                    return tuple()
            # Handle comma-separated format
            elif "," in state_str:
                try:
                    return tuple(int(v) for v in state_str.split(","))
                except:
                    return tuple()
        elif hasattr(state_str, "__iter__"):
            # Handle BasicState object
            return tuple(state_str)

        return tuple()

    def validate_batch_size(self, batch_size: int) -> tuple[bool, str]:
        """
        Check if batch size is valid for cloud execution.

        Args:
            batch_size: Proposed batch size

        Returns:
            Tuple of (is_valid, message)
        """
        if batch_size <= self.max_batch_size:
            return True, f"Batch size {batch_size} is within cloud limit"
        else:
            n_splits = (batch_size + self.max_batch_size - 1) // self.max_batch_size
            return False, (
                f"Batch size {batch_size} exceeds cloud limit {self.max_batch_size}. "
                f"Will be split into {n_splits} sub-batches."
            )

    @property
    def is_connected(self) -> bool:
        """Check if cloud connection is established."""
        return self._processor is not None

    @property
    def platform_info(self) -> dict:
        """Get platform information."""
        if self._processor:
            return {
                "name": self.platform,
                "specs": self._processor.specs,
                "status": self._processor.status,
                "constraints": self._processor.constraints,
                "performance": self._processor.performance,
                "max_batch_size": self.max_batch_size,
                "max_shots_per_call": self.max_shots_per_call,
                "no_bunching": self._no_bunching,
            }
        return {}

    def get_job_history(self) -> list[RemoteJob]:
        """Get history of executed jobs."""
        if self.job_group:
            return self.job_group.remote_jobs
        return []

    def clear_job_history(self) -> None:
        """Clear job history."""
        if self.job_group:
            # Create new job group
            self.job_group = JobGroup(f"merlin_{self.platform}_{int(time.time())}")


def deploy_to_cloud(
    quantum_layer: Any,
    platform: str,
    token: str | None = None,
    use_batching: bool = True,
    max_shots_per_call: int = 100000,
    wait_timeout: int = 60,
    **kwargs: Any,
) -> CloudProcessor:
    """
    Deploy a QuantumLayer to cloud.

    Example:
        >>> # Create circuit manually
        >>> c = create_quantum_circuit(m=7)
        >>> layer = QuantumLayer(
        ...     input_size=4,
        ...     output_size=3,
        ...     circuit=c,
        ...     trainable_parameters=["theta"],
        ...     input_parameters=["px"],
        ...     input_state=[1,0,1,0,1,0,1],
        ...     no_bunching=True,  # Supported!
        ... )
        >>> cloud_proc = deploy_to_cloud(layer, "sim:ascella", token="...")
        >>> model.eval()
        >>> output = layer(batch_input)  # Returns probabilities
    """
    cloud_proc = CloudProcessor(
        platform,
        token,
        use_job_group=use_batching,
        max_shots_per_call=max_shots_per_call,
        wait_timeout=wait_timeout,
        **kwargs,
    )
    cloud_proc.attach_layer(quantum_layer)
    return cloud_proc
