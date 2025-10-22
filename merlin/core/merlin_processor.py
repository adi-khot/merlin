"""
MerlinProcessor: PyTorch-friendly quantum processor with full async RPC semantics.

Optimal design changes:
- Traversal treats any module with `merlin_leaf == True` as an execution leaf
  (never recurses into its children like nn.Identity).
- Execution policy is delegated to the leaf:
    * If `should_offload(processor, shots)` exists → use it
    * Else fall back to legacy `_is_quantum_layer` (export_config & not force_simulation)
- If not offloading → call the leaf's own forward() locally (not an identity passthrough)

Key design points:
- Single async surface: forward_async(module, x, *, shots=..., timeout=...) -> Future[Tensor]
- Always return Tensors (no raw Perceval dicts leak out)
- Offloads allowed QuantumLayer instances in order (e.g., in nn.Sequential)
- Futures: fut.cancel_remote(), fut.status(), fut.job_ids
- Sync forward(...) waits on forward_async with timeout semantics
"""

import time
import threading
import warnings
from typing import Optional, Dict, List, Any, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.futures import Future
from math import comb

import perceval as pcvl
from perceval.runtime import RemoteProcessor, RemoteJob
from perceval.algorithm import Sampler


class MerlinProcessor:
    """
    Complete RPC-style processor for quantum execution.
    Returns raw probability distributions without output mapping.
    """

    MAX_BATCH_SIZE: int = 32
    DEFAULT_MAX_SHOTS: int = 100_000
    DEFAULT_SHOTS_PER_CALL: int = 10_000

    def __init__(
        self,
        remote_processor: RemoteProcessor,
        max_batch_size: int = 32,
        timeout: float = 60.0,
    ):
        if not isinstance(remote_processor, RemoteProcessor):
            raise TypeError(f"Expected pcvl.RemoteProcessor, got {type(remote_processor)}")

        self.remote_processor = remote_processor
        self.backend_name = getattr(remote_processor, "name", "unknown")

        if hasattr(remote_processor, "available_commands"):
            self.available_commands = remote_processor.available_commands
            if not self.available_commands:
                warnings.warn(
                    "Remote processor has no available commands. "
                    "Ensure the platform is properly configured.",
                    stacklevel=2,
                )
        else:
            self.available_commands = []

        self.max_batch_size = min(max_batch_size, self.MAX_BATCH_SIZE)
        self.default_timeout = timeout
        self.max_shots_per_call = self.DEFAULT_MAX_SHOTS

        if max_batch_size > self.MAX_BATCH_SIZE:
            warnings.warn(
                f"Requested batch size {max_batch_size} exceeds limit {self.MAX_BATCH_SIZE}. "
                f"Using maximum: {self.MAX_BATCH_SIZE}",
                stacklevel=2,
            )

        self._layer_cache: Dict[int, Tuple[Sampler, dict, RemoteProcessor]] = {}
        self._job_history: List[RemoteJob] = []

    @classmethod
    def from_platform(
        cls,
        platform: str,
        token: Optional[str] = None,
        url: str = "https://api.cloud.quandela.com",
        **kwargs,
    ) -> "MerlinProcessor":
        remote_proc = RemoteProcessor(name=platform, token=token, url=url)
        processor_kwargs = {
            "max_batch_size": kwargs.pop("max_batch_size", 32),
            "timeout": kwargs.pop("timeout", 60.0),
        }
        return cls(remote_proc, **processor_kwargs)

    # ---------------- Public forward APIs ----------------

    def forward(
        self,
        module: nn.Module,
        input: torch.Tensor,
        *,
        shots: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> torch.Tensor:
        timeout = self.default_timeout if timeout is None else timeout
        fut = self.forward_async(module, input, shots=shots, timeout=timeout)
        if timeout in (None, 0):
            return fut.wait()
        end = time.time() + float(timeout)
        while not fut.done():
            if time.time() >= end:
                try:
                    fut.cancel_remote()
                except Exception:
                    pass
                raise TimeoutError(f"Operation timed out after {timeout} seconds (remote cancel issued)")
            time.sleep(0.01)
        return fut.value()

    def forward_async(
        self,
        module: nn.Module,
        input: torch.Tensor,
        *,
        shots: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Future:
        if module.training:
            raise RuntimeError("Remote quantum execution requires `.eval()` mode. Call `module.eval()` before forward.")

        timeout = self.default_timeout if timeout is None else timeout
        original_device = input.device
        original_dtype = input.dtype

        layers = list(self._iter_layers_in_order(module))

        fut: Future = Future()
        state = {
            "cancel_requested": False,
            "current_job": None,       # type: Optional[RemoteJob]
            "current_status": None,    # type: Optional[Dict[str, Any]]
            "job_ids": [],             # list of job ids in order
        }

        # ---- helpers attached to the Future ----
        def _cancel_remote():
            state["cancel_requested"] = True
            job = state.get("current_job")
            if job is not None:
                try:
                    cancel = getattr(job, "cancel", None)
                    if callable(cancel):
                        cancel()
                except Exception:
                    pass
            if not fut.done():
                try:
                    from concurrent.futures import CancelledError
                except Exception:
                    class CancelledError(RuntimeError):
                        pass
                fut.set_exception(CancelledError("Remote call was cancelled"))

        def _status():
            js = state.get("current_status")
            if fut.done() and js is None:
                return {"state": "COMPLETE", "progress": 1.0, "message": None}
            if js is None:
                return {"state": "IDLE", "progress": 0.0, "message": None}
            return dict(js)

        fut.cancel_remote = _cancel_remote               # type: ignore[attr-defined]
        fut.status = _status                             # type: ignore[attr-defined]
        fut.job_ids = state["job_ids"]                   # type: ignore[attr-defined]

        # ---- background worker ----
        def _run_pipeline():
            try:
                deadline = None if timeout in (None, 0) else (time.time() + float(timeout))

                x = input
                batch = x.shape[0]
                for layer in layers:
                    if state["cancel_requested"]:
                        if not fut.done():
                            try:
                                from concurrent.futures import CancelledError
                            except Exception:
                                class CancelledError(RuntimeError):
                                    pass
                            fut.set_exception(CancelledError("Remote call was cancelled"))
                        return

                    # Decide offload policy
                    should_offload = None
                    if hasattr(layer, "should_offload") and callable(getattr(layer, "should_offload")):
                        try:
                            should_offload = bool(layer.should_offload(self.remote_processor, shots))
                        except Exception:
                            # defensive: fall back to legacy rule if policy accessor raises
                            should_offload = None

                    if should_offload is None:
                        # legacy fallback (export_config & not force_simulation)
                        should_offload = self._is_quantum_layer(layer)

                    if should_offload:
                        job = self._submit_quantum_layer(layer, x, shots)
                        state["current_job"] = job
                        job_id = getattr(job, "id", None) or getattr(job, "job_id", None)
                        if job_id is not None:
                            state["job_ids"].append(job_id)

                        sleep_ms = 50
                        while True:
                            if state["cancel_requested"]:
                                try:
                                    cancel = getattr(job, "cancel", None)
                                    if callable(cancel):
                                        cancel()
                                except Exception:
                                    pass
                                if not fut.done():
                                    try:
                                        from concurrent.futures import CancelledError
                                    except Exception:
                                        class CancelledError(RuntimeError):
                                            pass
                                    fut.set_exception(CancelledError("Remote call was cancelled"))
                                return

                            if deadline is not None and time.time() >= deadline:
                                try:
                                    cancel = getattr(job, "cancel", None)
                                    if callable(cancel):
                                        cancel()
                                except Exception:
                                    pass
                                if not fut.done():
                                    fut.set_exception(TimeoutError("Remote call timed out (remote cancel issued)"))
                                return

                            s = getattr(job, "status", None)
                            state["current_status"] = {
                                "state": getattr(s, "state", None) if s else None,
                                "progress": getattr(s, "progress", None) if s else None,
                                "message": getattr(s, "stop_message", None) if s else None,
                            }

                            if getattr(job, "is_failed", False):
                                msg = state["current_status"].get("message")
                                if not fut.done():
                                    fut.set_exception(RuntimeError(f"Remote call failed: {msg}"))
                                return

                            if getattr(job, "is_complete", False):
                                raw = job.get_results()
                                x = self._process_batch_results(raw, batch, layer, shots)
                                state["current_job"] = None
                                state["current_status"] = None
                                break

                            time.sleep(sleep_ms / 1000.0)
                            sleep_ms = min(sleep_ms * 2, 400)

                    else:
                        # NOT offloading → EXECUTE the leaf locally (call its forward), not its children.
                        with torch.no_grad():
                            x = layer(x)

                if not fut.done():
                    y = x.to(device=original_device, dtype=original_dtype)
                    fut.set_result(y)

            except BaseException as e:
                if not fut.done():
                    fut.set_exception(e)

        threading.Thread(target=_run_pipeline, daemon=True).start()
        return fut

    # ---------------- Resume API ----------------

    def resume(
        self,
        job_id: str,
        *,
        layer: Any,
        batch_size: int,
        shots: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        timeout: Optional[float] = None,
    ) -> Future:
        timeout = self.default_timeout if timeout is None else timeout

        fut: Future = Future()
        state = {
            "cancel_requested": False,
            "current_job": None,
            "current_status": None,
            "job_ids": [],
        }

        def _cancel_remote():
            state["cancel_requested"] = True
            job = state.get("current_job")
            if job is not None:
                try:
                    cancel = getattr(job, "cancel", None)
                    if callable(cancel):
                        cancel()
                except Exception:
                    pass
            if not fut.done():
                try:
                    from concurrent.futures import CancelledError
                except Exception:
                    class CancelledError(RuntimeError):
                        pass
                fut.set_exception(CancelledError("Remote call was cancelled"))

        def _status():
            js = state.get("current_status")
            if fut.done() and js is None:
                return {"state": "COMPLETE", "progress": 1.0, "message": None}
            if js is None:
                return {"state": "IDLE", "progress": 0.0, "message": None}
            return dict(js)

        fut.cancel_remote = _cancel_remote          # type: ignore[attr-defined]
        fut.status = _status                        # type: ignore[attr-defined]
        fut.job_ids = state["job_ids"]              # type: ignore[attr-defined]

        def _monitor():
            try:
                job = self.remote_processor.resume_job(job_id)
                state["current_job"] = job
                jid = getattr(job, "id", None) or getattr(job, "job_id", None) or job_id
                state["job_ids"].append(jid)

                deadline = None if timeout in (None, 0) else (time.time() + float(timeout))
                sleep_ms = 50

                while True:
                    if state["cancel_requested"]:
                        try:
                            cancel = getattr(job, "cancel", None)
                            if callable(cancel):
                                cancel()
                        except Exception:
                            pass
                        if not fut.done():
                            try:
                                from concurrent.futures import CancelledError
                            except Exception:
                                class CancelledError(RuntimeError):
                                    pass
                            fut.set_exception(CancelledError("Remote call was cancelled"))
                        return

                    if deadline is not None and time.time() >= deadline:
                        try:
                            cancel = getattr(job, "cancel", None)
                            if callable(cancel):
                                cancel()
                        except Exception:
                            pass
                        if not fut.done():
                            fut.set_exception(TimeoutError("Remote call timed out (remote cancel issued)"))
                        return

                    s = getattr(job, "status", None)
                    state["current_status"] = {
                        "state": getattr(s, "state", None) if s else None,
                        "progress": getattr(s, "progress", None) if s else None,
                        "message": getattr(s, "stop_message", None) if s else None,
                    }

                    if getattr(job, "is_failed", False):
                        msg = state["current_status"].get("message")
                        if not fut.done():
                            fut.set_exception(RuntimeError(f"Remote call failed: {msg}"))
                        return

                    if getattr(job, "is_complete", False):
                        raw = job.get_results()
                        t = self._process_batch_results(raw, batch_size, layer, shots)
                        if device is not None or dtype is not None:
                            t = t.to(device=device or t.device, dtype=dtype or t.dtype)
                        if not fut.done():
                            fut.set_result(t)
                        return

                    time.sleep(sleep_ms / 1000.0)
                    sleep_ms = min(sleep_ms * 2, 400)
            except BaseException as e:
                if not fut.done():
                    fut.set_exception(e)

        threading.Thread(target=_monitor, daemon=True).start()
        return fut

    # ---------------- Perceval integration (per-layer) ----------------

    def _submit_quantum_layer(self, layer: Any, input_data: torch.Tensor, shots: Optional[int]) -> RemoteJob:
        if input_data.is_cuda:
            input_data = input_data.cpu()

        batch_size = input_data.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds cloud limit {self.max_batch_size}. "
                "Please split into smaller chunks (≤ max_batch_size)."
            )

        layer_id = id(layer)
        if layer_id not in self._layer_cache:
            config = layer.export_config()

            layer_processor = RemoteProcessor(
                name=self.remote_processor.name,
                token=None
            )
            layer_processor.set_circuit(config["circuit"])

            if config.get("input_state"):
                input_state = pcvl.BasicState(config["input_state"])
                layer_processor.with_input(input_state)
                n_photons = sum(config["input_state"])
                layer_processor.min_detected_photons_filter(n_photons)

            sampler = Sampler(layer_processor, max_shots_per_call=self.max_shots_per_call)
            self._layer_cache[layer_id] = (sampler, config, layer_processor)
        else:
            sampler, config, layer_processor = self._layer_cache[layer_id]

        sampler.clear_iterations()

        input_param_names = self._extract_input_params(config)
        input_np = input_data.detach().cpu().numpy()

        for i in range(batch_size):
            circuit_params = {}
            for j, param_name in enumerate(input_param_names):
                if j < input_data.shape[1]:
                    circuit_params[param_name] = float(input_np[i, j] * np.pi)
                else:
                    circuit_params[param_name] = 0.0
            sampler.add_iteration(circuit_params=circuit_params)

        if "probs" in self.available_commands:
            job = sampler.probs.execute_async()
        elif "sample_count" in self.available_commands:
            use_shots = self.DEFAULT_SHOTS_PER_CALL if shots is None else int(shots)
            job = sampler.sample_count.execute_async(max_samples=use_shots)
        elif "samples" in self.available_commands:
            use_shots = self.DEFAULT_SHOTS_PER_CALL if shots is None else int(shots)
            job = sampler.samples.execute_async(max_samples=use_shots)
        else:
            use_shots = self.DEFAULT_SHOTS_PER_CALL if shots is None else int(shots)
            job = sampler.sample_count.execute_async(max_samples=use_shots)

        self._job_history.append(job)
        return job

    # ---------------- Results mapping & utilities ----------------

    def _iter_layers_in_order(self, module: nn.Module) -> Iterable[nn.Module]:
        """
        Depth-first, left-to-right traversal that yields *execution* leaves.

        RULE:
          - If a module declares `merlin_leaf == True`, treat it as a LEAF and DO NOT recurse.
          - Otherwise, recurse into children until we reach nn.Modules with no children.
        """
        if getattr(module, "merlin_leaf", False):
            yield module
            return

        children = list(module.children())
        if not children:
            yield module
            return

        for child in children:
            yield from self._iter_layers_in_order(child)

    def _extract_input_params(self, config: dict) -> List[str]:
        circuit = config["circuit"]
        all_params = [p.name for p in circuit.get_parameters()]
        input_param_names: List[str] = []

        for input_spec in config.get("input_parameters", []):
            if input_spec == "px":
                for p_name in all_params:
                    if p_name.startswith("px") and p_name[2:].isdigit():
                        input_param_names.append(p_name)
            else:
                for p_name in all_params:
                    if p_name.startswith(input_spec):
                        input_param_names.append(p_name)

        return sorted(input_param_names)

    def _process_batch_results(
        self,
        raw_results: dict,
        batch_size: int,
        layer: Any,
        shots: Optional[int] = None,
    ) -> torch.Tensor:
        dist_size, state_to_index, valid_states = self._get_state_mapping(layer)
        output_tensors: List[torch.Tensor] = []

        if "results_list" in raw_results:
            results_list = raw_results["results_list"]

            for i, result_item in enumerate(results_list):
                if i >= batch_size:
                    break

                if "results" in result_item:
                    state_counts = result_item["results"]
                    probs = torch.zeros(dist_size)

                    if state_counts:
                        if getattr(layer, "no_bunching", False) and valid_states is not None:
                            filtered_counts = {}
                            for state_str, count in state_counts.items():
                                state_tuple = self._parse_perceval_state(state_str)
                                if state_tuple in valid_states:
                                    filtered_counts[state_str] = count
                            state_counts = filtered_counts

                        if not state_counts:
                            output_tensors.append(torch.zeros(dist_size))
                            continue

                        first_value = next(iter(state_counts.values()))
                        is_probability = isinstance(first_value, float) and first_value <= 1.0
                        total = 1.0 if is_probability else sum(state_counts.values())

                        for state_str, value in state_counts.items():
                            state_tuple = self._parse_perceval_state(state_str)
                            if state_to_index and state_tuple in state_to_index:
                                idx = state_to_index[state_tuple]
                                if idx < dist_size:
                                    probs[idx] = value if is_probability else (value / total if total > 0 else 0)

                        prob_sum = probs.sum()
                        if prob_sum > 0 and abs(float(prob_sum) - 1.0) > 1e-6:
                            probs = probs / prob_sum

                        output_tensors.append(probs)
                else:
                    output_tensors.append(torch.zeros(dist_size))

        while len(output_tensors) < batch_size:
            output_tensors.append(torch.zeros(dist_size))

        return torch.stack(output_tensors[:batch_size])

    def _get_state_mapping(self, layer: Any) -> Tuple[int, Optional[Dict], Optional[set]]:
        if hasattr(layer, "computation_process") and hasattr(layer.computation_process, "simulation_graph"):
            graph = layer.computation_process.simulation_graph

            if hasattr(graph, "final_keys") and graph.final_keys:
                dist_size = len(graph.final_keys)
                state_to_index = {state: idx for idx, state in enumerate(graph.final_keys)}
                valid_states = set(graph.final_keys) if getattr(layer, "no_bunching", False) else None
            else:
                n_modes = layer.circuit.m if hasattr(layer, "circuit") else graph.m
                n_photons = sum(layer.input_state) if hasattr(layer, "input_state") else graph.n_photons

                if getattr(layer, "no_bunching", False):
                    dist_size = comb(n_modes, n_photons)
                    valid_states = set(self._generate_no_bunching_states(n_modes, n_photons))
                    state_to_index = {state: idx for idx, state in enumerate(sorted(valid_states))}
                else:
                    dist_size = comb(n_modes + n_photons - 1, n_photons)
                    state_to_index = None
                    valid_states = None
        else:
            if hasattr(layer, "circuit") and hasattr(layer, "input_state"):
                n_modes = layer.circuit.m
                n_photons = sum(layer.input_state)

                if getattr(layer, "no_bunching", False):
                    dist_size = comb(n_modes, n_photons)
                    valid_states = set(self._generate_no_bunching_states(n_modes, n_photons))
                    state_to_index = {state: idx for idx, state in enumerate(sorted(valid_states))}
                else:
                    dist_size = comb(n_modes + n_photons - 1, n_photons)
                    state_to_index = None
                    valid_states = None
            else:
                dist_size = 10
                state_to_index = None
                valid_states = None

        return dist_size, state_to_index, valid_states

    def _generate_no_bunching_states(self, n_modes: int, n_photons: int) -> List[Tuple[int, ...]]:
        valid_states: List[Tuple[int, ...]] = []

        def generate_states(current: List[int], remaining: int, start: int):
            if remaining == 0:
                valid_states.append(tuple(current))
                return
            for i in range(start, n_modes):
                if current[i] == 0:
                    current[i] = 1
                    generate_states(current, remaining - 1, i + 1)
                    current[i] = 0

        generate_states([0] * n_modes, n_photons, 0)
        return sorted(valid_states)

    def _parse_perceval_state(self, state_str: Any) -> tuple:
        if isinstance(state_str, str):
            if "|" in state_str and ">" in state_str:
                state_str = state_str.strip("|>")
                try:
                    return tuple(int(v) for v in state_str.split(","))
                except Exception:
                    return ()
            elif "," in state_str:
                try:
                    return tuple(int(v) for v in state_str.split(","))
                except Exception:
                    return ()
        elif hasattr(state_str, "__iter__"):
            return tuple(state_str)
        return ()

    # ---------------- Platform & diagnostics ----------------

    @property
    def platform_info(self) -> dict:
        info = {
            "name": self.backend_name,
            "available_commands": self.available_commands,
            "max_batch_size": self.max_batch_size,
            "max_shots_per_call": self.max_shots_per_call,
        }
        if hasattr(self.remote_processor, "specs"):
            info["specs"] = self.remote_processor.specs
        if hasattr(self.remote_processor, "status"):
            info["status"] = self.remote_processor.status
        if hasattr(self.remote_processor, "constraints"):
            info["constraints"] = self.remote_processor.constraints
        if hasattr(self.remote_processor, "performance"):
            info["performance"] = self.remote_processor.performance
        return info

    def get_job_history(self) -> List[RemoteJob]:
        return self._job_history

    def clear_job_history(self) -> None:
        self._job_history = []

    # ---------------- Legacy compatibility ----------------

    def _is_quantum_layer(self, module: Any) -> bool:
        """
        Back-compat rule for offload decision if a layer doesn't provide `should_offload`:
          - If module.force_simulation is True -> False (simulate locally).
          - Else -> offload iff export_config exists (capability check).
        """
        if getattr(module, "force_simulation", False):
            return False
        return hasattr(module, "export_config") and callable(getattr(module, "export_config"))