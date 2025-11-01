# merlin/core/merlin_processor.py

from __future__ import annotations

import logging
import threading
import time
import uuid
import warnings
import zlib
from collections.abc import Iterable
from contextlib import suppress
from math import comb
from typing import Any

import numpy as np
import perceval as pcvl
import torch
import torch.nn as nn
from perceval.algorithm import Sampler
from perceval.runtime import RemoteJob, RemoteProcessor
from torch.futures import Future

logger = logging.getLogger(__name__)


class MerlinProcessor:
    """
    RPC-style processor for quantum execution with:
      - Torch-friendly async interface (Future[Torch.Tensor])
      - Cloud offload of QuantumLayer leaves
      - Batch chunking per quantum leaf with limited concurrency
      - Cancellation (per-future and global)
      - Global timeouts that cancel in-flight jobs
      - Per-call RemoteProcessor pooling (no shared RPC handlers across threads)
      - Descriptive cloud job names (<= 50 chars) for traceability
    """

    DEFAULT_MAX_SHOTS: int = 100_000
    DEFAULT_SHOTS_PER_CALL: int = 10_000
    _JOB_NAME_MAX: int = 50

    def __init__(
        self,
        remote_processor: RemoteProcessor,
        microbatch_size: int = 32,
        timeout: float = 3600.0,
        max_shots_per_call: int | None = None,
        chunk_concurrency: int = 1,
    ):
        if not isinstance(remote_processor, RemoteProcessor):
            raise TypeError(
                f"Expected pcvl.RemoteProcessor, got {type(remote_processor)}"
            )

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

        self.microbatch_size = microbatch_size

        self.default_timeout = float(timeout)
        # When using RemoteProcessor, Perceval requires an explicit bound.
        self.max_shots_per_call = (
            None if max_shots_per_call is None else int(max_shots_per_call)
        )

        # Concurrency of chunk submissions inside a single quantum leaf
        self.chunk_concurrency = max(1, int(chunk_concurrency))

        # Caches & global tracking
        # id(layer) -> {"config": ...}  (we do NOT cache an RP anymore)
        self._layer_cache: dict[int, dict] = {}
        self._job_history: list[RemoteJob] = []

        # Lifecycle/cancellation
        self._lock = threading.Lock()
        self._active_jobs: set[RemoteJob] = set()
        self._closed = False

    # ---------------- Public APIs ----------------

    def __enter__(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("MerlinProcessor is closed")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.cancel_all()
        finally:
            with self._lock:
                self._closed = True

    def cancel_all(self) -> None:
        """Cancel all in-flight jobs across all futures."""
        with self._lock:
            jobs = list(self._active_jobs)
        for job in jobs:
            cancel = getattr(job, "cancel", None)
            if callable(cancel):
                with suppress(Exception):
                    cancel()

    def forward(
        self,
        module: nn.Module,
        input: torch.Tensor,
        *,
        nsample: int | None = None,
        timeout: float | None = None,
    ) -> torch.Tensor:
        """Synchronous convenience wrapper around forward_async()."""
        fut = self.forward_async(module, input, nsample=nsample, timeout=timeout)
        return fut.wait()

    def forward_async(
        self,
        module: nn.Module,
        input: torch.Tensor,
        *,
        nsample: int | None = None,
        timeout: float | None = None,
    ) -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("MerlinProcessor is closed")

        if module.training:
            raise RuntimeError(
                "Remote quantum execution requires `.eval()` mode. Call `module.eval()` before forward."
            )

        # Determine deadline
        effective_timeout = self.default_timeout if timeout is None else timeout
        deadline: float | None = (
            None
            if effective_timeout in (None, 0)
            else time.time() + float(effective_timeout)
        )

        original_device = input.device
        original_dtype = input.dtype
        layers = list(self._iter_layers_in_order(module))

        fut: Future = Future()
        state = {
            "cancel_requested": False,
            "current_status": None,
            "job_ids": [],  # accumulated across all chunk jobs
            "chunks_total": 0,
            "chunks_done": 0,
            "active_chunks": 0,
            "call_id": uuid.uuid4().hex[:8],  # tag all jobs of this forward() call
        }

        # --- Per-call RP pool context (per layer) ---
        # Built lazily on first use of a given layer during this forward call.
        pool_ctx = {
            "pools": {},  # layer_id -> list[RemoteProcessor]
            "cursors": {},  # layer_id -> int
            "locks": {},  # layer_id -> threading.Lock (for cursor bump)
            "pool_size": max(1, self.chunk_concurrency),
        }

        # ---- helpers attached to the Future ----
        def _cancel_remote():
            state["cancel_requested"] = True
            # Also propagate to global jobs to reduce wasted credits
            self.cancel_all()
            if not fut.done():
                try:
                    from concurrent.futures import CancelledError
                except Exception:  # pragma: no cover

                    class CancelledError(RuntimeError):
                        pass

                fut.set_exception(CancelledError("Remote call was cancelled"))

        def _status():
            js = state.get("current_status")
            base = {
                "state": "COMPLETE"
                if fut.done() and not js
                else (js.get("state") if js else "IDLE"),
                "progress": js.get("progress") if js else 0.0,
                "message": js.get("message") if js else None,
                "chunks_total": state["chunks_total"],
                "chunks_done": state["chunks_done"],
                "active_chunks": state["active_chunks"],
            }
            return base

        fut.cancel_remote = _cancel_remote  # type: ignore[attr-defined]
        fut.status = _status  # type: ignore[attr-defined]
        fut.job_ids = state["job_ids"]  # type: ignore[attr-defined]

        # ---- background worker ----
        def _run_pipeline():
            try:
                x = input
                for layer in layers:
                    # Policy: offload quantum leaves; else run locally
                    should_offload = None
                    if hasattr(layer, "should_offload") and callable(
                        layer.should_offload
                    ):
                        try:
                            should_offload = bool(
                                layer.should_offload(self.remote_processor, nsample)
                            )
                        except Exception:
                            should_offload = None
                    if should_offload is None:
                        should_offload = self._is_quantum_layer(layer)

                    if state["cancel_requested"]:
                        raise self._cancelled_error()

                    if should_offload:
                        x = self._offload_quantum_layer_with_chunking(
                            layer, x, nsample, state, deadline, pool_ctx
                        )
                    else:
                        with torch.no_grad():
                            x = layer(x)

                if not fut.done():
                    fut.set_result(x.to(device=original_device, dtype=original_dtype))
            except BaseException as e:
                if not fut.done():
                    fut.set_exception(e)

        threading.Thread(target=_run_pipeline, daemon=True).start()
        return fut

    # ---------------- Chunked offload per quantum leaf ----------------

    def _offload_quantum_layer_with_chunking(
        self,
        layer: Any,
        input_tensor: torch.Tensor,
        nsample: int | None,
        state: dict,
        deadline: float | None,
        pool_ctx: dict,
    ) -> torch.Tensor:
        """
        Split the batch into chunks of size <= max_batch_size,
        submit up to chunk_concurrency jobs concurrently, and stitch.
        """
        if input_tensor.is_cuda:
            input_tensor = input_tensor.cpu()

        B = input_tensor.shape[0]
        if B <= self.microbatch_size and self.chunk_concurrency == 1:
            # Fast path: single chunk, single job.
            return self._run_chunk_wrapper(
                layer, input_tensor, nsample, state, deadline, pool_ctx
            )

        # Ensure we have (and cache) layer config
        cache = self._layer_cache.get(id(layer))
        if cache is None:
            config = layer.export_config()
            self._layer_cache[id(layer)] = {"config": config}
        else:
            config = cache["config"]

        # Ensure a per-call pool exists for this layer
        self._ensure_pool_for_layer(config, layer, pool_ctx)

        # Build chunks
        chunks: list[tuple[int, int]] = []
        start = 0
        while start < B:
            end = min(start + self.microbatch_size, B)
            chunks.append((start, end))
            start = end

        state["chunks_total"] += len(chunks)
        outputs: list[torch.Tensor | None] = [None] * len(chunks)
        errors: list[BaseException] = []

        def _next_rp_for_layer() -> tuple[RemoteProcessor, int]:
            return self._pool_next_rp(id(layer), pool_ctx)

        total_chunks = len(chunks)
        layer_name = getattr(layer, "name", layer.__class__.__name__)

        def _call(s: int, e: int, idx: int):
            try:
                rp, pool_slot = _next_rp_for_layer()
                base_label = f"mer:{layer_name}:{state['call_id']}:{idx + 1}/{total_chunks}:{pool_slot}"
                t = self._run_chunk(
                    layer,
                    config,
                    rp,
                    input_tensor[s:e],
                    nsample,
                    state,
                    deadline,
                    job_base_label=base_label,
                )
                outputs[idx] = t
            except BaseException as ex:
                errors.append(ex)

        # Submit with limited concurrency
        in_flight = 0
        idx = 0
        futures: list[threading.Thread] = []
        while idx < len(chunks) or in_flight > 0:
            # Launch up to concurrency limit
            while idx < len(chunks) and in_flight < self.chunk_concurrency:
                s, e = chunks[idx]
                with self._lock:
                    state["active_chunks"] += 1
                th = threading.Thread(target=_call, args=(s, e, idx), daemon=True)
                th.start()
                futures.append(th)
                idx += 1
                in_flight += 1

            # Wait for any thread to finish
            for th in list(futures):
                if not th.is_alive():
                    futures.remove(th)
                    in_flight -= 1
                    with self._lock:
                        state["active_chunks"] = max(0, state["active_chunks"] - 1)
                        state["chunks_done"] += 1

            if deadline is not None and time.time() >= deadline:
                self.cancel_all()
                raise TimeoutError("Remote call timed out (remote cancel issued)")

            time.sleep(0.01)

        if errors:
            # Raise first error (others are likely the same)
            raise errors[0]

        return torch.cat(outputs, dim=0)  # type: ignore[arg-type]

    def _run_chunk_wrapper(
        self,
        layer: Any,
        input_chunk: torch.Tensor,
        nsample: int | None,
        state: dict,
        deadline: float | None,
        pool_ctx: dict,
    ) -> torch.Tensor:
        """Non-chunking simple path using the per-call RP pool."""
        cache = self._layer_cache.get(id(layer))
        if cache is None:
            config = layer.export_config()
            self._layer_cache[id(layer)] = {"config": config}
        else:
            config = cache["config"]

        # Ensure a per-call pool exists for this layer
        self._ensure_pool_for_layer(config, layer, pool_ctx)

        rp, pool_slot = self._pool_next_rp(id(layer), pool_ctx)
        layer_name = getattr(layer, "name", layer.__class__.__name__)
        base_label = f"mer:{layer_name}:{state['call_id']}:1/1:{pool_slot}"
        t = self._run_chunk(
            layer,
            config,
            rp,
            input_chunk,
            nsample,
            state,
            deadline,
            job_base_label=base_label,
        )
        state["chunks_total"] += 1
        state["chunks_done"] += 1
        return t

    def _run_chunk(
        self,
        layer: Any,
        config: dict,
        child_rp: RemoteProcessor,
        input_chunk: torch.Tensor,
        nsample: int | None,
        state: dict,
        deadline: float | None,
        job_base_label: str | None = None,
    ) -> torch.Tensor:
        """Submit a single chunk job for the given layer and return mapped tensor."""
        from concurrent.futures import CancelledError  # used for cancellation mapping

        batch_size = input_chunk.shape[0]
        if batch_size > self.microbatch_size:
            raise ValueError(
                f"Chunk size {batch_size} exceeds microbatch {self.microbatch_size}. "
                "Please report this bug."
            )

        # Prepare a fresh Sampler for THIS chunk (one sampler per worker for thread-safety)
        max_shots_arg = (
            self.DEFAULT_SHOTS_PER_CALL
            if self.max_shots_per_call is None
            else int(self.max_shots_per_call)
        )
        sampler = Sampler(child_rp, max_shots_per_call=max_shots_arg)

        # Build iterations
        sampler.clear_iterations()
        input_param_names = self._extract_input_params(config)
        input_np = input_chunk.detach().cpu().numpy()
        for i in range(batch_size):
            circuit_params = {}
            for j, param_name in enumerate(input_param_names):
                if j < input_chunk.shape[1]:
                    circuit_params[param_name] = float(input_np[i, j])
                else:
                    circuit_params[param_name] = 0.0
            sampler.add_iteration(circuit_params=circuit_params)

        # Choose execution primitive, set a descriptive (capped) name, then submit
        def _capped_name(base: str, cmd: str) -> str:
            # Compose and sanitize
            name = f"{base}:{cmd}"
            name = "".join(ch if ch.isalnum() or ch in "-_:/=." else "_" for ch in name)
            if len(name) <= self._JOB_NAME_MAX:
                return name
            # Truncate with a stable short hash suffix to preserve uniqueness
            h = f"{zlib.adler32(name.encode()):08x}"  # 8 hex chars
            keep = self._JOB_NAME_MAX - 1 - len(h)
            if keep < 1:
                # Extremely constrained; fall back to hash head
                return h[: self._JOB_NAME_MAX]
            return name[:keep] + "~" + h

        if "probs" in self.available_commands:
            job = sampler.probs  # not submitted yet
            cmd = "probs"
            if job_base_label:
                job.name = _capped_name(job_base_label, cmd)
            job = job.execute_async()
        elif "sample_count" in self.available_commands:
            use_shots = self.DEFAULT_SHOTS_PER_CALL if nsample is None else int(nsample)
            job = sampler.sample_count
            cmd = "sample_count"
            if job_base_label:
                job.name = _capped_name(job_base_label, cmd)
            job = job.execute_async(max_samples=use_shots)
        elif "samples" in self.available_commands:
            use_shots = self.DEFAULT_SHOTS_PER_CALL if nsample is None else int(nsample)
            job = sampler.samples
            cmd = "samples"
            if job_base_label:
                job.name = _capped_name(job_base_label, cmd)
            job = job.execute_async(max_samples=use_shots)
        else:
            use_shots = self.DEFAULT_SHOTS_PER_CALL if nsample is None else int(nsample)
            job = sampler.sample_count
            cmd = "sample_count"
            if job_base_label:
                job.name = _capped_name(job_base_label, cmd)
            job = job.execute_async(max_samples=use_shots)

        # Track globally for cancellation & history
        with self._lock:
            self._active_jobs.add(job)
            self._job_history.append(job)

        # Monitor until complete/failed/timeout
        sleep_ms = 50
        while True:
            if state.get("cancel_requested"):
                cancel = getattr(job, "cancel", None)
                if callable(cancel):
                    with suppress(Exception):
                        cancel()
                # Stop worker quickly; cloud job will transition to cancel
                raise CancelledError("Remote call was cancelled")

            if deadline is not None and time.time() >= deadline:
                cancel = getattr(job, "cancel", None)
                if callable(cancel):
                    with suppress(Exception):
                        cancel()
                raise TimeoutError("Remote call timed out (remote cancel issued)")

            s = getattr(job, "status", None)
            state["current_status"] = {
                "state": getattr(s, "state", None) if s else None,
                "progress": getattr(s, "progress", None) if s else None,
                "message": getattr(s, "stop_message", None) if s else None,
            }

            job_id = getattr(job, "id", None) or getattr(job, "job_id", None)
            if job_id is not None and job_id not in state["job_ids"]:
                state["job_ids"].append(job_id)

            if getattr(job, "is_failed", False):
                msg = state["current_status"].get("message")
                # Map Perceval "Cancel requested" to CancelledError
                if msg and "Cancel requested" in str(msg):
                    with self._lock:
                        self._active_jobs.discard(job)
                    raise CancelledError("Remote call was cancelled")
            # Success?
            if getattr(job, "is_complete", False):
                raw = job.get_results()
                with self._lock:
                    self._active_jobs.discard(job)
                return self._process_batch_results(raw, batch_size, layer, nsample)

            time.sleep(sleep_ms / 1000.0)
            sleep_ms = min(sleep_ms * 2, 400)

    # ---------------- Per-call RP pool helpers ----------------

    def _ensure_pool_for_layer(self, config: dict, layer: Any, pool_ctx: dict) -> None:
        """Create an RP pool for this layer within the current forward call, if absent."""
        lid = id(layer)
        if lid in pool_ctx["pools"]:
            return
        pool_size = int(pool_ctx["pool_size"])
        # Build independent RPs (each with its own handler)
        rps: list[RemoteProcessor] = []
        for _ in range(pool_size):
            rp = self._clone_remote_processor(self.remote_processor)
            rp.set_circuit(config["circuit"])
            if config.get("input_state"):
                input_state = pcvl.BasicState(config["input_state"])
                rp.with_input(input_state)
                n_photons = sum(config["input_state"])
                rp.min_detected_photons_filter(n_photons)
            rps.append(rp)
        pool_ctx["pools"][lid] = rps
        pool_ctx["cursors"][lid] = 0
        pool_ctx["locks"][lid] = threading.Lock()

    def _pool_next_rp(
        self, layer_id: int, pool_ctx: dict
    ) -> tuple[RemoteProcessor, int]:
        """Round-robin select an RP from the per-call pool for a given layer, returning (rp, slot_index)."""
        lock: threading.Lock = pool_ctx["locks"][layer_id]
        with lock:
            i = pool_ctx["cursors"][layer_id]
            rps = pool_ctx["pools"][layer_id]
            rp = rps[i]
            pool_ctx["cursors"][layer_id] = (i + 1) % len(rps)
            return rp, i

    # ---------------- Utilities & mapping ----------------

    def _clone_remote_processor(self, rp: RemoteProcessor) -> RemoteProcessor:
        """Create a sibling RemoteProcessor sharing auth/endpoint, with its OWN handler (avoid cross-thread sharing)."""
        # IMPORTANT: do NOT pass rpc_handler=... (avoid sharing handler across threads)
        return RemoteProcessor(
            name=rp.name,
            token=None,  # RemoteConfig pulls token from cache
            url=rp.get_rpc_handler().url
            if hasattr(rp.get_rpc_handler(), "url")
            else None,
            proxies=rp.proxies,
        )

    def _iter_layers_in_order(self, module: nn.Module) -> Iterable[nn.Module]:
        """Yield execution leaves in a deterministic order."""
        if getattr(module, "merlin_leaf", False):
            yield module
            return

        children = list(module.children())
        if not children:
            yield module
            return

        for child in children:
            yield from self._iter_layers_in_order(child)

    def _is_quantum_layer(self, module: Any) -> bool:
        if getattr(module, "force_local", False):
            return False
        return hasattr(module, "export_config") and callable(module.export_config)

    def _extract_input_params(self, config: dict) -> list[str]:
        circuit = config["circuit"]
        all_params = [p.name for p in circuit.get_parameters()]
        input_param_names: list[str] = []

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
        nsample: int | None = None,
    ) -> torch.Tensor:
        dist_size, state_to_index, valid_states = self._get_state_mapping(layer)
        output_tensors: list[torch.Tensor] = []

        if "results_list" in raw_results:
            results_list = raw_results["results_list"]

            for i, result_item in enumerate(results_list):
                if i >= batch_size:
                    break

                if "results" in result_item:
                    state_counts = result_item["results"]
                    probs = torch.zeros(dist_size)

                    if state_counts:
                        if (
                            getattr(layer, "no_bunching", False)
                            and valid_states is not None
                        ):
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
                        is_probability = (
                            isinstance(first_value, float) and first_value <= 1.0
                        )
                        total = 1.0 if is_probability else sum(state_counts.values())

                        for state_str, value in state_counts.items():
                            state_tuple = self._parse_perceval_state(state_str)
                            if state_to_index and state_tuple in state_to_index:
                                idx = state_to_index[state_tuple]
                                if idx < dist_size:
                                    probs[idx] = (
                                        value
                                        if is_probability
                                        else (value / total if total > 0 else 0)
                                    )

                        prob_sum = probs.sum()
                        if prob_sum > 0 and abs(float(prob_sum) - 1.0) > 1e-6:
                            probs = probs / prob_sum

                        output_tensors.append(probs)
                else:
                    output_tensors.append(torch.zeros(dist_size))

        while len(output_tensors) < batch_size:
            output_tensors.append(torch.zeros(dist_size))

        return torch.stack(output_tensors[:batch_size])

    def _get_state_mapping(self, layer: Any) -> tuple[int, dict | None, set | None]:
        if hasattr(layer, "computation_process") and hasattr(
            layer.computation_process, "simulation_graph"
        ):
            graph = layer.computation_process.simulation_graph

            if hasattr(graph, "final_keys") and graph.final_keys:
                dist_size = len(graph.final_keys)
                state_to_index = {
                    state: idx for idx, state in enumerate(graph.final_keys)
                }
                valid_states = (
                    set(graph.final_keys)
                    if getattr(layer, "no_bunching", False)
                    else None
                )
            else:
                n_modes = layer.circuit.m if hasattr(layer, "circuit") else graph.m
                n_photons = (
                    sum(layer.input_state)
                    if hasattr(layer, "input_state")
                    else graph.n_photons
                )

                if getattr(layer, "no_bunching", False):
                    dist_size = comb(n_modes, n_photons)
                    valid_states = set(
                        self._generate_no_bunching_states(n_modes, n_photons)
                    )
                    state_to_index = {
                        state: idx for idx, state in enumerate(sorted(valid_states))
                    }
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
                    valid_states = set(
                        self._generate_no_bunching_states(n_modes, n_photons)
                    )
                    state_to_index = {
                        state: idx for idx, state in enumerate(sorted(valid_states))
                    }
                else:
                    dist_size = comb(n_modes + n_photons - 1, n_photons)
                    state_to_index = None
                    valid_states = None
            else:
                dist_size = 10
                state_to_index = None
                valid_states = None

        return dist_size, state_to_index, valid_states

    def _generate_no_bunching_states(
        self, n_modes: int, n_photons: int
    ) -> list[tuple[int, ...]]:
        valid_states: list[tuple[int, ...]] = []

        def generate_states(current: list[int], remaining: int, start: int):
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

    # ---- Shot estimation (no remote jobs submitted) ----

    def estimate_required_shots_per_input(
        self,
        layer: nn.Module,
        input: torch.Tensor,
        desired_samples_per_input: int,
    ) -> list[int]:
        """
        Estimate required shots per input row for a QuantumLayer using the
        platform's RemoteProcessor estimator (transmittance, filters, etc.).

        - Accepts a single vector (shape: [D]) or a batch (shape: [B, D]).
        - Returns a list[int] with one entry per input row (0 means 'not viable').

        NOTE: This does not submit any cloud jobs; it only uses the estimator.
        """
        if not hasattr(layer, "export_config") or not callable(layer.export_config):
            raise TypeError("layer must provide export_config() like QuantumLayer")

        # Normalize input to [B, D]
        if input.dim() == 1:
            x = input.unsqueeze(0)
        elif input.dim() == 2:
            x = input
        else:
            raise ValueError("input must be 1D or 2D tensor")

        # Prepare a child RemoteProcessor mirroring user's processor (token/proxies)
        config = layer.export_config()
        child_rp = self._clone_remote_processor(self.remote_processor)
        child_rp.set_circuit(config["circuit"])

        # Mirror input_state & min_detected_photons if present in the exported config
        if config.get("input_state"):
            input_state = pcvl.BasicState(config["input_state"])
            child_rp.with_input(input_state)
            n_photons = sum(config["input_state"])
            # The estimator accounts for min_detected_photons via the processor setting
            child_rp.min_detected_photons_filter(
                n_photons if getattr(layer, "no_bunching", False) else 1
            )

        # Build param name list as Merlin does for execution
        input_param_names = self._extract_input_params(config)

        # For each row, map x -> param_values and query RP estimator
        x_np = x.detach().cpu().numpy()
        estimates: list[int] = []
        for i in range(x_np.shape[0]):
            row = x_np[i]
            param_values: dict[str, float] = {}
            for j, pname in enumerate(input_param_names):
                param_values[pname] = float(row[j] * np.pi) if j < row.shape[0] else 0.0

            # RemoteProcessor returns an int or None (if zero probability path)
            est = child_rp.estimate_required_shots(
                desired_samples_per_input, param_values=param_values
            )
            estimates.append(int(est) if est is not None else 0)

        return estimates

    # ---- Misc ----

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

    def get_job_history(self) -> list[RemoteJob]:
        return self._job_history

    def clear_job_history(self) -> None:
        self._job_history = []

    def _cancelled_error(self):
        from concurrent.futures import CancelledError

        return CancelledError("Remote call was cancelled")
