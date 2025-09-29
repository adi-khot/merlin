# quantum_bridge.py
#
# Generic PennyLane ↔ Merlin bridge that:
#   - calls a PennyLane state function/module to get ψ ∈ C^{2^n}
#   - maps |bitstring⟩ amplitudes to Perceval SLOS keys via one-photon-per-group encoding
#   - feeds a batched complex superposition tensor to a Merlin QuantumLayer
#
# Design :
#   -  No trainable mapping in the bridge (any variational behavior belongs to the PL/qubit side)
#   -  No design type,  no ancilla/postselected modes; m = Σ 2^group_size
#   -  No fallback QuantumLayer creation; a user-supplied `merlin_layer` is REQUIRED
#
# Notes:
#   - Uses the tensor superposition path (compute_superposition_state), so gradients
#     flow from Merlin back into the PL state-prep.
#   - We precompute an index_map from computational-basis order → Merlin's mapped_keys.
#   - Set `apply_sampling=False` at call to keep the graph fully differentiable.

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import perceval as pcvl

try:
    # Requires the 'merlin' package exposing QuantumLayer
    from merlin import QuantumLayer
except Exception as e:  # pragma: no cover
    raise ImportError("This bridge requires 'merlin' with QuantumLayer installed.") from e


# ----------------------------
# Helpers: qubit-groups <-> Fock (no ancillas)
# ----------------------------
def to_fock_state(qubit_state: str, group_sizes: List[int]) -> pcvl.BasicState:
    """
    Map a bitstring to a BasicState with one photon per qubit-group (one-hot over 2^k modes).
    No ancilla/postselected modes are added. The number of modes is m = Σ 2^group_size.
    """
    fock_state: List[int] = []
    bit_offset = 0
    for size in group_sizes:
        group_len = 2 ** size
        bits = qubit_state[bit_offset : bit_offset + size]
        idx = int(bits, 2)
        fock_state += [1 if i == idx else 0 for i in range(group_len)]
        bit_offset += size
    return pcvl.BasicState(fock_state)


def _to_occ_tuple(key: Union[pcvl.BasicState, Sequence[int]]) -> tuple:
    """Convert a BasicState or occupancy list to a tuple for dict keys."""
    if isinstance(key, pcvl.BasicState):
        return tuple(list(key))
    return tuple(list(key))


# ----------------------------
# The generic bridge
# ----------------------------
class QuantumBridge(nn.Module):
    """
    Plug-and-play bridge between a PennyLane state function/module and a Merlin QuantumLayer.

    REQUIRED:
      - merlin_layer: an already-configured QuantumLayer (we do not build one here)
      - qubit_groups: e.g., [2,2] means 4 qubits split into two groups → 2 photons over blocks of 4 modes each
      - Provide either:
          * pl_module: nn.Module whose forward(x) returns a complex statevector (B, 2^n) or (2^n,),
            or
          * pl_state_fn: a callable(x) -> complex statevector

    This module hides:
      - qubit-group → Fock encoding (one photon per group)
      - SLOS key ordering and complex amplitude scattering
      - batching, dtype/device handling

    No trainable mapping is performed here; any variational behavior should be implemented
    on the qubit/PennyLane side that produces ψ.
    """

    def __init__(
        self,
        *,
        qubit_groups: List[int],
        merlin_layer: QuantumLayer,                         # REQUIRED
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        # PennyLane side:
        pl_module: Optional[nn.Module] = None,
        pl_state_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        # encoding behavior:
        wires_order: str = "little",
        normalize: bool = True,
    ):
        super().__init__()
        if merlin_layer is None:
            raise ValueError("QuantumBridge requires a user-supplied 'merlin_layer' (QuantumLayer).")
        if wires_order not in ("little", "big"):
            raise ValueError("wires_order must be 'little' or 'big'.")

        self.group_sizes = qubit_groups
        self.n_photons = len(qubit_groups)
        self.wires_order = wires_order
        self.device = device
        self.dtype = dtype
        self.normalize = normalize

        # PennyLane state provider
        if pl_module is None and pl_state_fn is None:
            raise ValueError("Provide either 'pl_module' or 'pl_state_fn' that returns a complex statevector.")
        self.pl_module = pl_module
        self._pl_state_fn = pl_state_fn if pl_state_fn is not None else (pl_module.forward if pl_module is not None else None)
        if self._pl_state_fn is None:
            raise ValueError("Could not resolve a PennyLane state function.")

        # Use the provided Merlin layer as-is
        self.merlin_layer = merlin_layer

        # Lazily built on first forward (when we see the actual 2^n)
        self._initialized = False
        self.n_qubits: Optional[int] = None
        self.n_photonic_modes: Optional[int] = None

        # Buffers to fill later
        self.register_buffer("index_map", torch.tensor([], dtype=torch.long), persistent=False)

    # ------------- internal: building maps -------------
    def _ensure_sim_graph(self):
        dummy = self.merlin_layer._create_dummy_parameters()
        _ = self.merlin_layer.computation_process.compute(dummy)

    def _build_index_map(self, K: int):
        """Build the 2^n → mapped_keys index map (ordering matches computational basis)."""
        self._ensure_sim_graph()
        mapped_keys = self.merlin_layer.computation_process.simulation_graph.mapped_keys

        # build dict from BasicState -> index
        by_state = {_to_occ_tuple(k): i for i, k in enumerate(mapped_keys)}

        # ordered map: for each computational basis |bits⟩, map to its Fock BasicState index
        idx_map: List[int] = []
        n = int(round(math.log2(K)))
        for k in range(K):
            bits = format(k, f"0{n}b")
            if self.wires_order == "little":
                bits = bits[::-1]
            fock = to_fock_state(bits, self.group_sizes)
            tup = _to_occ_tuple(fock)
            if tup not in by_state:
                raise RuntimeError(
                    f"Fock state for bitstring {bits} not found in mapped_keys. "
                    f"Ensure the Merlin layer's circuit uses m = Σ 2^group_size modes and "
                    f"n_photons = len(qubit_groups), with no ancilla/postselected modes."
                )
            idx_map.append(by_state[tup])

        dev = self.device
        self.index_map = torch.tensor(idx_map, dtype=torch.long, device=dev)

    def _maybe_init(self, psi: torch.Tensor):
        """Initialize after seeing the first statevector."""
        # psi: (B,K) complex or (K,) complex
        K = psi.shape[-1]
        n = int(round(math.log2(K)))
        if 2**n != K:
            raise ValueError(f"PennyLane state length {K} is not a power of two.")
        if sum(self.group_sizes) != n:
            raise ValueError(f"sum(qubit_groups)={sum(self.group_sizes)} != inferred n_qubits={n}")
        self.n_qubits = n
        self.n_photonic_modes = sum(2**g for g in self.group_sizes)

        # sanity checks against Merlin config (when available)
        try:
            cp = self.merlin_layer.computation_process
            if hasattr(cp, "n_photons") and cp.n_photons is not None:
                if int(cp.n_photons) != self.n_photons:
                    raise ValueError(
                        f"Merlin layer expects n_photons={cp.n_photons}, but qubit_groups imply {self.n_photons}."
                    )
            if hasattr(cp, "m") and cp.m is not None:
                if int(cp.m) != self.n_photonic_modes:
                    raise ValueError(
                        f"Merlin layer has m={cp.m} modes, but qubit_groups imply {self.n_photonic_modes}."
                    )
        except Exception:
            # If attributes are not present, skip strict validation
            pass

        self._build_index_map(K)
        self._initialized = True

    # ------------- PL call helper (supports modules that aren't batched) -------------
    def _call_pl_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Call the PL provider; if it isn't batched, iterate over batch.
        Returns a tensor of shape (B, K) complex.
        """
        out = self._pl_state_fn(x)
        if isinstance(out, torch.Tensor):
            if out.ndim == 1:
                return out.unsqueeze(0)
            if out.ndim == 2:
                return out
        # Fallback: try per-sample evaluation
        if not isinstance(x, torch.Tensor) or x.ndim < 1:
            raise ValueError("pl_state_fn returned unsupported type/shape and input is not batchable.")
        outs = []
        for i in range(x.shape[0]):
            yi = self._pl_state_fn(x[i])
            if not isinstance(yi, torch.Tensor) or yi.ndim != 1:
                raise ValueError("pl_state_fn must return a 1D complex statevector per sample.")
            outs.append(yi)
        return torch.stack(outs, dim=0)

    # ------------- forward -------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x -> PennyLane state ψ (complex) -> superposition over SLOS keys -> Merlin output.
        """
        psi = self._call_pl_state(x)  # (B,K) complex
        # Unify dtype/device with the bridge/merlin side
        target_complex = torch.complex64 if self.dtype in (torch.float32, torch.bfloat16) else torch.complex128
        target_device = self.device if self.device is not None else psi.device
        psi = psi.to(dtype=target_complex, device=target_device)

        if self.normalize:
            psi = psi / (psi.norm(dim=1, keepdim=True) + 1e-20)

        if not self._initialized:
            self._maybe_init(psi)

        B, K = psi.shape
        self._ensure_sim_graph()
        M = len(self.merlin_layer.computation_process.simulation_graph.mapped_keys)

        # fixed mapping: scatter ψ to length-M at precomputed positions
        superpos = torch.zeros((B, M), dtype=psi.dtype, device=psi.device)
        idx = self.index_map.to(psi.device)
        superpos.scatter_(1, idx.unsqueeze(0).expand(B, -1), psi)

        # Feed to Merlin (tensor path → compute_superposition_state)
        self.merlin_layer.computation_process.input_state = superpos
        return self.merlin_layer(apply_sampling=False)  # keep differentiable
