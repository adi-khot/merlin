# quantum_bridge.py
#
# Generic PennyLane ↔ Merlin bridge that:
#   - receives a PennyLane statevector ψ ∈ C^{2^n}
#   - maps |bitstring⟩ amplitudes to Perceval SLOS occupancies via one-photon-per-group encoding
#   - emits a complex tensor consumed by a Merlin QuantumLayer
#
# Design :
#   - No trainable mapping in the bridge (any variational behavior belongs to the PL/qubit side)
#   - No design type, no ancilla/postselected modes; m = Σ 2^group_size
#   - The bridge is circuit-agnostic: it only needs the number of modes and photons
#
# Notes:
#   - Uses the tensor superposition path on the Merlin side (compute_superposition_state), so gradients
#     flow from Merlin back into the PL state-prep.
#   - We precompute the computational-basis → Fock occupancy list once lazily.
#   - Set `apply_sampling=False` at call time to keep the graph fully differentiable on the Merlin side.

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import perceval as pcvl
import torch
import torch.nn as nn


# ----------------------------
# Helpers: qubit-groups <-> Fock (no ancillas)
# ----------------------------
def to_fock_state(qubit_state: str, group_sizes: list[int]) -> pcvl.BasicState:
    """
    Map a bitstring to a BasicState with one photon per qubit-group (one-hot over 2^k modes).
    No ancilla/postselected modes are added. The number of modes is m = Σ 2^group_size.
    """
    fock_state: list[int] = []
    bit_offset = 0
    for size in group_sizes:
        group_len = 2**size
        bits = qubit_state[bit_offset : bit_offset + size]
        idx = int(bits, 2)
        fock_state += [1 if i == idx else 0 for i in range(group_len)]
        bit_offset += size
    return pcvl.BasicState(fock_state)


def _to_occ_tuple(key: pcvl.BasicState | Sequence[int]) -> tuple[int, ...]:
    """Convert a BasicState or occupancy list to a tuple for dict keys."""
    if isinstance(key, pcvl.BasicState):
        return tuple(key)
    return tuple(key)


def _tensor_metadata_key(tensor: torch.Tensor) -> tuple[int, torch.Size, torch.device]:
    """Return a key combining storage pointer, shape, and device for the given tensor."""
    return (
        int(tensor.untyped_storage().data_ptr()),
        tensor.shape,
        tensor.device,
    )


_BRIDGE_METADATA: dict[
    tuple[int, torch.Size, torch.device],
    tuple[tuple[tuple[int, ...], ...], tuple[Any, ...]],
] = {}


# ----------------------------
# The generic bridge
# ----------------------------
class QuantumBridge(nn.Module):
    """
    Plug-and-play bridge between a PennyLane state function/module and a Merlin QuantumLayer.

    REQUIRED:
      - qubit_groups: e.g., [2, 2] means 4 qubits split into two groups → 2 photons over blocks of 4 modes each
      - n_modes: Σ 2^group_size
      - n_photons: len(qubit_groups)

    Usage:
      - Place a PennyLane (or generic) module that outputs a complex statevector before this bridge.
      - Feed the resulting tensor (optionally alongside additional arguments meant for the Merlin layer)
        through this module.
      - The bridge emits a complex tensor of amplitudes that integrates naturally inside an
        ``nn.Sequential`` with a Merlin ``QuantumLayer``.

    This module hides:
      - qubit-group → Fock encoding (one photon per group)
      - batching, dtype/device handling

    No trainable mapping is performed here; any variational behavior should be implemented
    on the qubit/PennyLane side that produces ψ.
    """

    def __init__(
        self,
        *,
        qubit_groups: list[int],
        n_modes: int,
        n_photons: int,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        # encoding behavior:
        wires_order: str = "little",
        normalize: bool = True,
    ):
        super().__init__()
        if wires_order not in ("little", "big"):
            raise ValueError("wires_order must be 'little' or 'big'.")

        self.group_sizes = qubit_groups
        self.n_photons = len(qubit_groups)
        self.wires_order = wires_order
        self.device = device
        self.dtype = dtype
        self.normalize = normalize

        if n_photons != self.n_photons:
            raise ValueError(
                f"Provided n_photons={n_photons} does not match len(qubit_groups)={self.n_photons}."
            )
        expected_modes = sum(2**g for g in self.group_sizes)
        if expected_modes != n_modes:
            raise ValueError(
                f"Provided n_modes={n_modes} incompatible with qubit_groups (expected {expected_modes})."
            )
        self.n_photonic_modes = n_modes

        # Lazily built on first forward (when we see the actual 2^n)
        self._initialized = False
        self.n_qubits: int | None = None
        self._basis_occupancies: tuple[tuple[int, ...], ...] | None = None

    # ------------- internal: building basis occupancies -------------
    def _build_basis_occupancies(self, K: int):
        """Construct the list of occupancy tuples for each computational basis element."""
        occupancies: list[tuple[int, ...]] = []
        n = int(round(math.log2(K)))
        for k in range(K):
            bits = format(k, f"0{n}b")
            if self.wires_order == "little":
                bits = bits[::-1]
            fock = to_fock_state(bits, self.group_sizes)
            occupancies.append(_to_occ_tuple(fock))
        self._basis_occupancies = tuple(occupancies)

    def _maybe_init(self, psi: torch.Tensor):
        """Initialize after seeing the first statevector."""
        K = psi.shape[-1]
        n = int(round(math.log2(K)))
        if 2**n != K:
            raise ValueError(f"PennyLane state length {K} is not a power of two.")
        if sum(self.group_sizes) != n:
            raise ValueError(
                f"sum(qubit_groups)={sum(self.group_sizes)} != inferred n_qubits={n}"
            )
        self.n_qubits = n
        self._build_basis_occupancies(K)
        self._initialized = True

    # ------------- Input parsing helpers -------------
    @staticmethod
    def _extract_state_and_args(
        pl_output: torch.Tensor | Sequence[Any],
    ) -> tuple[torch.Tensor, tuple[Any, ...]]:
        """
        Accept PennyLane output that is either:
          - a complex tensor ψ (K,) or (B, K),
          - a sequence whose first element is ψ and whose remaining elements are Merlin args.
        """
        if isinstance(pl_output, torch.Tensor):
            return pl_output, ()

        if isinstance(pl_output, Sequence):
            if len(pl_output) == 0:
                raise ValueError(
                    "QuantumBridge received an empty sequence; expected a statevector as the first element."
                )
            state = pl_output[0]
            if not isinstance(state, torch.Tensor):
                raise TypeError(
                    "QuantumBridge expects the first element of the provided sequence to be a torch.Tensor statevector."
                )
            return state, tuple(pl_output[1:])

        raise TypeError(
            "QuantumBridge expects a torch.Tensor statevector or a sequence whose first element "
            "is that tensor."
        )

    # ------------- forward -------------
    def forward(
        self, pl_output: torch.Tensor | Sequence[Any]
    ) -> torch.Tensor:
        """
        PennyLane output → computational basis amplitudes.

        Accepts either a complex statevector ψ or a sequence whose first element is ψ and whose
        remaining elements should be forwarded as arguments to the Merlin layer.
        """
        psi, extra_args = self._extract_state_and_args(pl_output)

        if not isinstance(psi, torch.Tensor):
            raise TypeError("Statevector produced by PennyLane must be a torch.Tensor.")

        # Normalize shape to (B, K)
        if psi.ndim == 1:
            psi = psi.unsqueeze(0)
        elif psi.ndim != 2:
            raise ValueError(
                f"QuantumBridge expects statevector shape (K,) or (B, K); received {psi.shape}."
            )

        # Unify dtype/device with the bridge side
        target_complex = (
            torch.complex64
            if self.dtype in (torch.float32, torch.bfloat16)
            else torch.complex128
        )
        target_device = self.device if self.device is not None else psi.device
        psi = psi.to(dtype=target_complex, device=target_device)

        if self.normalize:
            psi = psi / (psi.norm(dim=1, keepdim=True) + 1e-20)

        if not self._initialized:
            self._maybe_init(psi)

        occupancies = self._basis_occupancies
        if occupancies is None:
            raise RuntimeError("QuantumBridge basis occupancies were not generated.")

        _BRIDGE_METADATA[_tensor_metadata_key(psi)] = (occupancies, extra_args)
        return psi


def pop_bridge_metadata(
    tensor: torch.Tensor,
) -> tuple[tuple[tuple[int, ...], ...] | None, tuple[Any, ...]]:
    """
    Retrieve and remove metadata associated with a QuantumBridge output tensor.

    Returns:
        (occupancies, extra_args) if available, otherwise (None, ()).
    """
    key = _tensor_metadata_key(tensor)
    if key in _BRIDGE_METADATA:
        occupancies, extra_args = _BRIDGE_METADATA.pop(key)
        return occupancies, extra_args
    return None, ()
