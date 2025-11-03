# MIT License
#
# Copyright (c) 2025 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to do so, subject to the
# following conditions:
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

"""Passive bridge between qubit statevectors and Merlin photonic computation spaces."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Literal

import perceval as pcvl
import torch
import torch.nn as nn

from ..core.computation_space import ComputationSpace
from ..utils.combinadics import Combinadics
from ..utils.dtypes import resolve_float_complex


class QuantumBridge(nn.Module):
    """
    Passive bridge between a qubit statevector (PyTorch tensor) and a Merlin QuantumLayer.

    The bridge applies a fixed transition matrix that maps computational-basis amplitudes
    into the selected photonic computation space (Fock, unbunched, or dual-rail).

    Args:
        n_photons: Number of logical photons (equals ``len(qubit_groups)``).
        n_modes: Total number of photonic modes that will be simulated downstream.
        qubit_groups: Logical grouping of qubits; ``[2, 1]`` means one photon is spread
            over ``2**2`` modes and another over ``2**1`` modes.
        wires_order: Endianness used to interpret computational basis strings.
        computation_space: Target photonic computation space. Accepts a
            :class:`ComputationSpace` enum or a string (``"fock"``, ``"unbunched"``,
            ``"dual_rail"``).
        normalize: Whether to L2-normalise input statevectors before applying the
            transition matrix.
        device: Optional device on which to place the output tensor.
        dtype: Real dtype that determines the corresponding complex dtype for amplitudes.
    """

    def __init__(
        self,
        n_photons: int,
        n_modes: int,
        *,
        qubit_groups: Sequence[int] | None = None,
        wires_order: Literal["little", "big"] = "little",
        computation_space: ComputationSpace = ComputationSpace.UNBUNCHED,
        normalize: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if wires_order not in ("little", "big"):
            raise ValueError("wires_order must be 'little' or 'big'.")

        if not isinstance(computation_space, ComputationSpace):
            raise TypeError(
                "'computation_space' must be a ComputationSpace enum value."
            )
        self.computation_space = computation_space
        self._device = device
        self._dtype = dtype
        self.normalize = normalize

        if qubit_groups is None:
            if n_modes != 2 * n_photons:
                raise ValueError(
                    "If qubit_groups are omitted, n_modes must equal 2 * n_photons (dual-rail default)."
                )
            qubit_groups = [1] * n_photons
        if len(qubit_groups) != n_photons:
            raise ValueError(
                f"Length of qubit_groups ({len(qubit_groups)}) must match n_photons ({n_photons})."
            )

        self.group_sizes: tuple[int, ...] = tuple(int(g) for g in qubit_groups)
        self._n_photons = n_photons
        self._n_modes = n_modes
        self.wires_order = wires_order

        expected_modes = sum(2**g for g in self.group_sizes)
        if expected_modes != n_modes:
            raise ValueError(
                f"Provided n_modes={n_modes} incompatible with qubit_groups (expected {expected_modes})."
            )

        self.n_qubits = sum(self.group_sizes)
        self.expected_state_dim = 2**self.n_qubits
        self._norm_epsilon = 1e-12

        self._output_enum = Combinadics(
            self.computation_space.value,
            self._n_photons,
            self._n_modes,
        )

        try:
            for occ in self._generate_qloq_basis():
                self._output_enum.fock_to_index(occ)
        except ValueError as exc:
            raise ValueError(
                "Selected computation space does not contain the QLOQ occupancies produced by the qubit groups."
            ) from exc

        self._output_size = self._output_enum.compute_space_size()
        self._transition_shape = (self._output_size, self.expected_state_dim)
        transition = self._build_transition_matrix()
        self.register_buffer("_transition", transition)
        self._transition = transition

    # ------------------------------------------------------------------
    # Basis construction
    # ------------------------------------------------------------------
    def _generate_qloq_basis(self) -> Iterator[tuple[int, ...]]:
        """Yield QLOQ occupancies in computational-basis order."""
        for idx in range(self.expected_state_dim):
            bits = format(idx, f"0{self.n_qubits}b")
            yield self._bitstring_to_occ(bits)

    @property
    def basis_occupancies(self) -> tuple[tuple[int, ...], ...]:
        """QLOQ occupancies indexed like the computational basis."""
        return tuple(self._generate_qloq_basis())

    @property
    def output_basis(self):  # type: ignore[override]
        """Iterator over occupancies enumerating the selected computation space."""
        return self._output_enum.iter_states()

    @property
    def n_modes(self) -> int:
        return self._n_modes

    @property
    def n_photons(self) -> int:
        return self._n_photons

    @property
    def output_size(self) -> int:
        return self._output_size

    def _build_transition_matrix(self) -> torch.Tensor:
        """Precompute the sparse transition matrix on the configured device."""
        rows: list[int] = []
        cols: list[int] = []
        for col, occ in enumerate(self._generate_qloq_basis()):
            row = self._output_enum.fock_to_index(occ)
            rows.append(row)
            cols.append(col)

        if rows:
            indices = torch.tensor([rows, cols], dtype=torch.long)
        else:
            indices = torch.empty((2, 0), dtype=torch.long)

        _, target_complex = resolve_float_complex(self._dtype)
        target_device = (
            self._device if self._device is not None else torch.device("cpu")
        )
        values = torch.ones(
            indices.shape[1], dtype=target_complex, device=target_device
        )
        indices = indices.to(device=target_device)
        matrix = torch.sparse_coo_tensor(
            indices,
            values,
            size=self._transition_shape,
            dtype=target_complex,
            device=target_device,
        )
        return matrix.coalesce()

    def _bitstring_to_occ(self, bitstring: str) -> tuple[int, ...]:
        """Convert a bitstring in computational order to an occupancy tuple."""
        if self.wires_order == "little":
            bitstring = bitstring[::-1]
        fock_state: list[int] = []
        bit_offset = 0
        for size in self.group_sizes:
            group_bits = bitstring[bit_offset : bit_offset + size]
            idx = int(group_bits, 2)
            fock_state.extend(1 if i == idx else 0 for i in range(2**size))
            bit_offset += size
        return tuple(fock_state)

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------
    def transition_matrix(self) -> torch.Tensor:
        r"""
        Return the precomputed transition matrix.

        Returns
        -------
        torch.Tensor
            Sparse COO tensor of shape ``(output_size, 2**n_qubits)`` mapping the qubit computational basis
            onto the selected photonic computation space.
        """
        return self._transition

    def qubit_to_fock_state(self, bitstring: str) -> pcvl.BasicState:
        r"""
        Convenience helper mirroring :func:`qubit_to_fock_state` with the bridge configuration.

        Parameters
        ----------
        bitstring : str
            Computational basis string. Its length must equal ``sum(self.group_sizes)``.

        Returns
        -------
        perceval.BasicState
            Photonic Fock state produced by the current qubit grouping convention.
        """
        if len(bitstring) != self.n_qubits:
            raise ValueError(
                f"Expected bitstring of length {self.n_qubits}, received {len(bitstring)}."
            )
        return pcvl.BasicState(self._bitstring_to_occ(bitstring))

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        r"""
        Project a qubit statevector onto the selected photonic computation space.

        Parameters
        ----------
        psi : torch.Tensor
            Input statevector with shape ``(2**n_qubits,)`` or ``(batch, 2**n_qubits)``. The tensor must
            reside on the bridge device.

        Returns
        -------
        torch.Tensor
            Amplitudes ordered according to the computation-space enumeration. A 1D input returns a 1D
            tensor; batched inputs preserve the leading batch dimension.

        Raises
        ------
        TypeError
            If ``psi`` is not a :class:`torch.Tensor`.
        ValueError
            If the tensor shape or device is inconsistent with the bridge configuration.
        """
        if not isinstance(psi, torch.Tensor):
            raise TypeError(
                "Statevector produced by the upstream module must be a torch.Tensor."
            )

        squeeze = False
        if psi.ndim == 1:
            psi = psi.unsqueeze(0)
            squeeze = True
        elif psi.ndim != 2:
            raise ValueError(
                f"QuantumBridge expects statevector shape (K,) or (B, K); received {psi.shape}."
            )

        _, target_complex = resolve_float_complex(self._dtype)

        expected_device = self._transition.device
        if psi.device != expected_device:
            raise ValueError(
                f"QuantumBridge expected input on device {expected_device}, received {psi.device}."
            )

        payload = psi.to(dtype=target_complex)

        if payload.shape[-1] != self.expected_state_dim:
            raise ValueError(
                f"Statevector dimension mismatch: expected {self.expected_state_dim}, "
                f"received {payload.shape[-1]}."
            )

        if self.normalize:
            norms = payload.norm(dim=1, keepdim=True)
            safe_norms = torch.where(
                norms > self._norm_epsilon,
                norms,
                torch.ones_like(norms),
            )
            payload = payload / safe_norms

        transformed = torch.sparse.mm(
            self._transition, payload.transpose(0, 1)
        ).transpose(0, 1)

        return transformed.squeeze(0) if squeeze else transformed


__all__ = ["QuantumBridge"]
