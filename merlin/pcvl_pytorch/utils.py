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

"""Utilities for converting between various perceval and torch representations."""

import perceval as pcvl  # type: ignore[import]
import torch

from ..core import ComputationSpace
from ..utils import Combinadics


def pcvl_to_tensor(
    state_vector: pcvl.StateVector,
    computation_space: ComputationSpace = ComputationSpace.FOCK,
    dtype: torch.dtype = torch.complex64,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Convert a Perceval StateVector into a torch Tensor.

    Args:
        state_vector: Perceval StateVector.
        computation_space: Computation space of the state vector following combinadics ordering.
        dtype: Desired torch dtype of the output Tensor.
        device: Desired torch device of the output Tensor.

    Returns:
        Equivalent torch Tensor.

    Raises:
        ValueError: If the StateVector includes states with incompatible photon number for the specified computation space,
            or non consistent number of photons across the states.

    """
    # Perceval StateVector.n is a set.
    ns_set = state_vector.n
    if len(ns_set) != 1:
        raise ValueError(
            "StateVector must have a fixed number of photons for conversion to tensor."
        )
    n_photons = ns_set.pop()

    n_modes = state_vector.m

    scheme = ComputationSpace.coerce(computation_space).value
    combinadics = Combinadics(scheme, n_photons, n_modes)
    tensor = torch.zeros(combinadics.compute_space_size(), dtype=dtype, device=device)

    # Perceval StateVector iteration yields (basic_state, amplitude)
    for bs, amplitude in state_vector:
        state = list(bs)

        # Validate constraints for restricted computation spaces
        if (
            computation_space is ComputationSpace.UNBUNCHED
            or computation_space is ComputationSpace.DUAL_RAIL
        ):
            if any(count > 1 for count in state):
                raise ValueError(
                    "unbunched and dual_rail compute spaces do not support basis states with photon bunching."
                )

        if computation_space is ComputationSpace.DUAL_RAIL:
            if n_photons * 2 != n_modes:
                raise ValueError(
                    "dual_rail compute space requires n_photons = m // 2 where m is the number of modes."
                )
            if any(state[i] + state[i + 1] != 1 for i in range(0, len(state), 2)):
                raise ValueError(
                    "dual_rail compute space requires each pair of modes to contain exactly one photon."
                )

        index = combinadics.fock_to_index(state)
        tensor[index] = amplitude

    return tensor
