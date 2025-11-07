# tests/core/cloud/_helpers.py
from __future__ import annotations

import time

from merlin.algorithms import QuantumLayer
from merlin.builder.circuit_builder import CircuitBuilder
from merlin.core.computation_space import ComputationSpace


def spin_until(pred, timeout_s: float = 10.0, sleep_s: float = 0.02) -> bool:
    start = time.time()
    while not pred():
        if time.time() - start > timeout_s:
            return False
        time.sleep(sleep_s)
    return True


def make_layer(
    m: int,
    n: int,
    input_size: int,
    *,
    no_bunching: bool = True,
    trainable: bool = True,
) -> QuantumLayer:
    b = CircuitBuilder(n_modes=m)
    if trainable:
        b.add_rotations(trainable=True, name="theta")
    if m >= 3:
        b.add_entangling_layer()
    b.add_angle_encoding(modes=list(range(input_size)), name="px")
    return QuantumLayer(
        input_size=input_size,
        builder=b,
        n_photons=n,
        computation_space=ComputationSpace.default(no_bunching=no_bunching),
    ).eval()
