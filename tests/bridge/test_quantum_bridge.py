import math

import torch
import perceval as pcvl
import pytest

from merlin import QuantumLayer, OutputMappingStrategy
from merlin.bridge.QuantumBridge import QuantumBridge, to_fock_state


def make_identity_layer(m: int, n_photons: int) -> QuantumLayer:
    c = pcvl.Circuit(m)  # identity unitary
    layer = QuantumLayer(
        input_size=0,
        circuit=c,
        n_photons=n_photons,
        output_mapping_strategy=OutputMappingStrategy.NONE,
        no_bunching=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return layer


def find_key_index(layer: QuantumLayer, basic_state: pcvl.BasicState) -> int:
    keys = layer.computation_process.simulation_graph.mapped_keys
    target = tuple(list(basic_state))
    for i, k in enumerate(keys):
        kt = tuple(list(k))
        if kt == target:
            return i
    raise AssertionError(f"BasicState {target} not found in mapped_keys")


@pytest.mark.parametrize(
    "basis_index,bits", [(0, "00"), (1, "01"), (2, "10"), (3, "11")]
)
def test_basis_state_mapping_little_endian(basis_index: int, bits: str):
    groups = [1, 1]  # 2 qubits -> two 1-qubit groups
    m = sum(2**g for g in groups)
    n_photons = len(groups)
    layer = make_identity_layer(m, n_photons)

    # PennyLane-like state provider (returns 1D statevector)
    def pl_state_fn(_x: torch.Tensor) -> torch.Tensor:
        psi = torch.zeros(2 ** sum(groups), dtype=torch.complex64)
        psi[basis_index] = 1.0 + 0.0j
        return psi

    bridge = QuantumBridge(
        qubit_groups=groups,
        merlin_layer=layer,
        pl_state_fn=pl_state_fn,
        wires_order="little",
        normalize=True,
    )

    out = bridge(torch.zeros(1, 1))  # dummy input, ignored by pl_state_fn
    assert out.shape[0] == 1

    # Expected Fock state for this basis (little-endian reverses the bitstring before grouping)
    expected_bs = to_fock_state(bits[::-1], groups)
    idx = find_key_index(layer, expected_bs)

    # Identity circuit should route basis |bits> to the same Fock key with prob ~1
    assert torch.isclose(out[0, idx], torch.tensor(1.0, dtype=out.dtype), atol=1e-6)
    assert torch.isclose(out[0].sum(), torch.tensor(1.0, dtype=out.dtype), atol=1e-6)


def test_superposition_and_normalization():
    groups = [1, 1]
    m = sum(2**g for g in groups)
    layer = make_identity_layer(m, n_photons=len(groups))

    # Unnormalized superposition |00> + i|11>
    def pl_state_fn(x: torch.Tensor) -> torch.Tensor:
        psi = torch.zeros(4, dtype=torch.complex64)
        psi[0] = 1.0 + 0.0j
        psi[3] = 0.0 + 1.0j
        # If called with batched input, return a batched statevector
        if isinstance(x, torch.Tensor) and x.shape[0] > 1:
            return psi.unsqueeze(0).expand(x.shape[0], -1)
        return psi

    bridge = QuantumBridge(
        qubit_groups=groups,
        merlin_layer=layer,
        pl_state_fn=pl_state_fn,
        wires_order="little",
        normalize=True,
    )

    out = bridge(torch.zeros(2, 2, 1))  # batched dummy input (batch size 2)
    assert out.shape[0] == 2

    idx_00 = find_key_index(layer, to_fock_state("00"[::-1], groups))
    idx_11 = find_key_index(layer, to_fock_state("11"[::-1], groups))

    # After normalization, each component should carry 0.5 probability on identity circuit
    for b in range(2):
        assert torch.isclose(
            out[b, idx_00], torch.tensor(0.5, dtype=out.dtype), atol=1e-5
        )
        assert torch.isclose(
            out[b, idx_11], torch.tensor(0.5, dtype=out.dtype), atol=1e-5
        )
        assert torch.isclose(
            out[b].sum(), torch.tensor(1.0, dtype=out.dtype), atol=1e-6
        )


def test_bridge_with_pennylane_qnode():
    qml = pytest.importorskip("pennylane")

    groups = [1]
    layer = make_identity_layer(m=2, n_photons=len(groups))

    dev = qml.device("default.qubit", wires=sum(groups), shots=None)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def pl_state(theta):
        qml.RY(theta, wires=0)
        return qml.state()

    def pl_state_fn(x: torch.Tensor) -> torch.Tensor:
        theta = x.squeeze().to(torch.get_default_dtype())
        psi = pl_state(theta)
        return psi.to(torch.complex64)

    bridge = QuantumBridge(
        qubit_groups=groups,
        merlin_layer=layer,
        pl_state_fn=pl_state_fn,
        wires_order="little",
        normalize=True,
    )

    angle = math.pi / 3
    inputs = torch.tensor([[angle]], dtype=torch.get_default_dtype())
    out = bridge(inputs)

    idx_0 = find_key_index(layer, to_fock_state("0", groups))
    idx_1 = find_key_index(layer, to_fock_state("1", groups))

    expected_zero = math.cos(angle / 2) ** 2
    expected_one = math.sin(angle / 2) ** 2

    assert torch.isclose(
        out[0, idx_0], torch.tensor(expected_zero, dtype=out.dtype), atol=1e-6
    )
    assert torch.isclose(
        out[0, idx_1], torch.tensor(expected_one, dtype=out.dtype), atol=1e-6
    )
    assert torch.isclose(out[0].sum(), torch.tensor(1.0, dtype=out.dtype), atol=1e-6)


def test_wires_order_big_endian_changes_mapping():
    groups = [1, 1]
    m = sum(2**g for g in groups)
    layer = make_identity_layer(m, n_photons=len(groups))

    # State |01> (index 1)
    def pl_state_fn(_x: torch.Tensor) -> torch.Tensor:
        psi = torch.zeros(4, dtype=torch.complex64)
        psi[1] = 1.0 + 0.0j
        return psi

    # Big-endian: do not reverse bits
    bridge_big = QuantumBridge(
        qubit_groups=groups,
        merlin_layer=layer,
        pl_state_fn=pl_state_fn,
        wires_order="big",
    )
    out_big = bridge_big(torch.zeros(1, 1))
    idx_big = find_key_index(layer, to_fock_state("01", groups))

    # Little-endian: reverse bits before grouping
    bridge_little = QuantumBridge(
        qubit_groups=groups,
        merlin_layer=layer,
        pl_state_fn=pl_state_fn,
        wires_order="little",
    )
    out_little = bridge_little(torch.zeros(1, 1))
    idx_little = find_key_index(layer, to_fock_state("01"[::-1], groups))

    assert torch.isclose(
        out_big[0, idx_big], torch.tensor(1.0, dtype=out_big.dtype), atol=1e-6
    )
    assert torch.isclose(
        out_little[0, idx_little], torch.tensor(1.0, dtype=out_little.dtype), atol=1e-6
    )
    # Ensure they target different keys for the same computational basis
    assert idx_big != idx_little


def test_error_when_qubit_groups_do_not_match_state_length():
    # Provide a 2-qubit state but groups sum to 1
    layer = make_identity_layer(m=2, n_photons=1)

    def pl_state_fn(_x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=torch.complex64
        )

    bridge = QuantumBridge(
        qubit_groups=[1], merlin_layer=layer, pl_state_fn=pl_state_fn
    )

    with pytest.raises(ValueError):
        _ = bridge(torch.zeros(1, 1))


def test_error_when_merlin_layer_mismatch_modes():
    # groups imply m=4, but build a layer with m=6
    bad_layer = make_identity_layer(m=6, n_photons=2)

    def pl_state_fn(_x: torch.Tensor) -> torch.Tensor:
        psi = torch.zeros(4, dtype=torch.complex64)
        psi[0] = 1.0 + 0.0j
        return psi

    bridge = QuantumBridge(
        qubit_groups=[1, 1], merlin_layer=bad_layer, pl_state_fn=pl_state_fn
    )

    with pytest.raises((ValueError, RuntimeError)):
        _ = bridge(torch.zeros(1, 1))
