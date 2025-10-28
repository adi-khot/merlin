Quantum Bridge (PennyLane ↔ Merlin)
===================================

Overview
--------
The Quantum Bridge lets you plug a PennyLane-style state preparation into a Merlin `QuantumLayer` by mapping qubit basis states into photonic Fock states using a one-photon-per-group encoding. This enables hybrid models where the differentiable statevector comes from a qubit simulator, and the photonic circuit and measurement are handled by Merlin.

Key ideas:
- You provide a preconfigured `QuantumLayer` (we do not build the circuit in the bridge).
- You insert the bridge between a PennyLane (or generic) module that outputs a complex statevector of size 2^n and the target `QuantumLayer`.
- You partition the n qubits into groups via `qubit_groups` (e.g., [2, 2] for 4 qubits), each group mapping to one photon spread over 2^group_size modes.
- You specify the photonic layout directly through ``n_modes`` (= Σ 2^group_size) and ``n_photons`` (= len(qubit_groups)) when constructing the bridge.
- The bridge keeps amplitudes in computational-basis order and emits a payload understood by the `QuantumLayer`, keeping the differentiable superposition path.

Minimal example
---------------
.. code-block:: python

    import torch
    import perceval as pcvl
    from merlin import QuantumLayer, OutputMappingStrategy
    from merlin.bridge.quantum_bridge import QuantumBridge

    # Build a simple identity photonic circuit with m = sum(2**g) modes
    qubit_groups = [1, 1]  # two groups of one qubit each → 2 photons, 4 modes
    m = sum(2**g for g in qubit_groups)

    circuit = pcvl.Circuit(m)
    layer = QuantumLayer(
        input_size=0,
        circuit=circuit,
        n_photons=len(qubit_groups),
        output_mapping_strategy=OutputMappingStrategy.NONE,
        no_bunching=True,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    class PennyLaneState(torch.nn.Module):
        """Return a PennyLane-like statevector |01>."""

        def forward(self, _x: torch.Tensor) -> torch.Tensor:
            psi = torch.zeros(4, dtype=torch.complex64)
            psi[1] = 1 + 0j  # |01>
            return psi

    state_prep = PennyLaneState()
    bridge = QuantumBridge(
        qubit_groups=qubit_groups,
        n_modes=m,
        n_photons=len(qubit_groups),
        wires_order="little",  # or "big"
        normalize=True,        # L2-normalize the input state
    )

    model = torch.nn.Sequential(state_prep, bridge, layer)

    x = torch.zeros(1, 1)  # dummy input; state prep ignores it
    y = model(x)           # probability distribution over photonic outcomes

The bridge returns a complex amplitude tensor. When you chain it with the same
``QuantumLayer`` instance, the layer automatically detects it, aligns the superposition with
its photonic ``mapped_keys`` ordering, and evaluates the circuit—no extra plumbing required.

Devices and dtypes
------------------
- The bridge output device follows the `QuantumLayer` device. Ensure the layer and bridge use the same device (CPU/CUDA).
- The bridge converts input states to complex64 for float32 and to complex128 for float64. The emitted probability distribution keeps the `QuantumLayer` real dtype (float32/float64).

Constraints
-----------
- No ancilla or postselected modes; total modes m must equal sum(2**group_size).
- Number of photons equals the number of groups.
- `QuantumLayer` must be provided; the bridge does not create circuits or perform validation against it.

API
---
.. autofunction:: merlin.bridge.QuantumBridge.to_fock_state

.. automodule:: merlin.bridge.QuantumBridge
   :members: QuantumBridge
   :undoc-members:
   :show-inheritance:
