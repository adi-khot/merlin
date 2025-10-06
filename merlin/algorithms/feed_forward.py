from itertools import product

import perceval as pcvl
import torch
from perceval.components import BS, PS

from merlin import (
    OutputMappingStrategy,
    QuantumLayer,
    CircuitGenerator,
    CircuitType,
)


def create_circuit(M, input_size):
    """Create a quantum photonic circuit with beam splitters and phase shifters.

    Args:
        M (int): Number of modes in the circuit.

    Returns:
        pcvl.Circuit: A quantum photonic circuit with alternating beam splitter layers and phase shifters.
    """
    circuit = pcvl.Circuit(M)

    def layer_bs(circuit, k, M, j):
        for i in range(k, M - 1, 2):
            theta = pcvl.P(f"phi_{i}_{j}")
            circuit.add(i, BS(theta=theta))

    layer_bs(circuit, 0, M, 0)
    layer_bs(circuit, 1, M, 1)
    layer_bs(circuit, 0, M, 2)
    layer_bs(circuit, 1, M, 3)
    layer_bs(circuit, 0, M, 4)
    for i in range(input_size):
        phi = pcvl.P(f"pl_{i}")
        circuit.add(i, PS(phi))
    layer_bs(circuit, 0, M, 5)
    layer_bs(circuit, 1, M, 6)
    layer_bs(circuit, 0, M, 7)
    layer_bs(circuit, 1, M, 8)
    layer_bs(circuit, 0, M, 9)
    return circuit


def define_layer_no_input(n_modes, n_photons, circuit_type=None):
    """Define a quantum layer for feed-forward processing.

    Args:
        n_modes (int): Number of optical modes.
        n_photons (int): Number of photons in the layer.

    Returns:
        QuantumLayer: A configured quantum layer with trainable parameters.
    """

    circuit = create_circuit(n_modes, 0)
    input_state = [1] * n_photons + [0] * (n_modes - n_photons)

    layer = QuantumLayer(
        input_size=0,
        output_size=None,
        circuit=circuit,
        n_photons=n_photons,
        input_state=input_state,  # Random Initial quantum state used only for initialization
        output_mapping_strategy=OutputMappingStrategy.NONE,
        trainable_parameters=["phi"],
        no_bunching=True,
    )
    return layer


def define_layer_with_input(M, N, input_size, circuit_type=None):
    """Define the first layer of the feed-forward network.

    Args:
        M (int): Number of modes in the circuit.
        N (int): Number of photons.

    Returns:
        QuantumLayer: The first quantum layer with input parameters.
    """

    circuit = create_circuit(M, input_size)
    input_state = [1] * N + [0] * (M - N)
    layer = QuantumLayer(
        input_size=input_size,
        output_size=None,
        circuit=circuit,
        n_photons=N,
        input_state=input_state,  # Random Initial quantum state used only for initialization
        output_mapping_strategy=OutputMappingStrategy.NONE,
        input_parameters=["pl"],  # Optional: Specify device
        trainable_parameters=["phi"],
        no_bunching=True,
    )
    return layer


class FeedForwardBlock(torch.nn.Module):
    """Feed-forward quantum neural network for photonic computation.

    This class implements a feed-forward architecture where quantum layers are
    conditionally activated based on photon detection measurements.

    Args:
        m (int): Total number of modes.
        n_photons (int): Number of photons in the system.
        conditional_mode (int): Mode index used for conditional measurement.
    """

    def __init__(
        self,
        input_size: int,
        n: int,
        m: int,
        depth: int = None,
        state_injection=False,
        conditional_modes: list[int] = None,
        layers: list[QuantumLayer] = None,
        circuit_type = None,
    ):
        super().__init__()
        self.m = m
        self.input_size = input_size
        self.n_photons = n
        self.state_injection = state_injection
        self.conditional_modes = conditional_modes or [0]  # default: single mode
        self.n_cond = len(self.conditional_modes)
        self.output_keys = None

        self.layers = {}
        if depth is None:
            depth = self.m - 1
        self.depth = depth

        if layers is None:
            self.define_layers(circuit_type)
        else:
            tuples = self.generate_possible_tuples()
            self.tuples = tuples
            assert len(tuples) == len(layers)
            self.layers = {tuples[k]: layers[k] for k in range(len(layers))}
            start = 0
            self.input_segments = {}
            for tup in tuples:
                input_size = self.layers[tup].input_size
                self.input_segments[tup] = (start, start + input_size)
                start += input_size
            assert start == self.input_size, f"Input size mismatch: {start}"

    # -------------------------------
    # Multi-mode tuple generation
    # -------------------------------
    def generate_possible_tuples(self):
        """Generate tuples of joint conditional outcomes for multiple modes."""
        n = self.n_photons
        m = self.m
        possible_tuples = []
        for depth in range(self.depth + 1):
            # Each depth step adds outcomes for all conditional modes
            for t in product([0, 1], repeat=depth * self.n_cond):
                if self.state_injection:
                    possible_tuples.append(t)
                elif t.count(1) <= n - 1 and t.count(0) <= (m - n - 1):
                    possible_tuples.append(t)
        return possible_tuples


    def define_layers(self, circuit_type):
        """Define all quantum layers for different measurement outcomes.

        Creates a dictionary mapping measurement tuples to corresponding quantum layers.
        Also creates mapping for input size distribution.
        """
        input_size = self.input_size
        tuples = self.generate_possible_tuples()
        self.tuples = tuples
        self.input_segments = {}  # Track input size for each layer
        start = 0
        for tup in tuples:
            n = sum(tup)
            m = len(tup)
            if self.state_injection:
                input = min(self.m, input_size)
            else:
                input = min(self.m - m, input_size)
            if input > 0:
                if self.state_injection:
                    self.layers[tup] = define_layer_with_input(
                        self.m, self.n_photons, input, circuit_type=circuit_type
                    )
                else:
                    self.layers[tup] = define_layer_with_input(
                        self.m - m, self.n_photons - n, input, circuit_type=circuit_type
                    )
                self.input_segments[tup] = (start, start + input)
            else:
                if self.state_injection:
                    self.layers[tup] = define_layer_no_input(self.m, self.n_photons)
                else:
                    self.layers[tup] = define_layer_no_input(
                        self.m - m, self.n_photons - n
                    )
                self.input_segments[tup] = (0, 0)
            input_size -= input
            start += input
        assert input_size == 0, f"The input size can't be higher than {start}"

    def parameters(self):
        """Return an iterator over all trainable parameters.

        Yields:
            torch.Tensor: Trainable parameters from all quantum layers.
        """
        for layer in self.layers.values():
            yield from layer.parameters()

    def iterate_feedforward(
        self,
        current_tuple,
        remaining_amplitudes,
        keys,
        accumulated_prob,
        intermediary,
        outputs,
        depth=0,
        x=None,
    ):
        if depth >= self.depth:
            fock_probs = remaining_amplitudes.abs().pow(2)
            for i, key in enumerate(keys):
                if key not in outputs:
                    outputs[key] = torch.zeros_like(accumulated_prob)
                outputs[key] += accumulated_prob * fock_probs[:, i]
            return

        # Generate all 2^n_cond measurement outcomes for this step
        outcome_combos = list(product([0, 1], repeat=self.n_cond))
        # Compute probabilities for each combination
        mode_indices = self._indices_by_values(keys, self.conditional_modes)

        for combo in outcome_combos:
            # find joint indices where all modes match combo
            idx_combo = mode_indices[combo]
            prob_combo = remaining_amplitudes[:, idx_combo].abs().pow(2).sum(dim=1)
            current_key = current_tuple + combo
            intermediary[current_key] = prob_combo

            layer = self.layers.get(current_key, None)
            if layer is not None:
                m = layer.computation_process.m
                if self.state_injection:
                    match_idx = idx_combo
                    keys_next = keys
                else:
                    keys_next = layer.computation_process.simulation_graph.mapped_keys
                    match_idx = self._match_indices_multi(
                        keys, keys_next, self.conditional_modes, combo
                    )

                layer.computation_process.input_state = remaining_amplitudes[
                    :, match_idx
                ]
                start, end = self.input_segments[current_key]

                if start != end:
                    probs_next, amps_next = layer(x[:, start:end], return_amplitudes=True)
                else:
                    probs_next, amps_next = layer(return_amplitudes=True)

                new_prob = accumulated_prob * prob_combo
                self.iterate_feedforward(
                    current_key,
                    amps_next,
                    keys_next,
                    new_prob,
                    intermediary,
                    outputs,
                    depth + 1,
                    x=x,
                )
            else:
                # End of branch
                final_tuple = current_key + (0,) * (
                    (self.depth - len(current_tuple)) * self.n_cond
                )
                outputs[final_tuple] = accumulated_prob * prob_combo

    # -------------------------------
    # Multi-mode index helpers
    # -------------------------------
    def _indices_by_values(self, keys, modes):
        """Return indices for all joint combinations of (0,1)^len(modes)."""
        t = torch.tensor(keys)
        combos = list(product([0, 1], repeat=len(modes)))
        out = {}
        for combo in combos:
            mask = torch.ones(len(keys), dtype=torch.bool)
            for j, mode in enumerate(modes):
                mask &= t[:, mode] == combo[j]
            out[combo] = torch.nonzero(mask, as_tuple=True)[0]
        return out

    def _match_indices_multi(self, data, data_out, modes, values):
        """Remove multiple modes and match to smaller Fock basis."""
        out_map = {tuple(row): i for i, row in enumerate(data_out)}
        idx = []
        for tup in data:
            # remove selected modes
            reduced = tuple(v for i, v in enumerate(tup) if i not in modes)
            if reduced in out_map and all(tup[m] == values[j] for j, m in enumerate(modes)):
                idx.append(out_map[reduced])
        return torch.tensor(idx)

    def forward(self, x):
        if x.shape[-1] != self.input_size:
            raise ValueError(f"The input should be of size {self.input_size}")
        intermediary = {}
        outputs = {}
        input_size = min(self.input_size, self.m)
        input = x[:, :input_size]
        layer = self.layers[()]
        probs, amplitudes = layer(input, return_amplitudes=True)
        keys = layer.computation_process.simulation_graph.mapped_keys
        self.iterate_feedforward((), amplitudes, keys, 1.0, intermediary, outputs, 0, x=x)
        self.output_keys = outputs.keys()
        return torch.stack(list(outputs.values()), dim=1)

    def get_output_size(self):
        x = torch.rand(1, self.input_size)
        return self.forward(x).shape[-1]

    def size_ff_layer(self, k: int):
        tuples_k = [1 for tup in self.tuples if len(tup) == k*self.n_cond]
        return len(tuples_k)

    def define_ff_layer(self, k: int, layers: list[QuantumLayer]):
        len_layers = self.size_ff_layer(k)
        assert len(layers) == len_layers, f"layers should be of length {len_layers}"
        for i, t in enumerate(product([0, 1], repeat=k)):
            if t in self.layers:
                self.layers[t] = layers[i]
        self._recompute_segments()

    def input_size_ff_layer(self, k: int):
        return [self.layers[tup].input_size for tup in self.tuples if len(tup) == k*self.n_cond]

    def get_output_keys(self):
        if self.output_keys is None:
            x = torch.rand(1, self.input_size)
            _ = self.forward(x)
        return self.output_keys

    def _recompute_segments(self):
        """Recompute input segments based on current layer configuration.

        This method recalculates the input_segments mapping and updates input_size
        based on the current layers, similar to the computation in define_layers.
        """
        start = 0
        total_input_size = 0
        self.input_segments = {}

        for tup in self.tuples:
            if tup in self.layers:
                input_size = self.layers[tup].input_size
                self.input_segments[tup] = (start, start + input_size)
                start += input_size
                total_input_size += input_size
            else:
                self.input_segments[tup] = (0, 0)

        # Update input_size and print new value
        self.input_size = total_input_size
        print(f"New input size: {self.input_size}")



if __name__ == "__main__":
    from itertools import chain

    import perceval as pcvl
    from perceval.components import BS, PS

    L = torch.nn.Linear(20, 20)
    feed_forward = FeedForwardBlock(
        20, 2, 6, depth=3, conditional_modes=[2, 5], state_injection=True, circuit_type=CircuitType.PARALLEL_COLUMNS
    )
    layers = list(feed_forward.layers.values())
    feed_forward = FeedForwardBlock(
        20, 2, 6, depth=3, state_injection=True, conditional_modes=[2, 5], layers=layers
    )
    params = chain(L.parameters(), feed_forward.parameters())
    optimizer = torch.optim.Adam(params)
    print(feed_forward.get_output_size())
    print(feed_forward.input_size_ff_layer(1))
    print(feed_forward.size_ff_layer(1))
    feed_forward.define_ff_layer(1, layers[1:5])
    x = torch.rand(1, 20)
    for _ in range(10):
        res = feed_forward(L(x))
        result = feed_forward(L(x)).pow(2).sum()
        print(result)
        result.backward()
        optimizer.step()
        optimizer.zero_grad()
