import torch
from itertools import product
from merlin import OutputMappingStrategy
from merlin import PhotonicBackend, CircuitType, StatePattern, AnsatzFactory, QuantumLayer
import perceval as pcvl
from perceval.components import BS, PS


def create_circuit(M):
    circuit = pcvl.Circuit(M)

    def layer_bs(circuit, k, M, j):
        for i in range(k, M - 1, 2):
            theta = pcvl.P(f"theta_{i}_{j}")
            circuit.add(i, BS(theta=theta))

    layer_bs(circuit, 0, M, 0)
    layer_bs(circuit, 1, M, 1)
    layer_bs(circuit, 0, M, 2)
    layer_bs(circuit, 1, M, 3)
    layer_bs(circuit, 0, M, 4)
    for i in range(M):
        phi = pcvl.P(f"phi_{i}")
        circuit.add(i, PS(phi))
    layer_bs(circuit, 0, M, 5)
    layer_bs(circuit, 1, M, 6)
    layer_bs(circuit, 0, M, 7)
    layer_bs(circuit, 1, M, 8)
    layer_bs(circuit, 0, M, 9)
    return circuit



def define_layer(n_modes, n_photons):
    circuit = create_circuit(n_modes)
    input_state = [1] * n_photons + [0] * (n_modes - n_photons)
    layer = QuantumLayer(
        input_size=0,
        output_size=None,
        circuit=circuit,
        n_photons=n_photons,
        input_state=input_state,  # Random Initial quantum state used only for initialization
        output_mapping_strategy=OutputMappingStrategy.NONE,
        trainable_parameters=["phi", "theta"],
        no_bunching=True,
    )
    return layer

def define_first_layer(M, N):
    circuit = create_circuit(M)
    input_state = [1] * N + [0] * (M - N)
    layer = QuantumLayer(
        input_size=M,
        output_size=None,
        circuit=circuit,
        n_photons=N,
        input_state=input_state,  # Random Initial quantum state used only for initialization
        output_mapping_strategy=OutputMappingStrategy.NONE,
        input_parameters=["phi"],  # Optional: Specify device
        trainable_parameters=["theta"],
        no_bunching=True,
    )
    return layer

class FeedForward(torch.nn.Module):
    def __init__(self, m:int, n_photons:int, conditional_mode:int):
        super().__init__()
        self.layer1 = define_first_layer(m, n_photons)
        self.conditional_mode = conditional_mode
        self.m = m
        self.n_photons = n_photons
        self.layers = {}
        self.define_layers()

    def generate_possible_tuples(self):
        n = self.n_photons
        m = self.m
        possible_tuples = set()

        for l in range(1, m):
            for t in product([0, 1], repeat=l):
                if t.count(1) <= n - 1 and t.count(0) <= (m - n - 1):
                    possible_tuples.add(t)

        return possible_tuples

    def define_layers(self):
        tuples = self.generate_possible_tuples()
        for tup in tuples:
            n = sum(tup)
            m = len(tup)
            self.layers[tup] = define_layer(self.m - m, self.n_photons - n)

    def parameters(self):
        for param in self.layer1.parameters():
            yield param
        for layer in self.layers.values():
            for param in layer.parameters():
                yield param



    def iterate_feedforward(self, current_tuple, remaining_amplitudes, keys, accumulated_prob, intermediary, outputs, depth=0):
        """

        """
        if depth >= self.m - 1:
            outputs[current_tuple] = accumulated_prob
            return
        
        next_position = len(current_tuple)

        layer_with_photon = self.layers.get(current_tuple + (1,), None)
        layer_without_photon = self.layers.get(current_tuple + (0,), None)
        
        layer_idx_not, layer_idx = self.indices_by_value(keys, 0)
        prob_not = remaining_amplitudes[:, layer_idx_not].abs().pow(2).sum(dim=1)
        prob_with = remaining_amplitudes[:, layer_idx].abs().pow(2).sum(dim=1)


        current_key_with = current_tuple + (1,)
        current_key_without = current_tuple + (0,)
        
        intermediary[current_key_with] = prob_with
        intermediary[current_key_without] = prob_not
        
        if layer_with_photon is not None:
            m = layer_with_photon.computation_process.m
            conditional_mode = min(self.conditional_mode, m)
            keys_with = layer_with_photon.computation_process.simulation_graph.mapped_keys
            match_idx_with = self.match_indices(keys, keys_with, conditional_mode, k_value=1)
            layer_with_photon.computation_process.input_state = remaining_amplitudes[:, match_idx_with]
            probs_with, amplitudes_with = layer_with_photon(return_amplitudes=True)
            
            new_prob_with = accumulated_prob * intermediary[current_key_with]
            
            self.iterate_feedforward(
                current_key_with, 
                amplitudes_with, 
                keys_with, 
                new_prob_with,
                intermediary, 
                outputs, 
                depth + 1
            )
        else:
            final_tuple_with = current_key_with + (0,) * (self.m - len(current_key_with))
            new_prob_with = accumulated_prob * intermediary[current_key_with]
            outputs[final_tuple_with] = new_prob_with
        
        if layer_without_photon is not None:
            m = layer_without_photon.computation_process.m
            conditional_mode = min(self.conditional_mode, m)
            keys_without = layer_without_photon.computation_process.simulation_graph.mapped_keys
            match_idx_without = self.match_indices(keys, keys_without, conditional_mode, k_value=0)
            layer_without_photon.computation_process.input_state = remaining_amplitudes[:, match_idx_without]
            probs_without, amplitudes_without = layer_without_photon(return_amplitudes=True)
            
            new_prob_without = accumulated_prob * intermediary[current_key_without]
            
            self.iterate_feedforward(
                current_key_without, 
                amplitudes_without, 
                keys_without, 
                new_prob_without,
                intermediary, 
                outputs, 
                depth + 1
            )
        else:
            final_tuple_without = current_key_without + (1,) * (self.m - len(current_key_without))
            new_prob_without = accumulated_prob * intermediary[current_key_without]
            outputs[final_tuple_without] = new_prob_without

    def forward(self, x):
        intermediary = {}
        outputs = {}
        probs, amplitudes = self.layer1(x, return_amplitudes=True)
        keys = self.layer1.computation_process.simulation_graph.mapped_keys
        layer_1_idx_not, layer_1_idx = self.indices_by_value(keys, self.conditional_mode)
        prob1_not = probs[:, layer_1_idx_not].sum(dim=1)
        prob1 = probs[:, layer_1_idx].sum(dim=1)
        intermediary[(1,)] = prob1
        intermediary[(0,)] = prob1_not
        self.iterate_feedforward((), amplitudes, keys, 1.0, intermediary, outputs)
        return torch.stack(list(outputs.values()))


    def indices_by_value(self, keys, k):
        """
        data : liste de tuples ou liste de listes
        k    : position à vérifier
        """
        # convertir en tenseur PyTorch
        t = torch.tensor(keys)
        # indices où la valeur vaut 0
        idx_0 = torch.nonzero(t[:, k] == 0, as_tuple=True)[0]

        # indices où la valeur vaut 1
        idx_1 = torch.nonzero(t[:, k] == 1, as_tuple=True)[0]

        return idx_0, idx_1


    def match_indices(self, data, data_out, k, k_value):
        """
        data      : list of tuples (length n)
        data_out  : list of tuples (length n-1)
        k         : index of the column to withdraw
        """
        # Convert to dict to optimize search
        out_map = {tuple(row): i for i, row in enumerate(data_out)}

        idx= []

        for i, tup in enumerate(data):
            removed = tup[:k] + tup[k + 1:]
            if removed in out_map:
                j = out_map[removed]
                if tup[k] == k_value:
                    idx.append(j)

        return torch.tensor(idx)



class FeedForwardBlock(torch.nn.Module):
    def __init__(self, layer1, layer2, layer2not, conditional_mode: int):
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer2not = layer2not
        self.conditional_mode = conditional_mode


    def forward(self, x):
        probs, amplitudes = self.layer1(x, return_amplitudes=True)
        keys = self.layer1.computation_process.simulation_graph.mapped_keys
        layer_1_idx_not, layer_1_idx = self.indices_by_value(keys, self.conditional_mode)
        prob1_not = probs[:, layer_1_idx_not].sum(dim=1)
        prob1 = probs[:, layer_1_idx].sum(dim=1)
        keys_2 = self.layer2.computation_process.simulation_graph.mapped_keys
        match_idx = self.match_indices(keys, keys_2, self.conditional_mode, k_value=1)
        keys_2_not = self.layer2not.computation_process.simulation_graph.mapped_keys
        match_idx_not = self.match_indices(keys, keys_2_not, self.conditional_mode, k_value=0)

        self.layer2.computation_process.input_state = amplitudes[:, match_idx]
        self.layer2not.computation_process.input_state = amplitudes[:, match_idx_not]
        return prob1 * self.layer2(), prob1_not * self.layer2not()



    def indices_by_value(self, keys, k):
        """
        data : liste de tuples ou liste de listes
        k    : position à vérifier
        """
        # convertir en tenseur PyTorch
        t = torch.tensor(keys)

        # indices où la valeur vaut 0
        idx_0 = torch.nonzero(t[:, k] == 0, as_tuple=True)[0]

        # indices où la valeur vaut 1
        idx_1 = torch.nonzero(t[:, k] == 1, as_tuple=True)[0]

        return idx_0, idx_1


    def match_indices(self, data, data_out, k, k_value):
        """
        data      : liste de tuples (longueur n)
        data_out  : liste de tuples (longueur n-1)
        k         : index de la colonne à retirer
        """
        # Conversion en dictionnaire pour retrouver rapidement les indices de data_out
        out_map = {tuple(row): i for i, row in enumerate(data_out)}

        idx= []

        for i, tup in enumerate(data):
            removed = tup[:k] + tup[k + 1:]  # on enlève l'élément k
            if removed in out_map:
                j = out_map[removed]  # index dans data_out
                if tup[k] == k_value:
                    idx.append(j)

        return torch.tensor(idx)



if __name__ == "__main__":
    import merlin as ML
    import perceval as pcvl
    from perceval.components import BS, PS
    from itertools import chain

    def create_circuit(M):

        circuit = pcvl.Circuit(M)

        def layer_bs(circuit, k, M, j):
            for i in range(k, M - 1, 2):
                theta = pcvl.P(f"theta_{i}_{j}")
                circuit.add(i, BS(theta=theta))

        layer_bs(circuit, 0, M, 0)
        layer_bs(circuit, 1, M, 1)
        layer_bs(circuit, 0, M, 2)
        layer_bs(circuit, 1, M, 3)
        layer_bs(circuit, 0, M, 4)
        for i in range(M):
            phi = pcvl.P(f"phi_{i}")
            circuit.add(i, PS(phi))
        layer_bs(circuit, 0, M, 5)
        layer_bs(circuit, 1, M, 6)
        layer_bs(circuit, 0, M, 7)
        layer_bs(circuit, 1, M, 8)
        layer_bs(circuit, 0, M, 9)
        return circuit

    M = 10
    N = 2
    circuit = create_circuit(M)
    input_state =  [1] * N + [0] * (M-N)
    layer = ML.QuantumLayer(
        input_size=M,
        output_size=None,
        circuit=circuit,
        n_photons=N,
        input_state=input_state,  # Random Initial quantum state used only for initialization
        output_mapping_strategy=OutputMappingStrategy.NONE,
        input_parameters=["phi"],  # Optional: Specify device
        trainable_parameters=["theta"],
        no_bunching=True,
    )
    input_size_2 = layer.output_size // 2
    input_state_2 = [1] * (N-1) + [0] * (M-N)
    input_state_2_not = [1] * N + [0] * (M-N-1)
    circuit_2 = create_circuit(M-1)
    circuit_2_not = create_circuit(M-1)
    layer_2 = ML.QuantumLayer(
        input_size=0,
        output_size=None,
        circuit=circuit_2,
        n_photons=N-1,
        input_state=input_state_2,  # Random Initial quantum state used only for initialization
        output_mapping_strategy=OutputMappingStrategy.NONE,
        trainable_parameters=["phi", "theta"],
        input_parameters=[],
        no_bunching=True,
    )

    layer_2_not = ML.QuantumLayer(
        input_size=0,
        output_size=None,
        circuit=circuit_2_not,
        n_photons=N,
        input_state=input_state_2_not,  # Random Initial quantum state used only for initialization
        output_mapping_strategy=OutputMappingStrategy.NONE,
        trainable_parameters=["phi", "theta"],
        input_parameters=[],
        no_bunching=True,
    )
    ff = FeedForwardBlock(layer, layer_2, layer_2_not, conditional_mode=1)
    x = torch.rand(M)
    L = torch.nn.Linear(M, M)
    params = chain(L.parameters(), ff.parameters())
    optimizer = torch.optim.Adam(params)
    for _ in range(10):
        result = (ff(L(x))[0]**2).sum() + (ff(L(x))[1]**2).sum()
        result.backward()
        optimizer.step()
        optimizer.zero_grad()
    x = torch.rand(M)
    L = torch.nn.Linear(12, 12)
    feed_forward = FeedForward(12, 3, 5)
    params = chain(L.parameters(), feed_forward.parameters())
    print(feed_forward.parameters())
    optimizer = torch.optim.Adam(params)
    x = torch.rand(1, 12)
    for _ in range(10):
        result = feed_forward(L(x)).pow(2).sum()
        print(result)
        result.backward()
        optimizer.step()
        optimizer.zero_grad()










