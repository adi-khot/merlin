:github_url: https://github.com/merlinquantum/merlin

=========================
Output Mapping Strategies
=========================

Overview
========

Output mapping strategies transform quantum probability distributions into classical neural network outputs. The choice of mapping strategy significantly impacts model performance, interpretability, and computational efficiency.

Merlin now uses `MeasurementStrategy` and `GroupingPolicy` instead of the deprecated `OutputMappingStrategy`. See migration notes below.


Mapping Strategies
==================

Migration Notes
---------------

**OutputMappingStrategy is deprecated.**
Use `MeasurementStrategy` and (optionally) `GroupingPolicy` instead. The mapping is:

- `OutputMappingStrategy.NONE` → `MeasurementStrategy.FOCKDISTRIBUTION`
- `OutputMappingStrategy.LINEAR` → `MeasurementStrategy.FOCKDISTRIBUTION` (add a torch.nn.Linear after the quantum layer)
- `OutputMappingStrategy.LEXGROUPING` → `MeasurementStrategy.FOCKGROUPING` + `GroupingPolicy.LEXGROUPING`
- `OutputMappingStrategy.MODGROUPING` → `MeasurementStrategy.FOCKGROUPING` + `GroupingPolicy.MODGROUPING`
- `OutputMappingStrategy.GROUPING` → `MeasurementStrategy.FOCKGROUPING` + `GroupingPolicy.LEXGROUPING`

For details, see the migration tests in `tests/sampling/test_measurement_strategy.py`.

GroupingPolicy defaults: If you use `MeasurementStrategy.FOCKGROUPING` and do not specify `GroupingPolicy`, it defaults to `GroupingPolicy.LEXGROUPING`.

Measurement Strategies
---------------------

FOCKDISTRIBUTION
----------------

**Description**: Maps quantum state amplitudes to the probability distribution over Fock states by applying $|a|^2$ to each amplitude.

**Mathematical Form**:

.. code-block:: text

    y_j = |a_j|^2

where $a_j$ is the amplitude for Fock state $j$, and $y_j$ is the probability for that state.

**Requirements**:

- Distribution size must equal desired output size
- No size transformation possible

**Characteristics**:

- No learnable parameters
- Converts amplitudes to probabilities
- Outputs are valid probability distributions
- Sum to 1 (normalized)

**Use Cases**:

- Probability estimation tasks
- When quantum probability distribution is the desired output
- Maximum efficiency requirements
- Quantum-native applications

.. code-block:: python

    ansatz = ML.AnsatzFactory.create(
        PhotonicBackend=experiment,
        input_size=4,
        output_size=dist_size,  # Must match
        measurement_strategy=ML.MeasurementStrategy.FOCKDISTRIBUTION
    )

**Advantages**:

- Pure quantum probability interpretation
- No additional parameters
- Maximum computational efficiency

**Disadvantages**:

- Rigid size constraints
- Limited output range [0,1]
- Sum constraint (sum of all y_i = 1)
- Not suitable for arbitrary outputs


FOCKGROUPING
------------

**Description**: Maps quantum state amplitudes to probabilities (via $|a|^2$), then groups the resulting probability distribution into buckets using a grouping policy.

**GroupingPolicy.LEXGROUPING** (default): Lexicographical grouping into equal-sized buckets. If not specified, this is the default.

For probability distribution p of size n mapping to output y of size m:

.. code-block:: text

    y_i = sum(p_j) for j from i*k to (i+1)*k-1

where k = ceiling(n/m) (bucket size)

**Padding Behavior**:

If n is not divisible by m, zeros are padded to make equal-sized groups.

**GroupingPolicy.MODGROUPING**: Groups indices by modulo arithmetic.

.. code-block:: text

    y_i = sum(p_j) for all j where j mod m = i

**Characteristics**:

- No learnable parameters
- Preserves probability mass
- Deterministic grouping scheme
- Output values in [0, 1] range

**Use Cases**:

- Probability-based outputs
- When preserving quantum measurement statistics
- Resource-constrained environments
- Interpretable quantum outputs

.. code-block:: python

    ansatz = ML.AnsatzFactory.create(
        PhotonicBackend=experiment,
        input_size=4,
        output_size=6,
        measurement_strategy=ML.MeasurementStrategy.FOCKGROUPING,
        grouping_policy=ML.GroupingPolicy.LEXGROUPING  # Optional, defaults to LEXGROUPING
    )

**Advantages**:

- No additional parameters
- Preserves quantum measurement structure
- Fast computation
- Interpretable outputs

**Disadvantages**:

- Limited flexibility
- May not capture optimal feature combinations
- Fixed grouping scheme


MODEEXPECTATION
---------------

**Description**: Maps quantum state amplitudes to probabilities (via $|a|^2$), then marginalizes the probability distribution to per-mode photon presence probabilities or expectations.

**Mathematical Form**:

.. code-block:: text

    y_k = sum_{s: s_k >= 1} |a_s|^2   (no_bunching=True)
    y_k = sum_{s} s_k * |a_s|^2       (no_bunching=False)

where s is a Fock state, s_k is the photon count in mode k, $|a_s|^2$ is the probability of state s.

**Characteristics**:

- No learnable parameters
- Output is per-mode probability or expectation
- Useful for threshold detectors or mode-wise analysis

**Use Cases**:

- Mode-wise probability estimation
- Quantum feature extraction
- Interpretable outputs for photonic hardware

.. code-block:: python

    keys = [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    ansatz = ML.AnsatzFactory.create(
        PhotonicBackend=experiment,
        input_size=len(keys),
        output_size=3,
        measurement_strategy=ML.MeasurementStrategy.MODEEXPECTATION,
        keys=keys,
        no_bunching=True,
    )

**Advantages**:

- Direct mode-wise interpretation
- Useful for hardware with threshold detectors
- No additional parameters

**Disadvantages**:

- Requires keys (Fock states)
- Output size must match number of modes


STATEVECTOR
-----------

**Description**: Returns the quantum state vector amplitudes directly (simulator only). No conversion to probabilities is performed.

**Mathematical Form**:

.. code-block:: text

    y = a

where $a$ is the amplitude vector for all Fock states.

**Characteristics**:

- No learnable parameters
- Only available in simulation (not hardware)
- Output is complex-valued amplitudes

**Use Cases**:

- Quantum simulation
- Amplitude-based analysis
- Research and debugging

.. code-block:: python

    ansatz = ML.AnsatzFactory.create(
        PhotonicBackend=experiment,
        input_size=5,
        output_size=5,
        measurement_strategy=ML.MeasurementStrategy.STATEVECTOR,
    )

**Advantages**:

- Direct access to quantum amplitudes
- Useful for simulation and research

**Disadvantages**:

- Not available on hardware
- Output may not be interpretable as probabilities

CUSTOMOBSERVABLE
----------------

**Description**: (TODO) Custom observable mapping. Placeholder for future implementation.

**Characteristics**:

- Will allow user-defined quantum-to-classical mappings
- Not yet implemented

**Use Cases**:

- Custom quantum measurement schemes
- Advanced research

.. code-block:: python

    ansatz = ML.AnsatzFactory.create(
        PhotonicBackend=experiment,
        input_size=2,
        output_size=2,
        measurement_strategy=ML.MeasurementStrategy.CUSTOMOBSERVABLE,
    )

**Advantages**:

- Will enable custom quantum-classical mappings

**Disadvantages**:

- Not yet implemented

Selection Guidelines
====================

Task-Based Recommendations
--------------------------

.. list-table::
     :header-rows: 1
     :widths: 20 20 20 20 20

     * - Task Type
         - Primary Choice
         - Alternative
         - Reasoning
         - Notes
     * - Classification
         - FOCKDISTRIBUTION + Linear
         - FOCKGROUPING (LEXGROUPING or MODGROUPING)
         - Need logits/flexible outputs
         - Add torch.nn.Linear after quantum layer for logits
     * - Regression
         - FOCKDISTRIBUTION + Linear
         - MODEEXPECTATION
         - Require arbitrary output ranges
         - Use Linear for flexible output, ModeExpectation for interpretable features
     * - Probability Estimation
         - FOCKDISTRIBUTION
         - FOCKGROUPING, MODEEXPECTATION
         - Want direct probabilities
         - Output is normalized probability distribution
     * - Structured Outputs
         - FOCKGROUPING (MODGROUPING)
         - FOCKGROUPING (LEXGROUPING), MODEEXPECTATION
         - Exploit pattern structure
         - Use MODGROUPING for cyclic/periodic data
     * - Quantum Feature Extraction
         - MODEEXPECTATION
         - STATEVECTOR
         - Extract mode-wise or amplitude features
         - STATEVECTOR for simulation/debugging only
     * - Custom Measurement
         - CUSTOMOBSERVABLE
         - (none)
         - Advanced research, user-defined mapping
         - Not yet implemented

Performance Considerations
--------------------------

.. list-table::
     :header-rows: 1
     :widths: 20 20 20 20 20

     * - Strategy
         - Parameter Cost
         - Computation Cost
         - Memory Usage
         - Output Type
     * - FOCKDISTRIBUTION
         - 0
         - O(input_size)
         - Minimal
         - Probability distribution
     * - FOCKGROUPING (LEXGROUPING)
         - 0
         - O(input_size)
         - Low
         - Grouped probabilities
     * - FOCKGROUPING (MODGROUPING)
         - 0
         - O(input_size)
         - Low
         - Grouped probabilities
     * - MODEEXPECTATION
         - 0
         - O(input_size × n_modes)
         - Low
         - Per-mode probabilities/expectations
     * - STATEVECTOR
         - 0
         - O(input_size)
         - Minimal
         - Amplitudes (complex, simulation only)
     * - CUSTOMOBSERVABLE
         - 0
         - TBD
         - TBD
         - User-defined (TODO)
     * - FOCKDISTRIBUTION + Linear
         - O(input_size × output_size)
         - O(input_size × output_size)
         - High
         - Arbitrary outputs

Size Compatibility
------------------

.. code-block:: python

        def check_mapping_compatibility(quantum_output_size, desired_output_size):
                """Check which MeasurementStrategy/GroupingPolicy are compatible."""
                compatible = []
                # FOCKDISTRIBUTION: Only if sizes match exactly
                if quantum_output_size == desired_output_size:
                        compatible.append('FOCKDISTRIBUTION')
                # FOCKGROUPING: Always compatible (uses padding or grouping)
                compatible.append('FOCKGROUPING')
                # MODEEXPECTATION: Only if output_size == n_modes
                # (user must provide keys and output_size = n_modes)
                compatible.append('MODEEXPECTATION')
                # STATEVECTOR: Only if sizes match exactly
                if quantum_output_size == desired_output_size:
                        compatible.append('STATEVECTOR')
                # CUSTOMOBSERVABLE: TBD
                compatible.append('CUSTOMOBSERVABLE')
                # FOCKDISTRIBUTION + Linear: Always compatible
                compatible.append('FOCKDISTRIBUTION + Linear')
                return compatible
     - LEXGROUPING
     - Want direct probabilities
   * - Structured Outputs
     - MODGROUPING
     - LEXGROUPING
     - Exploit pattern structure

Performance Considerations
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Strategy
     - Parameter Cost
     - Computation Cost
     - Memory Usage
   * - LINEAR
     - O(input_size × output_size)
     - O(input_size × output_size)
     - High
   * - LEXGROUPING
     - 0
     - O(input_size)
     - Low
   * - MODGROUPING
     - 0
     - O(input_size)
     - Low
   * - NONE
     - 0
     - 0
     - Minimal

Size Compatibility
------------------

.. code-block:: python

    def check_mapping_compatibility(quantum_output_size, desired_output_size):
        """Check which mapping strategies are compatible."""

        compatible = []

        # LINEAR: Always compatible
        compatible.append('LINEAR')

        # LEXGROUPING: Always compatible (uses padding)
        compatible.append('LEXGROUPING')

        # MODGROUPING: Always compatible
        compatible.append('MODGROUPING')

        # NONE: Only if sizes match exactly
        if quantum_output_size == desired_output_size:
            compatible.append('NONE')

        return compatible

Advanced Usage Patterns
=======================

Dynamic Strategy Selection
--------------------------

.. code-block:: python


    class AdaptiveOutputLayer(nn.Module):
        def __init__(self, quantum_layer, strategies=None):
            super().__init__()
            self.quantum_layer = quantum_layer
            self.strategies = strategies or [
                ('FOCKDISTRIBUTION', None),
                ('FOCKGROUPING', 'LEXGROUPING'),
                ('FOCKGROUPING', 'MODGROUPING'),
                ('MODEEXPECTATION', None),
                ('STATEVECTOR', None)
            ]
            self.current_strategy = 0

            # Create multiple output mappings
            self.mappers = nn.ModuleDict()
            for strategy, grouping_policy in self.strategies:
                if strategy == 'FOCKDISTRIBUTION':
                    self.mappers[f'{strategy}'] = nn.Identity()
                elif strategy == 'FOCKGROUPING':
                    self.mappers[f'{strategy}_{grouping_policy}'] = lambda x: ML.OutputMapper.create_mapping(
                        ML.MeasurementStrategy.FOCKGROUPING,
                        input_size=x.shape[-1],
                        output_size=output_size,
                        grouping_policy=getattr(ML.GroupingPolicy, grouping_policy) if grouping_policy else None
                    )(x)
                elif strategy == 'MODEEXPECTATION':
                    self.mappers[f'{strategy}'] = lambda x: ML.OutputMapper.create_mapping(
                        ML.MeasurementStrategy.MODEEXPECTATION,
                        input_size=x.shape[-1],
                        output_size=output_size,
                        keys=keys,
                        no_bunching=True
                    )(x)
                elif strategy == 'STATEVECTOR':
                    self.mappers[f'{strategy}'] = lambda x: ML.OutputMapper.create_mapping(
                        ML.MeasurementStrategy.STATEVECTOR,
                        input_size=x.shape[-1],
                        output_size=output_size
                    )(x)

        def forward(self, x):
            quantum_out = self.quantum_layer(x)
            strategy, grouping_policy = self.strategies[self.current_strategy]
            key = f'{strategy}' if not grouping_policy else f'{strategy}_{grouping_policy}'
            return self.mappers[key](quantum_out)

        def switch_strategy(self, new_strategy_idx):
            self.current_strategy = new_strategy_idx

Ensemble Output Mapping
-----------------------

.. code-block:: python


    class EnsembleOutputMapping(nn.Module):
        def __init__(self, input_size, output_size, keys=None):
            super().__init__()
            # Multiple mapping strategies
            self.linear = nn.Linear(input_size, output_size)
            self.fockgroup_lex = ML.OutputMapper.create_mapping(
                ML.MeasurementStrategy.FOCKGROUPING,
                input_size=input_size,
                output_size=output_size,
                grouping_policy=ML.GroupingPolicy.LEXGROUPING
            )
            self.fockgroup_mod = ML.OutputMapper.create_mapping(
                ML.MeasurementStrategy.FOCKGROUPING,
                input_size=input_size,
                output_size=output_size,
                grouping_policy=ML.GroupingPolicy.MODGROUPING
            )
            self.mode_expect = ML.OutputMapper.create_mapping(
                ML.MeasurementStrategy.MODEEXPECTATION,
                input_size=input_size,
                output_size=output_size,
                keys=keys,
                no_bunching=True
            ) if keys is not None else None
            self.state_vector = ML.OutputMapper.create_mapping(
                ML.MeasurementStrategy.STATEVECTOR,
                input_size=input_size,
                output_size=output_size
            )
            # Learnable combination weights
            self.combination_weights = nn.Parameter(torch.ones(5) / 5)

        def forward(self, quantum_distribution):
            outs = [
                self.linear(quantum_distribution),
                self.fockgroup_lex(quantum_distribution),
                self.fockgroup_mod(quantum_distribution),
            ]
            if self.mode_expect:
                outs.append(self.mode_expect(quantum_distribution))
            outs.append(self.state_vector(quantum_distribution))
            weights = torch.softmax(self.combination_weights, dim=0)
            combined = sum(w * o for w, o in zip(weights, outs))
            return combined

Hierarchical Mapping
--------------------

.. code-block:: python


    class HierarchicalMapping(nn.Module):
        def __init__(self, input_size, intermediate_size, output_size):
            super().__init__()
            # First stage: Reduce dimensionality with FockGrouping (LEXGROUPING)
            self.stage1 = ML.OutputMapper.create_mapping(
                ML.MeasurementStrategy.FOCKGROUPING,
                input_size=input_size,
                output_size=intermediate_size,
                grouping_policy=ML.GroupingPolicy.LEXGROUPING
            )
            # Second stage: Learn final transformation
            self.stage2 = nn.Linear(intermediate_size, output_size)

        def forward(self, quantum_distribution):
            intermediate = self.stage1(quantum_distribution)
            return self.stage2(intermediate)

Optimization Strategies
=======================

Gradient Flow Analysis
----------------------

Different mapping strategies affect gradient flow differently:

**LINEAR**:

- Full gradient backpropagation through learned weights
- May benefit from learning rate scheduling
- Standard optimization techniques apply

**LEXGROUPING/MODGROUPING**:

- Direct gradient flow through grouping operation
- Generally stable gradients
- May require careful quantum layer optimization

**NONE**:

- Direct gradients to quantum layer
- No intermediate transformation noise
- Depends entirely on quantum circuit optimization

Performance Tuning
------------------

.. code-block:: python

    def optimize_mapping_choice(model, val_loader, strategies):
        """Empirically determine best mapping strategy."""

        results = {}

        for strategy in strategies:
            # Create model variant with this strategy
            model_variant = create_model_with_strategy(strategy)

            # Evaluate performance
            val_loss = evaluate_model(model_variant, val_loader)
            train_time = measure_training_speed(model_variant)
            memory_usage = measure_memory_usage(model_variant)

            results[strategy] = {
                'val_loss': val_loss,
                'train_time': train_time,
                'memory_usage': memory_usage,
                'score': val_loss + 0.1 * train_time + 0.01 * memory_usage
            }

        # Return best strategy
        best_strategy = min(results.keys(), key=lambda k: results[k]['score'])
        return best_strategy, results

Integration with Classical Networks
===================================

Pre-quantum Processing
----------------------

.. code-block:: python

    class PreQuantumProcessor(nn.Module):
        def __init__(self, classical_size, quantum_size):
            super().__init__()
            self.processor = nn.Sequential(
                nn.Linear(classical_size, quantum_size * 2),
                nn.ReLU(),
                nn.Linear(quantum_size * 2, quantum_size),
                nn.Sigmoid()  # Normalize for quantum layer
            )

        def forward(self, x):
            return self.processor(x)

Post-quantum Processing
-----------------------

.. code-block:: python

    class PostQuantumProcessor(nn.Module):
        def __init__(self, quantum_size, final_size, mapping_strategy):
            super().__init__()

            # Quantum output mapping

            if mapping_strategy == 'FOCKDISTRIBUTION + Linear':
                self.mapper = nn.Linear(quantum_size, quantum_size // 2)
            elif mapping_strategy == 'FOCKGROUPING_LEXGROUPING':
                self.mapper = ML.OutputMapper.create_mapping(
                    ML.MeasurementStrategy.FOCKGROUPING,
                    input_size=quantum_size,
                    output_size=quantum_size // 2,
                    grouping_policy=ML.GroupingPolicy.LEXGROUPING
                )

            # Classical post-processing
            self.post_processor = nn.Sequential(
                nn.Linear(quantum_size // 2, final_size * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(final_size * 2, final_size)
            )

        def forward(self, quantum_output):
            mapped = self.mapper(quantum_output)
            return self.post_processor(mapped)

Debugging Output Mappings
=========================

Diagnostic Tools
----------------

.. code-block:: python


    def diagnose_output_mapping(layer, test_input):
        """Diagnose output mapping behavior."""
        # Get quantum amplitudes
        with torch.no_grad():
            amplitudes = layer.computation_process.compute(
                layer.prepare_parameters([test_input])
            )
        print(f"Quantum amplitudes shape: {amplitudes.shape}")
        print(f"Amplitude range: [{amplitudes.min():.6f}, {amplitudes.max():.6f}]")
        # Test mapping for each MeasurementStrategy
        for strategy in [
            ML.MeasurementStrategy.FOCKDISTRIBUTION,
            ML.MeasurementStrategy.FOCKGROUPING,
            ML.MeasurementStrategy.MODEEXPECTATION,
            ML.MeasurementStrategy.STATEVECTOR
        ]:
            try:
                output_mapping = ML.OutputMapper.create_mapping(
                    strategy,
                    input_size=amplitudes.shape[-1],
                    output_size=amplitudes.shape[-1]
                )
                final_output = output_mapping(amplitudes)
                print(f"{strategy} output shape: {final_output.shape}")
                print(f"Output range: [{final_output.min():.6f}, {final_output.max():.6f}]")
            except Exception as e:
                print(f"{strategy} mapping failed: {e}")
        # Check gradients (example for FockDistribution + Linear)
        output_mapping = nn.Linear(amplitudes.shape[-1], 4)
        amplitudes = amplitudes.clone().detach().requires_grad_(True)
        final_output = output_mapping(amplitudes)
        loss = final_output.sum()
        loss.backward()
        if hasattr(output_mapping, 'weight') and output_mapping.weight.grad is not None:
            grad_norm = output_mapping.weight.grad.norm()
            print(f"Output mapping gradient norm: {grad_norm:.6f}")