.. _feedforward_block:

Feedforward Circuits
====================

Feedforward is a key capability in photonic quantum circuits, where a *partial measurement* determines the configuration of the downstream circuit.
This mechanism is comparable to *dynamic circuits* in the gate-based model of quantum computing 
(see for example: `IBM Dynamic Circuits <https://research.ibm.com/blog/dynamic-circuits>`_).  

The main difference is in the physical implementation:

- **Gate-based circuits:** gates are applied consecutively, and adapting the circuit requires performing a measurement and determining follow-up gates *within the coherence time of the qubits* (typically ms–s).
- **Photonic circuits:** feedforward involves measuring some modes while the remaining modes travel through a *delay line*. The delay must be short enough to avoid photon loss, while still allowing the photonic chip to be reconfigured. Measurement and reconfiguration must therefore happen on *sub-microsecond timescales*.  

Feedforward circuits are thus essential in photonic platforms for enabling adaptive quantum protocols.

--------------------------
FeedForwardBlock in MerLin
--------------------------

In MerLin, feedforward circuits are modeled by the :class:`~merlin.algorithms.feed_forward.FeedForwardBlock` object.  
It provides a structured way to define a sequence of layers that adapt dynamically based on measurement outcomes.

**Key properties:**

- Initialization specifies:

  - ``input_size``: Input size of the feedforward block
  - ``depth``: number of feedforward layers.
  - ``n``: number of photons.
  - ``m``: number of modes.
  - ``state_injection``: Whether or not to enable state injection
  - ``conditional_modes``: Modes on which the measure will be done (default value is the first mode)

- Each feedforward layer is defined as a *collection of quantum layers* corresponding to the possible measurement outcomes.
  - Each successive layer acts on *one mode less*.
  - If no layers are defined, MerLin will automatically generate **universal interferometers**.

- **Output tensor size** corresponds to the probabilities of all possible configurations of ``n`` photons over ``m`` modes.

- **State injection option:** when enabled, a new photon is injected into the measured mode whenever a measurement occurs.

- **Input parameters:** all circuits within the same feedforward layer must use the same number of input parameters.
  - The total number of input parameters of the :class:`~merlin.algorithms.feed_forward.FeedForwardBlock` is the sum of the input parameters across all layers.

----------------------------
API Reference
----------------------------

.. autoclass:: merlin.algorithms.feed_forward.FeedForwardBlock
   :members:
   :undoc-members:
   :show-inheritance:

----------------------------
Example
----------------------------

.. code-block:: python

   from merlin.algorithms import FeedForwardBlock, QuantumLayer
   import torch

   # Initialize feedforward block
   ffb = FeedforwardBlock(input_size=20, n=2, m=6, depth=3, conditional_modes=[2, 5])

   # You can define the mode to measure (conditional_mode)
   # By default, the input_size is divided into the first layers of the ff block,
   # matching when possible the number of modes. For instance, here: [6, 5, 5, 4]

   # Define a feedforward layer
   ffb.define_ff_layer(
       0,
       [
           QuantumLayer(
               input_size=4,
               circuit=circuit,
               trainable_parameters=["theta"],   # Which parameters are trainable
               input_parameters=["px"],          # Which parameters are inputs
           )
       ]
   )   # This will update the quantum layer to an input size of 18 for the FeedForwardBlock
   # You can also define the layers as a list of Quantum Layers in the parameter layers in the init function.
   # Random input tensor
   t = torch.rand(18)

   # Forward pass
   o = ffb(t)

   ffb = FeedForwardBlock(input_size=20, n=2, m=6, depth=3)

   # Inspect output size
   print(ffb.get_output_size())   # 6


   # Inspect how many quantum layers are required for the first FF layer
   print(ffb.size_ff_layer(1))   # 2

   # Inspect what is the current input size of those layers, returns a list of integers
   print(ffb.input_size_ff_layer(1))  # [5, 5]


   # Feedforward block with state injection
   ffb = FeedForwardBlock(input_size=20, n=2, m=6, depth=3, state_injection=True)
   # Default state_injection mode is False

   # New output size
   print(ffb.get_output_size())   # 15

   # Inspect how many quantum layers are required for the first FF layer
   print(ffb.size_ff_layer(1))   # 2

   # Inspect what is the current input size of those layers, returns a list of integers
   print(ffb.input_size_ff_layer(1))  # [6, 6]



----------------------------
PoolingFeedforward in MerLin
----------------------------
Similarly to the FeedforwardBlock previously presented, the pooling feedforward creates a partial measurement that determines the configuration of the downstream circuit.
However, this measure will modify the input state in the circuit instead of creating dynamic circuits. This object involves a measure made on some of the modes and a report of all the measured photons on the remaining modes of the circuit.

**Key properties:**

- Initialization specifies:

  -  ``n_modes`` : number of modes before the pooling feedforward layer.
  -  ``n_photons`` : number of photons.
  - ``n_output_modes``: number of modes exiting the pooling feedforward layer.
  - ``pooling_modes``: List of modes to aggregate together. The length of this list must be equal to the number of output modes
  - ``no_bunching``: Whether bunched states are allowed are not (default is True i.e bunched states not allowed)
This pooling layer doesn't contain any parameters, but pools some modes. This method is based on the pooling layer used in the Convolutional Neural Network, and is then very similar to it, the number of photons on every mode here corresponding to the values of the tensor in the CNN.


----------------------------
Example
----------------------------

.. code-block:: python

   import torch
   import merlin as ML
   from merlin.algorithms.feed_forward import PoolingFeedForward

   quantum_dim = 10
   # Define Quantum Layer before pff layer
   experiment = ML.Experiment(
                circuit_type=ML.CircuitType.SERIES,
                n_modes=16,
                n_photons=4,
                use_bandwidth_tuning=True
            )
    ansatz = ML.AnsatzFactory.create(
        experiment=experiment,
        input_size=quantum_dim,
        output_size=quantum_dim * 2,
        measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
    )
    q_layer_pre_pff = ML.QuantumLayer(
        input_size=quantum_dim,
        ansatz=ansatz,
        measurement_strategy=ML.MeasurementStrategy.AMPLITUDES,
    )

    # Define pff layer
    pff = PoolingFeedForward(n_modes=16, n_photons=4, n_output_modes=8)

    # Define Quantum Layer after pff layer
    experiment = ML.Experiment(
                circuit_type=ML.CircuitType.SERIES,
                n_modes=8,
                n_photons=4,
                use_bandwidth_tuning=True
            )
    ansatz = ML.AnsatzFactory.create(
        experiment=experiment,
        input_size=0,
        output_size=quantum_dim * 2,
        measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
    )
    q_layer_post_pff = ML.QuantumLayer(
        input_size=quantum_dim,
        ansatz=ansatz,
        measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
    )

    # Example of a forward pass
    x = torch.rand(quantum_dim)
    # The AMPLITUDES strategy returns the simulated amplitudes directly
    amplitudes = q_layer_pre_pff(x)
    # Going from amplitudes in the space with 16 modes, 4 photons, to 8 modes, 4 photons
    amplitudes = pff(amplitudes)
    # Set input state : If we provide a tensor -> Every component of the tensor refers to the amplitude of a different fock state -> Entangled input state
    q_layer_post_pff.set_input_state(amplitudes)
    output = q_layer_post_pff()


----------------------------
Further Reading
----------------------------

- :ref:`circuit_specific_optimizations`
- :ref:`output_mappings`
- :ref:`basic_concepts`
