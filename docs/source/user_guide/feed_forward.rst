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

----------------------------
FeedforwardBlock in MerLin
----------------------------

In MerLin, feedforward circuits are modeled by the :class:`FeedforwardBlock` object.  
It provides a structured way to define a sequence of layers that adapt dynamically based on measurement outcomes.

**Key properties:**

- Initialization specifies:
  
  - ``depth``: number of feedforward layers.
  - ``n``: number of photons.
  - ``m``: number of modes.

- Each feedforward layer is defined as a *collection of quantum layers* corresponding to the possible measurement outcomes.
  - Each successive layer acts on *one mode less*.
  - If no layers are defined, MerLin will automatically generate **universal interferometers**.

- **Output tensor size** corresponds to the probabilities of all possible configurations of ``n`` photons over ``m`` modes.

- **State injection option:** when enabled, a new photon is injected into the measured mode whenever a measurement occurs.

- **Input parameters:** all circuits within the same feedforward layer must use the same number of input parameters.
  - The total number of input parameters of the :class:`FeedforwardBlock` is the sum of the input parameters across all layers.

----------------------------
API Reference
----------------------------

.. autoclass:: merlin.FeedforwardBlock
   :members:
   :undoc-members:
   :show-inheritance:

----------------------------
Example
----------------------------

.. code-block:: python

   from MerLin import QuantumLayer, FeedforwardBlock
   import torch

   # Initialize feedforward block
   ffb = FeedforwardBlock(n=2, m=6, depth=3)

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
   )

   # Random input tensor
   t = torch.rand(4)

   # Forward pass
   o = ffb(t)

   # Inspect output size
   print(ffb.get_output_size())   # 15

   # Inspect how many quantum layers are required for the first FF layer
   print(ffb.size_ff_layer(1))    # 2

----------------------------
Further Reading
----------------------------

- :ref:`circuit_specific_optimizations`
- :ref:`output_mappings`
- :ref:`basic_concepts`
