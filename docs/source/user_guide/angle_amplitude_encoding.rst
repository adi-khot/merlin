.. _angle_and_amplitude_encoding:

Angle Encoding and Amplitude Encoding
=====================================

This guide shows how to use **angle encoding** and **amplitude encoding** with
Merlin's :class:`~merlin.algorithms.QuantumLayer`. You'll find when to use each,
how to build circuits with :class:`~merlin.builder.CircuitBuilder` or native Perceval, and complete, runnable snippets.

Prerequisites
-------------

- Python, PyTorch, and Merlin installed.
- Basic familiarity with Merlin's :class:`~merlin.algorithms.QuantumLayer`.
- Optional: Perceval for custom circuits and experiments.

Conceptual Overview
-------------------

- **Angle encoding** maps a *real feature vector*
  into *circuit parameters* (e.g., phase shifter angles). The circuit unitary
  depends on your data.
- **Amplitude encoding** feeds a *statevector* directly to the layer. Instead of
  turning features into angles, you supply the input quantum state's amplitudes.

Angle Encoding
--------------

When to use
^^^^^^^^^^^

Use angle encoding for classical-quantum pipelines: feature maps, kernels, or
hybrid neural networks where your inputs are real-valued tensors.

With CircuitBuilder
^^^^^^^^^^^^^^^^^^^

:class:`~merlin.builder.CircuitBuilder` provides a declarative way to add an
angle-encoding stage into your photonic circuit.

1) Build a circuit with angle encoding:

.. code-block:: python

    import numpy as np
    from merlin.builder import CircuitBuilder

    # 1) Declare a circuit with 6 modes
    builder = CircuitBuilder(n_modes=6)

    # 2) Put trainable rotations (phase shifters) on every mode
    builder.add_rotations(modes=[0, 1, 2, 3, 4, 5], trainable=True)

    # 3) Add an angle-encoding layer; the 'name' will prefix the input parameters
    builder.add_angle_encoding(
        modes=[0, 1, 2, 3, 4, 5],
        name="input",
        scale=np.pi   # optional global scaling of features -> angles
    )

    # 4) Entangle some modes (e.g., MZI block between modes 0 and 5)
    builder.add_entangling_layer(modes=[0, 5], trainable=True, model="mzi")

    # 5) Add superposition/BS layers (increase expressivity)
    builder.add_superpositions(modes=[0, 1, 2, 3, 4, 5], trainable=True, depth=2)

2) Wrap it as a QuantumLayer and run a forward pass:

.. code-block:: python

    import torch
    from merlin.algorithms import QuantumLayer

    layer = QuantumLayer(
        input_size=6,           # number of *classical* features per sample
        builder=builder,        # the declarative circuit
        input_state=[1, 0, 1, 0, 1, 0]   # 5 photons in 10-mode equivalent => here 6 modes, so 3 photons example
    )

    x = torch.rand((4, 6))      # batch of 4 samples
    probs = layer(x)            # default MeasurementStrategy.PROBABILITIES

Parameter names and prefixes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:meth:`~merlin.builder.CircuitBuilder.add_angle_encoding` call registers
parameters prefixed by ``name`` (e.g., ``"input"``). Internally,
:class:`~merlin.algorithms.QuantumLayer` will consume your real-valued input
tensor and map each feature to the corresponding prefixed angle(s).

Tips and constraints
^^^^^^^^^^^^^^^^^^^^

- **Modes vs. features**: By construction you typically shouldn't encode more
  independent features than available modes in the encoding step.
- **Scaling and combinations**: You can use ``scale=...`` to rescale inputs
  before turning them into angles. If you create multiple encoding stages with
  different names (prefixes), the layer can split the input tensor across them.
- **Kernels**: For quantum kernels, consider :class:`~merlin.kernels.FeatureMap`
  and :class:`~merlin.kernels.FidelityKernel` if you need a reusable feature
  map object.

Angle encoding using QuantumLayer.simple
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want a quick start without designing the circuit:

.. code-block:: python

    import torch
    from merlin.algorithms import QuantumLayer

    layer = QuantumLayer.simple(
        input_size=6,   # number of classical features
        n_params=100,   # parameter budget for a 10-mode circuit
        output_size=10  # output dimensionality
    )

    x = torch.rand((2, 6))
    y = layer(x)  # probability vector of size `output_size`

Angle encoding with Perceval circuits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For full control, create a Perceval circuit, then expose input-parameter
prefix(es) that the layer will map features to:

.. code-block:: python

    import perceval as pcvl
    from merlin.algorithms import QuantumLayer

    # Build a 6-mode Perceval circuit
    circuit = pcvl.Circuit(6)

    # Example: add user-named input phase shifters (prefix 'input')
    for mode in range(6):
        circuit.add(mode, pcvl.PS(pcvl.P(f"input{mode}")))

    # (Add interferometers, MZIs, etc. as you like)
    # ...

    layer = QuantumLayer(
        input_size=6,
        circuit=circuit,
        input_state=[1, 0, 1, 0, 1, 0],
        input_parameters=["input"],    # map features -> parameters named 'input*'
        trainable_parameters=["theta"] # example trainable prefix used elsewhere in your circuit
    )

    import torch
    x = torch.rand((1, 6))
    probs = layer(x)

Common measurement choices (angle encoding)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :data:`~merlin.measurement.MeasurementStrategy.PROBABILITIES` (default):
  returns a probability vector aligned with :pyattr:`QuantumLayer.output_keys`.
- :data:`~merlin.measurement.MeasurementStrategy.MODE_EXPECTATIONS`:
  returns per-mode expected photon counts.
- :data:`~merlin.measurement.MeasurementStrategy.AMPLITUDES`:
  returns complex amplitudes (simulation-only; bypasses detectors and noise).

Amplitude Encoding
------------------

When to use
^^^^^^^^^^^

Choose amplitude encoding when you already have a prepared **quantum state** to
inject into the circuit (e.g., produced by an upstream simulator or another
photonic block). Here your input to ``forward`` is the **statevector**
amplitudes, not classical features.

Key differences
^^^^^^^^^^^^^^^

- Set ``amplitude_encoding=True`` on :class:`~merlin.algorithms.QuantumLayer`.
- Provide ``n_photons`` (to define the computational subspace).
- Do **not** pass ``input_size`` or ``input_parameters``- they are irrelevant
  because you are not mapping classical features to angles.
- The **input tensor shape** must match the layer's basis size:
  ``len(layer.output_keys)`` (or ``[batch, len(output_keys)]``).

Minimal example (amplitudes out)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from merlin.algorithms import QuantumLayer
    from merlin.measurement import MeasurementStrategy

    # Suppose you already have a circuit or builder; here we assume `circuit`
    # exists and is unitary with no post-selection.
    # For amplitude encoding, the optical layout defines the evolution,
    # but no classical input parameters are used.
    layer = QuantumLayer(
        circuit=circuit,              # or builder=..., or experiment=...
        n_photons=2,                  # required: defines the subspace
        amplitude_encoding=True,      # switch to amplitude input
        measurement_strategy=MeasurementStrategy.AMPLITUDES
    )

    # Build (or sample) an input statevector compatible with the layer basis
    num_states = len(layer.output_keys)     # basis size for 2 photons over the modes
    psi_in = torch.randn(num_states, dtype=torch.complex64)
    psi_in = psi_in / psi_in.norm()         # normalize

    # Forward: returns complex amplitudes after the circuit
    psi_out = layer(psi_in)

Batching amplitudes
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch

    B = 8
    num_states = len(layer.output_keys)
    psi_batch = torch.randn(B, num_states, dtype=torch.complex64)
    psi_batch = psi_batch / psi_batch.norm(dim=1, keepdim=True)

    amps_out = layer(psi_batch)  # shape: [B, num_states]

Probabilities or expectations from amplitudes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want classical outputs (e.g., probabilities) from amplitude-encoded
inputs, use the corresponding measurement strategy:

.. code-block:: python

    from merlin.measurement import MeasurementStrategy

    layer_probs = QuantumLayer(
        circuit=circuit,
        n_photons=2,
        amplitude_encoding=True,
        measurement_strategy=MeasurementStrategy.PROBABILITIES
    )

    probs = layer_probs(psi_in)  # shape: [num_states]

    # Or mode-level expectations
    layer_modes = QuantumLayer(
        circuit=circuit,
        n_photons=2,
        amplitude_encoding=True,
        measurement_strategy=MeasurementStrategy.MODE_EXPECTATIONS
    )

    mode_exp = layer_modes(psi_in)  # shape: [n_modes]

Detectors, noise, and shots
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- With :data:`~merlin.measurement.MeasurementStrategy.AMPLITUDES`, the layer
  **bypasses** detectors and noise; ``shots`` must be unset or zero.
- With probability-like strategies, detector/noise models (if present in a
  :class:`perceval.Experiment`) are applied *after* converting amplitudes to
  probabilities; shot sampling is supported when compatible.

Interoperability checklist
^^^^^^^^^^^^^^^^^^^^^^^^^^

- The amplitude vector must be **compatible with the basis** used by the layer.
  Check :pyattr:`QuantumLayer.output_keys` to see state ordering.
- Normalize your amplitude inputs to avoid exploding norms.

Choosing Between Angle and Amplitude
------------------------------------

+------------------------+----------------------------+----------------------------------+
| Aspect                 | Angle Encoding             | Amplitude Encoding               |
+========================+============================+==================================+
| Input to ``forward``   | Real features ``x``        | Complex statevector amplitudes   |
+------------------------+----------------------------+----------------------------------+
| Circuit dependence     | Features set parameters    | State defines input quantum      |
|                        | (phases/angles)            | state to propagate               |
+------------------------+----------------------------+----------------------------------+
| Setup knobs            | ``add_angle_encoding(...)``| ``amplitude_encoding=True``,     |
|                        | scales, multiple prefixes  | ``n_photons``                    |
+------------------------+----------------------------+----------------------------------+
| Typical use            | Feature maps, kernels,     | Passing prepared quantum states  |
|                        | hybrid NN layers           | through a photonic circuit       |
+------------------------+----------------------------+----------------------------------+
| Measurement options    | Probabilities, modes,      | Amplitudes (bypass detectors) or |
|                        | amplitudes (sim-only)      | probabilities/modes              |
+------------------------+----------------------------+----------------------------------+

Troubleshooting
---------------

- **Shape errors (angle encoding)**: Ensure ``input_size`` equals the number of
  features you feed into the layer and matches the encoding specification
  (number of encoded modes and prefixes).
- **Too many features**: If you attempt to encode more features than modes in
  your encoding stage, reduce features or expand the circuit's encoding modes.
- **Shape errors (amplitude encoding)**: The amplitude vector length must match
  the layer basis size: ``len(layer.output_keys)``. For batching, use
  ``[batch, len(output_keys)]``.
- **Incompatible measurement strategy**: When
  :data:`~merlin.measurement.MeasurementStrategy.AMPLITUDES` is selected, do not
  set nonzero ``shots`` or enable detectors/noise.
- **Unnormalized amplitudes**: Always normalize amplitude inputs to avoid
  unstable gradients and to ensure proper probability mass.

Complete Examples
-----------------

Angle encoding with builder + probabilities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    import torch
    from merlin.builder import CircuitBuilder
    from merlin.algorithms import QuantumLayer
    from merlin.measurement import MeasurementStrategy

    builder = CircuitBuilder(n_modes=6)
    builder.add_angle_encoding(modes=list(range(6)), name="input", scale=np.pi)
    builder.add_entangling_layer(trainable=True)
    builder.add_superpositions(modes=list(range(6)), trainable=True, depth=1)

    layer = QuantumLayer(
        input_size=6,
        builder=builder,
        input_state=[1, 0, 1, 0, 1, 0],
        measurement_strategy=MeasurementStrategy.PROBABILITIES
    )

    x = torch.rand((3, 6))
    probs = layer(x)  # shape: [3, layer.output_size]

Amplitude encoding with amplitudes out
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    import perceval as pcvl
    from merlin.algorithms import QuantumLayer
    from merlin.measurement import MeasurementStrategy

    # Simple unitary circuit placeholder; customize as needed
    circuit = pcvl.Circuit(4)
    # ... populate circuit ...

    layer = QuantumLayer(
        circuit=circuit,
        n_photons=2,
        amplitude_encoding=True,
        measurement_strategy=MeasurementStrategy.AMPLITUDES
    )

    num_states = len(layer.output_keys)
    psi_in = torch.randn(num_states, dtype=torch.complex64)
    psi_in = psi_in / psi_in.norm()

    amps_out = layer(psi_in)  # complex amplitudes

Amplitude encoding with probabilities out
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import torch
    from merlin.algorithms import QuantumLayer
    from merlin.measurement import MeasurementStrategy

    layer = QuantumLayer(
        circuit=circuit,     # same circuit as above
        n_photons=2,
        amplitude_encoding=True,
        measurement_strategy=MeasurementStrategy.PROBABILITIES
    )

    psi_in = torch.randn(len(layer.output_keys), dtype=torch.complex64)
    psi_in = psi_in / psi_in.norm()

    probs = layer(psi_in)   # classical probabilities


