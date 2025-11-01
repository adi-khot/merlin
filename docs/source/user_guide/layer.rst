.. _quantum_layer:

QuantumLayer Essentials
=======================

The :class:`~merlin.algorithms.layer.QuantumLayer` is MerLin’s core building
block for converting classical data into photonic probability distributions. It
combines a Perceval circuit (or experiment), optional classical parameters, and
detector logic into a single differentiable module.

---------------
Overview
---------------

- **Multiple construction paths** – Build layers from
  the convenience :meth:`~merlin.algorithms.layer.QuantumLayer.simple` factory,
  a :class:`~merlin.builder.circuit_builder.CircuitBuilder`, a custom
  :class:`perceval.Circuit` or a fully specified :class:`perceval.Experiment`.
- **Detector awareness** – Layers automatically derive detector transforms from
  the experiment, enabling threshold, PNR, or hybrid detection schemes.
- **Measurement strategies** – Select between probabilities, per-mode expectations,
  or raw amplitudes through :class:`~merlin.measurement.strategies.MeasurementStrategy`.
  The layer validates incompatible combinations (e.g. detectors with amplitude read-out).
- **Autograd ready** – QuantumLayer exposes a PyTorch ``Module`` interface,
  supports batching and differentiable forward passes, and plays nicely with
  optimisers or higher-level architectures.

-----------------------
Initialisation recipes
-----------------------

``QuantumLayer.simple()``
~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~merlin.algorithms.layer.QuantumLayer.simple` helper generates a
trainable, 10-mode interferometer with angle encoding and a configurable number
of parameters. It is convenient for quick experiments, baselines or for machine 
learning experts without any prior knowledge in quantum machine learning.

.. code-block:: python

   import merlin as ML

   layer = ML.QuantumLayer.simple(
       input_size=4,
       n_params=64,
       measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
       no_bunching=True,
   )

   x = torch.rand(16, 4)
   probs = layer(x)

CircuitBuilder
~~~~~~~~~~~~~~

Use MerLin’s :class:`CircuitBuilder` utilities to describe a circuit at a higher
level. The builder records trainable and input parameter prefixes for you. This is
an ideal tool for quantum machine learning experts who have do not have any experience 
with Perceval.

.. code-block:: python

   import torch
   import merlin as ML

   builder = ML.CircuitBuilder(n_modes=4)
   builder.add_superpositions(depth=1)
   builder.add_angle_encoding(modes=[0, 1], name="x")
   builder.add_rotations(trainable=True, name="theta")

   layer = ML.QuantumLayer(
       input_size=2,
       builder=builder,
       measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
       no_bunching=True,
   )

   x = torch.rand(4, 2)
   probs = layer(x)

Custom circuit
~~~~~~~~~~~~~~

When you already have a :class:`perceval.Circuit`, provide the classical input
layout and the trainable parameter prefixes explicitly. This initialization requires
a good understanding of Perceval.

.. code-block:: python

   import perceval as pcvl
   import torch
   import merlin as ML

   circuit = pcvl.Circuit(3)
   circuit.add((0, 1), pcvl.BS())
   circuit.add(0, pcvl.PS(pcvl.P("phi")))

   layer = ML.QuantumLayer(
       input_size=1,
       circuit=circuit,
       input_parameters=["phi"],
       trainable_parameters=["theta"],
       input_state=[1, 0, 0],
       measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
   )

   x = torch.linspace(0.0, 1.0, steps=8).unsqueeze(1)
   probs = layer(x)

Experiment-driven
~~~~~~~~~~~~~~~~~

For detector-heavy workflows, configure a :class:`perceval.Experiment` and pass
it directly. The layer inherits both the circuit and the detector selection. This scheme
is the one that gives the user the most options when utilizing a QuantumLayer.

.. code-block:: python

   import perceval as pcvl
   import torch
   import torch
   import merlin as ML

   circuit = pcvl.Circuit(2)
   circuit.add((0, 1), pcvl.BS())

   experiment = pcvl.Experiment(circuit)
   experiment.detectors[0] = pcvl.Detector.threshold()
   experiment.detectors[1] = pcvl.Detector.pnr()

   layer = ML.QuantumLayer(
       input_size=0,
       experiment=experiment,
       input_state=[1, 1],
       measurement_strategy=ML.MeasurementStrategy.PROBABILITIES,
   )

   probs = layer()
   detector_keys = layer.output_keys

-----------
Detector integration
-----------

- If any detector is set on the experiment, ``no_bunching`` must be ``False``.
  The layer enforces this by raising a ``RuntimeError`` when both are requested.
- Without an experiment, the layer defaults to ideal PNR detection on every
  mode, mirroring Perceval’s default behaviour.
- ``MeasurementStrategy.AMPLITUDES`` requires access to raw complex amplitudes
  and is therefore incompatible with custom detectors. Attempting this combination
  raises a ``RuntimeError``.
- Call :meth:`~merlin.algorithms.layer.QuantumLayer.output_keys` to inspect
  the classical outcomes produced by the detector transform.

-----------
Notes
-----------

- ``input_state`` must match the number of circuit modes. When unspecified,
  ``n_photons`` controls how photons are packed into the first modes.
- Sampling-based evaluations are available through ``shots`` and
  ``sampling_method``; the default returns exact SLOS probabilities.
- The layer registers trainable parameters (if any) with PyTorch so they appear
  in ``layer.parameters()``.

-----------
API Reference
-----------

.. autoclass:: merlin.algorithms.layer.QuantumLayer
   :members:
   :undoc-members:
   :show-inheritance:
