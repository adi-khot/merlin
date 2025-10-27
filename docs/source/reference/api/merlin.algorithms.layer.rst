merlin.algorithms.layer module
==============================

.. automodule:: merlin.algorithms.layer
   :members:
   :undoc-members:
   :show-inheritance:

Example: Quickstart QuantumLayer
--------------------------------

.. code-block:: python

    import torch.nn as nn
    from merlin import QuantumLayer

    simple_layer = QuantumLayer.simple(
        input_size=4,
        n_params=120,
    )

    model = nn.Sequential(
        simple_layer,
        nn.Linear(simple_layer.output_size, 3),
    )
    # Train and evaluate as a standard torch.nn.Module

.. note::
   :func:`QuantumLayer.simple` returns a thin ``SimpleSequential`` wrapper that behaves like a standard
   PyTorch module while exposing the inner quantum layer as ``.quantum_layer`` and any
   post-processing (:class:`~merlin.utils.grouping.ModGrouping` or :class:`~torch.nn.Identity`) as ``.post_processing``.
   The wrapper also forwards ``.circuit`` and ``.output_size`` so existing code that inspects these
   attributes continues to work.

.. image:: ../../_static/img/Circ_simple.png
   :alt: A Perceval Circuit built with QuantumLayer.simple
   :width: 600px
   :align: center

The simple quantum layer above implements a circuit of 10 modes and 5 photons with at least 90 trainable parameters. This circuit is made of:
- A first entangling layer (trainable)
- Angle encoding on the first N modes (for N input parameters with `input_size <= n_modes`)
- Add MZI blocks (two trainable parameters each) to match the requested number of trainable parameters

Additional trainable budget must therefore increase in multiples of two beyond the base interferometer (90 parameters).

Example: Declarative builder API
--------------------------------

.. code-block:: python

    import torch.nn as nn
    from merlin import LexGrouping, MeasurementStrategy, QuantumLayer
    from merlin.builder import CircuitBuilder
    builder = CircuitBuilder(n_modes=6)
    builder.add_entangling_layer(trainable=True, name="U1")
    builder.add_angle_encoding(modes=list(range(4)), name="input")
    builder.add_rotations(trainable=True, name="theta")
    builder.add_superpositions(depth=1)

    builder_layer = QuantumLayer(
        input_size=4,
        builder=builder,
        n_photons=3,  # is equivalent to input_state=[1,1,1,0,0,0]
        measurement_strategy=MeasurementStrategy.PROBABILITIES,
    )

    model = nn.Sequential(
        builder_layer,
        LexGrouping(builder_layer.output_size, 3),
    )
    # Train and evaluate as a standard torch.nn.Module

.. image:: ../../_static/img/Circ_builder.png
   :alt: A Perceval Circuit built with the CircuitBuilder
   :width: 600px
   :align: center

The circuit builder allows you to build your circuit layer by layer, with a high-level API. The example above implements a circuit of 6 modes and 3 photons.
This circuit is made of:
- A first entangling layer (trainable)
- Angle encoding on the first 4 modes (for 4 input parameters with the name "input")
- A trainable rotation layer to add more trainable parameters
- An entangling layer to add more expressivity

Other building blocks in the CircuitBuilder include:

- **add_rotations**: Add single or multiple phase shifters (rotations) to specific modes. Rotations can be fixed, trainable, or data-driven (input-encoded).
- **add_angle_encoding**: Encode classical data as quantum rotation angles, supporting higher-order feature combinations for expressive input encoding.
- **add_entangling_layer**: Insert a multi-mode entangling layer (implemented via a generic interferometer), optionally trainable, and tune its internal template with the ``model`` argument (``"mzi"`` or ``"bell"``) for different mixing behaviours.
- **add_superpositions**: Add one or more beam splitters (superposition layers) with configurable targets, depth, and trainability.

Example: Manual Perceval circuit (more control)
-----------------------------------------------

.. code-block:: python

    import torch.nn as nn
    import perceval as pcvl
    from merlin import LexGrouping, MeasurementStrategy, QuantumLayer
    modes = 6
    wl = pcvl.GenericInterferometer(
        modes,
        lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_li{i}")) //
        pcvl.BS() // pcvl.PS(pcvl.P(f"theta_lo{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    circuit = pcvl.Circuit(modes)
    circuit.add(0, wl)
    for mode in range(4):
        circuit.add(mode, pcvl.PS(pcvl.P(f"input{mode}")))
    wr = pcvl.GenericInterferometer(
        modes,
        lambda i: pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ri{i}")) //
        pcvl.BS() // pcvl.PS(pcvl.P(f"theta_ro{i}")),
        shape=pcvl.InterferometerShape.RECTANGLE,
    )
    circuit.add(0, wr)

    manual_layer = QuantumLayer(
        input_size=4,  # matches the number of phase shifters named "input{mode}"
        circuit=circuit,
        input_state=[1, 0, 1, 0, 1, 0],
        trainable_parameters=["theta"],
        input_parameters=["input"],
        measurement_strategy=MeasurementStrategy.PROBABILITIES,
    )

    model = nn.Sequential(
        manual_layer,
        LexGrouping(manual_layer.output_size, 3),
    )
    # Train and evaluate as a standard torch.nn.Module

.. image:: ../../_static/img/Circ_manual.png
   :alt: A Perceval Circuit built with the Perceval API
   :width: 600px
   :align: center

See the User guide and Notebooks for more advanced usage and training routines !
