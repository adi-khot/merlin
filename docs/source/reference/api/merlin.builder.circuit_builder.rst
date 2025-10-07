merlin.builder.circuit_builder module
=====================================

.. automodule:: merlin.builder.circuit_builder
   :members:
   :undoc-members:
   :show-inheritance:

--------------------------------------------------------------------------------
The different components of the builder with their arguments and example of code
--------------------------------------------------------------------------------

Below are the components available in the `CircuitBuilder` class:

1. **add_rotations**:
   - Adds one or multiple phase shifters across the specified modes.
   - Arguments:
     - `modes` (int | list[int] | ModuleGroup | None): Modes receiving the rotations. Defaults to all modes.
     - `axis` (str): Axis of rotation (default: "z").
     - `angle` (float): Fixed rotation angle for non-trainable cases.
     - `trainable` (bool): Promote the rotations to trainable parameters.
     - `name` (str): Optional stem used for generated parameter names.
     - `role` (:class:`~merlin.core.components.ParameterRole`): Explicitly set the parameter role.

.. code-block:: python
   builder = CircuitBuilder(n_modes=6)
   builder.add_rotations(modes=3, angle=np.pi / 4)

.. image:: ../../_static/img/builder_layer/rotation_comp.png
   :alt: A Rotation component built with CircuitBuilder
   :width: 200px
   :align: center

.. code-block:: python
   builder = CircuitBuilder(n_modes=6)
   builder.add_rotations(trainable=True, name="rotation")

.. image:: ../../_static/img/builder_layer/rotation_layer.png
   :alt: A Rotation layer built with CircuitBuilder
   :width: 200px
   :align: center

2. **add_angle_encoding**:
   - Adds angle-based input encoding to the circuit.
   - Arguments:
     - `modes` (list[int]): Modes to target for encoding.
     - `name` (str): Prefix for generated input parameters.
     - `scale` (float): Scaling factor for angle mapping.

.. code-block:: python
   builder = CircuitBuilder(n_modes=6)
   builder.add_angle_encoding(modes=list(range(6)), name="input", scale=np.pi)

This will show as a rotation layer as data is encoded in phase shifters.

3. **add_generic_interferometer**:
   - Adds a generic interferometer spanning a range of modes.
   - Arguments:
     - `modes` (list[int]): Modes to span.
     - `trainable` (bool): Whether the interferometer is trainable.
     - `name` (str): Optional prefix for parameter names.

.. code-block:: python
   builder = CircuitBuilder(n_modes=6)
   builder.add_generic_interferometer(trainable=True, name="U1")

.. image:: ../../_static/img/builder_layer/generic.png
   :alt: A Generic Interferometer built with CircuitBuilder
   :width: 200px
   :align: center

And to span it on different modes
.. code-block:: python
   builder = CircuitBuilder(n_modes=6)
   builder.add_generic_interferometer(trainable=True, name="U1", modes = [0,3])

.. image:: ../../_static/img/builder_layer/generic_0_3.png
   :alt: A Generic Interferometer built with CircuitBuilder
   :width: 200px
   :align: center

4. **add_superpositions**:
   - Adds one or more beam splitters with optional depth.
   - Arguments:
     - `targets` (tuple[int, int] | list[tuple[int, int]]): Explicit mode pairs receiving beam splitters. When omitted, nearest neighbours across ``modes`` (or all modes) are used.
     - `depth` (int): Number of successive passes to apply.
     - `theta` (float): Mixing angle for fixed beam splitters.
     - `phi` (float): Relative phase for fixed beam splitters.
     - `trainable` (bool): Convenience flag marking both parameters trainable.
     - `trainable_theta` (bool): Whether the mixing angle is trainable.
     - `trainable_phi` (bool): Whether the relative phase is trainable.
     - `modes` (list[int] or ModuleGroup): Mode span used when ``targets`` is omitted.
     - `name` (str): Optional prefix for generated parameter names.

.. code-block:: python
   builder = CircuitBuilder(n_modes=6)
   builder.add_superpositions(targets=(0, 1), trainable_theta=True, name="bs")

.. image:: ../../_static/img/builder_layer/supp_012.png
   :alt: A Superposition component (beam splitter)
   :width: 200px
   :align: center

.. code-block:: python
   builder = CircuitBuilder(n_modes=6)
   builder.add_superpositions(depth=2, name="mix")

.. image:: ../../_static/img/builder_layer/entangling_layer_depth2.png
   :alt: A Entangling layer of depth 2
   :width: 200px
   :align: center

5. **begin_section**:
   - Marks the beginning of a circuit section.
   - Arguments:
     - `name` (str): Name of the section.
     - `compute_adjoint` (bool): Whether to compute the adjoint of this section.
     - `reference` (str): Name of the section to copy structure from.
     - `share_trainable` (bool): Whether to share trainable parameters from the reference.
     - `share_input` (bool): Whether to share input parameters from the reference.

6. **end_section**:
   - Marks the end of the current circuit section.

.. code-block:: python
   builder = CircuitBuilder(n_modes=6)
   builder.begin_section(name="encoding", compute_adjoint=False)
   builder.add_angle_encoding(modes=list(range(X_train.shape[1])), name="input")
   builder.end_section()

.. image:: ../../_static/img/builder_layer/with_1_section.png
   :alt: A Section with an encoding layer
   :width: 200px
   :align: center

7. **add_adjoint_section**:
   - Adds the adjoint of an existing section.
   - Arguments:
     - `name` (str): Name of the new adjoint section.
     - `reference` (str): Existing section to mirror.
     - `share_trainable` (bool): Whether to reuse trainable parameters.
     - `share_input` (bool): Whether to reuse input parameters.

.. code-block:: python
   builder = CircuitBuilder(n_modes=6)
   builder.begin_section(name="trainable", compute_adjoint=False)
   builder.add_rotations(trainable=True, name="theta")
   builder.end_section()
   builder.add_adjoint_section(name="adjoint_trainable", reference="trainable", share_trainable=False, share_input=False)

.. image:: ../../_static/img/builder_layer/with_adjoint_sections.png
   :alt: A Section with an adjoint section
   :width: 200px
   :align: center

8. **build**:
    - Finalizes and returns the constructed circuit.
