:github_url: https://github.com/merlinquantum/merlin

========================
Internal Design Overview
========================

This page documents two of MerLin’s infrastructure components that often come up
in advanced workflows: the **partial** :class:`~merlin.measurement.detectors.DetectorTransform`
and the new experimental :class:`~merlin.algorithms.feed_forward.FeedForwardBlock`.

Partial DetectorTransform
=========================

MerLin’s detector transform normally converts a complete Fock probability
distribution into the classical detector outcomes dictated by the detector
model.  That path assumes *all* modes are detected simultaneously and it
operates on **real-valued probability tensors**.

`DetectorTransform` also supports a *partial measurement* mode, enabled by
passing ``partial_measurement=True``. In this configuration:

* You may pass ``None`` for the detectors attached to unmeasured modes.  Those
  modes remain quantum and their amplitudes are preserved.
* The forward pass now expects **complex-valued amplitude tensors** (instead of
  probabilities) and returns, for each measurement branch, the normalized
  amplitudes that correspond to the still-active modes.
* Internally the transform enumerates all measurement outcomes for the measured
  subset of modes, reweights the amplitudes by the corresponding detection
  probabilities, and yields:

  .. code-block:: python

      [
          {
              measurement_key: [
                  (probabilities, normalized_amplitudes),
                  ...
              ],
              ...
          },
          ...
      ]

  The outer list is indexed by the number of remaining photons. Each dictionary
  entry contains every measurement branch for that photon count. Each branch
  stores the accumulated probability weight and the normalized amplitudes for
  the unmeasured modes.

This partial interface is the backbone of feed-forward simulation where only a
subset of modes is observed at each stage and the remaining modes must be
propagated through additional circuits.

FeedForwardBlock
=================

``FeedForwardBlock`` is a new experimental block that consumes a full Perceval
experiment containing detectors and one or more
:class:`perceval.components.feed_forward_configurator.FFCircuitProvider`
instances. The block parses the experiment into a sequence of *stages*:

1. A unitary prefix (collapsed into a single :class:`~merlin.algorithms.layer.QuantumLayer`).
2. The detector set for the stage.
3. The matching feed-forward configurator that decides which circuit to insert
   based on the detector outcome.

The parser records these stages as ``FFStage`` instances. Each stage stores

* the collapsed unitary operating on the currently *active* optical modes,
* the tuple of global mode indices that remain active at the beginning of the stage,
* the subset of those modes that are measured, and
* the detector objects and ``FFCircuitProvider`` attached to the stage.

After a detector stage finishes, the measured modes are removed from the active
set so that every subsequent circuit is expressed solely in terms of the
remaining modes. This remapping guarantees that the :class:`QuantumLayer`
constructed for the stage matches the reduced dimensionality, while the
original mode identifiers are still available for bookkeeping.

For each ``FFStage`` the block builds a runtime bundle consisting of:

* A ``QuantumLayer`` configured in amplitude-encoding mode (for the pre-measurement unitary).
* A partial ``DetectorTransform`` tied to the stage’s measured modes.
* A dictionary of conditional ``QuantumLayer`` objects – one per feed-forward branch.

The current implementation expects noise-free experiments (a bare
``NoiseModel()`` or ``None``) and only the first stage is allowed to consume
classical inputs defined via ``input_parameters``. Once detectors fire, every
branch progresses in amplitude-encoding mode and additional classical tensors
are ignored.

During the forward pass, ``FeedForwardBlock`` iterates over the stages. Each
stage takes the incoming branch amplitudes, applies the unitary, runs the
partial detector transform, and spawns new branches for the next stage based on
the measured outcomes. Branch bookkeeping keeps track of:

* The amplitudes of the remaining (unmeasured) modes.
* The probability weight associated with that branch.
* The sequence of measurement results that led to the branch. These are exposed
  via dictionary keys in the *original* experiment mode ordering thanks to the
  stage-level remapping described above.

At the end of the execution the block can expose different classical views over
the surviving quantum modes. Much like :class:`~merlin.algorithms.layer.QuantumLayer`,
the ``measurement_strategy`` parameter switches between raw amplitudes, dense
probabilities, or per-mode expectations. ``measurement_strategy=AMPLITUDES``
returns a list of tuples ``(measurement_key, branch_probability,
remaining_photons, amplitudes)`` so callers can reason about the remaining
mixed state. ``PROBABILITIES`` collapses every branch into a tensor of shape
``(batch_size, len(output_keys))`` where the columns already align with the
fully specified Fock states recorded in
:pyattr:`~merlin.algorithms.feed_forward.FeedForwardBlock.output_keys`.
``MODE_EXPECTATIONS`` produces a ``(batch_size, num_modes)`` tensor describing
the photon expectations per mode. The result is already aggregated across all
measurement keys, so :pyattr:`~merlin.algorithms.feed_forward.FeedForwardBlock.output_keys`
is retained solely for metadata while
:pyattr:`~merlin.algorithms.feed_forward.FeedForwardBlock.output_state_sizes`
equals ``num_modes`` for every key so consumers can reason about downstream
reshaping without additional bookkeeping.

This design allows every stage to be simulated with amplitude access, while
still exposing convenient classical views. The mixed-state format (list of
tuples) is particularly useful for downstream probabilistic reasoning or for
feeding the remaining amplitudes into additional differentiable modules.
