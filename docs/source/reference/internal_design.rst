:github_url: https://github.com/merlinquantum/merlin

========================
Internal Design Overview
========================

This page documents two of MerLin’s infrastructure components that often come up
in advanced workflows: the **partial** :class:`~merlin.measurement.detectors.DetectorTransform`
and the new experimental :class:`~merlin.algorithms.feed_forward.FeedForwardBlock2`.

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

FeedForwardBlock2
=================

``FeedForwardBlock2`` is a new experimental block that consumes a full Perceval
experiment containing detectors and one or more
:class:`perceval.components.feed_forward_configurator.FFCircuitProvider`
instances. The block parses the experiment into a sequence of *stages*:

1. A unitary prefix (collapsed into a single :class:`~merlin.algorithms.layer.QuantumLayer`).
2. The detector set for the stage.
3. The matching feed-forward configurator that decides which circuit to insert
   based on the detector outcome.

The parser records these stages as ``FFStage`` instances. For each stage the
block builds a runtime bundle consisting of:

* A ``QuantumLayer`` configured in amplitude-encoding mode (for the pre-measurement unitary).
* A partial ``DetectorTransform`` tied to the stage’s measured modes.
* A dictionary of conditional ``QuantumLayer`` objects – one per feed-forward branch.

During the forward pass, ``FeedForwardBlock2`` iterates over the stages. Each
stage takes the incoming branch amplitudes, applies the unitary, runs the
partial detector transform, and spawns new branches for the next stage based on
the measured outcomes. Branch bookkeeping keeps track of:

* The amplitudes of the remaining (unmeasured) modes.
* The probability weight associated with that branch.
* The sequence of measurement results that led to the branch (exposed through
  the dictionary keys returned by ``forward``).

In the current implementation only the first stage executes; subsequent stages
are parsed and described via ``FeedForwardBlock2.describe()``, but the branch
propagation code is being implemented incrementally. This documentation reflects
the internal layout so that contributors can understand how partial detector
transforms and feed-forward runtimes fit together when extending the block.
