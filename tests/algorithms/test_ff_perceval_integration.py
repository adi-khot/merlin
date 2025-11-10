# MIT License
#
# Integration test ensuring FeedForwardBlock matches Perceval on a mixed-detector experiment.

import math

import perceval as pcvl
from perceval import BasicState, Circuit
from perceval.algorithm import Sampler

from merlin.algorithms.feed_forward import FeedForwardBlock


def _build_experiment():
    m = 4
    exp = pcvl.Experiment()
    root = Circuit(m)
    root.add(0, pcvl.BS())
    exp.add(0, root)
    exp.add(0, pcvl.Detector.pnr())

    # First FF level with balanced beam splitters
    v0 = Circuit(m - 1) // pcvl.BS()
    v1 = Circuit(m - 1) // pcvl.BS()
    v2 = Circuit(m - 1) // pcvl.BS()
    provider1 = pcvl.FFCircuitProvider(1, 0, v0)
    provider1.add_configuration([1], v1)
    provider1.add_configuration([2], v2)
    exp.add(0, provider1)

    # Second FF level with threshold detector
    exp.add(3, pcvl.Detector.threshold())
    provider2 = pcvl.FFCircuitProvider(1, -1, Circuit(m - 2))
    provider2.add_configuration([1], Circuit(m - 2) // pcvl.BS())
    exp.add(3, provider2)

    # Add PNR detectors on remaining inactive modes to align final measurement
    # basis with MerLin's classical outputs
    for mode in (1, 2):
        exp.add(mode, pcvl.Detector.pnr())

    return exp


def test_merlin_matches_perceval_for_mixed_detector_experiment():
    exp = _build_experiment()
    input_state = [1, 1, 0, 0]
    exp.with_input(BasicState(input_state))

    # Perceval reference
    processor = pcvl.Processor("SLOS", exp)
    processor.min_detected_photons_filter(0)
    sampler = Sampler(processor)
    results = sampler.probs()["results"]
    perceval_map = {
        tuple(int(v) for v in state): float(p) for state, p in results.items()
    }

    # MerLin may include zero-probability states not present (pruned) in Perceval results.
    # Extend Perceval map with explicit zeros for any MerLin-only keys so key sets match.

    # MerLin computation
    block = FeedForwardBlock(exp)
    outputs = block()  # no classical inputs for this experiment
    probs = outputs.squeeze(0)
    merlin_map = {block.output_keys[i]: float(probs[i]) for i in range(probs.shape[0])}
    for k in merlin_map.keys() - perceval_map.keys():
        perceval_map[k] = 0.0

    assert set(merlin_map.keys()) == set(perceval_map.keys())
    for k in merlin_map:
        a = merlin_map[k]
        b = perceval_map[k]
        assert math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5), (
            f"Mismatch for key {k}: MerLin={a}, Perceval={b}"
        )
