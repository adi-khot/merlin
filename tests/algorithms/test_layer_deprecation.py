import warnings

import perceval as pcvl
import pytest

from merlin.algorithms.layer import QuantumLayer
from merlin.core.computation_space import ComputationSpace


def test_no_bunching_deprecation_in_init():
    """Passing no_bunching explicitly to QuantumLayer.__init__ must issue a DeprecationWarning."""
    circuit = pcvl.Circuit(2)
    # Provide an explicit input_state so the layer can initialize from the custom circuit
    with pytest.warns(DeprecationWarning):
        layer = QuantumLayer(circuit=circuit, input_state=[1, 0], no_bunching=True)
    assert layer.computation_space is ComputationSpace.UNBUNCHED


def test_simple_no_bunching_converts_to_unbunched_and_no_warning():
    """QuantumLayer.simple should accept no_bunching and convert it to computation_space without warning."""
    # Capture warnings and ensure no DeprecationWarning is emitted by simple()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model = QuantumLayer.simple(input_size=1, n_params=10, no_bunching=True)

    # Ensure no DeprecationWarning in the captured warnings
    assert not any(
        isinstance(w.message, DeprecationWarning) or w.category is DeprecationWarning
        for w in rec
    )

    # The returned model wraps the actual QuantumLayer under attribute `quantum_layer`
    qlayer = model.quantum_layer
    assert qlayer.computation_space == ComputationSpace.UNBUNCHED
