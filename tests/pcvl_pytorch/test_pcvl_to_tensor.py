import math

import pytest
import torch
from perceval.utils import BasicState, StateVector

from merlin import ComputationSpace
from merlin.pcvl_pytorch.utils import pcvl_to_tensor


def test_pcvl_to_tensor_simple_fock():
    sv = 2 * BasicState([1, 0]) + 3 * BasicState([0, 1])

    tensor = pcvl_to_tensor(
        sv, computation_space=ComputationSpace.FOCK, dtype=torch.complex64
    )

    # For n=1, m=2 the Fock/combinadics ordering has two basis states; |1,0> should be first
    assert tensor.numel() == 2
    assert pytest.approx(2 / math.sqrt(13)) == tensor[0].item()
    assert pytest.approx(3 / math.sqrt(13)) == tensor[1].item()


def test_pcvl_to_tensor_ghz():
    sv = BasicState([0, 1, 0, 1]) + BasicState([1, 0, 1, 0])

    tensor = pcvl_to_tensor(
        sv, computation_space=ComputationSpace.DUAL_RAIL, dtype=torch.complex64
    )

    assert tensor.numel() == 4
    assert pytest.approx(1 / math.sqrt(2)) == tensor[0].item()
    assert pytest.approx(0) == tensor[1].item()
    assert pytest.approx(0) == tensor[2].item()
    assert pytest.approx(1 / math.sqrt(2)) == tensor[3].item()


def test_pcvl_to_tensor_unbunched_bunching_error():
    sv = StateVector()
    sv += StateVector(BasicState([2, 0])) * 1.0

    with pytest.raises(ValueError):
        pcvl_to_tensor(sv, computation_space=ComputationSpace.UNBUNCHED)


def test_pcvl_to_tensor_dualrail_bunching_error():
    # m=4, create a state with total n=2 but one mode has 2 photons -> invalid for dual_rail
    sv = StateVector()
    sv += StateVector(BasicState([2, 0, 0, 0])) * 1.0

    with pytest.raises(ValueError):
        pcvl_to_tensor(sv, computation_space=ComputationSpace.DUAL_RAIL)


def test_pcvl_to_tensor_inconsistent_photon_number():
    sv = StateVector()
    sv += StateVector(BasicState([0, 1])) + StateVector(BasicState([1, 1]))

    with pytest.raises(ValueError):
        pcvl_to_tensor(sv)
