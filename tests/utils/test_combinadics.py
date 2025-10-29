import math

import pytest

from merlin.utils.combinadics import Combinadics


def _enumerate_states(combo: Combinadics, total: int):
    states = [combo.index_to_fock(i) for i in range(total)]
    assert len(states) == total
    # ensure descending lexicographic order
    for prev, curr in zip(states, states[1:], strict=False):
        assert prev >= curr
    # check bidirectional consistency
    for idx, state in enumerate(states):
        assert combo.fock_to_index(state) == idx
    return states


def test_dual_rail_enumeration_desc_lex_and_size():
    n, m = 4, 8
    combo = Combinadics("dual_rail", n=n, m=m)
    expected_total = combo.compute_space_size()
    assert expected_total == 2**n
    states = _enumerate_states(combo, expected_total)
    assert len(states) == expected_total
    # each pair must contain exactly one photon
    for state in states:
        for i in range(0, m, 2):
            a, b = state[i], state[i + 1]
            assert a in (0, 1) and b in (0, 1) and a + b == 1


def test_unbunched_enumeration_desc_lex_and_size():
    n, m = 4, 8
    combo = Combinadics("unbunched", n=n, m=m)
    expected_total = combo.compute_space_size()
    assert expected_total == math.comb(m, n)
    states = _enumerate_states(combo, expected_total)
    assert len(states) == expected_total
    # unbunched: collision free
    for state in states:
        assert sum(state) == n
        for occ in state:
            assert occ in (0, 1)


def test_fock_enumeration_desc_lex_and_size():
    n, m = 4, 8
    combo = Combinadics("fock", n=n, m=m)
    expected_total = combo.compute_space_size()
    assert expected_total == math.comb(n + m - 1, m - 1)
    states = _enumerate_states(combo, expected_total)
    assert len(states) == expected_total
    # fock states: non-negative integers summing to n
    for state in states:
        assert sum(state) == n
        for occ in state:
            assert occ >= 0


def test_dual_rail_requires_twice_as_many_modes():
    with pytest.raises(ValueError):
        Combinadics("dual_rail", n=4, m=6)
