import numpy as np
import pytest

from raimon.replay import heterogeneity


def test_heterogeneity_zero_for_identical_edges():
    c = np.full((2, 2, 2), 100.0)
    assert heterogeneity([c, c.copy()]) == (0.0, 0.0)


def test_heterogeneity_matches_hand_computation():
    # Edge 1: P(G=1)=0.5; edge 2: P(G=1)=0.75 with twice the population. b=2/3, d=1/3, DH=0.5.
    c1 = np.full((2, 2, 2), 100.0)
    c2 = c1.copy()
    c2[1] *= 3
    sp, eo = heterogeneity([c1, c2])
    assert sp == pytest.approx(0.5)
    assert eo == pytest.approx(0.5)
