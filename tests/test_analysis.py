import numpy as np

from raimon.analysis import cluster_signflip, exact_signflip, holm


def test_holm_matches_definition():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])


def test_exact_signflip_floor_at_five_runs():
    assert exact_signflip(np.array([-3.0, -1.0, -2.0, -5.0, -1.0])) == 2 / 32
    assert exact_signflip(np.zeros(5)) == 1.0
    assert exact_signflip(np.array([-1.0, 1.0, -1.0, 1.0, 0.0])) == 1.0


def test_cluster_signflip_exact_and_monte_carlo():
    assert cluster_signflip(-np.ones(5), 25, np.random.default_rng(0)) == 2 / 32
    p = cluster_signflip(-np.ones(30), 150, np.random.default_rng(0))
    assert p < 1e-4
    assert cluster_signflip(np.array([1.0, -1.0] * 15), 150, np.random.default_rng(0)) == 1.0
