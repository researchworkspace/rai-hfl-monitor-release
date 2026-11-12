import numpy as np
import pytest

from raimon import stats
from raimon.opa import OpaServer


def counts(n0, pos0, n1, pos1):
    """Counts [G, Y, Yhat] with all examples labelled Y=1, for SP tests."""
    c = np.zeros((2, 2, 2))
    c[0, 1, 1], c[0, 1, 0] = pos0, n0 - pos0
    c[1, 1, 1], c[1, 1, 0] = pos1, n1 - pos1
    return c


def test_wilson_covers_estimate_and_shrinks():
    small = stats.wilson(5, 10)
    large = stats.wilson(500, 1000)
    assert small.lower < 0.5 < small.upper
    assert (large.upper - large.lower) < (small.upper - small.lower)


def test_newcombe_matches_reference_value():
    # Newcombe (1998), Table II example: 56/70 vs 48/80 gives 0.2 with interval (0.0524, 0.3339).
    iv = stats.newcombe_diff(56, 70, 48, 80)
    assert iv.estimate == pytest.approx(0.2)
    assert iv.lower == pytest.approx(0.0524, abs=1e-3)
    assert iv.upper == pytest.approx(0.3339, abs=1e-3)


def test_abs_interval_straddling_zero_has_zero_lower():
    iv = stats.abs_interval(stats.Interval(0.01, -0.05, 0.08))
    assert iv.lower == 0.0 and iv.upper == pytest.approx(0.08)


def test_sp_gap_large_sample_is_tight():
    iv = stats.sp_gap(counts(10000, 5000, 10000, 3000))
    assert iv.estimate == pytest.approx(0.2)
    assert 0.18 < iv.lower < iv.upper < 0.22


def test_worst_group_accuracy_takes_min_group():
    c = np.zeros((2, 2, 2))
    c[0, 0, 0], c[0, 1, 1] = 450, 450  # group 0 all correct
    c[1, 0, 0], c[1, 0, 1] = 300, 300  # group 1 half correct
    iv = stats.worst_group_accuracy(c)
    assert iv.estimate == pytest.approx(0.5)


def test_policy_agrees_with_interval_semantics():
    with OpaServer() as opa:
        iv = stats.sp_gap(counts(60, 30, 60, 22))  # small sample, estimate about 0.13
        chk = {"id": "R1", "domain": "*", "kind": "upper", "threshold": 0.1,
               "estimate": iv.estimate, "lower": iv.lower, "upper": iv.upper}
        out = opa.evaluate(["e0"], [chk])
        assert out["verdict"]["R1"] == "inconclusive"
        assert out["decision"]["e0"] == "escalate"
        p = stats.point(iv)
        chk.update(estimate=p.estimate, lower=p.lower, upper=p.upper)
        out = opa.evaluate(["e0"], [chk])
        assert out["verdict"]["R1"] == "fail"
        assert out["decision"]["e0"] == "hold"
