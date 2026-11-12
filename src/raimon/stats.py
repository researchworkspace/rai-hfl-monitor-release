"""Interval estimates for monitored statistics.

Every check produces an estimate and a two-sided (1 - alpha) interval. A policy turns the interval into a
three-valued verdict. Counts arrays have shape [G, Y, Yhat] with G = 2 sensitive groups, Y and Yhat binary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float


def wilson(successes: float, n: float, alpha: float = 0.05) -> Interval:
    """Wilson score interval for a binomial proportion. Empty samples give the uninformative [0, 1]."""
    if n <= 0:
        return Interval(float("nan"), 0.0, 1.0)
    z = norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, centre - half), min(1.0, centre + half))


def newcombe_diff(s1: float, n1: float, s2: float, n2: float, alpha: float = 0.05) -> Interval:
    """Newcombe hybrid score interval for p1 - p2 (method 10 in Newcombe, Stat. Med. 1998)."""
    a = wilson(s1, n1, alpha)
    b = wilson(s2, n2, alpha)
    if n1 <= 0 or n2 <= 0:
        return Interval(float("nan"), -1.0, 1.0)
    d = a.estimate - b.estimate
    lo = d - np.sqrt((a.estimate - a.lower) ** 2 + (b.upper - b.estimate) ** 2)
    hi = d + np.sqrt((a.upper - a.estimate) ** 2 + (b.estimate - b.lower) ** 2)
    return Interval(d, max(-1.0, lo), min(1.0, hi))


def abs_interval(iv: Interval) -> Interval:
    """Interval for |d| implied by an interval for d."""
    lo, hi = iv.lower, iv.upper
    upper = max(abs(lo), abs(hi))
    lower = 0.0 if lo <= 0.0 <= hi else min(abs(lo), abs(hi))
    return Interval(abs(iv.estimate), lower, upper)


def sp_gap(counts: np.ndarray, alpha: float = 0.05) -> Interval:
    """Statistical-parity gap |P(Yhat=1|G=1) - P(Yhat=1|G=0)|."""
    pos = counts[:, :, 1].sum(axis=1)
    n = counts.sum(axis=(1, 2))
    return abs_interval(newcombe_diff(pos[1], n[1], pos[0], n[0], alpha))


def eo_gap(counts: np.ndarray, alpha: float = 0.05) -> Interval:
    """Equalised-odds gap max(|TPR_1 - TPR_0|, |FPR_1 - FPR_0|), Bonferroni over the two rates."""
    parts = []
    for y in (1, 0):
        n = counts[:, y, :].sum(axis=1)
        pos = counts[:, y, 1]
        parts.append(abs_interval(newcombe_diff(pos[1], n[1], pos[0], n[0], alpha / 2)))
    est = max(p.estimate for p in parts)
    return Interval(est, max(p.lower for p in parts), max(p.upper for p in parts))


def worst_group_accuracy(counts: np.ndarray, alpha: float = 0.05) -> Interval:
    """min over groups of accuracy, Bonferroni over groups."""
    g = counts.shape[0]
    ivs = []
    for k in range(g):
        n = counts[k].sum()
        correct = counts[k, 0, 0] + counts[k, 1, 1]
        ivs.append(wilson(correct, n, alpha / g))
    return Interval(min(i.estimate for i in ivs), min(i.lower for i in ivs), min(i.upper for i in ivs))


def point(iv: Interval) -> Interval:
    """Collapse an interval to its point estimate (used by two-valued placements and ablation A1)."""
    return Interval(iv.estimate, iv.estimate, iv.estimate)


def bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, alpha: float = 0.05,
                        resamples: int = 2000) -> Interval:
    """Paired bootstrap interval for mean(a) - mean(b) over the same examples."""
    n = len(a)
    idx = rng.integers(0, n, size=(resamples, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return Interval(float(a.mean() - b.mean()), float(lo), float(hi))


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, alpha: float = 0.05,
                       resamples: int = 1000) -> Interval:
    """Interval for the Spearman correlation between mean |attribution| vectors.

    x, y: per-example attributions [N, F] of two model versions on the same examples. Examples are resampled,
    the per-feature mean absolute attribution is recomputed, and features are rank-correlated.
    """
    def spearman(u: np.ndarray, v: np.ndarray) -> float:
        ru = np.argsort(np.argsort(u))
        rv = np.argsort(np.argsort(v))
        return float(np.corrcoef(ru, rv)[0, 1])

    n = x.shape[0]
    est = spearman(np.abs(x).mean(0), np.abs(y).mean(0))
    vals = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        vals[i] = spearman(np.abs(x[idx]).mean(0), np.abs(y[idx]).mean(0))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return Interval(est, float(lo), float(hi))
