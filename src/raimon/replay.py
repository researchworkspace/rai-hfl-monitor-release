"""Offline replay of monitor placements on a logged trajectory.

Every placement turns round t's logged evidence into checks, OPA turns the checks into per-domain decisions,
and the decisions are scored against ground truth computed on test data that no monitor sees.

Scoring is per candidate. A *violating deployment* is a (domain, round) where candidate t was deployed to that
domain although it violates a requirement that applies there. A *false block* is a (domain, round) where a
non-violating candidate was held or left waiting for review.

Approver model (called reviewer in the code). Escalations go to an approver who knows the ground truth and
answers after `delay` rounds.
- fail-safe: the domain deploys nothing until the answer arrives; then candidate t is deployed if it is fine.
- fail-operational: the candidate is deployed with a warning while the review runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import stats
from .injections import fabricate
from .opa import OpaServer

REQS = ("R1", "R2", "R3", "R4", "R5", "R6", "R7")


@dataclass(frozen=True)
class Thresholds:
    sp: float = 0.10          # R1 upper
    eo: float = 0.10          # R2 upper
    acc_min: float = 0.70     # R3 lower
    eps_max: float = 8.0      # R4 upper (per edge, parallel composition)
    rob_drop: float = 0.03    # R5 upper
    attr_min: float = 0.80    # R6 lower
    trust_tol: float = 0.10   # R7 upper: |reported rate - audit rate|
    gt_min_cell: int = 20     # ground truth: a per-domain rate statistic is scored only with this support
    alpha: float = 0.05


@dataclass(frozen=True)
class Setting:
    placement: str                    # P1 | P2 | P2s | P2c | P3 | P4 | P4R
    ablation: str = ""                # "" | A1 | A2a | A2b | A4
    n_eval: int = 0                   # 0 = all eval rows of a client; otherwise a fixed per-client subsample
    delay: int = 0
    mode: str = "fail_safe"           # fail_safe | fail_operational

    @property
    def label(self) -> str:
        return "-".join(str(v) for v in (self.placement, self.ablation or "full", self.n_eval, self.delay, self.mode))


@dataclass
class Trajectory:
    path: Path
    z: dict
    meta: dict
    t0: int
    kind: str
    n_edges: int
    n_clients: int
    liars: set[int] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "Trajectory":
        npz = np.load(path / "trajectory.npz")
        z = {k: npz[k] for k in npz.files}
        meta = json.loads((path / "meta.json").read_text())
        inj = meta["config"]["injection"]
        return cls(path, z, meta, inj["t0"], inj["kind"], len(meta["edge_names"]), len(meta["client_names"]),
                   set(meta["liars"]))

    @property
    def rounds(self) -> int:
        return self.z["pred_test"].shape[0]

    def active(self, t: int) -> bool:
        return self.kind != "V0" and t >= self.t0

    def edge_of_client(self) -> np.ndarray:
        return self.z["client_edge"]


def counts_of(pred: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    c = np.zeros((2, 2, 2))
    np.add.at(c, (g.astype(np.int64), y.astype(np.int64), pred.astype(np.int64)), 1)
    return c


def spearman(u: np.ndarray, v: np.ndarray) -> float:
    ru, rv = np.argsort(np.argsort(u)), np.argsort(np.argsort(v))
    return float(np.corrcoef(ru, rv)[0, 1])


# ---------------------------------------------------------------- ground truth

def ground_truth(tr: Trajectory, th: Thresholds) -> np.ndarray:
    """viol[t, e, r] = candidate t violates requirement r for domain e (global requirements on every domain).

    A per-domain R1-R3 statistic counts only if every cell it uses has at least `th.gt_min_cell` ground-truth
    examples (SP and worst-group accuracy: each group; EO: each group-label cell). Below that support the
    domain's own value is noise (for example 4 malignant test images in a whole centre), so the requirement is
    not assessable there and only its global part applies to the domain.
    """
    z = tr.z
    T, E = tr.rounds, tr.n_edges
    viol = np.zeros((T, E, len(REQS)), bool)
    edge_of_row = tr.edge_of_client()[z["test_client"]]
    for t in range(T):
        keep = z["keep_test"] if tr.active(t) else np.ones_like(z["keep_test"])
        p, y, g = z["pred_test"][t], z["test_y"], z["test_g"]
        glob = counts_of(p[keep], y[keep], g[keep])
        g_sp, g_eo, g_acc = stats.sp_gap(glob).estimate, stats.eo_gap(glob).estimate, \
            stats.worst_group_accuracy(glob).estimate
        drop = z["rob_test_clean"][t].mean() - z["rob_test_pert"][t].mean()
        attr_bad = False
        if t > 0:
            a, b = np.abs(z["attr_test"][t].astype(float)).mean(0), np.abs(z["attr_test"][t - 1].astype(float)).mean(0)
            attr_bad = spearman(a, b) < th.attr_min
        for e in range(E):
            m = keep & (edge_of_row == e)
            c = counts_of(p[m], y[m], g[m])
            group_ok = c.sum(axis=(1, 2)).min() >= th.gt_min_cell
            cell_ok = c.sum(axis=2).min() >= th.gt_min_cell
            viol[t, e, 0] = g_sp > th.sp or (group_ok and stats.sp_gap(c).estimate > th.sp)
            viol[t, e, 1] = g_eo > th.eo or (cell_ok and stats.eo_gap(c).estimate > th.eo)
            viol[t, e, 2] = g_acc < th.acc_min or (group_ok and stats.worst_group_accuracy(c).estimate < th.acc_min)
            viol[t, e, 3] = z["eps_actual"][t, e] > th.eps_max
            viol[t, e, 4] = drop > th.rob_drop
            viol[t, e, 5] = attr_bad
            # R7 is a means (trustworthy evidence), not a property of the model; it has no ground-truth column.
    return viol


# ---------------------------------------------------------------- evidence

class Evidence:
    """Per-round statistics shared by all settings of one trajectory and one n_eval."""

    def __init__(self, tr: Trajectory, n_eval: int, seed: int = 0):
        self.tr = tr
        z = tr.z
        rng = np.random.default_rng(seed)
        ec = z["eval_client"]
        self.client_rows = []
        for k in range(tr.n_clients):
            rows = np.flatnonzero(ec == k)
            if n_eval and len(rows) > n_eval:
                rows = np.sort(rng.choice(rows, n_eval, replace=False))
            self.client_rows.append(rows)
        self.audit_rows = [np.flatnonzero((ec == k) & z["audit"]) for k in range(tr.n_clients)]
        self.boot_rng = np.random.default_rng(seed + 1)

    def reported_counts(self, t: int) -> np.ndarray:
        z, tr = self.tr.z, self.tr
        keep = z["keep_eval"] if tr.active(t) else np.ones_like(z["keep_eval"])
        p, y, g = z["pred_eval"][t], z["eval_y"], z["eval_g"]
        out = np.zeros((tr.n_clients, 2, 2, 2))
        for k, rows in enumerate(self.client_rows):
            r = rows[keep[rows]]
            c = counts_of(p[r], y[r], g[r])
            if tr.active(t) and tr.kind == "V4" and k in tr.liars:
                c = fabricate(c)
            out[k] = c
        return out

    def audit_counts(self, t: int) -> np.ndarray:
        z, tr = self.tr.z, self.tr
        keep = z["keep_eval"] if tr.active(t) else np.ones_like(z["keep_eval"])
        p, y, g = z["pred_eval"][t], z["eval_y"], z["eval_g"]
        out = np.zeros((tr.n_clients, 2, 2, 2))
        for k, rows in enumerate(self.audit_rows):
            r = rows[keep[rows]]
            out[k] = counts_of(p[r], y[r], g[r])
        return out

    def ref_counts(self, t: int) -> np.ndarray:
        z = self.tr.z
        return counts_of(z["pred_ref"][t], z["ref_y"], z["ref_g"])

    @lru_cache(maxsize=None)
    def robustness(self, t: int) -> stats.Interval:
        z = self.tr.z
        return stats.bootstrap_mean_diff(z["rob_ref_clean"][t].astype(float), z["rob_ref_pert"][t].astype(float),
                                         self.boot_rng)

    @lru_cache(maxsize=None)
    def attribution(self, t: int) -> stats.Interval:
        z = self.tr.z
        if t == 0:
            return stats.Interval(1.0, 1.0, 1.0)
        return stats.bootstrap_spearman(z["attr_ref"][t].astype(float), z["attr_ref"][t - 1].astype(float),
                                        self.boot_rng, resamples=300)


# ---------------------------------------------------------------- checks per placement

BIG = 1e9


def _finite(v: float, fallback: float) -> float:
    return float(v) if np.isfinite(v) else fallback


def _chk(cid: str, domain: str, kind: str, threshold: float, iv: stats.Interval, point: bool) -> dict:
    """One check for OPA. A non-finite estimate (an empty group) means no evidence: the interval is unbounded,
    and a point estimate falls back to the interval's midpoint, which for no evidence is uninformative."""
    lo, hi = _finite(iv.lower, -BIG), _finite(iv.upper, BIG)
    est = _finite(iv.estimate, (lo + hi) / 2 if abs(lo) < BIG and abs(hi) < BIG else 0.0)
    if point:
        lo = hi = est
    return {"id": cid, "domain": domain, "kind": kind, "threshold": _finite(threshold, -BIG),
            "estimate": est, "lower": lo, "upper": hi}


def _fair_checks(prefix: str, domain: str, c: np.ndarray, th: Thresholds, point: bool) -> list[dict]:
    if c.sum() == 0:
        return []
    return [
        _chk(f"R1/{prefix}", domain, "upper", th.sp, stats.sp_gap(c, th.alpha), point),
        _chk(f"R2/{prefix}", domain, "upper", th.eo, stats.eo_gap(c, th.alpha), point),
        _chk(f"R3/{prefix}", domain, "lower", th.acc_min, stats.worst_group_accuracy(c, th.alpha), point),
    ]


def _trust_check(k: int, domain: str, rep: np.ndarray, aud: np.ndarray, th: Thresholds, point: bool) -> dict | None:
    """Largest gap between reported and audit prediction rates over (group, label) cells."""
    worst = None
    for g in (0, 1):
        for y in (0, 1):
            n_r, n_a = rep[g, y].sum(), aud[g, y].sum()
            if n_r < 1 or n_a < 1:
                continue
            iv = stats.abs_interval(stats.newcombe_diff(rep[g, y, 1], n_r, aud[g, y, 1], n_a, th.alpha / 4))
            if worst is None or iv.lower > worst.lower:
                worst = iv
    return None if worst is None else _chk(f"R7/c{k}", domain, "alarm_upper", th.trust_tol, worst, point)


def heterogeneity(edge_counts: list[np.ndarray]) -> tuple[float, float]:
    """FedGFT data-heterogeneity coefficients (Wang et al., arXiv 2305.09931, Def. 4.5) for SP and EO.

    For a proper metric |a/b - c/d| with model-independent b and d, DH = max_k |d * b_k / (b * d_k) - 1|, where
    b, d are population-weighted averages of the edges' b_k, d_k. SP uses b = P(G=1), d = P(G=0). EO uses the
    same construction within Y=1 and within Y=0 and takes the larger value.
    """
    n = np.array([c.sum() for c in edge_counts], float)
    w = n / n.sum()

    def dh(bk: np.ndarray, dk: np.ndarray) -> float:
        b, d = (w * bk).sum(), (w * dk).sum()
        ok = (bk > 0) & (dk > 0)
        return float(np.max(np.abs(d * bk[ok] / (b * dk[ok]) - 1))) if ok.any() else float("inf")

    tot = np.maximum(n, 1)
    sp_b = np.array([c[1].sum() for c in edge_counts]) / tot
    sp_d = np.array([c[0].sum() for c in edge_counts]) / tot
    eo = max(dh(np.array([c[1, y].sum() for c in edge_counts]) / tot,
                np.array([c[0, y].sum() for c in edge_counts]) / tot) for y in (0, 1))
    return dh(sp_b, sp_d), eo


def _restricted_global_checks(edge_counts: list[np.ndarray], s: Setting, th: Thresholds, point: bool) -> list[dict]:
    """Restricted regime: per-group counts stay inside each edge. Each edge checks a local sufficient condition.

    By FedGFT Thm 4.6, local gap <= tau - beta at every edge implies global gap <= tau. A local failure says nothing
    about the global property, so it is inconclusive. Ablation A3 checks tau instead and treats the local verdict
    as the global one, which is unsound. Worst-group accuracy is not a proper metric and is not checked globally.
    """
    beta_sp, beta_eo = heterogeneity(edge_counts)
    out = []
    for e, c in enumerate(edge_counts):
        if c.sum() == 0:
            continue
        for rid, iv, tau, beta in (("R1", stats.sp_gap(c, th.alpha), th.sp, beta_sp),
                                   ("R2", stats.eo_gap(c, th.alpha), th.eo, beta_eo)):
            if s.ablation == "A3":
                out.append(_chk(f"{rid}/global@e{e}", "*", "upper", tau, iv, point))
            else:
                out.append(_chk(f"{rid}/global@e{e}", "*", "sufficient_upper", tau - beta, iv, point))
    return out


def build_checks(s: Setting, t: int, ev: Evidence, th: Thresholds) -> list[dict]:
    tr, z = ev.tr, ev.tr.z
    E = tr.n_edges
    edge_of = tr.edge_of_client()
    rep = ev.reported_counts(t)
    point = s.placement not in ("P4", "P4R") or s.ablation == "A1"
    checks: list[dict] = []
    if s.placement == "P1":
        checks += _fair_checks("global", "*", ev.ref_counts(t), th, True)
        checks.append(_chk("R4/global", "*", "upper", th.eps_max,
                           stats.Interval(*(float(z["eps_declared"][t].max()),) * 3), True))
    elif s.placement == "P2s":
        # P2s (Uniform-scoped): P2 with domain-scoped holds. Each edge applies the global thresholds to its own
        # domain and a local failure holds only that domain. There is no global fairness check.
        for e in range(E):
            ce = rep[edge_of == e].sum(0)
            checks += _fair_checks(f"e{e}", f"e{e}", ce, th, True)
            checks.append(_chk(f"R4/e{e}", f"e{e}", "upper", th.eps_max,
                               stats.Interval(*(float(z["eps_actual"][t, e]),) * 3), True))
    elif s.placement in ("P2", "P2c"):
        # P2: each edge applies the global thresholds to its own domain and a local failure holds every domain.
        # P2c additionally applies them at every client.
        if s.placement == "P2c":
            for k in range(tr.n_clients):
                checks += _fair_checks(f"c{k}", "*", rep[k], th, True)
        for e in range(E):
            ce = rep[edge_of == e].sum(0)
            checks += _fair_checks(f"e{e}", "*", ce, th, True)
            checks += [dict(c, id=c["id"] + "/dom", domain=f"e{e}") for c in _fair_checks(f"e{e}", "", ce, th, True)]
            checks.append(_chk(f"R4/e{e}", f"e{e}", "upper", th.eps_max,
                               stats.Interval(*(float(z["eps_actual"][t, e]),) * 3), True))
    elif s.placement == "P3":
        checks += _fair_checks("global", "*", rep.sum(0), th, True)
        checks.append(_chk("R4/global", "*", "upper", th.eps_max,
                           stats.Interval(*(float(z["eps_declared"][t].max()),) * 3), True))
    elif s.placement in ("P4", "P4R"):
        trusted = np.ones(tr.n_clients, bool)
        if s.ablation != "A4":
            aud = ev.audit_counts(t)
            for k in range(tr.n_clients):
                c = _trust_check(k, f"e{edge_of[k]}", rep[k], aud[k], th, point)
                if c is not None:
                    checks.append(c)
                    if c["lower"] > th.trust_tol:
                        trusted[k] = False
        edge_counts = [rep[trusted & (edge_of == e)].sum(0) for e in range(E)]
        if s.placement == "P4":
            checks += _fair_checks("global", "*", rep[trusted].sum(0), th, point)
        else:
            checks += _restricted_global_checks(edge_counts, s, th, point)
        for e in range(E):
            checks += _fair_checks(f"e{e}", f"e{e}", edge_counts[e], th, point)
            checks.append(_chk(f"R4/e{e}", f"e{e}", "upper", th.eps_max,
                               stats.Interval(*(float(z["eps_actual"][t, e]),) * 3), True))
    else:
        raise ValueError(s.placement)
    # Model-global checks at the cloud on its reference subset (all placements).
    checks.append(_chk("R5/global", "*", "upper", th.rob_drop, ev.robustness(t), point))
    checks.append(_chk("R6/global", "*", "lower", th.attr_min, ev.attribution(t), point))
    return checks


# ---------------------------------------------------------------- decisions and scoring

def decide(s: Setting, decisions: dict[str, str]) -> dict[str, str]:
    """Apply ablation A2 (no escalation): inconclusive becomes deploy (A2a) or hold (A2b)."""
    if s.ablation == "A2a":
        return {d: ("deploy" if v == "escalate" else v) for d, v in decisions.items()}
    if s.ablation == "A2b":
        return {d: ("hold" if v == "escalate" else v) for d, v in decisions.items()}
    return decisions


def score(s: Setting, tr: Trajectory, decisions: list[dict[str, str]], viol: np.ndarray) -> dict:
    T, E = tr.rounds, tr.n_edges
    bad = viol[:, :, :6].any(axis=2)                       # [T, E] candidate violates something there
    deployed = np.zeros((T, E), bool)
    blocked = np.zeros((T, E), bool)
    escalations = 0
    for e in range(E):
        frozen_until = -1
        for t in range(T):
            if t < frozen_until:
                blocked[t, e] = True
                continue
            d = decisions[t][f"e{e}"]
            if d == "deploy":
                deployed[t, e] = True
            elif d == "hold":
                blocked[t, e] = True
            else:
                escalations += 1
                if s.mode == "fail_operational":
                    deployed[t, e] = True
                elif s.delay == 0:
                    deployed[t, e] = not bad[t, e]
                    blocked[t, e] = bad[t, e]
                else:
                    blocked[t, e] = True
                    frozen_until = t + s.delay
    viol_dep = deployed & bad
    false_block = blocked & ~bad
    out = {
        "violating_deployments": int(viol_dep.sum()),
        "violating_candidates": int(bad.sum()),
        "false_blocks": int(false_block.sum()),
        "clean_candidates": int((~bad).sum()),
        "escalations": escalations,
        "domain_rounds": T * E,
    }
    for i, r in enumerate(REQS[:6]):
        out[f"viol_dep_{r}"] = int((deployed & viol[:, :, i]).sum())
        out[f"viol_cand_{r}"] = int(viol[:, :, i].sum())
    # Detection delay: first round at or after t0 in which some domain whose candidate violates is not deployed.
    delay = None
    if tr.kind != "V0":
        for t in range(tr.t0, T):
            if (bad[t] & ~deployed[t]).any():
                delay = t - tr.t0
                break
    out["detection_delay"] = delay
    return out


def paired_v0(tr: Trajectory) -> Path:
    """The V0 trajectory with the same dataset, learner and seed."""
    ds, learner, _, seed = tr.path.name.split("__")
    return tr.path.parent / f"{ds}__{learner}__V0__{seed}"


def ground_truth_pre_from_v0(tr: Trajectory, th: Thresholds) -> np.ndarray:
    """Ground truth with rounds t < t0 taken from the paired V0 trajectory.

    The R5/R6 test subsample of an injected run is drawn from the rows kept after t0 and used in every round, so
    for injections that remove rows (V1, V2) its own arrays describe the post-injection population. Before t0 an
    injected trajectory's models equal the paired V0 run's (checked here), so V0's ground truth is the ground
    truth on the population before the injection.
    """
    viol = ground_truth(tr, th)
    if tr.kind == "V0":
        return viol
    v0 = Trajectory.load(paired_v0(tr))
    for t in range(tr.t0):
        if not np.array_equal(tr.z["pred_test"][t], v0.z["pred_test"][t]):
            raise ValueError(f"{tr.path.name} differs from its V0 pair before t0 (round {t})")
    viol[:tr.t0] = ground_truth(v0, th)[:tr.t0]
    return viol


def replay(tr: Trajectory, settings: list[Setting], th: Thresholds, opa: OpaServer) -> list[dict]:
    viol = ground_truth_pre_from_v0(tr, th)
    domains = [f"e{e}" for e in range(tr.n_edges)]
    rows = []
    evidence: dict[int, Evidence] = {}
    for s in settings:
        ev = evidence.setdefault(s.n_eval, Evidence(tr, s.n_eval))
        decisions = []
        for t in range(tr.rounds):
            res = opa.evaluate(domains, build_checks(s, t, ev, th))
            decisions.append(decide(s, res["decision"]))
        row = {"trajectory": str(tr.path), "setting": s.label, **s.__dict__, **score(s, tr, decisions, viol)}
        cfg = tr.meta["config"]
        row.update(learner=cfg["learner"]["name"], injection=tr.kind, seed=cfg["seed"])
        rows.append(row)
    return rows
