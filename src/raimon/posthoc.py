"""Post-hoc analyses, from the replay tables, the per-round decisions and the trajectories.

python -I -m raimon.posthoc --runs acs=outputs/replay/acs.csv isic=outputs/replay/isic.csv \
    --decisions acs=outputs/decisions/acs isic=outputs/decisions/isic \
    --trajectories acs=outputs/trajectories/acs isic=outputs/trajectories/isic \
    --decidability acs=outputs/decidability/acs isic=outputs/decidability/isic --out outputs/posthoc

In the order of the paper:
- decision rules: placements and Tiered variants on exposure and false blocks (rules.csv);
- localisation of targeted injections and report-integrity alarms (localisation.csv, localisation_split.csv,
  integrity.csv), and which checks hold domains under point estimates (point_estimate_holds.csv);
- an imperfect approver who answers each escalation correctly with probability q (approver.csv);
- the evaluation data a domain needs, from the design-time decidability analysis (sample_size.csv);
- interval coverage in simulation and against the test split (simulated_coverage.csv, coverage.csv.gz);
- exposure of the model in service (retained.csv);
- domains excluded from per-domain scoring by the support rule (support.csv).
The tests of the ablations are in `analysis.py`.
"""

from __future__ import annotations

import argparse
import json
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats
from .analysis import load, select
from .replay import Evidence, Thresholds, Trajectory, counts_of, paired_v0, spearman

QS = (1.0, 0.95, 0.9, 0.8, 0.7, 0.5)
DRAWS = 20
EXPECTED_REQ = {"V1": {"R1", "R2"}, "V5": {"R4"}, "V6": {"R3"}}


# ---------------------------------------------------------------- decision rules

RULES = {"P1": ("P1", {}, False), "P2": ("P2", {}, True), "P2s": ("P2s", {}, True), "P3": ("P3", {}, True),
         "P4-point": ("P4", {"ablation": "A1"}, True), "P4-hold": ("P4", {"ablation": "A2b"}, True),
         "P4-deploy": ("P4", {"ablation": "A2a"}, True), "P4-reviewer": ("P4", {}, True)}


def rules(r: pd.DataFrame) -> pd.DataFrame:
    """Means over learners and runs. P1 uses the cloud's reference set and does not depend on n_eval."""
    rows = []
    for n_eval in (0, 50):
        for ds, c in r.groupby("dataset"):
            for lab, (p, kw, by_n) in RULES.items():
                s = select(c, p, **({"n_eval": n_eval} if by_n else {}), **kw)
                inj = s[s.inj != "V0"]
                rows.append({"n_eval": n_eval or "all", "dataset": ds, "setting": lab, "runs": len(s),
                             "exposure_injected": inj.violating_deployments.mean(),
                             "violating_candidates": inj.violating_candidates.mean(),
                             "fb_injected": inj.fb_rate.mean(), "fb_V0": s[s.inj == "V0"].fb_rate.mean(),
                             "esc100": s.esc100.mean()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- localisation and integrity alarms

def scope_edge(cid: str, client_edge: list[int]) -> int | None:
    part = cid.split("/")[1]
    if part.startswith("e"):
        return int(part[1:].split("@")[0])
    if part.startswith("c"):
        return int(client_edge[int(part[1:])])
    return None


def localisation(dec_dirs: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, r7 = [], []
    for ds, d in dec_dirs.items():
        for f in sorted(Path(d).glob("*.json")):
            name = f.name[:-5]
            meta = json.loads(f.read_text())
            kind, t0, tgt, ce = meta["kind"], meta["t0"], meta["target_edge"], meta["client_edge"]
            z = np.load(f"{Path(d) / name}.npz")
            bad = z["bad"]
            T, E = bad.shape
            if kind in EXPECTED_REQ:
                for p in ("P2", "P4"):
                    label = f"{p}-full-0-0-fail_safe"
                    dec = z[f"dec_{label}"]
                    by_t: dict[int, list] = {}
                    for t, cid, dom, v in meta["nonpass"][label]:
                        by_t.setdefault(t, []).append((cid, v))
                    caught = loc = req_ok = 0
                    off_n = off_alarm = off_fail = 0
                    for t in range(t0, T):
                        scoped = [(scope_edge(c, ce), c.split("/")[0]) for c, _ in by_t.get(t, [])]
                        failed = [scope_edge(c, ce) for c, v in by_t.get(t, []) if v == "fail"]
                        if bad[t, tgt] and dec[t, tgt] != 0:
                            caught += 1
                            at = [rq for e, rq in scoped if e == tgt]
                            loc += bool(at)
                            req_ok += bool(set(at) & EXPECTED_REQ[kind])
                        for e in range(E):
                            if e != tgt and not bad[t, e]:
                                off_n += 1
                                off_alarm += any(se == e for se, _ in scoped)
                                off_fail += e in failed
                    rows.append({"dataset": ds, "trajectory": name, "injection": kind, "placement": p,
                                 "target_violating_rounds": int(bad[t0:, tgt].sum()), "caught": caught,
                                 "localised": loc, "requirement_correct": req_ok,
                                 "offtarget_clean": off_n, "offtarget_alarms": off_alarm, "offtarget_fails": off_fail})
            if kind in ("V4", "V0"):
                liars = set(meta["liars"])
                alarms = [(t, int(cid.split("/c")[1])) for t, cid, dom, v in meta["nonpass"]["P4-full-0-0-fail_safe"]
                          if cid.startswith("R7/") and v == "fail" and (t >= t0 or kind == "V0")]
                on_liars = sum(k in liars for _, k in alarms)
                r7.append({"dataset": ds, "trajectory": name, "injection": kind, "liars": len(liars),
                           "alarms": len(alarms), "alarms_on_liars": on_liars,
                           "liars_alarmed": len({k for _, k in alarms if k in liars}),
                           "clients": len(ce), "rounds": T - (t0 if kind == "V4" else 0)})
    return pd.DataFrame(rows), pd.DataFrame(r7)


def localisation_split(dec_dirs: dict) -> pd.DataFrame:
    """For stopped violating target-domain rounds (as in `localisation`): was a check scoped to the target domain
    failing, or only inconclusive?"""
    rows = []
    for ds, d in dec_dirs.items():
        for f in sorted(Path(d).glob("*.json")):
            meta = json.loads(f.read_text())
            kind, t0, tgt, ce = meta["kind"], meta["t0"], meta["target_edge"], meta["client_edge"]
            if kind not in EXPECTED_REQ:
                continue
            z = np.load(f"{Path(d) / f.name[:-5]}.npz")
            bad = z["bad"]
            for p in ("P2", "P4"):
                label = f"{p}-full-0-0-fail_safe"
                dec = z[f"dec_{label}"]
                by_t: dict = {}
                for t, cid, dom, v in meta["nonpass"][label]:
                    by_t.setdefault(t, []).append((scope_edge(cid, ce), v))
                caught = with_fail = inc_only = 0
                for t in range(t0, bad.shape[0]):
                    if bad[t, tgt] and dec[t, tgt] != 0:
                        caught += 1
                        vs = [v for e, v in by_t.get(t, []) if e == tgt]
                        with_fail += "fail" in vs
                        inc_only += bool(vs) and "fail" not in vs
                rows.append({"dataset": ds, "injection": kind, "placement": p, "caught": caught,
                             "target_failing": with_fail, "target_inconclusive_only": inc_only})
    return pd.DataFrame(rows).groupby(["dataset", "injection", "placement"]).sum().reset_index()


def point_estimate_holds(traj_dirs: dict, th_by_ds: dict) -> pd.DataFrame:
    """Tiered with point estimates (ablation A1, all evaluation rows) on every control run: for each domain-round,
    which requirements have a failing check that applies to the domain. With zero-width intervals no check is
    undecided, so a domain holds exactly when one applicable check fails (the rules of policies/verdict.rego)."""
    from .replay import Setting, build_checks
    rows = []
    for ds, d in traj_dirs.items():
        th = th_by_ds[ds]
        for p in sorted(Path(d).glob("*__V0__*")):
            tr = Trajectory.load(p)
            ev = Evidence(tr, 0)
            s = Setting("P4", ablation="A1")
            for t in range(tr.rounds):
                fails = [c for c in build_checks(s, t, ev, th)
                         if (c["kind"] in ("upper", "alarm_upper") and c["lower"] > c["threshold"])
                         or (c["kind"] == "lower" and c["upper"] < c["threshold"])]
                for e in range(tr.n_edges):
                    reqs = sorted({c["id"].split("/")[0] for c in fails if c["domain"] in ("*", f"e{e}")})
                    rows.append({"dataset": ds, "trajectory": p.name, "round": t, "domain": e, "held": bool(reqs),
                                 "only_integrity": reqs == ["R7"], "integrity": "R7" in reqs})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- imperfect approver

def imperfect_approver(r: pd.DataFrame, dec_dirs: dict[str, str]) -> tuple[pd.DataFrame, list]:
    rng = np.random.default_rng(0)
    rows, mism = [], []
    ref = r[(r.placement == "P4") & (r.delay == 0) & (r["mode"] == "fail_safe")]
    for ds, d in dec_dirs.items():
        for f in sorted(Path(d).glob("*.npz")):
            name = f.name[:-4]
            z = np.load(f)
            bad = z["bad"]
            kind = name.split("__")[2].split("-")[0]
            for n_eval in (0, 200, 50):
                dec = z[f"dec_P4-full-{n_eval}-0-fail_safe"]
                esc = dec == 2
                auto = dec == 0
                rr = ref[(ref.dataset == ds) & (ref.trajectory.str.endswith("/" + name)) & (ref.n_eval == n_eval)]
                for lab, deployed in (("hold", auto), ("deploy", auto | esc)):
                    ab = rr[rr.ablation == ("A2b" if lab == "hold" else "A2a")]
                    if len(ab):
                        ok = (int((deployed & bad).sum()) == int(ab.violating_deployments.iloc[0])
                              and int((~deployed & ~bad).sum()) == int(ab.false_blocks.iloc[0]))
                        if not ok:
                            mism.append((name, n_eval, lab))
                for q in QS:
                    exp, fb = [], []
                    for _ in range(DRAWS if q < 1 else 1):
                        right = rng.random(bad.shape) < q
                        says_deploy = np.where(bad, ~right, right)
                        deployed = auto | (esc & says_deploy)
                        exp.append(int((deployed & bad).sum()))
                        fb.append(int((~deployed & ~bad).sum()))
                    if q == 1.0:
                        full = rr[rr.ablation == ""]
                        if (exp[0] != int(full.violating_deployments.iloc[0])
                                or fb[0] != int(full.false_blocks.iloc[0])):
                            mism.append((name, n_eval, "q=1"))
                    rows.append({"dataset": ds, "trajectory": name, "injection": kind, "n_eval": n_eval, "q": q,
                                 "exposure": float(np.mean(exp)), "fb_rate": float(np.mean(fb)) / max(1, int((~bad).sum())),
                                 "esc100": 100 * esc.mean()})
    t = pd.DataFrame(rows)
    t["arm"] = np.where(t.injection == "V0", "V0", "injected")
    agg = t.groupby(["dataset", "n_eval", "q", "arm"]).agg(exposure=("exposure", "mean"), fb_rate=("fb_rate", "mean"),
                                                         esc100=("esc100", "mean")).reset_index()
    return agg, mism


# ---------------------------------------------------------------- evaluation data needed, from the decidability analysis

FACTORS = (1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def sample_size(decid_dir: str, plan: str, target: float = 0.1, draws: int = 2000) -> pd.DataFrame:
    """Invert the design-time analysis: the smallest factor k by which a domain's evaluation cells must grow, with
    unchanged proportions and pilot rates, for its predicted undecided probability to fall to `target` or below.
    The global checks use all evaluation records and stay fixed."""
    from .decidability import domain_cells, pilot_rates, predict
    meta = json.loads((Path(decid_dir) / "summary.json").read_text())
    th = Thresholds(**json.loads(Path(plan).read_text())["thresholds"])
    pilot = [Trajectory.load(Path(d)) for d in meta["pilot_runs"]]
    rng = np.random.default_rng(0)
    ev = Evidence(pilot[0], 0)
    edge_of = pilot[0].edge_of_client()
    _, v_glob = predict(domain_cells(ev, ev.client_rows), pilot_rates(pilot, None, meta["burn_in"]), th, draws, rng)
    rows = []
    for e in range(pilot[0].n_edges):
        n = domain_cells(ev, [ev.client_rows[k] for k in np.flatnonzero(edge_of == e)])
        rate = pilot_rates(pilot, e, meta["burn_in"])
        row = {"domain": e, "eval_records": int(n.sum()), "min_cell": int(n.min()),
               "positives_by_group": [int(n[0, 1]), int(n[1, 1])]}
        cexp = np.stack([n * (1 - rate), n * rate], axis=-1)
        row.update(pilot_sp=stats.sp_gap(cexp).estimate, pilot_eo=stats.eo_gap(cexp).estimate,
                   pilot_wga=stats.worst_group_accuracy(cexp).estimate)
        for k in FACTORS:
            res, v = predict(np.round(n * k), rate, th, draws, rng)
            vv = np.concatenate([v, v_glob], axis=1)
            und = float(((vv == "inconclusive").any(1) & ~(vv == "fail").any(1)).mean())
            row[f"k{k}"] = und
            row[f"hold{k}"] = float((vv == "fail").any(1).mean())
            if und <= target and "factor" not in row:
                row["factor"], row["records_needed"] = k, int(round(n.sum() * k))
                row["p_hold_at_factor"] = row[f"hold{k}"]
                row["eo_inconclusive_at_factor"] = res["R2"]["p_inconclusive"]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- interval coverage

def simulated_coverage(decid_dirs: dict, plans: dict, draws: int = 2000) -> pd.DataFrame:
    """Coverage of the per-domain and global intervals against known population statistics. The population of a
    scope is its pilot cell proportions P(y | g) and pilot prediction rates P(yhat = 1 | g, y), taken as true; each
    draw keeps the scope's evaluation cell sizes and draws the positive predictions per cell from the binomial
    distribution, as the decidability analysis does."""
    from .decidability import domain_cells, pilot_rates
    fns = {"SP gap": stats.sp_gap, "EO gap": stats.eo_gap, "Worst-group accuracy": stats.worst_group_accuracy}
    rows = []
    rng = np.random.default_rng(1)
    for ds, dd in decid_dirs.items():
        meta = json.loads((Path(dd) / "summary.json").read_text())
        th = Thresholds(**json.loads(Path(plans[ds]).read_text())["thresholds"])
        pilot = [Trajectory.load(Path(x)) for x in meta["pilot_runs"]]
        ev = Evidence(pilot[0], 0)
        edge_of = pilot[0].edge_of_client()
        scopes = {f"e{e}": (e, [ev.client_rows[k] for k in np.flatnonzero(edge_of == e)]) for e in range(pilot[0].n_edges)}
        scopes["global"] = (None, ev.client_rows)
        for scope, (edge, crows) in scopes.items():
            n = domain_cells(ev, crows)
            rate = pilot_rates(pilot, edge, meta["burn_in"])
            truth_counts = np.stack([n * (1 - rate), n * rate], axis=-1)   # expected table: the population values
            for req, fn in fns.items():
                truth = fn(truth_counts).estimate
                cov = []
                for _ in range(draws):
                    pos = rng.binomial(n.astype(int), rate)
                    iv = fn(np.stack([n - pos, pos], axis=-1), th.alpha)
                    lo = iv.lower if np.isfinite(iv.lower) else -np.inf
                    hi = iv.upper if np.isfinite(iv.upper) else np.inf
                    cov.append(lo <= truth <= hi)
                rows.append({"dataset": ds, "scope": scope, "requirement": req, "min_cell": int(n.min()),
                             "truth": truth, "coverage": float(np.mean(cov)), "draws": draws})
    return pd.DataFrame(rows)


Z = {"SP gap": 1.959964, "EO gap": 2.241403, "Worst-group accuracy": 2.241403}   # EO, worst group: Bonferroni over 2


def coverage(traj_dirs: dict, th_by_ds: dict) -> pd.DataFrame:
    """Coverage of Tiered's per-domain 95% intervals (all evaluation records, reported counts) for the test-split
    statistic of the same domain and round, on control runs; cells below the ground-truth support rule are skipped."""
    rows = []
    fns = {"SP gap": stats.sp_gap, "EO gap": stats.eo_gap, "Worst-group accuracy": stats.worst_group_accuracy}
    for ds, d in traj_dirs.items():
        th = th_by_ds[ds]
        for p in sorted(Path(d).glob("*__V0__*")):
            if not (p / "completion.json").exists():
                continue
            tr = Trajectory.load(p)
            z = tr.z
            ev = Evidence(tr, 0)
            edge_of = tr.edge_of_client()
            edge_of_row = edge_of[z["test_client"]]
            for t in range(tr.rounds):
                rep = ev.reported_counts(t)
                pt = z["pred_test"][t]
                for e in range(tr.n_edges):
                    ce = rep[edge_of == e].sum(0)
                    m = edge_of_row == e
                    ct = counts_of(pt[m], z["test_y"][m], z["test_g"][m])
                    group_ok = ct.sum(axis=(1, 2)).min() >= th.gt_min_cell
                    cell_ok = ct.sum(axis=2).min() >= th.gt_min_cell
                    for req, fn in fns.items():
                        if not (cell_ok if req == "EO gap" else group_ok):
                            continue
                        iv, it = fn(ce, th.alpha), fn(ct, th.alpha)
                        truth = it.estimate
                        if not all(np.isfinite(v) for v in (iv.lower, iv.upper, it.lower, it.upper, truth)):
                            continue
                        h_e, h_t = (iv.upper - iv.lower) / 2, (it.upper - it.lower) / 2
                        # The test value is a second sample of the domain's population. If the interval is
                        # calibrated, test - eval ~ N(0, se_e^2 + se_t^2) and P(test value inside the interval)
                        # = 2 Phi(z h_e / sqrt(h_e^2 + h_t^2)) - 1, with z the interval's critical value.
                        zc = Z[req]
                        expected = erf(zc * h_e / np.sqrt(h_e ** 2 + h_t ** 2) / sqrt(2)) if h_e > 0 else 0.0
                        rows.append({"dataset": ds, "trajectory": p.name, "round": t, "domain": e, "requirement": req,
                                     "covered": bool(iv.lower <= truth <= iv.upper), "expected": float(expected),
                                     "width": iv.upper - iv.lower, "n_eval": int(ce.sum()), "n_test": int(ct.sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- model in service

STATIC = ("V0", "V2", "V3", "V4", "V5")


def model_truth(tr: Trajectory, v0: Trajectory, th: Thresholds) -> np.ndarray:
    """gt[m, s, e, r]: model m violates requirement r for domain e on population s (0 before t0, 1 from t0 on).

    Same rules as replay.ground_truth: population s = 0 uses the paired V0 run's arrays (identical models before t0,
    full population), s = 1 uses the injected run's arrays and kept rows. R4 (privacy loss) and R6 (change from the
    previous candidate) are properties of model m.
    """
    T, E = tr.rounds, tr.n_edges
    gt = np.zeros((T, 2, E, 6), bool)
    for s, src in ((0, v0), (1, tr)):
        z = src.z
        keep = z["keep_test"] if (s == 1 and tr.kind != "V0") else np.ones_like(z["keep_test"])
        edge_of_row = src.edge_of_client()[z["test_client"]]
        y, g = z["test_y"], z["test_g"]
        for m in range(T):
            if s == 0 and m >= tr.t0 and tr.kind != "V0":
                continue                                 # a model from t0 on never serves before t0
            p = z["pred_test"][m]
            glob = counts_of(p[keep], y[keep], g[keep])
            g_sp, g_eo, g_acc = stats.sp_gap(glob).estimate, stats.eo_gap(glob).estimate, \
                stats.worst_group_accuracy(glob).estimate
            drop = z["rob_test_clean"][m].mean() - z["rob_test_pert"][m].mean()
            attr_bad = False
            if m > 0:
                a = np.abs(z["attr_test"][m].astype(float)).mean(0)
                b = np.abs(z["attr_test"][m - 1].astype(float)).mean(0)
                attr_bad = spearman(a, b) < th.attr_min
            for e in range(E):
                mm = keep & (edge_of_row == e)
                c = counts_of(p[mm], y[mm], g[mm])
                group_ok = c.sum(axis=(1, 2)).min() >= th.gt_min_cell
                cell_ok = c.sum(axis=2).min() >= th.gt_min_cell
                gt[m, s, e, 0] = g_sp > th.sp or (group_ok and stats.sp_gap(c).estimate > th.sp)
                gt[m, s, e, 1] = g_eo > th.eo or (cell_ok and stats.eo_gap(c).estimate > th.eo)
                gt[m, s, e, 2] = g_acc < th.acc_min or (group_ok and stats.worst_group_accuracy(c).estimate < th.acc_min)
                gt[m, s, e, 3] = z["eps_actual"][m, e] > th.eps_max
                gt[m, s, e, 4] = drop > th.rob_drop
                gt[m, s, e, 5] = attr_bad
    return gt


def deployed_of(dec: np.ndarray, bad: np.ndarray, rule: str) -> np.ndarray:
    if rule == "oracle":                                  # escalation answered at once by a correct approver
        return (dec == 0) | ((dec == 2) & ~bad)
    if rule == "hold":
        return dec == 0
    raise ValueError(rule)


RETAINED = {"Central": ("P1-full-0-0-fail_safe", "hold"), "Uniform": ("P2-full-0-0-fail_safe", "hold"),
            "Pooled": ("P3-full-0-0-fail_safe", "hold"), "Tiered, fail-safe default": ("P4-full-0-0-fail_safe", "hold"),
            "Tiered + oracle approver": ("P4-full-0-0-fail_safe", "oracle")}


def retained(dec_dirs: dict, traj_dirs: dict, th_by_ds: dict) -> tuple[pd.DataFrame, list]:
    rows, mism = [], []
    for ds, d in dec_dirs.items():
        th = th_by_ds[ds]
        for f in sorted(Path(d).glob("*.npz")):
            name = f.name[:-4]
            inj = name.split("__")[2].replace("byz_frac", "f")
            kind = inj.split("-")[0]
            if kind not in STATIC:
                continue
            z = np.load(f)
            bad = z["bad"]
            tr = Trajectory.load(Path(traj_dirs[ds]) / name)
            v0 = tr if tr.kind == "V0" else Trajectory.load(paired_v0(tr))
            gt = model_truth(tr, v0, th)
            T, E = bad.shape
            state = np.array([int(tr.kind != "V0" and t >= tr.t0) for t in range(T)])
            # sanity: the candidate scored at its own round must reproduce the logged ground truth
            own = gt[np.arange(T), state].any(axis=2)
            if not np.array_equal(own, bad):
                mism.append((ds, name, int((own != bad).sum())))
            for lab, (key, rule) in RETAINED.items():
                dep = deployed_of(z[f"dec_{key}"], bad, rule)
                in_service = np.full(E, -1)
                viol = age = served = 0
                for t in range(T):
                    in_service = np.where(dep[t], t, in_service)
                    for e in range(E):
                        m = in_service[e]
                        if m < 0:
                            continue
                        served += 1
                        age += t - m
                        viol += bool(gt[m, state[t], e].any())
                rows.append({"dataset": ds, "trajectory": name, "injection": inj, "setting": lab,
                             "retained_violating": viol, "served_domain_rounds": served,
                             "never_deployed_domain_rounds": T * E - served, "mean_model_age": age / max(served, 1),
                             "candidate_violating_deployments": int((dep & bad).sum())})
    return pd.DataFrame(rows), mism


# ---------------------------------------------------------------- support rule

def support(traj_dirs: dict, th_by_ds: dict) -> pd.DataFrame:
    """Which domains the ground-truth support rule excludes from per-domain scoring, from the test-split cell sizes
    (model-independent) of one control run per dataset; before any injection."""
    rows = []
    for ds, d in traj_dirs.items():
        th = th_by_ds[ds]
        p = sorted(Path(d).glob("*__fedavg__V0__s1"))[0]
        tr = Trajectory.load(p)
        z = tr.z
        edge_of_row = tr.edge_of_client()[z["test_client"]]
        for e in range(tr.n_edges):
            m = edge_of_row == e
            n_gy = np.zeros((2, 2))
            np.add.at(n_gy, (z["test_g"][m].astype(int), z["test_y"][m].astype(int)), 1)
            rows.append({"dataset": ds, "domain": e, "test_records": int(m.sum()), "min_group": int(n_gy.sum(1).min()),
                         "min_cell": int(n_gy.min()), "sp_wga_scored": bool(n_gy.sum(1).min() >= th.gt_min_cell),
                         "eo_scored": bool(n_gy.min() >= th.gt_min_cell)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--decisions", nargs="+", required=True)
    ap.add_argument("--trajectories", nargs="+", required=True)
    ap.add_argument("--decidability", nargs="+", required=True)
    ap.add_argument("--plans", nargs="+", default=["acs=configs/acs.json", "isic=configs/isic.json"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kv = lambda xs: dict(x.split("=", 1) for x in xs)
    runs, dec, traj, decid, plans = kv(a.runs), kv(a.decisions), kv(a.trajectories), kv(a.decidability), kv(a.plans)
    th_by_ds = {ds: Thresholds(**json.loads(Path(p).read_text())["thresholds"]) for ds, p in plans.items()}
    r = load(runs)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    rules(r).to_csv(out / "rules.csv", index=False)

    loc, r7 = localisation(dec)
    loc.to_csv(out / "localisation.csv", index=False)
    r7.to_csv(out / "integrity.csv", index=False)
    localisation_split(dec).to_csv(out / "localisation_split.csv", index=False)
    ph = point_estimate_holds(traj, th_by_ds)
    g = ph.groupby("dataset").agg(domain_rounds=("held", "size"), held=("held", "sum"),
                                  only_integrity=("only_integrity", "sum"), integrity=("integrity", "sum"))
    g.reset_index().to_csv(out / "point_estimate_holds.csv", index=False)
    summary["point_estimate_holds"] = g.reset_index().to_dict("records")

    ap_, mism = imperfect_approver(r, dec)
    ap_.to_csv(out / "approver.csv", index=False)
    summary["approver_mismatches_with_replay"] = mism

    pd.concat([sample_size(decid[ds], plans[ds]).assign(dataset=ds) for ds in plans]).to_csv(
        out / "sample_size.csv", index=False)
    simulated_coverage(decid, plans).to_csv(out / "simulated_coverage.csv", index=False)
    cv = coverage(traj, th_by_ds)
    cv.to_csv(out / "coverage.csv.gz", index=False)
    summary["coverage"] = cv.groupby(["dataset", "requirement"]).agg(
        checks=("covered", "size"), coverage=("covered", "mean"), expected=("expected", "mean"),
        median_width=("width", "median")).reset_index().to_dict("records")

    rt, mism = retained(dec, traj, th_by_ds)
    rt.to_csv(out / "retained.csv", index=False)
    summary["retained_ground_truth_mismatches"] = mism
    support(traj, th_by_ds).to_csv(out / "support.csv", index=False)

    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
