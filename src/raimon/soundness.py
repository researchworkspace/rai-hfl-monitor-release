"""RQ2: local-pass/global-fail and global-pass/local-fail rounds, which placements catch them, and P4's misses.

python -I -m raimon.soundness --plan configs/acs.json --runs outputs/trajectories/acs \
    --replay outputs/replay/acs.csv --out outputs/rq2/acs

Definitions, per round t and requirement r in R1-R3 on the ground-truth test split:
- local-pass/global-fail (LPGF): the global statistic violates r while every assessable domain passes. Every
  domain's candidate violates (global requirements apply everywhere).
- global-pass/local-fail (GPLF): the global statistic passes while at least one assessable domain violates r.
  The violating domain-rounds are those domains.
For each type, the violating domain-rounds each placement (full sample, d = 0, fail-safe) did not deploy, split
by how: held by the monitor, or escalated (the simulated reviewer then holds them). Observed beta: FedGFT DH for
SP and EO from the edges' test (G, Y) counts, before and after t0. P4 misses: every (round, domain) where P4
deployed a violating candidate, in every injection, with P4's verdicts on the violated requirement.

Replay is reproduced exactly: placements run in the replay's order on a fresh Evidence, so the bootstrap draws
match, and per-trajectory totals are checked against the replay table. Rounds before t0 take the ground truth of
the paired V0 run, as in the replay.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats
from .opa import OpaServer
from .replay import (Evidence, Setting, Thresholds, Trajectory, build_checks, counts_of,
                     ground_truth_pre_from_v0, heterogeneity, paired_v0, spearman)

PLACEMENTS = ("P1", "P2", "P3", "P4")
RQ2_INJ = ("V0", "V1", "V2")
STAT = {0: stats.sp_gap, 1: stats.eo_gap, 2: stats.worst_group_accuracy}


def gt_parts(tr: Trajectory, th: Thresholds):
    """glob[t, r] and dom[t, e, r] (assessable domains only) for R1-R3, with the values, as in ground_truth."""
    z = tr.z
    T, E = tr.rounds, tr.n_edges
    glob = np.zeros((T, 3), bool)
    dom = np.zeros((T, E, 3), bool)
    gval = np.full((T, 3), np.nan)
    dval = np.full((T, E, 3), np.nan)
    edge_of_row = tr.edge_of_client()[z["test_client"]]
    for t in range(T):
        keep = z["keep_test"] if tr.active(t) else np.ones_like(z["keep_test"])
        p, y, g = z["pred_test"][t], z["test_y"], z["test_g"]
        c = counts_of(p[keep], y[keep], g[keep])
        gval[t] = [STAT[i](c).estimate for i in range(3)]
        glob[t] = [gval[t, 0] > th.sp, gval[t, 1] > th.eo, gval[t, 2] < th.acc_min]
        for e in range(E):
            m = keep & (edge_of_row == e)
            ce = counts_of(p[m], y[m], g[m])
            ok = [ce.sum(axis=(1, 2)).min() >= th.gt_min_cell, ce.sum(axis=2).min() >= th.gt_min_cell,
                  ce.sum(axis=(1, 2)).min() >= th.gt_min_cell]
            v = [STAT[i](ce).estimate for i in range(3)]
            dval[t, e] = [v[i] if ok[i] else np.nan for i in range(3)]
            dom[t, e] = [ok[0] and v[0] > th.sp, ok[1] and v[1] > th.eo, ok[2] and v[2] < th.acc_min]
    return glob, dom, gval, dval


def betas(tr: Trajectory) -> dict:
    z = tr.z
    edge_of_row = tr.edge_of_client()[z["test_client"]]
    out = {}
    for phase, keep in (("pre", np.ones_like(z["keep_test"])), ("post", z["keep_test"])):
        if phase == "post" and tr.kind == "V0":
            continue
        y, g = z["test_y"], z["test_g"]
        cs = [counts_of(np.zeros(int((keep & (edge_of_row == e)).sum())), y[keep & (edge_of_row == e)],
                        g[keep & (edge_of_row == e)]) for e in range(tr.n_edges)]
        out[f"beta_sp_{phase}"], out[f"beta_eo_{phase}"] = heterogeneity(cs)
    return out


def outcome(decision: str, bad: bool) -> tuple[bool, str]:
    """(deployed, how) under d = 0 fail-safe with the ground-truth reviewer."""
    if decision == "deploy":
        return True, "deploy"
    if decision == "hold":
        return False, "hold"
    return (not bad), "escalate"


def one(path: str, th_d: dict, rq2: bool) -> dict:
    th = Thresholds(**th_d)
    tr = Trajectory.load(Path(path))
    viol = ground_truth_pre_from_v0(tr, th)
    bad = viol[:, :, :6].any(axis=2)
    glob, dom, gval, dval = gt_parts(tr, th)
    if tr.kind != "V0":                                    # R1-R3 parts before t0 from the V0 pair as well
        g0, d0, gv0, dv0 = gt_parts(Trajectory.load(paired_v0(tr)), th)
        glob[:tr.t0], dom[:tr.t0], gval[:tr.t0], dval[:tr.t0] = g0[:tr.t0], d0[:tr.t0], gv0[:tr.t0], dv0[:tr.t0]
    assert np.array_equal(glob[:, None, :] | dom, viol[:, :, :3]), path
    T, E = tr.rounds, tr.n_edges
    domains = [f"e{e}" for e in range(E)]
    ev = Evidence(tr, 0)
    how = {}
    totals = {}
    misses = []
    with OpaServer() as opa:
        for p in (PLACEMENTS if rq2 else ("P1", "P4")):     # P1 first: it fixes the bootstrap draw order
            s = Setting(placement=p)
            h = np.empty((T, E), object)
            dep = np.zeros((T, E), bool)
            for t in range(T):
                checks = build_checks(s, t, ev, th)
                res = opa.evaluate(domains, checks)
                for e in range(E):
                    dep[t, e], h[t, e] = outcome(res["decision"][f"e{e}"], bad[t, e])
                    if p == "P4" and dep[t, e] and bad[t, e]:
                        for ri in np.flatnonzero(viol[t, e, :6]):
                            rid = f"R{ri + 1}"
                            rel = [c for c in checks if c["id"].split("/")[0] == rid
                                   and c["id"].split("/")[1] in ("global", f"e{e}")]
                            gt = {0: gval[t, 0], 1: gval[t, 1], 2: gval[t, 2]}.get(ri)
                            entry = {"trajectory": tr.path.name, "t": t, "domain": f"e{e}", "req": rid,
                                     "gt_global": None if gt is None else float(gt),
                                     "gt_domain": float(dval[t, e, ri]) if ri < 3 else None,
                                     "gt_global_viol": bool(glob[t, ri]) if ri < 3 else None}
                            if ri == 4:
                                entry["gt_global"] = float(tr.z["rob_test_clean"][t].mean() - tr.z["rob_test_pert"][t].mean())
                            if ri == 5:
                                a = np.abs(tr.z["attr_test"][t].astype(float)).mean(0)
                                b = np.abs(tr.z["attr_test"][t - 1].astype(float)).mean(0)
                                entry["gt_global"] = spearman(a, b)
                            entry["checks"] = [{k: c[k] for k in ("id", "threshold", "estimate", "lower", "upper")}
                                               | {"verdict": res["verdict"].get(c["id"])} for c in rel]
                            misses.append(entry)
            how[p] = h
            totals[p] = int((dep & bad).sum())
    rows = []
    if rq2:
        for ri in range(3):
            lpgf = glob[:, ri] & ~dom[:, :, ri].any(1)
            gplf = ~glob[:, ri] & dom[:, :, ri].any(1)
            for kind, rounds, mask in (("LPGF", lpgf, np.repeat(lpgf[:, None], E, 1)),
                                       ("GPLF", gplf, gplf[:, None] & dom[:, :, ri])):
                row = {"trajectory": tr.path.name, "req": f"R{ri + 1}", "type": kind, "rounds": int(rounds.sum()),
                       "domain_rounds": int(mask.sum()),
                       "rounds_after_t0": int(rounds[tr.t0:].sum()) if tr.kind != "V0" else None}
                for p in PLACEMENTS:
                    hm = how[p][mask]
                    row[f"{p}_caught"] = int(np.isin(hm, ["hold", "escalate"]).sum())
                    row[f"{p}_held"] = int((hm == "hold").sum())
                    row[f"{p}_escalated"] = int((hm == "escalate").sum())
                rows.append(row)
    cfg = tr.meta["config"]
    return {"name": tr.path.name, "learner": cfg["learner"]["name"], "injection": tr.kind, "seed": cfg["seed"],
            "rows": rows, "totals": totals, "beta": betas(tr), "misses": misses}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    th = json.loads(Path(a.plan).read_text())["thresholds"]
    rep = pd.read_csv(a.replay, keep_default_na=False)
    rep["name"] = rep["trajectory"].str.split("/").str[-1]
    full = rep[(rep.n_eval == 0) & (rep.delay == 0) & (rep["mode"] == "fail_safe") & (rep.ablation == "")]
    p4miss = set(full[(full.placement == "P4") & (full.violating_deployments > 0)].name)
    jobs = []
    for d in sorted(Path(a.runs).iterdir()):
        if not (d / "completion.json").exists():
            continue
        kind = d.name.split("__")[2].split("-")[0]
        if kind in RQ2_INJ or d.name in p4miss:
            jobs.append((str(d), kind in RQ2_INJ))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, [j[0] for j in jobs], [th] * len(jobs), [j[1] for j in jobs]))
    # exact reproduction of the replay
    mism = []
    for r in res:
        for p, v in r["totals"].items():
            ref = int(full[(full.name == r["name"]) & (full.placement == p)].violating_deployments.iloc[0])
            if ref != v:
                mism.append({"name": r["name"], "placement": p, "replay": ref, "here": v})
    rows = pd.DataFrame([dict(row, learner=r["learner"], injection=r["injection"], seed=r["seed"])
                         for r in res for row in r["rows"]])
    beta = pd.DataFrame([dict(r["beta"], name=r["name"], learner=r["learner"], injection=r["injection"],
                              seed=r["seed"]) for r in res if r["rows"]])
    misses = [dict(m, learner=r["learner"], injection=r["injection"], seed=r["seed"]) for r in res for m in r["misses"]]
    rows.to_csv(out / "rounds.csv", index=False)
    beta.to_csv(out / "beta.csv", index=False)
    (out / "p4_misses.json").write_text(json.dumps(misses, indent=1))
    agg = rows.groupby(["injection", "req", "type"])[
        ["rounds", "domain_rounds"] + [f"{p}_{k}" for p in PLACEMENTS for k in ("caught", "held", "escalated")]].sum()
    agg.to_csv(out / "rq2_summary.csv")
    summary = {"trajectories": len(res), "replay_mismatches": mism, "p4_misses": len(misses),
               "p4_misses_by": {f"{i}/{r}": int(n) for (i, r), n in
                                pd.DataFrame(misses).groupby(["injection", "req"]).size().items()} if misses else {},
               "beta": beta.drop(columns=["name", "seed"]).groupby("injection").median(numeric_only=True).to_dict()}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(agg.to_string())
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
