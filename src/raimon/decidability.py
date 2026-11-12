"""Design-time decidability of P4's rate checks, compared with what P4 observed.

python -I -m raimon.decidability --plan configs/isic.json --pilot outputs/pilot/isic \
    --main outputs/trajectories/isic --out outputs/decidability/isic

Predicted: from the monitor-visible cell counts of each domain (known before deployment) and per-cell prediction
rates measured on the pilot's V0 runs, draw binomial counts and run the same interval functions and verdict rule
as P4. Observed: P4's OPA verdicts and decisions on the main V0 trajectories. Rounds >= `burn_in` only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats
from .opa import OpaServer
from .replay import Evidence, Setting, Thresholds, Trajectory, build_checks

REQ = {"R1": ("upper", "sp"), "R2": ("upper", "eo"), "R3": ("lower", "acc_min")}


def interval(rid: str, c: np.ndarray, alpha: float) -> stats.Interval:
    return {"R1": stats.sp_gap, "R2": stats.eo_gap, "R3": stats.worst_group_accuracy}[rid](c, alpha)


def verdict(kind: str, iv: stats.Interval, tau: float) -> str:
    lo = iv.lower if np.isfinite(iv.lower) else -1e9
    hi = iv.upper if np.isfinite(iv.upper) else 1e9
    if kind == "upper":
        return "pass" if hi <= tau else ("fail" if lo > tau else "inconclusive")
    return "pass" if lo >= tau else ("fail" if hi < tau else "inconclusive")


def domain_cells(ev: Evidence, scope_rows: list[np.ndarray]) -> np.ndarray:
    """Monitor-visible (g, y) counts of the union of the given client rows, before any injection."""
    z = ev.tr.z
    r = np.concatenate(scope_rows)
    n = np.zeros((2, 2))
    np.add.at(n, (z["eval_g"][r], z["eval_y"][r]), 1)
    return n


def pilot_rates(pilot: list[Trajectory], edge: int | None, burn_in: int) -> np.ndarray:
    """P(Yhat=1 | g, y) of a domain (None = global) on the eval split, averaged over rounds and pilot runs."""
    pos, n = np.zeros((2, 2)), np.zeros((2, 2))
    for tr in pilot:
        z = tr.z
        m = np.ones(len(z["eval_y"]), bool) if edge is None else tr.edge_of_client()[z["eval_client"]] == edge
        g, y = z["eval_g"][m], z["eval_y"][m]
        for t in range(burn_in, tr.rounds):
            p = z["pred_eval"][t][m]
            np.add.at(pos, (g, y), p)
            np.add.at(n, (g, y), 1)
    return pos / np.maximum(n, 1)


def predict(n: np.ndarray, rate: np.ndarray, th: Thresholds, draws: int, rng: np.random.Generator) -> dict:
    out = {rid: {"inconclusive": 0, "fail": 0, "half": []} for rid in REQ}
    v_all = np.empty((draws, len(REQ)), dtype=object)
    for i in range(draws):
        pos = rng.binomial(n.astype(int), rate)
        c = np.stack([n - pos, pos], axis=-1)                              # [g, y, yhat]
        for j, (rid, (kind, key)) in enumerate(REQ.items()):
            iv = interval(rid, c, th.alpha)
            v = verdict(kind, iv, getattr(th, key))
            v_all[i, j] = v
            out[rid]["inconclusive"] += v == "inconclusive"
            out[rid]["fail"] += v == "fail"
            if np.isfinite(iv.upper) and np.isfinite(iv.lower):
                out[rid]["half"].append((iv.upper - iv.lower) / 2)
    res = {rid: {"p_inconclusive": d["inconclusive"] / draws, "p_fail": d["fail"] / draws,
                 "half_width": float(np.median(d["half"])) if d["half"] else float("inf")} for rid, d in out.items()}
    return res, v_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--pilot", required=True)
    ap.add_argument("--main", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--burn_in", type=int, default=10)
    ap.add_argument("--draws", type=int, default=2000)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    th = Thresholds(**plan["thresholds"])
    rng = np.random.default_rng(0)
    pilot = [Trajectory.load(d) for d in sorted(Path(a.pilot).glob("*__V0__*"))]
    main_runs = [Trajectory.load(d) for d in sorted(Path(a.main).glob("*__V0__*"))]
    E = pilot[0].n_edges
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pred_rows, dom_rows = [], []
    for n_eval in (0, 200, 50):
        ev = Evidence(pilot[0], n_eval)                       # the subsample depends only on the eval split
        edge_of = pilot[0].edge_of_client()
        scopes = {f"e{e}": (e, [ev.client_rows[k] for k in np.flatnonzero(edge_of == e)]) for e in range(E)}
        scopes["global"] = (None, ev.client_rows)
        verdicts = {}
        for scope, (edge, rows) in scopes.items():
            n = domain_cells(ev, rows)
            res, v_all = predict(n, pilot_rates(pilot, edge, a.burn_in), th, a.draws, rng)
            verdicts[scope] = v_all
            for rid, r in res.items():
                kind, key = REQ[rid]
                pred_rows.append({"n_eval": n_eval or "all", "scope": scope, "req": rid, "min_cell": int(n.min()),
                                  "min_group": int(n.sum(1).min()), "tau": getattr(th, key), **r})
        for e in range(E):
            v = np.concatenate([verdicts[f"e{e}"], verdicts["global"]], axis=1)
            fails = (v == "fail").any(1)
            opens = (v == "inconclusive").any(1)
            dom_rows.append({"n_eval": n_eval or "all", "domain": f"e{e}", "p_escalate_pred": float((~fails & opens).mean()),
                             "p_hold_pred": float(fails.mean())})
    pred = pd.DataFrame(pred_rows)
    dom = pd.DataFrame(dom_rows)

    obs_rows, obs_dom = [], []
    with OpaServer() as opa:
        for tr in main_runs:
            for n_eval in (0, 200, 50):
                ev = Evidence(tr, n_eval)
                s = Setting(placement="P4", n_eval=n_eval)
                for t in range(a.burn_in, tr.rounds):
                    checks = build_checks(s, t, ev, th)
                    res = opa.evaluate([f"e{e}" for e in range(tr.n_edges)], checks)
                    for c in checks:
                        rid, scope = c["id"].split("/")[0], c["id"].split("/")[1]
                        if rid in REQ:
                            obs_rows.append({"n_eval": n_eval or "all", "scope": scope, "req": rid,
                                             "inconclusive": res["verdict"][c["id"]] == "inconclusive",
                                             "fail": res["verdict"][c["id"]] == "fail"})
                    for d, dec in res["decision"].items():
                        obs_dom.append({"n_eval": n_eval or "all", "domain": d, "escalate": dec == "escalate",
                                        "hold": dec == "hold"})
    obs = pd.DataFrame(obs_rows).groupby(["n_eval", "scope", "req"], sort=False).mean().rename(
        columns={"inconclusive": "obs_inconclusive", "fail": "obs_fail"}).reset_index()
    obsd = pd.DataFrame(obs_dom).groupby(["n_eval", "domain"], sort=False).mean().rename(
        columns={"escalate": "obs_escalate", "hold": "obs_hold"}).reset_index()
    for df in (pred, dom, obs, obsd):
        df["n_eval"] = df["n_eval"].astype(str)
    checks = pred.merge(obs, on=["n_eval", "scope", "req"], how="left")
    doms = dom.merge(obsd, on=["n_eval", "domain"], how="left")
    checks.to_csv(out / "checks.csv", index=False)
    doms.to_csv(out / "domains.csv", index=False)

    def agree(p, o):
        p, o = np.asarray(p, float), np.asarray(o, float)
        return {"mae": float(np.abs(p - o).mean()), "binary_agreement": float(((p >= 0.5) == (o >= 0.5)).mean()),
                "spearman": float(pd.Series(p).corr(pd.Series(o), method="spearman")), "n": int(len(p))}

    summary = {"checks_inconclusive": agree(checks.p_inconclusive, checks.obs_inconclusive),
               "domains_escalate": agree(doms.p_escalate_pred, doms.obs_escalate),
               "pilot_runs": [str(t.path) for t in pilot], "main_runs": len(main_runs), "burn_in": a.burn_in,
               "draws": a.draws, "thresholds": plan["thresholds"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    pd.set_option("display.width", 220)
    print(checks.round(3).to_string(index=False))
    print(doms.round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k in ("checks_inconclusive", "domains_escalate")}, indent=1))


if __name__ == "__main__":
    main()
