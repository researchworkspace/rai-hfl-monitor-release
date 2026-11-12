"""Pilot calibration: natural levels of every monitored statistic, measured on ground-truth test data.

python -I -m raimon.calibrate --runs outputs/pilot/acs

For each trajectory, prints quantiles over rounds >= `--burn_in` of the global and per-domain statistics, and
for injected runs, the change after t0 in the targeted domain. Thresholds are then set from V0 runs only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats
from .replay import Trajectory, counts_of, spearman


def levels(tr: Trajectory, burn_in: int) -> pd.DataFrame:
    z = tr.z
    edge_of_row = tr.edge_of_client()[z["test_client"]]
    rows = []
    for t in range(burn_in, tr.rounds):
        keep = z["keep_test"] if tr.active(t) else np.ones_like(z["keep_test"])
        p, y, g = z["pred_test"][t], z["test_y"], z["test_g"]
        scopes = {"global": keep} | {f"e{e}": keep & (edge_of_row == e) for e in range(tr.n_edges)}
        for scope, m in scopes.items():
            c = counts_of(p[m], y[m], g[m])
            rows.append({"t": t, "scope": scope, "post": tr.active(t),
                         "sp": stats.sp_gap(c).estimate, "eo": stats.eo_gap(c).estimate,
                         "wacc": stats.worst_group_accuracy(c).estimate, "acc": (p[m] == y[m]).mean()})
        a, b = np.abs(z["attr_test"][t].astype(float)).mean(0), np.abs(z["attr_test"][t - 1].astype(float)).mean(0)
        rows.append({"t": t, "scope": "model", "post": tr.active(t),
                     "drop": z["rob_test_clean"][t].mean() - z["rob_test_pert"][t].mean(), "attr": spearman(a, b)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--burn_in", type=int, default=10)
    args = ap.parse_args()
    frames = []
    for d in sorted(Path(args.runs).iterdir()):
        if not (d / "completion.json").exists():
            continue
        tr = Trajectory.load(d)
        df = levels(tr, args.burn_in)
        df["run"], df["kind"] = d.name, tr.kind
        df["learner"] = tr.meta["config"]["learner"]["name"]
        df["sec_per_round"] = float(tr.z["round_seconds"].mean())
        frames.append(df)
    df = pd.concat(frames)
    pd.set_option("display.width", 200)
    v0 = df[df.kind == "V0"]
    q = [0.5, 0.9, 0.99]
    print("== V0 global statistics (quantiles over rounds and runs)")
    print(v0[v0.scope == "global"].groupby("learner")[["sp", "eo", "wacc", "acc"]].quantile(q).round(3))
    print("== V0 per-domain statistics")
    dom = v0[v0.scope.str.startswith("e")]
    print(dom.groupby("learner")[["sp", "eo", "wacc"]].quantile(q).round(3))
    print("== V0 model-level statistics")
    print(v0[v0.scope == "model"].groupby("learner")[["drop", "attr"]].quantile([0.01, 0.1, 0.5, 0.9, 0.99]).round(3))
    print("== injected runs: mean before vs after t0")
    inj = df[(df.kind != "V0") & (df.scope != "model")]
    print(inj.groupby(["kind", "learner", "scope", "post"])[["sp", "eo", "wacc"]].mean().round(3)
          .unstack("post").to_string())
    print("== seconds per round", df.groupby("learner").sec_per_round.mean().round(1).to_dict())


if __name__ == "__main__":
    main()
