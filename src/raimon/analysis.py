"""Research questions RQ1, RQ3 and RQ5 and the ablation tests, from the replay tables.

python -I -m raimon.analysis --runs acs=outputs/replay/acs.csv isic=outputs/replay/isic.csv --out outputs/analysis

RQ2 is in `soundness.py` and RQ4 in `overhead_report.py`.

Tests. Each configuration (dataset, learner, injection) has five runs, paired by seed because two placements replay
the same trajectory. Every comparison is reported
- per configuration: the mean paired difference, the number of runs in which each side is better, and the exact
  two-sided sign-flip test over the 2^5 sign vectors (smallest p = 0.0625);
- over the configurations of a dataset: a sign-flip test of the mean paired difference that flips the signs of
  cluster sums. Runs with the same seed share their trajectory before t0 and, across learners, their initial
  model and random streams, so the primary clusters are the random states (dataset, seed), with exact enumeration.
  A supplementary test uses (dataset, learner, seed) clusters.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

FULL = dict(n_eval=0, delay=0, mode="fail_safe", ablation="")
N_MC = 200_000
COMPARATORS = ("P1", "P2", "P2s", "P3")
TIERED = {"oracle": ("P4", {}), "failsafe": ("P4", {"ablation": "A2b"})}
ABLATIONS = {  # name: (ablated, full, injections or None for all)
    "A1": (("P4", dict(ablation="A1", n_eval=50)), ("P4", dict(n_eval=50)), None),
    "A2a": (("P4", dict(ablation="A2a", n_eval=50)), ("P4", dict(n_eval=50)), None),
    "A2b": (("P4", dict(ablation="A2b", n_eval=50)), ("P4", dict(n_eval=50)), None),
    "A3": (("P4R", dict(ablation="A3")), ("P4R", {}), ["V2"]),
    "A4": (("P4", dict(ablation="A4")), ("P4", {}), ["V4"]),
}


# ---------------------------------------------------------------- loading and selection

def load(runs: dict[str, str]) -> pd.DataFrame:
    frames = []
    for ds, path in runs.items():
        r = pd.read_csv(path, keep_default_na=False, na_values={"detection_delay": [""]})
        r["dataset"] = ds
        r["name"] = r["trajectory"].str.rsplit("/", n=1).str[-1]
        frac = r["trajectory"].str.extract(r"byz_frac([\d.]+)")[0]
        r["inj"] = np.where(frac.notna(), "V3-f" + frac.fillna(""), r["injection"])
        r["fb_rate"] = r["false_blocks"] / r["clean_candidates"].clip(lower=1)
        r["esc100"] = 100 * r["escalations"] / r["domain_rounds"]
        r["deployed"] = r["violating_deployments"] + r["clean_candidates"] - r["false_blocks"]
        r["no_update"] = r["domain_rounds"] - r["deployed"]
        frames.append(r)
    r = pd.concat(frames, ignore_index=True)
    for k in ("n_eval", "delay", "seed"):
        r[k] = r[k].astype(int)
    return r


def select(r: pd.DataFrame, placement: str, **kw) -> pd.DataFrame:
    spec = {**FULL, **kw}
    m = r["placement"] == placement
    for k, v in spec.items():
        m &= r[k] == v
    return r[m]


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str) -> np.ndarray:
    """a - b per seed; both frames hold one row per seed of the same configuration."""
    x = a.set_index("seed")[col].sort_index()
    y = b.set_index("seed")[col].sort_index()
    assert list(x.index) == list(y.index) and len(x) == 5, (x.index, y.index)
    return (x - y).to_numpy()


def cells(r: pd.DataFrame, injections=None):
    for (ds, ln, inj), _ in r.groupby(["dataset", "learner", "inj"], sort=True):
        if injections is None or inj in injections:
            yield ds, ln, inj


# ---------------------------------------------------------------- tests

def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def exact_signflip(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    if not np.any(d):
        return 1.0
    signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(d))))
    return float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-12).mean())


def cluster_signflip(sums: np.ndarray, n_pairs: int, rng: np.random.Generator) -> float:
    """Two-sided sign-flip test of the mean paired difference, flipping cluster sums."""
    sums = np.asarray(sums, float)
    obs = abs(sums.sum()) / n_pairs
    if obs == 0:
        return 1.0
    if len(sums) <= 16:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(sums))))
        return float((np.abs(signs @ sums) / n_pairs >= obs - 1e-12).mean())
    k = 0
    for _ in range(N_MC // 20_000):
        signs = rng.choice([-1.0, 1.0], size=(20_000, len(sums)))
        k += int((np.abs(signs @ sums) / n_pairs >= obs - 1e-12).sum())
    return (k + 1) / (N_MC + 1)


def per_config(r: pd.DataFrame, a_sel: tuple, b_sel: tuple, col: str, injections) -> pd.DataFrame:
    """Paired differences a - b per configuration (dataset, learner, injection), one value per seed."""
    rows = []
    for (ds, ln, inj), c in r.groupby(["dataset", "learner", "inj"], sort=True):
        if inj not in injections:
            continue
        a, b = select(c, a_sel[0], **a_sel[1]), select(c, b_sel[0], **b_sel[1])
        d = paired(a, b, col)
        rows.append({"dataset": ds, "learner": ln, "injection": inj, "a_mean": a[col].mean(), "b_mean": b[col].mean(),
                     "mean_diff": float(d.mean()), "min_diff": float(d.min()), "max_diff": float(d.max()),
                     "runs_a_lower": int((d < 0).sum()), "runs_a_higher": int((d > 0).sum()),
                     "p_exact": exact_signflip(d), "d": d.tolist(), "seeds": sorted(a.seed)})
    return pd.DataFrame(rows)


def pooled(t: pd.DataFrame, rng) -> dict:
    out = {}
    for scope, s in [("both", t)] + list(t.groupby("dataset")):
        clusters: dict = {}
        n = 0
        for x in s.itertuples():
            for seed, v in zip(x.seeds, x.d):
                clusters[(x.dataset, x.learner, seed)] = clusters.get((x.dataset, x.learner, seed), 0.0) + v
                n += 1
        sums = np.array(list(clusters.values()))
        by_seed: dict = {}
        for (ds_, _, seed), v in clusters.items():
            by_seed[(ds_, seed)] = by_seed.get((ds_, seed), 0.0) + v
        seed_sums = np.array(list(by_seed.values()))
        out[scope] = {"mean_diff": float(sums.sum() / n), "pairs": n, "clusters": len(sums),
                      "seed_clusters": len(seed_sums), "seed_clusters_a_lower": int((seed_sums < 0).sum()),
                      "seed_clusters_a_higher": int((seed_sums > 0).sum()),
                      "p_seed_cluster": cluster_signflip(seed_sums, n, rng),
                      "clusters_a_lower": int((sums < 0).sum()), "clusters_a_higher": int((sums > 0).sum()),
                      "p_cluster": cluster_signflip(sums, n, rng),
                      "configs": int(len(s)), "configs_mean_lower": int((s.mean_diff < 0).sum()),
                      "configs_mean_higher": int((s.mean_diff > 0).sum()),
                      "configs_all_runs_lower": int((s.runs_a_lower == 5).sum()),
                      "configs_no_run_higher": int(((s.runs_a_higher == 0) & (s.runs_a_lower > 0)).sum()),
                      "configs_p_exact_lt_0.05": int((s.p_exact < 0.05).sum())}
    return out


def tests(r: pd.DataFrame, rng) -> tuple[pd.DataFrame, dict]:
    """RQ1 (Tiered against each comparator, with the oracle approver and with the fail-safe default) and the
    ablations (ablated minus full, Tiered with the oracle approver as reference)."""
    viol = [i for i in sorted(r.inj.unique()) if i != "V0"]
    allinj = sorted(r.inj.unique())
    frames, summary = [], {}
    for tname, tsel in TIERED.items():
        for cmp in COMPARATORS:
            for metric, col, injs in (("exposure", "violating_deployments", viol), ("false_block_rate", "fb_rate", allinj)):
                t = per_config(r, tsel, (cmp, {}), col, injs)
                t = t.assign(family="rq1", tiered=tname, comparator=cmp, metric=metric)
                frames.append(t)
                summary[f"rq1/{tname}/{cmp}/{metric}"] = pooled(t, rng)
    for name, (a, f, injs) in ABLATIONS.items():
        for metric, col in (("exposure", "violating_deployments"), ("false_block_rate", "fb_rate")):
            t = per_config(r, a, f, col, injs or allinj).assign(family="ablation", tiered=name, comparator="full",
                                                                metric=metric)
            frames.append(t)
            summary[f"ablation/{name}/{metric}"] = pooled(t, rng)
    t = pd.concat(frames, ignore_index=True)
    # the prespecified RQ1 family: oracle approver against Central, Uniform and Pooled, exposure, 126 tests
    rq1 = t[(t.family == "rq1") & (t.tiered == "oracle") & (t.metric == "exposure") & (t.comparator != "P2s")]
    summary["prespecified_rq1_family"] = {"tests": int(len(rq1)), "min_p_exact": float(rq1.p_exact.min()),
                                          "holm_below_0.05": int((holm(rq1.p_exact) < 0.05).sum())}
    return t, summary


# ---------------------------------------------------------------- descriptive tables

def table_a(r: pd.DataFrame) -> pd.DataFrame:
    """Mean exposure (injected) or false-block rate (control) per configuration and placement."""
    rows = []
    for ds, ln, inj in cells(r):
        c = r[(r.dataset == ds) & (r.learner == ln) & (r.inj == inj)]
        row = {"dataset": ds, "learner": ln, "injection": inj,
               "violating_candidates": c[c.placement == "P1"].violating_candidates.mean()}
        for p in ("P1", "P2", "P2c", "P3", "P4", "P4R"):
            s = select(c, p)
            row[p] = s["fb_rate" if inj == "V0" else "violating_deployments"].mean()
            if inj == "V0":
                row[f"{p}_exposure"] = s.violating_deployments.mean()
        row["metric"] = "false_block_rate" if inj == "V0" else "violating_deployments"
        rows.append(row)
    return pd.DataFrame(rows)


def rq3(r: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p4 = r[(r.placement == "P4") & (r.ablation == "")]
    t = p4.groupby(["dataset", "learner", "n_eval", "delay", "mode"]).agg(
        esc100=("esc100", "mean"), exposure=("violating_deployments", "mean"),
        no_update=("no_update", "mean"), domain_rounds=("domain_rounds", "mean"),
        false_blocks=("false_blocks", "mean")).reset_index()
    base = t[(t["delay"] == 0) & (t["mode"] == "fail_safe")]
    ok = base[base.n_eval.isin([0, 200])]
    summary = {"esc100_d0": {f"{a.dataset}/{a.learner}/n_eval={a.n_eval or 'all'}": float(a.esc100)
                             for a in base.itertuples()},
               "usable": bool((ok.esc100 <= 10).all()),
               "usable_by_dataset": {ds: bool((ok[ok.dataset == ds].esc100 <= 10).all()) for ds in ok.dataset.unique()}}
    return t, summary


def rq5(r: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    viol = [i for i in sorted(r["inj"].unique()) if i != "V0"]
    for (ds, ln), c in r[r.inj.isin(viol)].groupby(["dataset", "learner"]):
        diffs = [paired(select(c[c.inj == i], "P4"), select(c[c.inj == i], "P3"), "violating_deployments")
                 for i in viol]
        d = np.concatenate(diffs)
        rows.append({"dataset": ds, "learner": ln, "mean_diff_P4_minus_P3": float(d.mean()),
                     "p4_mean": float(select(c, "P4").violating_deployments.mean()),
                     "p3_mean": float(select(c, "P3").violating_deployments.mean()), "n": len(d)})
    t = pd.DataFrame(rows)
    signs = {f"{x.dataset}/{x.learner}": int(np.sign(x.mean_diff_P4_minus_P3)) for x in t.itertuples()}
    return t, {"signs": signs, "supported": bool((t.mean_diff_P4_minus_P3 < 0).all())}


def delays(r: pd.DataFrame) -> pd.DataFrame:
    s = r[(r.n_eval == 0) & (r.delay == 0) & (r["mode"] == "fail_safe") & (r.ablation == "") & (r.inj != "V0")
          & r.placement.isin(["P1", "P2", "P3", "P4"])]
    return s[["dataset", "learner", "inj", "seed", "placement", "detection_delay"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="dataset=path/to/replay.csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = load(dict(x.split("=", 1) for x in a.runs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t, st = tests(r, np.random.default_rng(0))
    t.drop(columns=["d", "seeds"]).to_csv(out / "tests_per_config.csv", index=False)
    t[["family", "tiered", "comparator", "metric", "dataset", "learner", "injection", "seeds", "d"]].to_json(
        out / "paired_differences.json", orient="records")
    table_a(r).to_csv(out / "table_a.csv", index=False)
    t3, s3 = rq3(r)
    t3.to_csv(out / "rq3.csv", index=False)
    t5, s5 = rq5(r)
    t5.to_csv(out / "rq5.csv", index=False)
    delays(r).to_csv(out / "detection_delay.csv", index=False)
    summary = {"rq3": s3, "rq5": s5, "tests": st}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"rq3": s3, "rq5": s5, "prespecified_rq1_family": st["prespecified_rq1_family"]}, indent=1))


if __name__ == "__main__":
    main()
