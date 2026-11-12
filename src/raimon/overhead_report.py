"""Summarise RQ4 overhead measurements (outputs of scripts/overhead.sh).

python -I -m raimon.overhead_report --root outputs/overhead/mlp --out outputs/overhead/mlp/summary.json

Per round, the critical path of a synchronous round is: the slowest client's local training, the edge's
monitoring, the edge model upload to the cloud, the cloud's monitoring, and the reply. Monitoring time on that
path is the slowest edge's monitoring plus the cloud's monitoring; the monitoring message itself travels with
the round's other traffic, and its transfer time at the emulated rate is added from its size. Round
wall-clock = training + edge monitoring + model upload + message transfer + cloud monitoring. Medians are taken
over rounds within a repetition, then the median over repetitions is reported with the range.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def rate_bits_per_s(netem: str) -> float:
    m = re.search(r"rate\s+([\d.]+)\s*([kmg]?)bit", netem)
    scale = {"": 1, "k": 1e3, "m": 1e6, "g": 1e9}[m.group(2)]
    return float(m.group(1)) * scale


def one_rep(d: Path, bps: float) -> dict:
    cloud = json.loads((d / "cloud.json").read_text())
    edges = [json.loads(p.read_text()) for p in sorted(d.glob("edge*.json"))]
    train = json.loads((d / "train.json").read_text())
    T = cloud["rounds"]
    edge_ms = np.array([e["raw"]["edge_compute_ms"] for e in edges])          # [E, T]
    msg_bytes = np.array([e["raw"]["message_bytes"] for e in edges])
    upload_ms = np.array([e["raw"]["model_upload_ms"] for e in edges])
    cloud_ms = np.array(cloud["raw"]["cloud_compute_ms"])
    train_ms = float(np.median(train["raw"]))
    msg_ms = 1e3 * 8 * msg_bytes.max(0) / bps
    monitor = edge_ms.max(0) + cloud_ms + msg_ms
    round_ms = train_ms + edge_ms.max(0) + upload_ms.max(0) + msg_ms + cloud_ms
    out = {
        "edges": len(edges), "clients_per_edge": [e["clients"] for e in edges],
        "edge_compute_ms_median": float(np.median(edge_ms)), "edge_compute_ms_slowest_edge": float(np.median(edge_ms.max(0))),
        "cloud_compute_ms": float(np.median(cloud_ms)),
        "train_ms_largest_client": train_ms, "model_upload_ms": float(np.median(upload_ms)),
        "message_bytes_per_edge_median": float(np.median(msg_bytes)), "message_bytes_per_round_total": float(np.median(msg_bytes.sum(0))),
        "model_bytes": edges[0]["model_bytes"],
        "monitoring_share_of_round": float(np.median(monitor / round_ms)),
        "edge_share_of_edge_round": float(np.median(edge_ms.max(0) / (train_ms + edge_ms.max(0) + upload_ms.max(0)))),
        "peak_rss_mb_edge_max": max(e["peak_rss_mb"] for e in edges), "peak_rss_mb_cloud": cloud["peak_rss_mb"],
        "rounds": T,
    }
    for k in ("audit_forward_ms", "checks_ms", "accountant_ms", "opa_ms"):
        out[f"edge_{k}"] = float(np.median([np.median(e["raw"][k]) for e in edges]))
    for k in ("global_checks_ms", "r5_ms", "r6_ms", "opa_ms"):
        out[f"cloud_{k}"] = float(np.median(cloud["raw"][k]))
    out["checks_per_edge_median"] = float(np.median([np.median(e["raw"]["checks"]) for e in edges]))
    out["n_ref"] = cloud["n_ref"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(a.root)
    env = json.loads((root / "environment.json").read_text())
    bps = rate_bits_per_s(env["netem"])
    summary = {"environment": env, "datasets": {}}
    for ds_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "workloads"):
        reps = [one_rep(d, bps) for d in sorted(ds_dir.glob("rep*")) if (d / "completion.json").exists()]
        if not reps:
            continue
        agg = {}
        for k, v in reps[0].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                vals = [r[k] for r in reps]
                agg[k] = {"median": float(np.median(vals)), "min": float(min(vals)), "max": float(max(vals))}
            else:
                agg[k] = v
        agg["repetitions"] = len(reps)
        summary["datasets"][ds_dir.name] = agg
    Path(a.out).write_text(json.dumps(summary, indent=1))
    rows = ["edge_compute_ms_median", "edge_compute_ms_slowest_edge", "edge_audit_forward_ms", "edge_checks_ms",
            "edge_accountant_ms", "edge_opa_ms", "cloud_compute_ms", "cloud_global_checks_ms", "cloud_r5_ms", "cloud_r6_ms",
            "cloud_opa_ms", "train_ms_largest_client", "model_upload_ms", "message_bytes_per_edge_median",
            "message_bytes_per_round_total", "model_bytes", "monitoring_share_of_round", "edge_share_of_edge_round",
            "peak_rss_mb_edge_max", "peak_rss_mb_cloud"]
    names = list(summary["datasets"])
    print("| quantity | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for r in rows:
        cells = []
        for n in names:
            v = summary["datasets"][n][r]
            cells.append(f"{v['median']:.3g} ({v['min']:.3g}–{v['max']:.3g})")
        print(f"| {r} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
