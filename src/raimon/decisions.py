"""Per-round decisions and non-passing checks of selected placements, for the post-hoc analyses.

python -I -m raimon.decisions --plan configs/acs.json --runs outputs/trajectories/acs --out outputs/decisions/acs

For every trajectory, writes <name>.npz with
  bad[t, e], viol[t, e, r]     ground truth (rounds before t0 from the paired V0 run, see replay.ground_truth_pre_from_v0)
  dec_<setting>[t, e]          0 deploy, 1 hold, 2 escalate (OPA decisions, before any reviewer)
and <name>.json with the non-passing checks per setting: [t, check id, domain, verdict].
Placements run in the replay's order (P1 first), so the bootstrap draws for R5/R6 match the main replay.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .opa import OpaServer
from .replay import Evidence, Setting, Thresholds, Trajectory, build_checks, ground_truth_pre_from_v0

SETTINGS = [Setting("P1"), Setting("P2"), Setting("P3"), Setting("P4"),
            Setting("P4", n_eval=200), Setting("P4", n_eval=50)]
CODE = {"deploy": 0, "hold": 1, "escalate": 2}


def one(path: str, th_d: dict, out: str) -> str:
    th = Thresholds(**th_d)
    tr = Trajectory.load(Path(path))
    dst = Path(out) / tr.path.name                       # names contain dots (byz_frac0.1): append suffixes
    if Path(f"{dst}.json").exists():
        return tr.path.name
    viol = ground_truth_pre_from_v0(tr, th)
    arrays = {"bad": viol[:, :, :6].any(axis=2), "viol": viol}
    nonpass = {}
    domains = [f"e{e}" for e in range(tr.n_edges)]
    evidence: dict[int, Evidence] = {}
    with OpaServer() as opa:
        for s in SETTINGS:
            ev = evidence.setdefault(s.n_eval, Evidence(tr, s.n_eval))
            dec = np.zeros((tr.rounds, tr.n_edges), np.int8)
            npass = []
            for t in range(tr.rounds):
                checks = build_checks(s, t, ev, th)
                res = opa.evaluate(domains, checks)
                for e in range(tr.n_edges):
                    dec[t, e] = CODE[res["decision"][f"e{e}"]]
                for c in checks:
                    v = res["verdict"].get(c["id"])
                    if v in ("fail", "inconclusive"):
                        npass.append([t, c["id"], c["domain"], v])
            arrays[f"dec_{s.label}"] = dec
            nonpass[s.label] = npass
    np.savez_compressed(f"{dst}.npz", **arrays)
    tmp = Path(f"{dst}.json.tmp")
    tmp.write_text(json.dumps({"t0": tr.t0, "kind": tr.kind, "liars": sorted(tr.liars),
                               "target_edge": tr.meta["config"]["injection"].get("target_edge"),
                               "client_edge": tr.edge_of_client().tolist(), "nonpass": nonpass}))
    tmp.rename(f"{dst}.json")
    return tr.path.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    th = json.loads(Path(a.plan).read_text())["thresholds"]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    paths = [str(d) for d in sorted(Path(a.runs).iterdir()) if (d / "completion.json").exists()]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, name in enumerate(ex.map(one, paths, [th] * len(paths), [a.out] * len(paths))):
            print(f"{i + 1}/{len(paths)} {name}", flush=True)


if __name__ == "__main__":
    main()
