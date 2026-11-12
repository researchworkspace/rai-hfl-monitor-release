"""Command-line entry points.

python -I -m raimon.cli simulate --plan configs/acs.json --task 7 --out outputs/trajectories/acs
python -I -m raimon.cli replay --plan configs/acs.json --runs outputs/trajectories/acs --out outputs/replay/acs.csv

A plan file lists the trajectory grid (dataset x learner x injection x seed) and the run and monitor settings.
Task i of `simulate` runs the i-th trajectory of the grid, so a Slurm array maps one task to one trajectory.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import pandas as pd

from .fl import LearnerConfig
from .injections import InjectionConfig
from .simulate import RunConfig


def grid(plan: dict) -> list[dict]:
    g = plan["grid"]
    items = []
    for dataset, learner, inj, seed in itertools.product(g["datasets"], g["learners"], g["injections"], g["seeds"]):
        items.append({"dataset": dataset, "learner": learner, "injection": inj, "seed": seed})
    return items


def task_name(item: dict) -> str:
    inj = item["injection"]
    tag = inj["kind"] + "".join(f"-{k}{v}" for k, v in sorted(inj.items()) if k != "kind")
    return f"{item['dataset']}__{item['learner']}__{tag}__s{item['seed']}"


def load_data(dataset: str, plan: dict):
    from .data import load_acs, load_isic
    root = Path(os.environ.get("RAIHFL_DATA_ROOT", plan["data_root"])).expanduser()
    if dataset == "acs":
        return load_acs(root / "acs", **plan.get("acs", {}))
    if dataset == "isic":
        return load_isic(root / "isic", **plan.get("isic", {}))
    raise ValueError(f"unknown dataset {dataset}")


def cmd_simulate(args) -> None:
    from .simulate import run
    plan = json.loads(Path(args.plan).read_text())
    items = grid(plan)
    if args.task is not None:
        tasks = [args.task]
    elif args.chunk is not None:
        tasks = [i for i in range(len(items)) if i % args.n_chunks == args.chunk]
    else:
        tasks = range(len(items))
    data_cache = {}
    for i in tasks:
        item = items[i]
        out = Path(args.out) / task_name(item)
        if (out / "completion.json").exists():
            print(f"skip {out.name} (complete)")
            continue
        if item["dataset"] not in data_cache:
            data_cache[item["dataset"]] = load_data(item["dataset"], plan)
        inj = {**plan.get("injection_defaults", {}), **item["injection"]}
        cfg = RunConfig(dataset=item["dataset"], learner=LearnerConfig(item["learner"]),
                        injection=InjectionConfig(**inj), seed=item["seed"], **plan.get("run", {}))
        print(f"run {i}: {out.name}", flush=True)
        run(cfg, data_cache[item["dataset"]], out)


def cmd_replay(args) -> None:
    from .opa import OpaServer
    from .replay import Setting, Thresholds, Trajectory, replay
    plan = json.loads(Path(args.plan).read_text())
    th = Thresholds(**plan.get("thresholds", {}))
    settings = [Setting(**s) for s in plan["settings"]]
    rows = []
    dirs = [d for d in sorted(Path(args.runs).iterdir()) if (d / "completion.json").exists()]
    if args.name:
        dirs = [d for d in dirs if d.name == args.name]
        if not dirs:
            raise SystemExit(f"no completed trajectory named {args.name}")
    with OpaServer() as opa:
        for d in dirs:
            rows += replay(Trajectory.load(d), settings, th, opa)
            print(f"replayed {d.name}", flush=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(rows)} rows)")


def cmd_grid(args) -> None:
    plan = json.loads(Path(args.plan).read_text())
    for i, item in enumerate(grid(plan)):
        print(i, task_name(item))


def main() -> None:
    p = argparse.ArgumentParser(prog="raimon")
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--plan", required=True)
    s.add_argument("--task", type=int)
    s.add_argument("--chunk", type=int, help="run tasks i with i % n_chunks == chunk")
    s.add_argument("--n_chunks", type=int, default=1)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_simulate)
    r = sub.add_parser("replay")
    r.add_argument("--plan", required=True)
    r.add_argument("--runs", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--name", help="replay only this trajectory folder")
    r.set_defaults(func=cmd_replay)
    g = sub.add_parser("grid")
    g.add_argument("--plan", required=True)
    g.set_defaults(func=cmd_grid)
    sub.add_parser("isic-embed", help="precompute Fed-ISIC2019 embeddings; options as in raimon.isic")
    sub.add_parser("overhead", help="RQ4 overhead harness; options as in raimon.overhead")
    if sys.argv[1:2] == ["overhead"]:
        from .overhead import main as overhead_main
        return overhead_main(sys.argv[2:])
    if sys.argv[1:2] == ["isic-embed"]:
        from .isic import main as isic_main
        return isic_main(sys.argv[2:])
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
