"""Run one training trajectory and log everything the monitors and the ground truth need.

Monitoring only gates deployment, so training is identical under every placement. One logged trajectory is
replayed offline for all placements, ablations, sample sizes and reviewer delays (see `replay.py`).

Logged per round t (the candidate model is the global model after round t's aggregation):
  pred_eval[t, N_eval], pred_test[t, N_test], pred_ref[t, N_ref]       candidate predictions
  rep_counts[t, K, G, Y, Yhat], rep_n[t, K]                            what clients report
  eps_actual[t, E], eps_declared[t, E]                                 RDP accountant, actual and declared noise
  rob_{ref,test}_{clean,pert}[t, n_sub]                                per-example correctness for R5
  attr_{ref,test}[t, n_sub, F_groups]                                  Integrated-Gradients attributions for R6
Static: masks before and after t0, row metadata, topology.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

from .data import FedData
from .fl import (MLP, Aggregator, ClientReport, EdgeState, LearnerConfig, get_vec, integrated_gradients,
                 local_train_dp, predict, set_vec)
from .injections import Injection, InjectionConfig


@dataclass
class RunConfig:
    dataset: str = "acs"
    learner: LearnerConfig = field(default_factory=lambda: LearnerConfig("fedavg"))
    injection: InjectionConfig = field(default_factory=InjectionConfig)
    seed: int = 0
    rounds: int = 60
    local_epochs: int = 1
    lr: float = 0.1
    batch: int = 256
    clip: float = 1.0                # per-example clipping norm for DP-SGD
    target_eps: float = 6.0          # each client's noise multiplier is set so that T rounds give this epsilon
    delta: float = 1e-5
    n_sub: int = 2000                # examples for R5 and R6
    rho: float = 0.5                 # R5 perturbation: Gaussian noise std on numeric features
    ig_steps: int = 32
    hidden: int = 64


def counts_of(pred: np.ndarray, y: np.ndarray, g: np.ndarray) -> np.ndarray:
    c = np.zeros((2, 2, 2))
    np.add.at(c, (g, y, pred.astype(np.int64)), 1)
    return c


def feature_groups(names: list[str]) -> tuple[list[str], np.ndarray]:
    groups = sorted({n.split("_")[0] for n in names})
    idx = np.array([groups.index(n.split("_")[0]) for n in names])
    return groups, idx


def run(cfg: RunConfig, data: FedData, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "completion.json").exists():
        return out
    t_start = time.time()
    rng = np.random.default_rng(cfg.seed)
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    inj = Injection(cfg.injection, data, np.random.default_rng(cfg.seed + 10_000))

    F = data.splits["train"].x.shape[1]
    model = MLP(F, cfg.hidden)
    theta = get_vec(model)
    K, E = data.n_clients, data.n_edges
    edges = [EdgeState([k for k in range(K) if data.client_edge[k] == e]) for e in range(E)]
    n_train = np.bincount(data.splits["train"].client, minlength=K)
    q_k = np.minimum(1.0, cfg.batch / np.maximum(n_train, 1))
    steps_k = [cfg.local_epochs * max(1, round(int(n) / cfg.batch)) for n in n_train]
    sigma_cache: dict[tuple, float] = {}
    sigma = np.zeros(K)
    for k in range(K):
        key = (float(q_k[k]), steps_k[k])
        if key not in sigma_cache:
            sigma_cache[key] = get_noise_multiplier(target_epsilon=cfg.target_eps, target_delta=cfg.delta,
                                                    sample_rate=key[0], steps=cfg.rounds * key[1],
                                                    accountant="rdp")
        sigma[k] = sigma_cache[key]
    actual_acc = [RDPAccountant() for _ in range(K)]
    declared_acc = [RDPAccountant() for _ in range(K)]
    label_freq = np.bincount(data.splits["train"].y, minlength=2) / len(data.splits["train"].y)
    agg = Aggregator(cfg.learner, label_freq)

    ev, te, ref = data.splits["eval"], data.splits["test"], data.reference
    rows_of = {s: [np.flatnonzero(data.splits[s].client == k) for k in range(K)] for s in ("train", "eval")}

    # Fixed subsets for R5 and R6: test rows kept after t0, reference rows.
    post_keep = inj.keep["test"]
    test_sub = rng.choice(np.flatnonzero(post_keep), min(cfg.n_sub, post_keep.sum()), replace=False)
    ref_sub = rng.choice(len(ref.y), min(cfg.n_sub, len(ref.y)), replace=False)
    num_cols = list(cfg.injection.numeric_cols)
    z_ref = rng.standard_normal((len(ref_sub), len(num_cols))).astype(np.float32)
    z_test = rng.standard_normal((len(test_sub), len(num_cols))).astype(np.float32)
    groups, gidx = feature_groups(data.feature_names)

    T = cfg.rounds
    log = {
        "pred_eval": np.zeros((T, len(ev.y)), np.uint8),
        "pred_test": np.zeros((T, len(te.y)), np.uint8),
        "pred_ref": np.zeros((T, len(ref.y)), np.uint8),
        "rep_counts": np.zeros((T, K, 2, 2, 2), np.float32),
        "rep_n": np.zeros((T, K), np.float32),
        "eps_actual": np.zeros((T, E)), "eps_declared": np.zeros((T, E)),
        "rob_ref_clean": np.zeros((T, len(ref_sub)), np.uint8), "rob_ref_pert": np.zeros((T, len(ref_sub)), np.uint8),
        "rob_test_clean": np.zeros((T, len(test_sub)), np.uint8),
        "rob_test_pert": np.zeros((T, len(test_sub)), np.uint8),
        "attr_ref": np.zeros((T, len(ref_sub), len(groups)), np.float16),
        "attr_test": np.zeros((T, len(test_sub), len(groups)), np.float16),
        "round_seconds": np.zeros(T),
    }

    def evaluate_split(split: str, t: int) -> np.ndarray:
        rows = np.arange(len(data.splits[split].y))
        return predict(model, theta, inj.features(split, t, rows))

    # Reports of the initial model, used for the first round's weights.
    pred0 = evaluate_split("eval", 0)
    m0 = inj.mask("eval", 0)
    reports = {}
    for k in range(K):
        r = rows_of["eval"][k][m0[rows_of["eval"][k]]]
        reports[k] = ClientReport(len(r), counts_of(pred0[r], ev.y[r], ev.g[r]))

    for t in range(T):
        t_round = time.time()
        tr_mask, ev_mask = inj.mask("train", t), inj.mask("eval", t)
        audit_hist = {}
        for k in range(K):
            r = rows_of["eval"][k]
            r = r[ev_mask[r] & data.audit[r]]
            h = np.bincount(ev.y[r], minlength=2).astype(float) + 1.0
            audit_hist[k] = h / h.sum()
        edge_updates, edge_n, edge_counts = {}, {}, {}
        for e, edge in enumerate(edges):
            weights = agg.client_weights(edge, reports, audit_hist)
            updates = {}
            for k in edge.clients:
                r = rows_of["train"][k]
                r = r[tr_mask[r]]
                if len(r) == 0:
                    continue
                x = inj.features("train", t, r)
                y = inj.train_labels(k, t, data.splits["train"].y[r], data.splits["train"].g[r])
                s_act = sigma[k] * inj.sigma_scale(e, t)
                updates[k] = local_train_dp(model, theta, x, y, q_k[k], steps_k[k], cfg.lr, cfg.clip,
                                            s_act, actual_acc[k], gen)
                # The declared accountant records what the edge reports: the configured noise multiplier.
                for _ in range(steps_k[k]):
                    declared_acc[k].step(noise_multiplier=sigma[k], sample_rate=q_k[k])
            weights = {k: weights[k] for k in updates}
            s = sum(weights.values())
            weights = {k: v / s for k, v in weights.items()}
            edge_updates[e] = agg.aggregate_edge(updates, weights)
            edge_n[e] = sum(reports[k].n for k in edge.clients)
            edge_counts[e] = sum(reports[k].counts for k in edge.clients)
            log["eps_actual"][t, e] = max(actual_acc[k].get_epsilon(cfg.delta) for k in edge.clients)
            log["eps_declared"][t, e] = max(declared_acc[k].get_epsilon(cfg.delta) for k in edge.clients)
        ew = agg.edge_weights(edge_n, edge_counts)
        theta = theta + sum(ew[e] * edge_updates[e] for e in edge_updates)

        # Candidate evaluation.
        pe = evaluate_split("eval", t)
        log["pred_eval"][t] = pe
        log["pred_test"][t] = evaluate_split("test", t)
        log["pred_ref"][t] = predict(model, theta, ref.x)
        for k in range(K):
            r = rows_of["eval"][k]
            r = r[ev_mask[r]]
            n_rep, c_rep = inj.report(k, t, len(r), counts_of(pe[r], ev.y[r], ev.g[r]))
            reports[k] = ClientReport(n_rep, c_rep)
            log["rep_counts"][t, k] = c_rep
            log["rep_n"][t, k] = n_rep

        x_ref = ref.x[ref_sub]
        x_test = inj.features("test", t, test_sub)
        if data.alt is not None:
            # Image data: the R5 perturbation is the fixed per-image noise at the row's current blur level.
            xp_ref = data.view("reference", ref_sub, 0, 1)
            xp_test = data.view("test", test_sub, inj.levels("test", t, test_sub), 1)
        for name, x, y, z in (("ref", x_ref, ref.y[ref_sub], z_ref), ("test", x_test, te.y[test_sub], z_test)):
            if data.alt is not None:
                xp = xp_ref if name == "ref" else xp_test
            else:
                xp = x.copy()
                xp[:, num_cols] += cfg.rho * z
            log[f"rob_{name}_clean"][t] = predict(model, theta, x) == y
            log[f"rob_{name}_pert"][t] = predict(model, theta, xp) == y
            set_vec(model, theta)
            model.eval()
            a = integrated_gradients(model, torch.from_numpy(x), cfg.ig_steps).numpy()
            grouped = np.zeros((len(x), len(groups)), np.float32)
            np.add.at(grouped.T, gidx, a.T)
            log[f"attr_{name}"][t] = grouped.astype(np.float16)
        log["round_seconds"][t] = time.time() - t_round
        print(f"round {t + 1}/{T} {log['round_seconds'][t]:.1f}s", flush=True)

    static = {
        "eval_client": ev.client, "eval_y": ev.y, "eval_g": ev.g, "audit": data.audit,
        "test_client": te.client, "test_y": te.y, "test_g": te.g,
        "ref_y": ref.y, "ref_g": ref.g,
        "keep_eval": inj.keep["eval"], "keep_test": inj.keep["test"],
        "client_edge": data.client_edge, "test_sub": test_sub, "ref_sub": ref_sub,
    }
    np.savez_compressed(out / "trajectory.npz", **log, **static)
    meta = {
        "config": asdict(cfg), "sigma": sigma.tolist(), "edge_names": data.edge_names, "client_names": data.client_names,
        "feature_groups": groups, "byz": sorted(inj.byz), "liars": sorted(inj.liars),
        "seconds": time.time() - t_start,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    (out / "completion.json").write_text(json.dumps({"complete": True, "seconds": meta["seconds"]}))
    return out
