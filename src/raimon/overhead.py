"""RQ4 overhead: the per-tier work of placement P4, run as separate processes on emulated nodes.

Each edge node (a container with one CPU) does, every round, what P4 places at an edge:
  - evaluates the candidate model on its audit sample (forward pass),
  - R7 trust checks of its clients' reports against the audit counts,
  - R1-R3 checks of its domain on trusted pooled counts,
  - R4: one RDP accountant per client, stepped and queried,
  - OPA evaluation of its checks,
  - one JSON message (domain counts and checks) to the cloud, and waits for the decision.
The cloud node pools the edges' counts, runs the global R1-R3 checks, R5 (clean and perturbed forward passes on
its reference set, paired bootstrap) and R6 (Integrated Gradients on its reference set, bootstrap Spearman
against the previous candidate), and lets OPA decide every domain.

Client counts per round come from a real trajectory (`make-workload`). Model weights and the inputs of forward
passes are synthetic with the real shapes: their cost does not depend on the values.

  raimon overhead make-workload --plan P --trajectory DIR --out W.npz      (host)
  raimon overhead cloud --workload W.npz --port 8080 --out cloud.json      (cloud container)
  raimon overhead edge --workload W.npz --edge e --cloud URL --out e.json  (edge container)
  raimon overhead train --workload W.npz --out train.json                  (client container)

`--arch effnet` (full-model configuration) replaces the head-only MLP with EfficientNet-B0 on 224 x 224
images at every node: client training (one local epoch of non-private SGD, a lower bound on DP-SGD), edge audit
forward passes, the cloud's R5 forward passes and R6 Integrated Gradients on pixels (8 x 8 patch groups).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import requests
import torch

from . import stats
from .fl import MLP, get_vec, integrated_gradients, local_train_dp
from .replay import Thresholds, _chk, _fair_checks, _trust_check


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 if sys.platform != "darwin" else r / 2**20     # Linux reports KiB, macOS bytes


# ------------------------------------------------------------------ workload (host side)

IMG = 224
MICRO = 16          # micro-batch for EfficientNet-B0 (memory of the 2 GB nodes)


def effnet() -> torch.nn.Module:
    from torchvision.models import efficientnet_b0
    return efficientnet_b0(weights=None, num_classes=1)


def images(n: int, seed: int) -> torch.Tensor:
    """Synthetic uint8 images of the real shape (cost does not depend on the values)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n, 3, IMG, IMG), dtype=torch.uint8, generator=g)


def batched_forward(model: torch.nn.Module, x: torch.Tensor, bs: int = 32, noise: float = 0.0) -> torch.Tensor:
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = x[i:i + bs].float() / 255
            if noise:
                xb = (xb + noise * torch.randn(xb.shape, generator=torch.Generator().manual_seed(i))).clamp(0, 1)
            outs.append(model(xb).reshape(-1))
    return torch.cat(outs)


def make_workload(plan_path: str, trajectory: str, out: str, arch: str = "mlp", rounds: int | None = None) -> None:
    from .cli import load_data
    from .replay import Evidence, Trajectory
    plan = json.loads(Path(plan_path).read_text())
    tr = Trajectory.load(Path(trajectory))
    ev = Evidence(tr, 0)
    cfg = tr.meta["config"]
    data = load_data(cfg["dataset"], plan)
    n_train = np.bincount(data.splits["train"].client, minlength=data.n_clients)
    batch = cfg["batch"]
    q = np.minimum(1.0, batch / np.maximum(n_train, 1))
    steps = np.array([cfg["local_epochs"] * max(1, round(int(n) / batch)) for n in n_train])
    T = tr.rounds if rounds is None else rounds
    np.savez(out, rep=np.stack([ev.reported_counts(t) for t in range(T)]).astype(np.float32),
             aud=np.stack([ev.audit_counts(t) for t in range(T)]).astype(np.float32), arch=arch,
             client_edge=tr.edge_of_client(), sigma=np.array(tr.meta["sigma"]), q=q, steps=steps, n_train=n_train,
             n_audit=np.array([len(r) for r in ev.audit_rows]), n_ref=len(tr.z["ref_sub"]),
             n_features=data.splits["train"].x.shape[1], hidden=cfg["hidden"], ig_steps=cfg["ig_steps"],
             n_groups=tr.z["attr_ref"].shape[2], lr=cfg["lr"], batch=batch, clip=cfg["clip"],
             thresholds=json.dumps(plan["thresholds"]), dataset=cfg["dataset"], trajectory=str(trajectory))
    print("wrote", out, "clients", data.n_clients, "edges", data.n_edges, "F", data.splits["train"].x.shape[1])


class Workload:
    def __init__(self, path: str):
        z = np.load(path, allow_pickle=True)
        self.z = z
        self.rep, self.aud, self.client_edge = z["rep"], z["aud"], z["client_edge"]
        self.T, self.K = self.rep.shape[:2]
        self.E = int(self.client_edge.max()) + 1
        self.F, self.H = int(z["n_features"]), int(z["hidden"])
        self.th = Thresholds(**json.loads(str(z["thresholds"])))
        self.arch = str(z["arch"]) if "arch" in z.files else "mlp"

    def model(self, seed: int) -> torch.nn.Module:
        torch.manual_seed(seed)
        return (effnet() if self.arch == "effnet" else MLP(self.F, self.H)).eval()

    def forward(self, model: torch.nn.Module, x: torch.Tensor, noise: float = 0.0) -> torch.Tensor:
        if self.arch == "effnet":
            return batched_forward(model, x, noise=noise)
        with torch.no_grad():
            return model(x if not noise else x + noise * torch.randn_like(x))


class AdditiveRDP:
    """R4 at an edge: each client's per-round RDP curve is computed once; RDP composes additively over rounds,
    so the cumulative curve after round t is (t + 1) times it. Equal to opacus' RDPAccountant (tested)."""

    def __init__(self, sigma: float, q: float, steps: int):
        from opacus.accountants import RDPAccountant
        from opacus.accountants.analysis.rdp import compute_rdp
        self.orders = RDPAccountant.DEFAULT_ALPHAS
        self.per_round = np.asarray(compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=self.orders))
        self.rounds = 0

    def step_round(self) -> None:
        self.rounds += 1

    def epsilon(self, delta: float) -> float:
        from opacus.accountants.analysis.rdp import get_privacy_spent
        eps, _ = get_privacy_spent(orders=self.orders, rdp=self.rounds * self.per_round, delta=delta)
        return float(eps)


def _summary(xs: list[float]) -> dict:
    a = np.asarray(xs, float)
    return {"median": float(np.median(a)), "p95": float(np.quantile(a, 0.95)), "max": float(a.max()), "n": len(a)}


# ------------------------------------------------------------------ edge

def edge_main(workload: str, edge: int, cloud: str, out: str, opa_bin: str | None = None) -> None:
    from .opa import OpaServer
    w = Workload(workload)
    th = w.th
    clients = np.flatnonzero(w.client_edge == edge)
    acc = {int(k): AdditiveRDP(float(w.z["sigma"][k]), float(w.z["q"][k]), int(w.z["steps"][k])) for k in clients}
    update = b"\0" * (4 * sum(p.numel() for p in w.model(0).parameters()))   # the edge model, fp32
    n_audit = int(w.z["n_audit"][clients].sum())
    x_audit = images(n_audit, edge) if w.arch == "effnet" else torch.randn(n_audit, w.F)
    timeout = 14400 if w.arch == "effnet" else 600
    sess = requests.Session()
    rec: dict[str, list[float]] = {k: [] for k in ("audit_forward_ms", "checks_ms", "accountant_ms", "opa_ms",
                                                    "edge_compute_ms", "message_bytes", "model_upload_ms", "rtt_ms",
                                                    "checks")}
    with OpaServer(opa_bin) as opa:
        for t in range(w.T):
            model = w.model(t)
            t0 = time.perf_counter()
            w.forward(model, x_audit)
            t1 = time.perf_counter()
            rep, aud = w.rep[t], w.aud[t]
            checks, trusted = [], []
            for k in clients:
                c = _trust_check(int(k), f"e{edge}", rep[k], aud[k], th, False)
                if c is not None:
                    checks.append(c)
                trusted.append(c is None or c["lower"] <= th.trust_tol)
            counts = rep[clients[np.array(trusted, bool)]].sum(0)
            checks += _fair_checks(f"e{edge}", f"e{edge}", counts, th, False)
            t2 = time.perf_counter()
            eps = 0.0
            for k in clients:
                acc[int(k)].step_round()
                eps = max(eps, acc[int(k)].epsilon(1e-5))
            checks.append(_chk(f"R4/e{edge}", f"e{edge}", "upper", th.eps_max, stats.Interval(eps, eps, eps), True))
            t3 = time.perf_counter()
            opa.evaluate([f"e{edge}"], checks)
            t4 = time.perf_counter()
            # The edge model upload that every round needs anyway, on the same link (denominator of the ratio).
            sess.post(f"{cloud}/model", data=update, timeout=timeout).raise_for_status()
            t4b = time.perf_counter()
            body = json.dumps({"round": t, "edge": edge, "counts": counts.tolist(), "checks": checks}).encode()
            r = sess.post(f"{cloud}/report", data=body, headers={"Content-Type": "application/json"}, timeout=timeout)
            r.raise_for_status()
            t5 = time.perf_counter()
            rec["audit_forward_ms"].append(1e3 * (t1 - t0))
            rec["checks_ms"].append(1e3 * (t2 - t1))
            rec["accountant_ms"].append(1e3 * (t3 - t2))
            rec["opa_ms"].append(1e3 * (t4 - t3))
            rec["edge_compute_ms"].append(1e3 * (t4 - t0))
            rec["message_bytes"].append(len(body))
            rec["model_upload_ms"].append(1e3 * (t4b - t4))
            rec["rtt_ms"].append(1e3 * (t5 - t4b))      # includes waiting for the slowest edge and the cloud
            rec["checks"].append(len(checks))
    result = {"edge": edge, "clients": len(clients), "audit_rows": len(x_audit), "peak_rss_mb": peak_rss_mb(),
              "model_bytes": len(update),
              "rounds": w.T, "raw": rec, **{k: _summary(v) for k, v in rec.items()}}
    Path(out).write_text(json.dumps(result, indent=1))


# ------------------------------------------------------------------ cloud

def cloud_main(workload: str, port: int, out: str, opa_bin: str | None = None) -> None:
    from .opa import OpaServer
    w = Workload(workload)
    th = w.th
    n_ref, n_steps, n_groups = int(w.z["n_ref"]), int(w.z["ig_steps"]), int(w.z["n_groups"])
    if w.arch == "effnet":
        x_ref = images(n_ref, 10_000)
    else:
        x_ref = torch.randn(n_ref, w.F)
        x_pert = x_ref + 0.5 * torch.randn(n_ref, w.F)
        gidx = np.arange(w.F) * n_groups // w.F
    y_ref = torch.randint(0, 2, (n_ref,))
    rng = np.random.default_rng(0)
    rec: dict[str, list[float]] = {k: [] for k in ("global_checks_ms", "r5_ms", "r6_ms", "opa_ms", "cloud_compute_ms",
                                                    "wait_for_edges_ms", "reply_bytes")}
    state = {"round": 0, "msgs": {}, "decision": None, "prev_attr": None, "t_first": None}
    cond = threading.Condition()
    opa = OpaServer(opa_bin)
    done = threading.Event()

    def decide(t: int, msgs: dict) -> bytes:
        model = w.model(t)
        c0 = time.perf_counter()
        counts = np.sum([np.array(m["counts"]) for m in msgs.values()], axis=0)
        checks = [c for m in msgs.values() for c in m["checks"]]
        checks += _fair_checks("global", "*", counts, th, False)
        c1 = time.perf_counter()
        if w.arch == "effnet":
            clean = ((w.forward(model, x_ref) > 0).long() == y_ref).float().numpy()
            pert = ((w.forward(model, x_ref, noise=0.01) > 0).long() == y_ref).float().numpy()
        else:
            with torch.no_grad():
                clean = ((model(x_ref) > 0).long() == y_ref).float().numpy()
                pert = ((model(x_pert) > 0).long() == y_ref).float().numpy()
        r5 = stats.bootstrap_mean_diff(clean, pert, rng)
        checks.append(_chk("R5/global", "*", "upper", th.rob_drop, r5, False))
        c2 = time.perf_counter()
        if w.arch == "effnet":
            side = IMG // 8
            parts = []
            for i in range(0, n_ref, 8):
                a = integrated_gradients(model, x_ref[i:i + 8].float() / 255, n_steps).sum(1)     # [b, H, W]
                parts.append(a.reshape(len(a), 8, side, 8, side).sum((2, 4)).reshape(len(a), 64).numpy())
            grouped = np.concatenate(parts).astype(np.float32)
        else:
            a = integrated_gradients(model, x_ref, n_steps).numpy()
            grouped = np.zeros((n_ref, n_groups), np.float32)
            np.add.at(grouped.T, gidx, a.T)
        if state["prev_attr"] is not None:
            r6 = stats.bootstrap_spearman(grouped, state["prev_attr"], rng, resamples=300)
            checks.append(_chk("R6/global", "*", "lower", th.attr_min, r6, False))
        state["prev_attr"] = grouped
        c3 = time.perf_counter()
        res = opa.evaluate([f"e{e}" for e in range(w.E)], checks)
        c4 = time.perf_counter()
        reply = json.dumps({"round": t, "decision": res["decision"]}).encode()
        rec["global_checks_ms"].append(1e3 * (c1 - c0))
        rec["r5_ms"].append(1e3 * (c2 - c1))
        rec["r6_ms"].append(1e3 * (c3 - c2))
        rec["opa_ms"].append(1e3 * (c4 - c3))
        rec["cloud_compute_ms"].append(1e3 * (c4 - c0))
        rec["reply_bytes"].append(len(reply))
        return reply

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            if self.path == "/model":
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with cond:
                t = msg["round"]
                if not state["msgs"]:
                    state["t_first"] = time.perf_counter()
                state["msgs"][msg["edge"]] = msg
                if len(state["msgs"]) == w.E:
                    rec["wait_for_edges_ms"].append(1e3 * (time.perf_counter() - state["t_first"]))
                    state["decision"] = decide(t, state["msgs"])
                    state["msgs"], state["round"] = {}, t + 1
                    cond.notify_all()
                else:
                    cond.wait_for(lambda: state["round"] > t)
                reply = state["decision"]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)
            if state["round"] >= w.T:
                done.set()

    class Server(ThreadingHTTPServer):
        request_queue_size = 256      # every edge's request stays open until the round is decided
        daemon_threads = True

    srv = Server(("0.0.0.0", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"cloud listening on {port}", flush=True)
    done.wait()
    time.sleep(1)
    srv.shutdown()
    opa.close()
    result = {"edges": w.E, "n_ref": n_ref, "peak_rss_mb": peak_rss_mb(), "rounds": w.T, "raw": rec,
              **{k: _summary(v) for k, v in rec.items()}}
    Path(out).write_text(json.dumps(result, indent=1))


# ------------------------------------------------------------------ client training baseline

def train_main(workload: str, out: str, rounds: int = 10) -> None:
    """One client's local DP-SGD round on synthetic data with the largest client's shape (the critical path)."""
    from opacus.accountants import RDPAccountant
    w = Workload(workload)
    torch.set_num_threads(1)
    k = int(np.argmax(w.z["n_train"]))
    n = int(w.z["n_train"][k])
    if w.arch == "effnet":
        train_effnet(w, n, out)
        return
    x, y = torch.randn(n, w.F).numpy(), np.random.default_rng(0).integers(0, 2, n)
    model = w.model(0)
    theta = get_vec(model)
    gen = torch.Generator().manual_seed(0)
    acc = RDPAccountant()
    secs = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        local_train_dp(model, theta, x, y, float(w.z["q"][k]), int(w.z["steps"][k]), float(w.z["lr"]),
                       float(w.z["clip"]), float(w.z["sigma"][k]), acc, gen)
        secs.append(1e3 * (time.perf_counter() - t0))
    n_params = int(theta.numel())
    Path(out).write_text(json.dumps({"client_rows": n, "steps": int(w.z["steps"][k]), "n_params": n_params,
                                     "update_bytes_fp32": 4 * n_params, "train_ms": _summary(secs), "raw": secs,
                                     "peak_rss_mb": peak_rss_mb()}, indent=1))


def train_effnet(w: Workload, n: int, out: str) -> None:
    """One local epoch of non-private SGD over n images, batches of `batch` built from micro-batches of MICRO."""
    batch = int(w.z["batch"])
    model = effnet().train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    t0 = time.perf_counter()
    steps = 0
    for b in range(0, n, batch):
        opt.zero_grad()
        m = min(batch, n - b)
        for i in range(0, m, MICRO):
            xb = images(min(MICRO, m - i), b + i).float() / 255
            yb = torch.randint(0, 2, (len(xb), 1)).float()
            (loss_fn(model(xb), yb) * len(xb) / m).backward()
        opt.step()
        steps += 1
    secs = [1e3 * (time.perf_counter() - t0)]
    n_params = sum(p.numel() for p in model.parameters())
    Path(out).write_text(json.dumps({"client_rows": n, "steps": steps, "n_params": n_params, "private": False,
                                     "update_bytes_fp32": 4 * n_params, "train_ms": _summary(secs), "raw": secs,
                                     "peak_rss_mb": peak_rss_mb()}, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="raimon overhead")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-workload")
    m.add_argument("--plan", required=True)
    m.add_argument("--trajectory", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--arch", default="mlp", choices=["mlp", "effnet"])
    m.add_argument("--rounds", type=int)
    c = sub.add_parser("cloud")
    c.add_argument("--workload", required=True)
    c.add_argument("--port", type=int, default=8080)
    c.add_argument("--out", required=True)
    e = sub.add_parser("edge")
    e.add_argument("--workload", required=True)
    e.add_argument("--edge", type=int, required=True)
    e.add_argument("--cloud", required=True)
    e.add_argument("--out", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--workload", required=True)
    tr.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("RAIMON_THREADS", "1")))
    if a.cmd == "make-workload":
        make_workload(a.plan, a.trajectory, a.out, a.arch, a.rounds)
    elif a.cmd == "cloud":
        cloud_main(a.workload, a.port, a.out)
    elif a.cmd == "edge":
        edge_main(a.workload, a.edge, a.cloud, a.out)
    else:
        train_main(a.workload, a.out)


if __name__ == "__main__":
    main()
