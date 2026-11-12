"""Hierarchical federated learning: local DP-SGD training, edge aggregation, cloud aggregation.

Learners differ only in how an edge (and the cloud) weights the updates it receives:

- FedAvg: weights proportional to reported sample counts.
- FairFed (Ezzeldin et al., AAAI 2023), reimplemented: weights move away from clients whose local
  statistical-parity value deviates from the aggregate value, with step `beta_ff`.
- EA-FCL-c: fairness-aware adaptive weighting. Entropy and inverse-frequency
  terms come from the edge's audit sample, fairness terms enter as deviations from their fair values, weights are
  floored to stay positive, and Multi-Krum selects updates before the weights are renormalised over the kept set.

Record-level DP: every client trains with DP-SGD (Poisson sampling at rate q, per-example clipping to `clip`,
Gaussian noise with standard deviation `sigma * clip`) and keeps its own RDP accountant. Clients hold disjoint
records, so by parallel composition a patient's privacy loss is that of their own client, and an edge's privacy
statistic is the maximum epsilon over its clients. Aggregation is post-processing and adds no privacy cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from opacus.accountants import RDPAccountant
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def integrated_gradients(model: nn.Module, x: torch.Tensor, n_steps: int = 32) -> torch.Tensor:
    """Integrated Gradients from a zero baseline with Gauss-Legendre quadrature (Captum's default rule).

    The model's outputs are per-example, so one backward pass of the summed output gives every example's
    gradient. Captum unbinds a 1-D output into one tensor per element, which makes its backward pass
    quadratic in the number of examples (about 100 s for 10,000 rows at 32 steps).
    """
    from captum.attr._utils.approximation_methods import approximation_parameters

    step_sizes, alphas = approximation_parameters("gausslegendre")
    total = torch.zeros_like(x)
    for a, w in zip(alphas(n_steps), step_sizes(n_steps)):
        xi = (a * x).detach().requires_grad_(True)
        (g,) = torch.autograd.grad(model(xi).sum(), xi)
        total += w * g
    return total * x


def get_vec(model: nn.Module) -> torch.Tensor:
    return parameters_to_vector(model.parameters()).detach().clone()


def set_vec(model: nn.Module, vec: torch.Tensor) -> None:
    """Load a copy of `vec` into the model. vector_to_parameters makes parameters views of its argument, so
    passing `vec` itself would let training modify the caller's vector."""
    vector_to_parameters(vec.detach().clone(), model.parameters())


def local_train_dp(model: nn.Module, start: torch.Tensor, x: np.ndarray, y: np.ndarray, q: float, steps: int,
                   lr: float, clip_norm: float, sigma: float, accountant: RDPAccountant | None,
                   gen: torch.Generator) -> torch.Tensor:
    """DP-SGD from `start` with Poisson sampling at rate q for `steps` steps; returns the update.

    The client fixes q and steps from its initial data size. The guarantee of Poisson-sampled DP-SGD depends on
    q, sigma and steps only, so it holds when the client's data later shrinks or grows.

    Per-example gradients of the MLP are computed exactly from activations and back-propagated errors. Each
    example's weight gradient in a linear layer is the outer product d_i a_i^T, so its squared norm is
    |d_i|^2 (|a_i|^2 + 1) including the bias, and clipping needs no per-example gradient tensors.
    """
    lins = [m for m in model.net if isinstance(m, nn.Linear)]
    set_vec(model, start)
    W = [l.weight.detach().clone() for l in lins]
    b = [l.bias.detach().clone() for l in lins]
    xt, yt = torch.from_numpy(x), torch.from_numpy(y).float()
    n = len(yt)
    denom = q * n
    for _ in range(steps):
        idx = torch.nonzero(torch.rand(n, generator=gen) < q).squeeze(1)
        gW = [torch.zeros_like(w) for w in W]
        gb = [torch.zeros_like(v) for v in b]
        if len(idx) > 0:
            a0 = xt[idx]
            z1 = a0 @ W[0].T + b[0]
            a1 = torch.relu(z1)
            z2 = a1 @ W[1].T + b[1]
            a2 = torch.relu(z2)
            z3 = (a2 @ W[2].T + b[2]).squeeze(-1)
            d3 = (torch.sigmoid(z3) - yt[idx]).unsqueeze(1)
            d2 = (d3 @ W[2]) * (z2 > 0)
            d1 = (d2 @ W[1]) * (z1 > 0)
            pairs = ((d1, a0), (d2, a1), (d3, a2))
            sq = sum((d * d).sum(1) * ((a * a).sum(1) + 1) for d, a in pairs)
            f = torch.clamp(clip_norm / (sq.sqrt() + 1e-6), max=1.0).unsqueeze(1)
            gW = [(d * f).T @ a for d, a in pairs]
            gb = [(d * f).sum(0) for d, _ in pairs]
        for i in range(3):
            W[i] = W[i] - lr * (gW[i] + torch.randn(W[i].shape, generator=gen) * sigma * clip_norm) / denom
            b[i] = b[i] - lr * (gb[i] + torch.randn(b[i].shape, generator=gen) * sigma * clip_norm) / denom
    if accountant is not None:
        for _ in range(steps):
            accountant.step(noise_multiplier=sigma, sample_rate=q)
    with torch.no_grad():
        for i, l in enumerate(lins):
            l.weight.copy_(W[i])
            l.bias.copy_(b[i])
    return get_vec(model) - start


@torch.no_grad()
def predict(model: nn.Module, vec: torch.Tensor, x: np.ndarray, batch: int = 65536) -> np.ndarray:
    set_vec(model, vec)
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        out.append((model(torch.from_numpy(x[i:i + batch])) > 0).numpy())
    return np.concatenate(out).astype(np.uint8) if out else np.zeros(0, np.uint8)


def clip(update: torch.Tensor, c: float) -> torch.Tensor:
    norm = update.norm().item()
    return update * min(1.0, c / (norm + 1e-12))


def multi_krum(updates: list[torch.Tensor], n_byz: int) -> list[int]:
    """Indices of the m = n - n_byz updates with the lowest Krum scores (Blanchard et al., NeurIPS 2017)."""
    n = len(updates)
    m = max(1, n - n_byz)
    if n <= 2 * n_byz + 2:
        return list(range(n))
    u = torch.stack(updates)
    d = torch.cdist(u, u) ** 2
    closest = n - n_byz - 2
    scores = torch.sort(d, dim=1).values[:, 1:closest + 1].sum(dim=1)
    return torch.argsort(scores)[:m].tolist()


def sp_signed(counts: np.ndarray) -> float:
    """Signed statistical-parity difference P(Yhat=1|G=1) - P(Yhat=1|G=0) from counts [G, Y, Yhat]."""
    n = counts.sum(axis=(1, 2))
    pos = counts[:, :, 1].sum(axis=1)
    if n[0] == 0 or n[1] == 0:
        return 0.0
    return float(pos[1] / n[1] - pos[0] / n[0])


def fairness_terms(counts: np.ndarray) -> tuple[float, float, float]:
    """Deviations from fairness: |1 - DI|, |SP|, EO gap (absolute TPR and FPR gaps, max)."""
    n = counts.sum(axis=(1, 2))
    pos = counts[:, :, 1].sum(axis=1)
    if n.min() == 0:
        return 0.0, 0.0, 0.0
    r1, r0 = pos[1] / n[1], pos[0] / n[0]
    di_dev = abs(1 - r1 / r0) if r0 > 0 else 1.0
    gaps = []
    for y in (1, 0):
        ny = counts[:, y, :].sum(axis=1)
        if ny.min() > 0:
            gaps.append(abs(counts[1, y, 1] / ny[1] - counts[0, y, 1] / ny[0]))
    return float(min(di_dev, 1.0)), float(abs(r1 - r0)), float(max(gaps) if gaps else 0.0)


@dataclass
class LearnerConfig:
    name: str                      # "fedavg" | "fairfed" | "eafclc"
    beta_ff: float = 0.1           # FairFed step (pilot-tuned for 10-35 clients per edge)
    beta: float = 0.5              # EA-FCL-c: entropy vs inverse-frequency balance
    lam: float = 0.5               # EA-FCL-c: weight of |1 - DI|
    mu: float = 0.5                # EA-FCL-c: weight of |SP|
    nu: float = 0.5                # EA-FCL-c: weight of EO gap
    floor: float = 0.05            # EA-FCL-c: minimum raw score before normalisation
    byz_frac: float = 0.2          # EA-FCL-c: Krum's assumed Byzantine fraction per edge


@dataclass
class ClientReport:
    """What a client sends with its update. Liars may misreport every field."""
    n: float
    counts: np.ndarray             # [G, Y, Yhat] of the current global model on the client's eval split


@dataclass
class EdgeState:
    clients: list[int]
    ff_weights: dict[int, float] = field(default_factory=dict)


class Aggregator:
    def __init__(self, cfg: LearnerConfig, global_label_freq: np.ndarray):
        self.cfg = cfg
        self.global_label_freq = global_label_freq     # [C] global class frequencies (public)
        self.cloud_ff: dict[int, float] = {}

    def client_weights(self, edge: EdgeState, reports: dict[int, ClientReport],
                       audit_hist: dict[int, np.ndarray]) -> dict[int, float]:
        ks = edge.clients
        c = self.cfg
        if c.name == "fedavg":
            raw = {k: reports[k].n for k in ks}
        elif c.name == "fairfed":
            pooled = sum(reports[k].counts for k in ks)
            f_glob = sp_signed(pooled)
            if not edge.ff_weights:
                tot = sum(reports[k].n for k in ks)
                edge.ff_weights = {k: reports[k].n / tot for k in ks}
            delta = {k: abs(f_glob - sp_signed(reports[k].counts)) for k in ks}
            mean = np.mean(list(delta.values()))
            raw = {k: max(edge.ff_weights[k] - c.beta_ff * (delta[k] - mean), 1e-6) for k in ks}
            s = sum(raw.values())
            edge.ff_weights = {k: v / s for k, v in raw.items()}
        elif c.name == "eafclc":
            raw = {}
            for k in ks:
                p = audit_hist[k]
                h = float(-(p[p > 0] * np.log(p[p > 0])).sum())
                gamma = float((p / self.global_label_freq).sum())
                di, sp, eo = fairness_terms(reports[k].counts)
                score = c.beta * h + (1 - c.beta) * gamma - c.lam * di - c.mu * sp - c.nu * eo
                raw[k] = max(score, c.floor)
        else:
            raise ValueError(c.name)
        s = sum(raw.values())
        return {k: v / s for k, v in raw.items()}

    def aggregate_edge(self, updates: dict[int, torch.Tensor], weights: dict[int, float]) -> torch.Tensor:
        """Weighted sum of client updates; EA-FCL-c first keeps the Multi-Krum selection."""
        ks = list(updates)
        if self.cfg.name == "eafclc":
            n_byz = int(np.floor(self.cfg.byz_frac * len(ks)))
            keep = [ks[i] for i in multi_krum([updates[k] for k in ks], n_byz)]
            s = sum(weights[k] for k in keep)
            weights = {k: weights[k] / s for k in keep}
            ks = keep
        return sum(weights[k] * updates[k] for k in ks)

    def edge_weights(self, edge_n: dict[int, float], edge_counts: dict[int, np.ndarray]) -> dict[int, float]:
        es = list(edge_n)
        if self.cfg.name == "fairfed":
            pooled = sum(edge_counts[e] for e in es)
            f_glob = sp_signed(pooled)
            if not self.cloud_ff:
                tot = sum(edge_n.values())
                self.cloud_ff = {e: edge_n[e] / tot for e in es}
            delta = {e: abs(f_glob - sp_signed(edge_counts[e])) for e in es}
            mean = np.mean(list(delta.values()))
            raw = {e: max(self.cloud_ff[e] - self.cfg.beta_ff * (delta[e] - mean), 1e-6) for e in es}
            s = sum(raw.values())
            self.cloud_ff = {e: v / s for e, v in raw.items()}
            return dict(self.cloud_ff)
        tot = sum(edge_n.values())
        return {e: edge_n[e] / tot for e in es}
