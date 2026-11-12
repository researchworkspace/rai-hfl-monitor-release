"""Violations injected at round t0 with known ground truth.

V0 none. V1 per-domain fairness drift. V2 Simpson construction. V3 targeted poisoning. V4 lying clients.
V5 one edge halves its noise multiplier. V6 gradual acquisition shift in one domain.

All random choices are drawn once from the run's generator, so the same rows are affected in every round.
Row removal (V1, V2, V4) applies to all three splits so that monitors and ground truth see the same population.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import FedData


def fabricate(counts: np.ndarray) -> np.ndarray:
    """A liar's report: group 0's prediction rate within each label is copied onto group 1."""
    fake = counts.astype(float).copy()
    for y in (0, 1):
        n1 = fake[1, y].sum()
        n0 = fake[0, y].sum()
        if n0 > 0 and n1 > 0:
            fake[1, y, 1] = np.round(fake[0, y, 1] / n0 * n1)
            fake[1, y, 0] = n1 - fake[1, y, 1]
    return fake


@dataclass
class InjectionConfig:
    kind: str = "V0"
    t0: int = 20
    target_edge: int = 0                  # V1, V5, V6
    drop_frac: float = 0.7                # V1, V4: share of (G=1, Y=1) rows removed
    shift_std: float = 1.0                # V1: feature noise on protected-group rows of the target domain
    simpson_frac: float = 0.6             # V2: share of rows removed to skew group composition
    simpson_label_frac: float = 0.5       # V2: share of rows removed to separate domain base rates
    byz_frac: float = 0.1                 # V3: share of clients that poison
    liar_frac: float = 0.1                # V4: share of clients that misreport
    n_inflation: float = 3.0              # V4: factor on reported sample counts
    sigma_scale: float = 0.5              # V5
    noise_max: float = 2.0                # V6: final noise std on numeric features
    ramp: int = 20                        # V6: rounds to reach noise_max
    numeric_cols: tuple[int, ...] = (0, 1, 2)
    v1_level: int = 3                     # image data: blur level for V1's protected-group rows
    shift_train: bool = True              # V1/V6 feature shifts also reach training inputs; False models label
                                          # lag: shifted inputs reach deployment before they are labelled


@dataclass
class Injection:
    cfg: InjectionConfig
    data: FedData
    rng: np.random.Generator
    keep: dict[str, np.ndarray] = field(default_factory=dict)        # rows kept from t0 on
    z: dict[str, np.ndarray] = field(default_factory=dict)           # fixed noise draws
    byz: set[int] = field(default_factory=set)
    liars: set[int] = field(default_factory=set)

    def __post_init__(self):
        c, d = self.cfg, self.data
        edge_of = d.client_edge
        for s, sp in d.splits.items():
            self.keep[s] = np.ones(len(sp.y), bool)
        if c.kind == "V1":
            for s, sp in d.splits.items():
                in_e = edge_of[sp.client] == c.target_edge
                u = self.rng.random(len(sp.y))
                self.keep[s] &= ~(in_e & (sp.g == 1) & (sp.y == 1) & (u < c.drop_frac))
                if d.alt is None:
                    self.z[s] = self.rng.standard_normal((len(sp.y), len(c.numeric_cols))).astype(np.float32)
        elif c.kind == "V2":
            # Base rate of each edge from the training split decides where each group is thinned.
            tr = d.splits["train"]
            rate = np.array([tr.y[edge_of[tr.client] == e].mean() for e in range(d.n_edges)])
            high = rate >= np.median(rate)
            for s, sp in d.splits.items():
                e = edge_of[sp.client]
                u = self.rng.random(len(sp.y))
                v = self.rng.random(len(sp.y))
                # Widen the gap between high- and low-rate domains, then put group 1 in the low-rate ones.
                drop_y = (high[e] & (sp.y == 0)) | (~high[e] & (sp.y == 1))
                drop_g = (high[e] & (sp.g == 1)) | (~high[e] & (sp.g == 0))
                self.keep[s] &= ~(drop_y & (v < c.simpson_label_frac)) & ~(drop_g & (u < c.simpson_frac))
        elif c.kind == "V3":
            n_byz = max(1, int(round(c.byz_frac * d.n_clients)))
            self.byz = set(self.rng.choice(d.n_clients, n_byz, replace=False).tolist())
        elif c.kind == "V4":
            n_liar = max(1, int(round(c.liar_frac * d.n_clients)))
            self.liars = set(self.rng.choice(d.n_clients, n_liar, replace=False).tolist())
            for s, sp in d.splits.items():
                liar_row = np.isin(sp.client, list(self.liars))
                u = self.rng.random(len(sp.y))
                self.keep[s] &= ~(liar_row & (sp.g == 1) & (sp.y == 1) & (u < c.drop_frac))
        elif c.kind == "V6":
            for s, sp in d.splits.items():
                if d.alt is None:
                    self.z[s] = self.rng.standard_normal((len(sp.y), len(c.numeric_cols))).astype(np.float32)
        elif c.kind not in ("V0", "V5"):
            raise ValueError(c.kind)

    def active(self, t: int) -> bool:
        return self.cfg.kind != "V0" and t >= self.cfg.t0

    def mask(self, split: str, t: int) -> np.ndarray:
        if not self.active(t):
            return np.ones(len(self.data.splits[split].y), bool)
        return self.keep[split]

    def levels(self, split: str, t: int, rows: np.ndarray) -> np.ndarray:
        """Image data: corruption level of each row (0 = clean). V1 blurs the protected group of the target
        domain at a fixed level; V6 blurs the whole target domain at a level that rises over `ramp` rounds."""
        sp, c = self.data.splits[split], self.cfg
        lv = np.zeros(len(rows), np.int64)
        if not self.active(t) or c.kind not in ("V1", "V6") or (split == "train" and not c.shift_train):
            return lv
        in_e = self.data.client_edge[sp.client[rows]] == c.target_edge
        if c.kind == "V1":
            lv[in_e & (sp.g[rows] == 1)] = c.v1_level
        else:
            L = self.data.n_levels
            lv[in_e] = int(np.ceil(L * min(1.0, (t - c.t0 + 1) / c.ramp)))
        return lv

    def features(self, split: str, t: int, rows: np.ndarray) -> np.ndarray:
        sp = self.data.splits[split]
        c = self.cfg
        if self.data.alt is not None:
            if not self.active(t) or c.kind not in ("V1", "V6"):
                return sp.x[rows]
            return self.data.view(split, rows, self.levels(split, t, rows), 0)
        x = sp.x[rows]
        if not self.active(t) or c.kind not in ("V1", "V6") or (split == "train" and not c.shift_train):
            return x
        x = x.copy()
        cols = list(c.numeric_cols)
        if c.kind == "V1":
            sel = (self.data.client_edge[sp.client[rows]] == c.target_edge) & (sp.g[rows] == 1)
            x[np.ix_(sel, cols)] += c.shift_std * self.z[split][rows[sel]]
        else:
            level = c.noise_max * min(1.0, (t - c.t0 + 1) / c.ramp)
            sel = self.data.client_edge[sp.client[rows]] == c.target_edge
            x[np.ix_(sel, cols)] += level * self.z[split][rows[sel]]
        return x

    def train_labels(self, k: int, t: int, y: np.ndarray, g: np.ndarray) -> np.ndarray:
        if self.active(t) and self.cfg.kind == "V3" and k in self.byz:
            y = y.copy()
            y[g == 1] = 0
        return y

    def report(self, k: int, t: int, n: float, counts: np.ndarray) -> tuple[float, np.ndarray]:
        """Reported sample count and counts. Liars copy group 0's prediction rates onto group 1."""
        if not (self.active(t) and self.cfg.kind == "V4" and k in self.liars):
            return n, counts
        fake = fabricate(counts)
        return n * self.cfg.n_inflation, fake

    def sigma_scale(self, e: int, t: int) -> float:
        if self.active(t) and self.cfg.kind == "V5" and e == self.cfg.target_edge:
            return self.cfg.sigma_scale
        return 1.0
