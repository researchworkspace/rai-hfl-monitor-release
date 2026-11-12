"""Federated datasets with a client -> edge -> cloud topology.

A `FedData` holds one array per split for all clients, with an integer client index per row. Clients belong to
exactly one edge. The cloud reference set is drawn from the pooled data before partitioning and is excluded
from every client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# US Census divisions over the 50 states. DC is not in folktables; Puerto Rico is excluded.
CENSUS_DIVISIONS: dict[str, list[str]] = {
    "new_england": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "middle_atlantic": ["NJ", "NY", "PA"],
    "east_north_central": ["IL", "IN", "MI", "OH", "WI"],
    "west_north_central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "south_atlantic": ["DE", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "east_south_central": ["AL", "KY", "MS", "TN"],
    "west_south_central": ["AR", "LA", "OK", "TX"],
    "mountain": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "pacific": ["AK", "CA", "HI", "OR", "WA"],
}

SPLITS = ("train", "eval", "test")


@dataclass
class Split:
    x: np.ndarray          # [N, F] float32
    y: np.ndarray          # [N] int64, binary label
    g: np.ndarray          # [N] int64, sensitive group (1 = protected)
    client: np.ndarray     # [N] int64, client index


@dataclass
class FedData:
    name: str
    splits: dict[str, Split]
    reference: Split                   # cloud reference set (client = -1)
    client_edge: np.ndarray            # [K] edge index of each client
    edge_names: list[str]
    client_names: list[str]
    audit: np.ndarray                  # [N_eval] bool, rows of the eval split the edge holds as audit sample
    feature_names: list[str] = field(default_factory=list)
    # Image datasets: precomputed embeddings under corruption views, alt[split or "reference"][level, noise, row].
    alt: dict[str, np.ndarray] | None = None
    alt_mu: np.ndarray | None = None
    alt_sd: np.ndarray | None = None

    @property
    def n_levels(self) -> int:
        return 0 if self.alt is None else self.alt["train"].shape[0] - 1

    def view(self, split: str, rows: np.ndarray, level: np.ndarray | int = 0, noise: int = 0) -> np.ndarray:
        """Standardised embeddings of `rows` at the given corruption level (per row) and noise flag."""
        assert self.alt is not None
        x = self.alt[split][level, noise, rows].astype(np.float32)
        return (x - self.alt_mu) / self.alt_sd

    @property
    def n_clients(self) -> int:
        return len(self.client_edge)

    @property
    def n_edges(self) -> int:
        return len(self.edge_names)

    def rows(self, split: str, k: int) -> np.ndarray:
        return np.flatnonzero(self.splits[split].client == k)


def _encode_acs(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Numeric and one-hot encoding of the ten standard ACSIncome features (SEX and RAC1P included)."""
    num = pd.DataFrame({
        "AGEP": df["AGEP"], "WKHP": df["WKHP"], "SCHL": df["SCHL"],
    }).astype(float)
    num = (num - num.mean()) / num.std()
    cats = pd.DataFrame({
        "COW": df["COW"].astype(int),
        "MAR": df["MAR"].astype(int),
        "RELP": df["RELP"].astype(int),
        "OCCP": (df["OCCP"] // 100).astype(int),        # occupation major group
        "POBP": np.where(df["POBP"] < 100, 0, df["POBP"] // 100).astype(int),  # US-born or world region
        "SEX": df["SEX"].astype(int),
        "RAC1P": df["RAC1P"].astype(int),
    })
    onehot = pd.get_dummies(cats.astype(str), prefix=cats.columns.tolist()).astype(float)
    x = pd.concat([num, onehot], axis=1)
    return x.to_numpy(np.float32), x.columns.tolist()


def load_acs(root: Path, year: str = "2018", shard_rows: int = 10_000, reference_frac: float = 0.02,
             audit_frac: float = 0.10, seed: int = 0, states: list[str] | None = None) -> FedData:
    """ACS Income. Each state is split into shards of at most `shard_rows` rows; each shard is a client.

    Edges are the nine census divisions. The sensitive group is SEX (1 = female).
    """
    from folktables import ACSDataSource

    rng = np.random.default_rng(seed)
    division_of = {s: d for d, ss in CENSUS_DIVISIONS.items() for s in ss}
    wanted = states or sorted(division_of)
    source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person", root_dir=str(root))
    raw = source.get_data(states=wanted, download=True)
    df = raw[(raw["AGEP"] > 16) & (raw["PINCP"] > 100) & (raw["WKHP"] > 0) & (raw["PWGTP"] >= 1)]
    df = df.reset_index(drop=True)
    st_codes = _state_fips_to_abbr()
    state = df["ST"].astype(int).map(st_codes).to_numpy()
    keep = np.isin(state, wanted)
    df, state = df[keep].reset_index(drop=True), state[keep]

    x, names = _encode_acs(df)
    y = (df["PINCP"].to_numpy() > 50_000).astype(np.int64)
    g = (df["SEX"].to_numpy().astype(int) == 2).astype(np.int64)

    # Cloud reference set, drawn before partitioning.
    is_ref = rng.random(len(df)) < reference_frac
    ref = Split(x[is_ref], y[is_ref], g[is_ref], np.full(is_ref.sum(), -1))

    edges = [d for d in CENSUS_DIVISIONS if any(s in wanted for s in CENSUS_DIVISIONS[d])]
    client_edge, client_names = [], []
    split_rows: dict[str, list[np.ndarray]] = {s: [] for s in SPLITS}
    split_client: dict[str, list[np.ndarray]] = {s: [] for s in SPLITS}
    for e, div in enumerate(edges):
        for st in CENSUS_DIVISIONS[div]:
            if st not in wanted:
                continue
            rows = np.flatnonzero((state == st) & ~is_ref)
            rng.shuffle(rows)
            n_shards = max(1, int(np.ceil(len(rows) / shard_rows)))
            for j, shard in enumerate(np.array_split(rows, n_shards)):
                k = len(client_edge)
                client_edge.append(e)
                client_names.append(f"{st}-{j}")
                n = len(shard)
                cut1, cut2 = int(0.70 * n), int(0.85 * n)
                for s, part in zip(SPLITS, (shard[:cut1], shard[cut1:cut2], shard[cut2:])):
                    split_rows[s].append(part)
                    split_client[s].append(np.full(len(part), k))

    splits = {}
    for s in SPLITS:
        rows = np.concatenate(split_rows[s])
        splits[s] = Split(x[rows], y[rows], g[rows], np.concatenate(split_client[s]))
    audit = rng.random(len(splits["eval"].y)) < audit_frac
    return FedData("acs", splits, ref, np.array(client_edge), edges, client_names, audit, names)


def load_isic(root: Path, clients_per_centre: int = 3, reference_frac: float = 0.02, audit_frac: float = 0.10,
              seed: int = 0, group: str = "sex", embeddings: str = "embeddings.npz") -> FedData:
    """Fed-ISIC2019 (FLamby centres) on frozen-backbone embeddings (see `isic.py`).

    Each of the six centres is an edge, split at random into `clients_per_centre` clients. The label is
    malignant (MEL, BCC, SCC). The sensitive group is female (group="sex") or age >= 60 (group="age");
    images without that attribute are dropped. Features are standardised with the reference set's statistics.
    """
    from .isic import CENTRES

    z = np.load(Path(root) / embeddings, allow_pickle=True)
    known = z["sex_known"] if group == "sex" else z["age_known"]
    idx = np.flatnonzero(known)
    emb = z["emb"][:, :, idx]
    y = z["y"][idx].astype(np.int64)
    g = (z["g"] if group == "sex" else z["age60"])[idx].astype(np.int64)
    centre = z["center"][idx].astype(int)
    rng = np.random.default_rng(seed)
    is_ref = rng.random(len(idx)) < reference_frac

    client_edge, client_names = [], []
    split_rows: dict[str, list[np.ndarray]] = {s: [] for s in SPLITS}
    split_client: dict[str, list[np.ndarray]] = {s: [] for s in SPLITS}
    for e in range(len(CENTRES)):
        rows = np.flatnonzero((centre == e) & ~is_ref)
        rng.shuffle(rows)
        for j, part_rows in enumerate(np.array_split(rows, clients_per_centre)):
            k = len(client_edge)
            client_edge.append(e)
            client_names.append(f"{CENTRES[e]}-{j}")
            n = len(part_rows)
            cut1, cut2 = int(0.70 * n), int(0.85 * n)
            for s, part in zip(SPLITS, (part_rows[:cut1], part_rows[cut1:cut2], part_rows[cut2:])):
                split_rows[s].append(part)
                split_client[s].append(np.full(len(part), k))

    ref_rows = np.flatnonzero(is_ref)
    clean_ref = emb[0, 0, ref_rows].astype(np.float32)
    mu, sd = clean_ref.mean(0), clean_ref.std(0) + 1e-6
    alt, splits = {"reference": emb[:, :, ref_rows]}, {}
    for s in SPLITS:
        rows = np.concatenate(split_rows[s])
        alt[s] = emb[:, :, rows]
        x = (alt[s][0, 0].astype(np.float32) - mu) / sd
        splits[s] = Split(x, y[rows], g[rows], np.concatenate(split_client[s]))
    ref = Split((clean_ref - mu) / sd, y[ref_rows], g[ref_rows], np.full(len(ref_rows), -1))
    audit = rng.random(len(splits["eval"].y)) < audit_frac
    names = [f"emb{i // 20:02d}_{i}" for i in range(emb.shape[-1])]
    return FedData("isic", splits, ref, np.array(client_edge), list(CENTRES), client_names, audit, names,
                   alt=alt, alt_mu=mu, alt_sd=sd)


def _state_fips_to_abbr() -> dict[int, str]:
    return {
        1: "AL", 2: "AK", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE", 11: "DC", 12: "FL",
        13: "GA", 15: "HI", 16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA",
        23: "ME", 24: "MD", 25: "MA", 26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE",
        32: "NV", 33: "NH", 34: "NJ", 35: "NM", 36: "NY", 37: "NC", 38: "ND", 39: "OH", 40: "OK",
        41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD", 47: "TN", 48: "TX", 49: "UT", 50: "VT",
        51: "VA", 53: "WA", 54: "WV", 55: "WI", 56: "WY", 72: "PR",
    }
