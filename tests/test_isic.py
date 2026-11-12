import json

import numpy as np
import pytest

from raimon.data import load_isic
from raimon.fl import LearnerConfig
from raimon.injections import Injection, InjectionConfig
from raimon.isic import color_constancy, views


def fake_embeddings(path, n=1800, f=40, seed=0):
    rng = np.random.default_rng(seed)
    centre = rng.integers(0, 6, n)
    y = rng.integers(0, 2, n)
    g = rng.integers(0, 2, n)
    base = rng.standard_normal((n, f)) + y[:, None] * 0.8
    emb = np.zeros((6, 2, n, f), np.float16)
    for level in range(6):
        for noise in range(2):
            emb[level, noise] = base + 0.3 * level + 0.5 * noise    # each view is distinguishable
    sex_known = rng.random(n) > 0.02
    np.savez(path / "embeddings.npz", emb=emb, image=np.array([f"ISIC_{i:07d}" for i in range(n)]),
             center=centre, y=y, g=g, sex_known=sex_known, age60=g, age_known=sex_known)
    return sex_known


def test_loader_topology_and_standardisation(tmp_path):
    known = fake_embeddings(tmp_path)
    d = load_isic(tmp_path)
    assert d.n_edges == 6 and d.n_clients == 18 and d.n_levels == 5
    n_rows = sum(len(s.y) for s in d.splits.values()) + len(d.reference.y)
    assert n_rows == known.sum()                                    # unknown sex dropped, nothing lost
    assert np.allclose(d.reference.x.mean(0), 0, atol=1e-3)        # standardised on the reference set
    rows = np.arange(5)
    assert np.allclose(d.view("train", rows, 0, 0), d.splits["train"].x[rows], atol=1e-5)
    shift = d.view("train", rows, 2, 1) - d.view("train", rows, 0, 0)
    assert np.allclose(shift, (0.3 * 2 + 0.5) / d.alt_sd, atol=1e-2)


def test_v6_ramps_and_v1_targets_protected_group(tmp_path):
    fake_embeddings(tmp_path)
    d = load_isic(tmp_path)
    rows = np.arange(len(d.splits["eval"].y))
    in_e = d.client_edge[d.splits["eval"].client] == 2
    v6 = Injection(InjectionConfig("V6", t0=20, target_edge=2, ramp=20), d, np.random.default_rng(0))
    assert v6.levels("eval", 19, rows).max() == 0
    assert set(v6.levels("eval", 20, rows)[in_e]) == {1}
    assert set(v6.levels("eval", 39, rows)[in_e]) == {5}
    assert v6.levels("eval", 39, rows)[~in_e].max() == 0
    x = v6.features("eval", 39, rows)
    assert np.allclose(x[in_e], d.view("eval", rows[in_e], 5, 0), atol=1e-5)
    lag = Injection(InjectionConfig("V6", t0=20, target_edge=2, ramp=20, shift_train=False), d, np.random.default_rng(0))
    tr_rows = np.arange(len(d.splits["train"].y))
    assert lag.levels("train", 39, tr_rows).max() == 0 and set(lag.levels("eval", 39, rows)[in_e]) == {5}
    v1 = Injection(InjectionConfig("V1", t0=20, target_edge=2, v1_level=3), d, np.random.default_rng(0))
    lv = v1.levels("eval", 25, rows)
    prot = in_e & (d.splits["eval"].g == 1)
    assert set(lv[prot]) == {3} and lv[~prot].max() == 0


def test_views_and_colour_constancy():
    import torch
    grey = np.full((8, 8, 3), 128, np.uint8)
    assert np.abs(color_constancy(grey).astype(int) - 128).max() <= 1
    v = views(torch.rand(2, 3, 32, 32), torch.zeros(2, 3, 32, 32))
    assert len(v) == 12 and {(lv, nf) for lv, nf, _ in v} == {(lv, nf) for lv in range(6) for nf in (0, 1)}


@pytest.mark.parametrize("kind", ["V0", "V6"])
def test_simulate_and_replay_end_to_end(tmp_path, kind):
    from raimon.opa import OpaServer
    from raimon.replay import Setting, Thresholds, Trajectory, replay
    from raimon.simulate import RunConfig, run
    fake_embeddings(tmp_path)
    d = load_isic(tmp_path)
    # An injected run takes its ground truth before t0 from the control (V0) run with the same seed.
    for k in dict.fromkeys(["V0", kind]):
        cfg = RunConfig(dataset="isic", learner=LearnerConfig("fedavg"),
                        injection=InjectionConfig(k, t0=1, target_edge=2, ramp=2), seed=1, rounds=3,
                        lr=0.5, batch=32, n_sub=200, ig_steps=4)
        out = run(cfg, d, tmp_path / "traj" / f"isic__fedavg__{k}__s1")
        assert json.loads((out / "completion.json").read_text())["complete"]
    tr = Trajectory.load(out)
    assert tr.z["attr_test"].shape[2] == 2                          # 40 dims in blocks of 20
    with OpaServer() as opa:
        rows = replay(tr, [Setting(placement="P4"), Setting(placement="P3")], Thresholds(), opa)
    assert len(rows) == 2 and all(r["domain_rounds"] == 3 * 6 for r in rows)
