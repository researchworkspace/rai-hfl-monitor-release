import numpy as np
from opacus.accountants import RDPAccountant

from raimon.overhead import AdditiveRDP


def test_additive_rdp_matches_opacus_accountant():
    for sigma, q, steps in ((1.1, 0.05, 27), (0.8, 1.0, 1), (2.5, 0.3, 4)):
        ours, ref = AdditiveRDP(sigma, q, steps), RDPAccountant()
        for _ in range(12):
            ours.step_round()
            for _ in range(steps):
                ref.step(noise_multiplier=sigma, sample_rate=q)
        assert np.isclose(ours.epsilon(1e-5), ref.get_epsilon(1e-5), rtol=1e-9)


def test_effnet_nodes_end_to_end(tmp_path):
    """D2-full path: cloud, one edge and the client baseline on a tiny EfficientNet-B0 workload."""
    import json
    import socket
    import threading

    from raimon.overhead import cloud_main, edge_main, train_main
    rng = np.random.default_rng(0)
    T, K = 2, 2
    counts = rng.integers(5, 30, (T, K, 2, 2, 2)).astype(np.float32)
    wl = tmp_path / "w.npz"
    np.savez(wl, rep=counts, aud=counts, arch="effnet", client_edge=np.zeros(K, int), sigma=np.ones(K), q=np.full(K, 0.5),
             steps=np.ones(K, int), n_train=np.array([3, 2]), n_audit=np.array([2, 1]), n_ref=4, n_features=1280,
             hidden=8, ig_steps=2, n_groups=64, lr=0.5, batch=2, clip=1.0,
             thresholds=json.dumps({"sp": 0.26, "eo": 0.25, "acc_min": 0.6}), dataset="isic", trajectory="test")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    th = threading.Thread(target=cloud_main, args=(str(wl), port, str(tmp_path / "cloud.json")), daemon=True)
    th.start()
    import time
    time.sleep(1.0)
    edge_main(str(wl), 0, f"http://127.0.0.1:{port}", str(tmp_path / "edge0.json"))
    th.join(timeout=120)
    train_main(str(wl), str(tmp_path / "train.json"))
    cloud = json.loads((tmp_path / "cloud.json").read_text())
    edge = json.loads((tmp_path / "edge0.json").read_text())
    train = json.loads((tmp_path / "train.json").read_text())
    assert len(cloud["raw"]["r6_ms"]) == T and cloud["n_ref"] == 4
    assert edge["model_bytes"] > 15e6 and edge["audit_rows"] == 3          # EfficientNet-B0 in fp32
    assert train["client_rows"] == 3 and train["steps"] == 2 and train["private"] is False
