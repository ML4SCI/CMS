"""End-to-end check on small synthetic data (no download, runs on CPU in minutes).

    python tests/smoke_test.py --root /tmp/superres_smoke [--dataset qg|calo|both]

Writes fake raw files in the formats the real loaders expect, then runs every
pipeline stage with the SMOKE config.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fake_qg(raw, n=700, seed=0):
    import torch
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = np.zeros((n, 125, 125, 3), np.float32)  # channel-last on purpose
    yy, xx = np.mgrid[:125, :125]
    for i in range(n):
        k = 3 + 4 * (1 - y[i])  # "gluons" have more, wider deposits
        for _ in range(k):
            cy, cx = rng.normal(62, 6 + 4 * (1 - y[i]), 2)
            w = rng.uniform(1, 3)
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * w ** 2))
            X[i, :, :, rng.integers(0, 3)] += rng.exponential(5) * blob * (blob > 0.05)
    os.makedirs(raw, exist_ok=True)
    torch.save({"X_jets": torch.from_numpy(X), "y": torch.from_numpy(y)}, os.path.join(raw, "fake_jets.pt"))


def fake_calo(raw, n=(2000, 500), seed=0):
    import h5py
    rng = np.random.default_rng(seed)
    os.makedirs(raw, exist_ok=True)
    layer = np.arange(45)[:, None, None]
    r = np.arange(9)[None, None, :]
    for name, m in zip(("dataset_2_1.hdf5", "dataset_2_2.hdf5"), n):
        e = np.exp(rng.uniform(np.log(1e3), np.log(1e6), (m, 1)))
        depth = 8 + 2 * np.log10(e[:, 0] / 1e3)
        showers = np.zeros((m, 45, 16, 9), np.float32)
        for i in range(m):
            prof = np.exp(-0.5 * ((layer - depth[i]) / 5) ** 2) * np.exp(-r / 1.5) * rng.gamma(2, 0.5, (1, 16, 1))
            showers[i] = e[i, 0] * 0.9 * prof / prof.sum() * rng.lognormal(0, 0.3, prof.shape)
        showers[showers < 0.0151] = 0
        with h5py.File(os.path.join(raw, name), "w") as f:
            f["showers"] = showers.reshape(m, -1)
            f["incident_energies"] = e


def check_taggers():
    """ResNet-18 on HR (128) and LR (64) shapes; a config without tagger.arch means the small CNN."""
    import torch
    from superres.models import TAGGERS, ResNet18Tagger, Tagger
    for shape in [(128, 128), (64, 64)]:
        m = ResNet18Tagger(2, 3, shape, None, 1, 0.1, (), 3).eval()
        assert m(torch.randn(2, 3, *shape), torch.randn(2, 3)).shape == (2, 1)
    print("ResNet18Tagger parameters:", sum(q.numel() for q in m.parameters()))
    assert TAGGERS[{}.get("arch", "cnn")] is Tagger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/tmp/superres_smoke")
    p.add_argument("--dataset", default="both", choices=["qg", "calo", "both"])
    a = p.parse_args()
    from superres.pipeline import run
    check_taggers()
    paths = {"paths.drive_root": os.path.join(a.root, "results"), "paths.raw_root": os.path.join(a.root, "raw"),
             "paths.cache_root": os.path.join(a.root, "cache"), "num_workers": 0}
    stages = ["prepare", "tune_vqvae", "train_vqvae", "tune_var", "train_var", "generate", "eval_sr",
              "tune_tagger", "train_taggers", "tagger_xeval", "summarize"]
    for ds in (["qg", "calo"] if a.dataset == "both" else [a.dataset]):
        raw = os.path.join(a.root, "raw", ds)
        if not os.listdir(raw) if os.path.isdir(raw) else True:
            (fake_qg if ds == "qg" else fake_calo)(raw)
        run(ds, stages, overrides=paths, smoke=True)
        if ds == "qg":  # the same taggers again with the small CNN, in their own folders
            run(ds, ["train_taggers", "tagger_xeval"], overrides={**paths, "tagger.arch": "cnn"}, smoke=True)
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
