"""Checks of the jet observables on synthetic images (no data, runs in seconds).

    python -m pytest tests/test_observables.py
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superres.metrics import paired_metrics  # noqa: E402
from superres.observables import jet_observables  # noqa: E402
from superres.physics_ops import sum_pool  # noqa: E402

DCFG = {"pixel_size": 0.0174, "center_index": 63, "occupancy_threshold": 1e-3,
        "channels": ["tracks", "ecal", "hcal"]}


def _jets(n=50, seed=0):
    rng = np.random.default_rng(seed)
    E = np.zeros((n, 3, 128, 128), np.float32)
    for i in range(n):
        for _ in range(rng.integers(5, 40)):
            y, x = np.clip(rng.normal(63, 8, 2).astype(int), 0, 127)
            E[i, rng.integers(0, 3), y, x] += rng.exponential(5)
    return E


def _obs(E, f=(1, 1)):
    return jet_observables(E, f, DCFG, nsub_max=0)[0]


def test_same_image_same_values():
    E = _jets()
    a, b = _obs(E), _obs(E.copy())
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_two_body_mass():
    """Two massless pixels: m^2 = 2 E1 E2 (cosh(d_eta) - cos(d_phi))."""
    E = np.zeros((1, 3, 128, 128), np.float32)
    E[0, 0, 63, 53], E[0, 1, 63, 73] = 30.0, 20.0  # 20 cells apart in phi
    dphi = 20 * DCFG["pixel_size"]
    expected = np.sqrt(2 * 30.0 * 20.0 * (1 - np.cos(dphi)))
    o = _obs(E)
    assert o["jet_mass"][0] == pytest.approx(expected, rel=1e-6)
    assert o["sum_all"][0] == pytest.approx(50.0)
    # girth: energy-weighted distance from the centroid, which sits 8 cells from the 30 GeV pixel
    assert o["girth"][0] == pytest.approx((30 * 8 + 20 * 12) * DCFG["pixel_size"] / 50, rel=1e-6)
    assert o["dR_std"][0] == pytest.approx(np.sqrt((30 * 8 ** 2 + 20 * 12 ** 2) / 50 - 9.6 ** 2)
                                           * DCFG["pixel_size"], rel=1e-5)
    assert o["n_hits"][0] == 2 and o["n_hits_tracks"][0] == 1 and o["n_hits_ecal"][0] == 1


def test_pooled_grid_keeps_energy_and_kinematics():
    E = _jets()
    hr, lr = _obs(E), _obs(sum_pool(E, (2, 2)), (2, 2))
    np.testing.assert_allclose(lr["sum_all"], hr["sum_all"], rtol=1e-5)
    np.testing.assert_allclose(lr["jet_pt"], hr["jet_pt"], rtol=0.01)
    assert np.all(lr["jet_mass"] <= hr["jet_mass"] * 1.05)
    np.testing.assert_allclose(lr["jet_mass"], hr["jet_mass"], rtol=0.15)


def test_threshold_cells_survive_storage():
    """The decoder writes hit cells at exactly the threshold; after the log / exp round trip and
    fp16 / fp32 storage they must still count as hits."""
    import torch
    thr = DCFG["occupancy_threshold"]
    E = np.zeros((1, 3, 128, 128), np.float32)
    rng = np.random.default_rng(1)
    for c in range(3):
        s = np.float32(rng.uniform(0.01, 5))
        level = torch.tensor(np.log1p(thr / s), dtype=torch.float32)  # hit_levels()
        v = (torch.tensor(s) * torch.expm1(level)).item()  # to_energy()
        E[0, c, 10 + c, 10:60] = v
    for dtype in (np.float32, np.float16):
        o = _obs(E.astype(dtype).astype(np.float32))
        assert o["n_hits_tracks"][0] == o["n_hits_ecal"][0] == o["n_hits_hcal"][0] == 50


def test_paired_metrics():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    p = paired_metrics(a, 1.1 * a)
    assert p["bias"] == pytest.approx(0.1)
    assert p["pearson_r"] == pytest.approx(1.0)
