"""Cambridge–Aachen coarsening augmentation.

The transformer is unchanged: C/A is used only to replace a training jet
with a random-resolution set of E-scheme pseudojets (HERON §4 / J-JEPA).
"""

from __future__ import annotations

import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch

from dataloader.ca_augment import cambridge_aachen_augment_batch
from variants.tests.strategies import assert_loader_contract

_PID_CHARGED_HADRON = 11  # index in the 16-feature x vector


def _wrap(dphi: float) -> float:
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def _particle(pt, y, phi, mass=0.14, charge=0.0, pid_idx=_PID_CHARGED_HADRON):
    """One physically valid constituent; kinematics from (pt, y, phi, mass)."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(y)
    energy = float(np.sqrt(px * px + py * py + pz * pz + mass * mass))
    p = float(np.sqrt(px * px + py * py + pz * pz))
    eta = float(np.arctanh(np.clip(pz / p, -0.999999, 0.999999)))
    x = np.zeros(16, dtype=np.float32)
    x[0] = pt
    x[1] = eta
    x[2] = phi
    x[3] = energy
    x[10] = charge
    x[pid_idx] = 1.0
    v = np.array([px, py, pz, energy], dtype=np.float32)
    return x, v


def _jet_batch(particles, pad_to=None):
    """``particles`` is a list of per-jet lists of ``(x, v)`` pairs."""
    lengths = [len(p) for p in particles]
    p_max = pad_to if pad_to is not None else max(lengths)
    b = len(particles)
    x = torch.zeros(b, 16, p_max)
    v = torch.zeros(b, 4, p_max)
    mask = torch.zeros(b, 1, p_max)
    for i, jet in enumerate(particles):
        jet_px = sum(float(pv[0]) for _, pv in jet)
        jet_py = sum(float(pv[1]) for _, pv in jet)
        jet_pz = sum(float(pv[2]) for _, pv in jet)
        jet_p = np.sqrt(jet_px**2 + jet_py**2 + jet_pz**2)
        jet_eta = float(
            np.arctanh(np.clip(jet_pz / max(jet_p, 1e-12), -0.999999, 0.999999))
        )
        jet_phi = float(np.arctan2(jet_py, jet_px))
        for k, (fx, fv) in enumerate(jet):
            fx = fx.copy()
            fx[4] = fx[1] - jet_eta
            fx[5] = _wrap(float(fx[2] - jet_phi))
            x[i, :, k] = torch.from_numpy(fx)
            v[i, :, k] = torch.from_numpy(fv)
            mask[i, 0, k] = 1.0
    return x, v, mask


def _sum_p4(v, mask):
    weights = mask[:, 0, :].unsqueeze(1)
    return (v * weights).sum(dim=-1)


def test_radius_zero_is_identity():
    x, v, mask = _jet_batch(
        [[_particle(20.0, 0.0, 0.0), _particle(10.0, 0.0, 0.05)]]
    )
    x2, v2, mask2 = cambridge_aachen_augment_batch(
        x, v, mask, prob=1.0, radius=0.0, min_particles=1
    )
    torch.testing.assert_close(x2, x)
    torch.testing.assert_close(v2, v)
    torch.testing.assert_close(mask2, mask)


def test_prob_zero_is_identity():
    x, v, mask = _jet_batch(
        [[_particle(20.0, 0.0, 0.0), _particle(10.0, 0.0, 0.05)]]
    )
    x2, v2, mask2 = cambridge_aachen_augment_batch(
        x, v, mask, prob=0.0, rmax=10.0
    )
    torch.testing.assert_close(x2, x)
    torch.testing.assert_close(v2, v)


def test_close_pair_merges_above_their_delta_r():
    """ΔR ≈ 0.05; a C/A cut of 0.1 must merge them into one pseudojet."""
    hard = _particle(30.0, 0.0, 0.0, charge=1.0, pid_idx=_PID_CHARGED_HADRON)
    soft = _particle(5.0, 0.0, 0.05, charge=0.0, pid_idx=13)  # photon bit
    x, v, mask = _jet_batch([[hard, soft]])
    x2, v2, mask2 = cambridge_aachen_augment_batch(
        x, v, mask, prob=1.0, radius=0.1, min_particles=1
    )
    assert int(mask2[0, 0].sum()) == 1
    torch.testing.assert_close(
        _sum_p4(v2, mask2), _sum_p4(v, mask), atol=1e-5, rtol=1e-5
    )
    # Harder leaf's PID / charge survive.
    assert float(x2[0, 10, 0]) == 1.0
    assert float(x2[0, _PID_CHARGED_HADRON, 0]) == 1.0
    assert float(x2[0, 13, 0]) == 0.0


def test_close_pair_does_not_merge_below_their_delta_r():
    x, v, mask = _jet_batch(
        [[_particle(30.0, 0.0, 0.0), _particle(5.0, 0.0, 0.05)]]
    )
    x2, v2, mask2 = cambridge_aachen_augment_batch(
        x, v, mask, prob=1.0, radius=0.01, min_particles=1
    )
    assert int(mask2[0, 0].sum()) == 2
    torch.testing.assert_close(v2, v)


def test_min_particles_floor():
    """A huge radius would otherwise collapse the jet to one constituent."""
    particles = [_particle(10.0 + i, 0.0, 0.02 * i) for i in range(6)]
    x, v, mask = _jet_batch([particles])
    x2, v2, mask2 = cambridge_aachen_augment_batch(
        x, v, mask, prob=1.0, radius=10.0, min_particles=3
    )
    assert int(mask2[0, 0].sum()) == 3
    torch.testing.assert_close(
        _sum_p4(v2, mask2), _sum_p4(v, mask), atol=1e-4, rtol=1e-4
    )


def test_unmerged_particles_keep_features():
    """A far-away third particle must come through byte-identical."""
    close_a = _particle(20.0, 0.0, 0.0, charge=1.0)
    close_b = _particle(8.0, 0.0, 0.04, charge=-1.0)
    far = _particle(12.0, 0.4, 1.2, charge=0.0, pid_idx=13)
    x, v, mask = _jet_batch([[close_a, close_b, far]])
    x2, v2, mask2 = cambridge_aachen_augment_batch(
        x, v, mask, prob=1.0, radius=0.1, min_particles=2
    )
    assert int(mask2[0, 0].sum()) == 2
    far_v = torch.from_numpy(far[1])
    matched = False
    for k in range(2):
        if torch.allclose(v2[0, :, k], far_v, atol=1e-5):
            torch.testing.assert_close(x2[0, 10, k], torch.tensor(0.0))
            torch.testing.assert_close(x2[0, 13, k], torch.tensor(1.0))
            matched = True
    assert matched


def test_batch_contract_and_padding_stay_zero():
    jets = [
        [
            _particle(20.0, 0.0, 0.0),
            _particle(10.0, 0.0, 0.05),
            _particle(7.0, 0.2, 0.4),
        ],
        [_particle(15.0, 0.1, -0.3)],
    ]
    x, v, mask = _jet_batch(jets, pad_to=5)
    y = torch.zeros(2, 10)
    y[0, 0] = 1.0
    y[1, 3] = 1.0
    x2, v2, mask2 = cambridge_aachen_augment_batch(
        x, v, mask, prob=1.0, radius=0.2, min_particles=1
    )
    assert_loader_contract(x2, v2, mask2, y)
    lengths = mask2[:, 0, :].sum(dim=-1)
    for b in range(2):
        n = int(lengths[b])
        if n < x2.shape[-1]:
            assert torch.count_nonzero(x2[b, :, n:]) == 0
            assert torch.count_nonzero(v2[b, :, n:]) == 0


def test_recomputed_kinematics_match_merged_four_vector():
    x, v, mask = _jet_batch(
        [[_particle(30.0, 0.0, 0.0), _particle(5.0, 0.0, 0.05)]]
    )
    x2, v2, mask2 = cambridge_aachen_augment_batch(
        x, v, mask, prob=1.0, radius=0.1, min_particles=1
    )
    px, py, pz, energy = (float(v2[0, i, 0]) for i in range(4))
    pt = float(np.hypot(px, py))
    assert abs(float(x2[0, 0, 0]) - pt) < 1e-4
    assert abs(float(x2[0, 3, 0]) - energy) < 1e-4
    jet_eta = float(x[0, 1, 0] - x[0, 4, 0])
    jet_phi = _wrap(float(x[0, 2, 0] - x[0, 5, 0]))
    p = np.sqrt(px * px + py * py + pz * pz)
    eta = float(np.arctanh(np.clip(pz / p, -0.999999, 0.999999)))
    phi = float(np.arctan2(py, px))
    assert abs(float(x2[0, 4, 0]) - (eta - jet_eta)) < 1e-4
    assert abs(_wrap(float(x2[0, 5, 0]) - _wrap(phi - jet_phi))) < 1e-4


def test_single_particle_jet_is_untouched():
    x, v, mask = _jet_batch([[_particle(40.0, 0.1, -0.2)]])
    x2, v2, mask2 = cambridge_aachen_augment_batch(
        x, v, mask, prob=1.0, radius=10.0, min_particles=2
    )
    torch.testing.assert_close(x2, x)
    torch.testing.assert_close(v2, v)


def test_train_collate_is_picklable():
    """DataLoader workers pickle the collate; a closure would hang the loader."""
    from ablation.data import NormStats, _train_collate

    stats = NormStats(mean=np.ones(16), std=np.ones(16))
    fn = partial(
        _train_collate, stats=stats, ca_prob=0.5, ca_rmax=0.2, ca_min_particles=2
    )
    pickle.loads(pickle.dumps(fn))


def test_baseline_ca_config_loads():
    from ablation.config import load_config

    path = (
        Path(__file__).resolve().parents[2]
        / "ablation"
        / "configs"
        / "baseline_ca.yaml"
    )
    cfg = load_config(str(path))
    assert cfg.ca_augment is True
    assert cfg.run_name == "baseline_ca"
    assert cfg.arm == "baseline"
    assert cfg.ca_rmax == 0.2
    assert 0.0 < cfg.ca_augment_prob <= 1.0


def test_ca_augment_off_by_default():
    from ablation.config import AblationConfig

    cfg = AblationConfig(arm="baseline")
    assert cfg.ca_augment is False
