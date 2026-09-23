"""Cambridge–Aachen coarsening as a training-time jet augmentation.

The HERON plan (``knowledge/heron_plan_and_novelty.md`` §4, task 3, and the
"first C/A implementation") treats C/A as a physical resolution of the jet,
not an architecture change: repeatedly merge the pair with smallest angular
distance, E-scheme ``p_k = p_i + p_j``.  J-JEPA (arXiv:2412.05333) uses a
fixed C/A radius ``R = 0.2`` to build subjets.  Here the same clustering is
applied as a *random-resolution* augmentation: each training jet is, with
probability ``prob``, reclustered with a radius drawn uniformly from
``[0, rmax]`` and replaced by the surviving pseudojets.

Validation is left at the original constituents so the train/val mismatch is
the usual augmentation one (the model must still tag the un-coarsened jet).

Clustering metric is rapidity–azimuth ``ΔR² = Δy² + Δφ²`` (HERON §4.4: merged
pseudojets are massive, so ``y`` not ``η``).  Stored ``part_eta`` stays
pseudorapidity so the 16-feature contract matches JetClass / ParT.

A merged pseudojet has no unique PID.  Displacement, charge and the five PID
one-hots are inherited from the harder original leaf (higher ``p_T``), which
is the usual subjet-tagging convention and stays inside ParT's training
support (charge in ``{-1,0,+1}``, a single PID bit).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor

__all__ = ["cambridge_aachen_augment_batch"]

_EPS = 1e-12
_ETA_CLIP = 1.0 - 1e-7


def _wrap_dphi(dphi: np.ndarray) -> np.ndarray:
    return (dphi + np.pi) % (2.0 * np.pi) - np.pi


def _rapidity(pz: np.ndarray, energy: np.ndarray) -> np.ndarray:
    return 0.5 * np.log(
        np.maximum(energy + pz, _EPS) / np.maximum(energy - pz, _EPS)
    )


def _pseudorapidity(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
    p = np.sqrt(px * px + py * py + pz * pz)
    return np.arctanh(
        np.clip(pz / np.maximum(p, _EPS), -_ETA_CLIP, _ETA_CLIP)
    )


def _coarsen_one(
    x: np.ndarray,
    v: np.ndarray,
    n: int,
    r_cut: float,
    min_particles: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """C/A-coarsen one jet. ``x`` is ``(16, P)``, ``v`` is ``(4, P)``."""
    if n <= min_particles or r_cut <= 0.0 or n < 2:
        return x[:, :n], v[:, :n], n

    px = v[0, :n].copy()
    py = v[1, :n].copy()
    pz = v[2, :n].copy()
    energy = v[3, :n].copy()
    alive = np.ones(n, dtype=bool)
    was_merged = np.zeros(n, dtype=bool)
    leading = np.arange(n, dtype=np.int64)
    orig_pt = np.hypot(px, py)

    y = _rapidity(pz, energy)
    phi = np.arctan2(py, px)
    r_cut_sq = r_cut * r_cut
    n_alive = n

    while n_alive > min_particles:
        idx = np.flatnonzero(alive)
        yi = y[idx]
        phii = phi[idx]
        dy = yi[:, None] - yi[None, :]
        dphi = _wrap_dphi(phii[:, None] - phii[None, :])
        dist = dy * dy + dphi * dphi
        np.fill_diagonal(dist, np.inf)
        flat = int(np.argmin(dist))
        i_loc, j_loc = divmod(flat, n_alive)
        if dist[i_loc, j_loc] >= r_cut_sq:
            break
        i = int(idx[i_loc])
        j = int(idx[j_loc])
        if i > j:
            i, j = j, i
        if orig_pt[leading[j]] > orig_pt[leading[i]]:
            leading[i] = leading[j]
        px[i] += px[j]
        py[i] += py[j]
        pz[i] += pz[j]
        energy[i] += energy[j]
        y[i] = _rapidity(pz[i], energy[i])
        phi[i] = np.arctan2(py[i], px[i])
        alive[j] = False
        was_merged[i] = True
        n_alive -= 1

    if n_alive == n:
        return x[:, :n], v[:, :n], n

    jet_eta = float(x[1, 0] - x[4, 0])
    jet_phi = float(_wrap_dphi(np.asarray(x[2, 0] - x[5, 0])))

    keep = np.flatnonzero(alive)
    out_x = np.zeros((16, n_alive), dtype=np.float32)
    out_v = np.zeros((4, n_alive), dtype=np.float32)
    for t, src in enumerate(keep):
        if not was_merged[src]:
            out_x[:, t] = x[:, src]
            out_v[:, t] = v[:, src]
            continue
        pt = np.hypot(px[src], py[src])
        eta = _pseudorapidity(px[src], py[src], pz[src])
        ph = np.arctan2(py[src], px[src])
        out_v[0, t] = px[src]
        out_v[1, t] = py[src]
        out_v[2, t] = pz[src]
        out_v[3, t] = energy[src]
        out_x[0, t] = pt
        out_x[1, t] = eta
        out_x[2, t] = ph
        out_x[3, t] = energy[src]
        out_x[4, t] = eta - jet_eta
        out_x[5, t] = float(_wrap_dphi(np.asarray(ph - jet_phi)))
        out_x[6:, t] = x[6:, leading[src]]
    return out_x, out_v, n_alive


def cambridge_aachen_augment_batch(
    x: Tensor,
    v: Tensor,
    mask: Tensor,
    *,
    prob: float = 0.5,
    rmax: float = 0.2,
    min_particles: int = 2,
    radius: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Random-R C/A coarsening on a channel-first batch.

    Parameters
    ----------
    x, v, mask
        Loader-contract tensors ``(B, 16, P)``, ``(B, 4, P)``, ``(B, 1, P)``.
    prob
        Per-jet probability of applying any coarsening.  ``0`` is a no-op.
    rmax
        Upper bound on the C/A radius.  The actual cut is ``U(0, rmax)``
        unless ``radius`` is set.  ``0.2`` is J-JEPA's subjet radius.
    min_particles
        Never coarsen below this multiplicity.  Default 2 avoids weaver
        BatchNorm degeneracy on a 1×1 pair grid in train mode.
    radius
        If given, use this C/A cut instead of sampling.  Tests and a
        fixed-R (J-JEPA-style) ablation pass this; training leaves it
        ``None``.
    rng
        Optional NumPy generator.  ``None`` uses the process-global
        ``np.random`` so DataLoader ``worker_init_fn`` seeding applies.

    Returns
    -------
    tuple
        ``(x, v, mask)`` with the same contract, packed to the new
        (possibly smaller) max multiplicity.
    """
    if prob <= 0.0:
        return x, v, mask
    if radius is not None and radius <= 0.0:
        return x, v, mask
    if radius is None and rmax <= 0.0:
        return x, v, mask

    x_np = x.detach().cpu().numpy()
    v_np = v.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    batch_size = x_np.shape[0]
    counts = mask_np[:, 0, :].sum(axis=1).astype(np.int64)

    out_x = [None] * batch_size
    out_v = [None] * batch_size
    out_n = np.empty(batch_size, dtype=np.int64)

    def _draw_radius() -> float:
        if radius is not None:
            return float(radius)
        if rng is None:
            return float(np.random.uniform(0.0, rmax))
        return float(rng.uniform(0.0, rmax))

    def _coin() -> bool:
        if rng is None:
            return bool(np.random.random() < prob)
        return bool(rng.random() < prob)

    for b in range(batch_size):
        n = int(counts[b])
        if n < 1:
            raise RuntimeError(f"ca_augment: jet {b} has no valid particles")
        if n <= min_particles or not _coin():
            out_x[b] = x_np[b, :, :n]
            out_v[b] = v_np[b, :, :n]
            out_n[b] = n
            continue
        cx, cv, cn = _coarsen_one(
            x_np[b], v_np[b], n, _draw_radius(), min_particles
        )
        out_x[b] = cx
        out_v[b] = cv
        out_n[b] = cn

    new_p = int(out_n.max())
    x_out = np.zeros((batch_size, 16, new_p), dtype=np.float32)
    v_out = np.zeros((batch_size, 4, new_p), dtype=np.float32)
    mask_out = np.zeros((batch_size, 1, new_p), dtype=np.float32)
    for b in range(batch_size):
        n = int(out_n[b])
        x_out[b, :, :n] = out_x[b]
        v_out[b, :, :n] = out_v[b]
        mask_out[b, 0, :n] = 1.0

    return (
        torch.from_numpy(x_out),
        torch.from_numpy(v_out),
        torch.from_numpy(mask_out),
    )
