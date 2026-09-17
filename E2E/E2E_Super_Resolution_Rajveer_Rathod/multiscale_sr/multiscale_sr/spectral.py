"""Spectral Energy Mover's Distance (SEMD) for calorimeter images.

Implements the closed-form ``p=2`` SEMD of

    R. Gambhir, A. Larkoski, J. Thaler,
    "SPECTER: Efficient Evaluation of the Spectral Energy Mover's Distance",
    arXiv:2410.05379v3 (MIT-CTP 5771),

adapted from particle lists to calorimeter images.

Why this module exists
----------------------
Every physics quantity currently logged by ``engine.py`` — ``energy_response``
(sum ratio), ``peak_ratio`` (max ratio), ``nonzero_ratio`` (count ratio), and
``physics_loss`` (|response - 1|) — is **invariant under arbitrary permutation
of the pixels**. Shuffle an SR image's pixels and not one of them changes. They
constrain how much energy exists and how bright the brightest pixel is, never
*where* the energy sits. Jet tagging, meanwhile, is almost entirely a question
of radiation geometry (prong count, subjet splitting, angular spread). SEMD is
built from energy-weighted *pairwise angles*, so pixel permutation does change
it — that is the entire point of using it here.

The spectral representation (paper Eq. 2.1) maps an event to a 1-D distribution
of pairwise angles weighted by energy products::

    s(w) = sum_{i,j} E_i E_j delta(w - w_ij)
         = sum_i E_i^2 delta(w) + sum_{i<j} 2 E_i E_j delta(w - w_ij)

and SEMD_p is the p-th Wasserstein distance between two such distributions
(Eq. 2.4), which for ``p=2`` has the closed form (Eq. 2.19)::

    SEMD_{beta,p=2}(s_A, s_B) = sum_{i<j in A} 2 E_i E_j w_ij^2
                              + sum_{i<j in B} 2 E_i E_j w_ij^2
                              - 2 sum_{n in A^2, l in B^2} w_n w_l ReLU(S_nl)

with (Eq. 2.16)::

    S_nl = min(S_A^+(w_n), S_B^+(w_l)) - max(S_A^-(w_n), S_B^-(w_l))

where ``S^-``/``S^+`` are the cumulative spectral functions excluding/including
the delta at that angle (Eqs. 2.13/2.14). The first two terms are two-point
energy correlation functions (Eq. 2.11). No optimal-transport solve is needed,
the cost is O(N^2 log N), and every step — including the sort — is
differentiable, so the same code serves as both a metric and a loss.

The image adaptation (stated openly; see also ``docs/SEED_STABILITY_REPORT.md``)
-------------------------------------------------------------------------------
1. **Pixels as particles.** Each pixel above ``threshold`` is a "particle" with
   energy ``E_i`` = its raw (denormalized) energy and position the pixel centre
   in ``(eta, phi)`` index units. Channels are summed to a single energy map
   first, matching how the sample-grid figures render a jet.

2. **Ground metric.** ``w_ij = (d_eta^2 + d_phi^2)^(beta/2)``, i.e. Euclidean
   pixel distance for the default ``beta=2``... note that the paper's ``beta``
   enters as ``w_ij = theta_ij^beta`` with ``theta`` an angle; here ``beta=1``
   gives plain Euclidean distance and is the default. This is admissible
   because the paper only requires ``w_ij`` to be *any symmetric matrix with
   ``w_ii = 0``* — it need not even be a proper metric. ``phi`` is wrapped
   modulo the image width when ``periodic_phi=True``.

3. **Energy balancing is mandatory.** Eq. 2.7 assumes ``E_tot^A == E_tot^B``.
   SR and HR do *not* have equal total energy — that discrepancy is literally
   what ``energy_response != 1`` measures. Following the paper's prescription,
   the deficit is added to the lighter event as an extra "particle" placed a
   distance ``omega_R`` away from everything else. ``omega_R`` is an explicit,
   logged parameter, not a hidden constant.

4. **Top-K truncation.** A 128x128 image has 16384 pixels, i.e. ~2.7e8 pairs —
   far too many. Only the ``topk`` brightest pixels are kept (default 128,
   matching the N=125 points-per-shape the paper uses in its own benchmarks).
   The discarded tail is *not* silently dropped: its energy is folded into the
   balancing term of step 3, so total energy is conserved.

Because of (4) this is an approximation, and is labelled **"SEMD (top-K pixel
approximation)"** everywhere it is reported. It is not the paper's exact
observable and must not be presented as such.
"""
from __future__ import annotations

import torch
from torch import Tensor


__all__ = [
    "extract_particles",
    "pairwise_omega",
    "spectral_function",
    "cumulative_spectral",
    "semd_p2",
    "semd_images",
]


# --------------------------------------------------------------------------- #
# Image -> particle list
# --------------------------------------------------------------------------- #
def extract_particles(
    images: Tensor,
    topk: int = 128,
    threshold: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert a batch of raw-energy images into padded top-K particle lists.

    Args:
        images: ``(B, C, H, W)`` raw (denormalized, non-negative) energies.
            Channels are summed to a single energy map.
        topk: number of brightest pixels to keep per image.
        threshold: pixels at or below this energy are discarded.

    Returns:
        ``(energies, coords, residual)`` where ``energies`` is ``(B, K)`` with
        zeros in padding slots, ``coords`` is ``(B, K, 2)`` holding
        ``(row, col)`` pixel-centre positions, and ``residual`` is ``(B,)``
        holding the total energy of every pixel *not* kept (below threshold or
        outside the top-K). ``residual`` is what the caller must feed to the
        energy-balancing term so no energy is lost.

    Zero-energy padding slots are harmless downstream: every term of Eq. 2.19
    is multiplied by ``E_i E_j``, so a slot with ``E = 0`` contributes nothing
    regardless of the position assigned to it.
    """
    if images.ndim != 4:
        raise ValueError(f"expected (B,C,H,W) images, got shape {tuple(images.shape)}")
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")

    energy_map = images.sum(dim=1).clamp_min(0.0)  # (B,H,W)
    b, h, w = energy_map.shape
    flat = energy_map.reshape(b, h * w)

    kept = min(topk, h * w)
    vals, idx = torch.topk(flat, k=kept, dim=1)  # (B,K)

    # Drop sub-threshold pixels by zeroing their energy (keeps shapes static,
    # which matters for batching and for torch.compile-ability).
    vals = torch.where(vals > threshold, vals, torch.zeros_like(vals))

    rows = torch.div(idx, w, rounding_mode="floor").to(images.dtype)
    cols = (idx % w).to(images.dtype)
    coords = torch.stack((rows, cols), dim=-1)  # (B,K,2)

    residual = (flat.sum(dim=1) - vals.sum(dim=1)).clamp_min(0.0)  # (B,)
    return vals, coords, residual


def pairwise_omega(
    coords: Tensor,
    beta: float = 1.0,
    periodic_phi: bool = False,
    phi_period: float | None = None,
) -> Tensor:
    """Pairwise ground metric ``w_ij`` between particle positions.

    Args:
        coords: ``(B, N, 2)`` positions as ``(eta_index, phi_index)``.
        beta: exponent, ``w_ij = dist_ij ** beta``.
        periodic_phi: wrap the second coordinate (phi) to the shorter arc.
        phi_period: the phi period; required when ``periodic_phi`` is True.

    Returns:
        ``(B, N, N)`` symmetric matrix with an exactly-zero diagonal.

    The paper notes it "suffices for ``w_ij`` to be any symmetric matrix such
    that ``w_ii = 0``", so a pixel-grid distance is admissible even though it is
    not an angle on a sphere. The diagonal is forced to exactly zero rather than
    left to floating-point luck, because ``S^-``/``S^+`` treat the ``w = 0``
    delta specially.
    """
    d = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B,N,N,2)
    if periodic_phi:
        if phi_period is None:
            raise ValueError("phi_period is required when periodic_phi=True")
        dphi = d[..., 1]
        dphi = dphi - phi_period * torch.round(dphi / phi_period)
        d = torch.stack((d[..., 0], dphi), dim=-1)

    sq = (d ** 2).sum(dim=-1)
    # Clamp before sqrt: the exact-zero diagonal makes d(sqrt)/dx infinite at 0,
    # which would put NaNs in the gradient of an otherwise finite loss.
    dist = torch.sqrt(sq.clamp_min(1e-12))
    omega = dist ** beta if beta != 1.0 else dist

    n = coords.shape[1]
    eye = torch.eye(n, device=coords.device, dtype=torch.bool)
    return omega.masked_fill(eye, 0.0)


# --------------------------------------------------------------------------- #
# Spectral representation
# --------------------------------------------------------------------------- #
def spectral_function(
    energies: Tensor,
    omega: Tensor,
) -> tuple[Tensor, Tensor]:
    """Spectral function of Eq. 2.1, as a sorted list of weighted deltas.

    Args:
        energies: ``(B, N)`` particle energies.
        omega: ``(B, N, N)`` pairwise ground metric.

    Returns:
        ``(omega_sorted, weights_sorted)``, each ``(B, N*N)``, giving the
        support points and weights of ``s(w)`` in ascending ``w``. The full
        ``N x N`` outer product is used (rather than only ``i < j``) so that the
        ``i == j`` self-pairs supply the ``sum_i E_i^2 delta(w)`` term and each
        off-diagonal pair is counted twice, reproducing the ``2 E_i E_j``
        coefficient of Eq. 2.1 exactly.

    The sort is differentiable in the sense that matters here: gathering by the
    returned permutation is a linear operation on the weights, so gradients flow
    to ``energies`` and (through ``omega``) to positions. The paper makes the
    same observation about its own implementation.
    """
    weights = energies.unsqueeze(2) * energies.unsqueeze(1)  # (B,N,N)
    b = energies.shape[0]
    flat_omega = omega.reshape(b, -1)
    flat_weights = weights.reshape(b, -1)

    order = torch.argsort(flat_omega, dim=1)
    return torch.gather(flat_omega, 1, order), torch.gather(flat_weights, 1, order)


def cumulative_spectral(weights_sorted: Tensor) -> tuple[Tensor, Tensor]:
    """Cumulative spectral functions ``S^-`` and ``S^+`` (Eqs. 2.13 / 2.14).

    Args:
        weights_sorted: ``(B, M)`` delta weights ordered by ascending ``w``.

    Returns:
        ``(S_minus, S_plus)``, each ``(B, M)``. ``S_plus[n]`` is the cumulative
        weight *including* the delta at index ``n``; ``S_minus[n]`` excludes it.
        Both are needed because the closed form measures the overlap of the
        half-open intervals ``[S^-, S^+)`` that each delta occupies in
        ``E^2``-space.
    """
    s_plus = torch.cumsum(weights_sorted, dim=1)
    s_minus = s_plus - weights_sorted
    return s_minus, s_plus


# --------------------------------------------------------------------------- #
# Closed-form p=2 SEMD
# --------------------------------------------------------------------------- #
def _self_term(omega_sorted: Tensor, weights_sorted: Tensor) -> Tensor:
    """``sum_{i<j} 2 E_i E_j w_ij^2`` — the two-point correlator of Eq. 2.11.

    Computed over the full sorted delta list, where the ``i == j`` entries carry
    ``w = 0`` and so drop out automatically.
    """
    return (weights_sorted * omega_sorted ** 2).sum(dim=1)


def semd_p2(
    energies_a: Tensor,
    coords_a: Tensor,
    energies_b: Tensor,
    coords_b: Tensor,
    beta: float = 1.0,
    periodic_phi: bool = False,
    phi_period: float | None = None,
) -> Tensor:
    """Closed-form ``p=2`` SEMD between two batched particle clouds (Eq. 2.19).

    Both clouds must already carry equal total energy per sample — use
    :func:`semd_images`, which handles balancing, rather than calling this
    directly on unbalanced inputs. ``semd_p2`` does not balance for you, since
    the correct place to add the deficit depends on ``omega_R``, which is a
    property of the comparison and not of either cloud.

    Args:
        energies_a: ``(B, Na)``; ``coords_a``: ``(B, Na, 2)``.
        energies_b: ``(B, Nb)``; ``coords_b``: ``(B, Nb, 2)``.
        beta, periodic_phi, phi_period: passed to :func:`pairwise_omega`.

    Returns:
        ``(B,)`` non-negative distances.

    The cross term is written as ``omega_n * omega_l * ReLU(S_nl)`` over the
    ``M_A x M_B`` pairs of deltas. That looks like ``O(N^4)``, but ``ReLU``
    zeroes all but an ``O(N^2)``-sized band of terms (the two sorted cumulative
    ladders only overlap locally), which is the observation the paper's
    ``O(N^2 log N)`` scaling rests on. This implementation materializes the full
    band matrix for clarity and batching; with the default ``topk=128`` that is
    ``(128*128)^2`` entries per sample, so it is evaluated in chunks over the
    batch by :func:`semd_images`.
    """
    omega_a = pairwise_omega(coords_a, beta, periodic_phi, phi_period)
    omega_b = pairwise_omega(coords_b, beta, periodic_phi, phi_period)

    w_a, wt_a = spectral_function(energies_a, omega_a)
    w_b, wt_b = spectral_function(energies_b, omega_b)

    self_a = _self_term(w_a, wt_a)
    self_b = _self_term(w_b, wt_b)

    sm_a, sp_a = cumulative_spectral(wt_a)
    sm_b, sp_b = cumulative_spectral(wt_b)

    # S_nl = min(S_A^+, S_B^+) - max(S_A^-, S_B^-)   (Eq. 2.16)
    overlap = torch.minimum(sp_a.unsqueeze(2), sp_b.unsqueeze(1)) - torch.maximum(
        sm_a.unsqueeze(2), sm_b.unsqueeze(1)
    )
    cross = (w_a.unsqueeze(2) * w_b.unsqueeze(1) * torch.relu(overlap)).sum(dim=(1, 2))

    # Clamp at zero: the closed form is non-negative analytically, but float32
    # cancellation between three large, nearly-equal terms can land a hair below.
    return (self_a + self_b - 2.0 * cross).clamp_min(0.0)


def _balance(
    energies: Tensor,
    coords: Tensor,
    deficit: Tensor,
    omega_R: float,
    far_offset: float,
) -> tuple[Tensor, Tensor]:
    """Append one balancing particle carrying ``deficit`` energy.

    Placed ``far_offset`` away along the eta axis so its distance to every real
    pixel is dominated by ``omega_R``. This follows the paper's prescription of
    adding the energy discrepancy "a distance ``omega_R`` away" rather than
    rescaling either event — rescaling would erase the very energy-response
    mismatch we want the metric to notice.
    """
    pad_e = deficit.clamp_min(0.0).unsqueeze(1)  # (B,1)
    ref = coords[:, :1, :]  # (B,1,2), an arbitrary real position
    offset = torch.tensor(
        [far_offset, 0.0], device=coords.device, dtype=coords.dtype
    ).view(1, 1, 2)
    pad_c = ref + offset * omega_R
    return torch.cat((energies, pad_e), dim=1), torch.cat((coords, pad_c), dim=1)


def semd_images(
    pred_raw: Tensor,
    target_raw: Tensor,
    topk: int = 128,
    threshold: float = 0.0,
    beta: float = 1.0,
    omega_R: float = 1.0,
    periodic_phi: bool = False,
    chunk: int = 8,
) -> Tensor:
    """SEMD (top-K pixel approximation) between two batches of raw-energy images.

    This is the entry point for both evaluation and the optional training loss.

    Args:
        pred_raw: ``(B, C, H, W)`` denormalized SR (or LR) energies.
        target_raw: ``(B, C, H, W)`` denormalized HR energies.
        topk: brightest pixels kept per image (default 128, cf. the paper's
            N=125 benchmark shapes). Logged with every result — a K-dependent
            metric never checked for K-sensitivity is a trap.
        threshold: raw-energy cut applied before the top-K selection.
        beta: ground-metric exponent, ``w_ij = dist_ij ** beta``.
        omega_R: angular scale at which the energy imbalance between ``pred``
            and ``target`` is deposited. Explicit because the two do *not*
            carry equal energy by construction, while Eq. 2.7 assumes they do.
        periodic_phi: wrap the phi (column) axis at the image width.
        chunk: samples per cross-term evaluation, bounding peak memory.

    Returns:
        ``(B,)`` per-sample distances, in units of ``energy^2 * length^(2*beta)``.
    """
    if pred_raw.shape != target_raw.shape:
        raise ValueError(
            f"shape mismatch: pred {tuple(pred_raw.shape)} vs target {tuple(target_raw.shape)}"
        )

    e_p, c_p, res_p = extract_particles(pred_raw, topk=topk, threshold=threshold)
    e_t, c_t, res_t = extract_particles(target_raw, topk=topk, threshold=threshold)

    # Energy balancing (mandatory, see module docstring step 3). Each side gets
    # a slot: its own truncation residual plus whatever it lacks relative to the
    # other side's total. Both sides therefore end at the same E_tot.
    tot_p = e_p.sum(dim=1) + res_p
    tot_t = e_t.sum(dim=1) + res_t
    tot = torch.maximum(tot_p, tot_t)

    far = float(max(pred_raw.shape[-2], pred_raw.shape[-1]))
    e_p, c_p = _balance(e_p, c_p, tot - e_p.sum(dim=1), omega_R, far)
    e_t, c_t = _balance(e_t, c_t, tot - e_t.sum(dim=1), omega_R, far)

    phi_period = float(pred_raw.shape[-1]) if periodic_phi else None

    out = []
    for start in range(0, e_p.shape[0], max(chunk, 1)):
        stop = start + max(chunk, 1)
        out.append(
            semd_p2(
                e_p[start:stop],
                c_p[start:stop],
                e_t[start:stop],
                c_t[start:stop],
                beta=beta,
                periodic_phi=periodic_phi,
                phi_period=phi_period,
            )
        )
    return torch.cat(out, dim=0)
