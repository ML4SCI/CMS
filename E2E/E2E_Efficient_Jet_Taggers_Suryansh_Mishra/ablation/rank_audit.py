"""Rank audit of the pairwise interaction bias U — hypothesis T0.1.

Decides whether the HERON N8 contribution (factorized pair bias) is worth
building, *before* any model code is written.  See
``knowledge/heron_plan_and_novelty.md`` §0.1 (T0.1) and §8.7a.

The question
------------
ParT-family models add a learned pairwise bias to the attention logits::

    logit[b,h,i,j] = (Q K^T / sqrt(d))[b,h,i,j] + U[b,h,i,j]

``U`` is produced by ``PairEmbed`` — an MLP over the four Lorentz pair
features ``[ln kt, ln z, ln delta, ln m2]`` — and costs ``O(B N^2 C_pair)``
activation memory plus ``O(B N^2 C_pair^2)`` FLOPs.  N8 proposes to delete it
by folding the pairwise structure into extra Q/K dimensions::

    U[i,j] ~= phi(p_i)^T M psi(p_j)

which is possible **iff U is low rank**.  This script measures that.

Two different numbers, both reported
------------------------------------
1. **Spectral rank** — singular values of ``U[b,h]``.  By Eckart-Young this is
   the *best possible* rank-r approximation, i.e. an upper bound on how well
   any factorization can do.  If the spectral rank is high, N8 is dead
   outright and no clever choice of features rescues it.

2. **Achievable rank with physical features** — least-squares fit of
   ``U ~= Phi M Phi^T`` where ``Phi`` holds moment features built from the raw
   four-momenta:

   ===========  ======  =====================================================
   ``degree``   dims    features
   ===========  ======  =====================================================
   1            4       ``p^mu``
   2            14      ``p^mu`` + symmetric ``p^mu p^nu`` (10)
   3            34      degree 2 + symmetric ``p^mu p^nu p^rho`` (20)
   ===========  ======  =====================================================

   This is what N8 can actually implement: appending ``A p_i`` to the query
   and ``B p_j`` to the key yields ``p_i^T (A^T B) p_j``, so **any** ``M`` of
   rank <= r is reachable with r appended dimensions — the fit is therefore
   left unconstrained rather than being pinned to ``M = lambda * eta``.

   Gap between (1) and (2) = how much is lost by insisting the features be
   polynomial in the four-momenta rather than arbitrary.

Decision rule (registered in §8.7a before running)
--------------------------------------------------
- rank 8 captures > 90% Frobenius energy  -> build N8 as the primary
  efficiency route.
- rank > 40 needed                        -> N8 is dead, fall back to N7
  (sparse execution).
- in between                              -> build the degree-2 variant and
  re-gate on measured quality.

Correctness notes
-----------------
- ``U`` is shared across all encoder blocks.  Weaver computes
  ``attn_mask = self.pair_embed(v, uu=uu, mask=mask)`` **once** outside the
  block loop (``ParticleTransformer.py`` line 1173) and passes the identical
  tensor to every block, so this is one measurement, not one per layer.  The
  per-block variation lives in ``Q K^T``, which is not what N8 replaces.
- Padded pairs in ``U`` are **exactly 0.0**, not ``-1e9`` and not ``-inf``
  (established by the ``pair-sentinel-batchnorm-fix`` spec; exclusion of
  padded keys is key-side masking only).  Every slice is therefore cut to the
  jet's valid length ``U[b, h, :n, :n]`` — leaving the zero rows in would
  manufacture spurious zero singular values and flatter the rank.
- With random weights the pair MLP is an arbitrary smooth function of the four
  pair features and its spectrum says nothing about what a *trained* model
  relies on.  Runs without ``--checkpoint`` are therefore labelled
  ``untrained`` in the output and serve only as a null baseline.  The gate
  above applies to trained weights.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "MOMENT_DEGREES",
    "ENERGY_FRACTIONS",
    "PROBE_RANKS",
    "AuditResult",
    "moment_features",
    "pair_bias",
    "slice_valid",
    "spectral_profile",
    "bilinear_residual",
    "audit_batch",
    "summarize",
    "format_report",
]

#: Moment-feature degrees to fit, and the resulting number of appended Q/K
#: dimensions.  Degree d contributes ``C(d+3, d)`` symmetric monomials.
MOMENT_DEGREES = (1, 2, 3)

#: Cumulative Frobenius-energy thresholds to report a rank for.
ENERGY_FRACTIONS = (0.90, 0.95, 0.99)

#: Relative singular-value cutoff for the moment-fit pseudo-inverse. See
#: :func:`bilinear_residual` -- without it the degree-3 fit diverges on real jets.
PINV_RTOL = 1e-6

#: Iteration cap for the off-diagonal (masked) moment fit. The EM iteration in
#: :func:`bilinear_residual` converges in ~10-35 steps at every size measured here; the cap only
#: bounds the pathological case.
MASKED_FIT_MAX_ITERS = 64

#: Truncation ranks to report residuals at.  4/14/34 are the moment-feature
#: dimensions; 8/16/32 are round numbers for the Q/K budget.
PROBE_RANKS = (4, 8, 14, 16, 32, 34)


# ---------------------------------------------------------------------------
# Moment features
# ---------------------------------------------------------------------------

def _monomial_exponents(degree: int) -> list[tuple[int, ...]]:
    """Index tuples of the symmetric monomials of a given degree in 4 vars.

    Returns sorted non-decreasing index tuples, so each distinct product
    ``p^mu p^nu ...`` appears exactly once (10 for degree 2, 20 for degree 3).
    """
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")

    combos: list[tuple[int, ...]] = [()]
    for _ in range(degree):
        combos = [
            (*combo, idx)
            for combo in combos
            for idx in range(4)
            if not combo or idx >= combo[-1]
        ]
    return combos


def moment_features(p: Tensor, max_degree: int) -> Tensor:
    """Build symmetric moment features from four-momenta.

    Parameters
    ----------
    p : Tensor
        ``(N, 4)`` four-momenta ``[px, py, pz, E]`` (weaver's ``v`` ordering).
    max_degree : int
        Include all symmetric monomials of degree ``1..max_degree``.

    Returns
    -------
    Tensor
        ``(N, r)`` float64 features with **columns normalized to unit L2 norm**.

        Column scaling is deliberate and free: ``span(Phi)`` is invariant under
        invertible column scaling and the scale is absorbed into the fitted
        ``M``, so the achievable approximation is unchanged — but without it the
        degree-3 block sits at ``O(p^3) ~ 1e6`` while degree 1 is ``O(1e2)``, and
        the least-squares problem becomes numerically hopeless.
    """
    p64 = p.to(torch.float64)
    columns = [
        torch.prod(p64[:, list(exps)], dim=1)
        for degree in range(1, max_degree + 1)
        for exps in _monomial_exponents(degree)
    ]
    phi = torch.stack(columns, dim=1)  # (N, r)

    norms = phi.norm(dim=0, keepdim=True)
    norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    return phi / norms


def moment_dim(max_degree: int) -> int:
    """Number of feature columns :func:`moment_features` produces."""
    return sum(len(_monomial_exponents(d)) for d in range(1, max_degree + 1))


# ---------------------------------------------------------------------------
# Extracting U
# ---------------------------------------------------------------------------

def pair_bias(model: torch.nn.Module, v: Tensor, mask: Tensor) -> Tensor:
    """Return the pairwise interaction bias ``U`` for one batch.

    Calls ``model.pair_embed`` directly rather than hooking the encoder blocks:
    weaver builds the bias once outside the block loop, so the module call *is*
    the authoritative value and no hook is needed.

    Parameters
    ----------
    model : nn.Module
        A weaver ``ParticleTransformer`` (or anything exposing ``.pair_embed``).
        Put it in ``eval()`` mode first — ``PairEmbed`` contains BatchNorm and
        its ``sparse_eval`` dispatch differs between train and eval.
    v : Tensor
        ``(B, 4, N)`` raw four-vectors.
    mask : Tensor
        ``(B, 1, N)`` float, ``1.0`` = real particle.

    Returns
    -------
    Tensor
        ``(B, H, N, N)`` float bias, detached on CPU.
    """
    pair_embed = getattr(model, "pair_embed", None)
    if pair_embed is None:
        raise AttributeError(
            "model has no .pair_embed — the rank audit only applies to arms "
            "that use a pairwise interaction bias"
        )
    with torch.no_grad():
        u = pair_embed(v, uu=None, mask=mask)
    if u.dim() != 4:
        raise RuntimeError(f"expected (B, H, N, N) bias, got shape {tuple(u.shape)}")
    return u.detach().float().cpu()


def slice_valid(u: Tensor, lengths: Sequence[int]) -> list[Tensor]:
    """Cut each jet's bias down to its valid block ``U[b, :, :n, :n]``.

    Padded entries are exact zeros, so keeping them would add ``N - n`` zero
    singular values per head and make any bias look artificially low rank.
    """
    return [u[b, :, : int(n), : int(n)] for b, n in enumerate(lengths)]


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def spectral_profile(matrix: Tensor) -> dict:
    """Singular-value profile of one ``(n, n)`` bias block.

    Returns cumulative Frobenius energy (normalized ``sum sigma_k^2``), the rank
    needed for each :data:`ENERGY_FRACTIONS` threshold, the relative residual of
    the best rank-r truncation for each :data:`PROBE_RANKS`, and an asymmetry
    measure (weaver's ``PairEmbed`` is symmetric, so a large value flags a
    changed configuration).
    """
    m = matrix.to(torch.float64)
    n = m.shape[-1]
    total_sq = float((m**2).sum())

    if n == 0 or total_sq == 0.0:
        return {
            "n": int(n),
            "degenerate": True,
            "frobenius": 0.0,
            "asymmetry": 0.0,
            "rank_for_energy": {f"{f:.2f}": 0 for f in ENERGY_FRACTIONS},
            "residual_at_rank": {str(r): 0.0 for r in PROBE_RANKS},
            "cumulative_energy": [],
        }

    sv = torch.linalg.svdvals(m).to(torch.float64)
    energy = sv**2
    cumulative = torch.cumsum(energy, dim=0) / energy.sum()

    rank_for_energy = {}
    for frac in ENERGY_FRACTIONS:
        hits = torch.nonzero(cumulative >= frac, as_tuple=False)
        rank_for_energy[f"{frac:.2f}"] = (
            int(hits[0].item()) + 1 if hits.numel() else int(n)
        )

    # Eckart-Young: the best rank-r error is the tail of the spectrum.
    residual_at_rank = {}
    for r in PROBE_RANKS:
        if r >= n:
            residual_at_rank[str(r)] = 0.0
        else:
            tail = float(energy[r:].sum())
            residual_at_rank[str(r)] = math.sqrt(max(tail, 0.0) / total_sq)

    asymmetry = float((m - m.T).norm() / m.norm())

    return {
        "n": int(n),
        "degenerate": False,
        "frobenius": math.sqrt(total_sq),
        "asymmetry": asymmetry,
        "rank_for_energy": rank_for_energy,
        "residual_at_rank": residual_at_rank,
        "cumulative_energy": [float(c) for c in cumulative[: max(PROBE_RANKS)]],
    }


# ---------------------------------------------------------------------------
# Gauge and self-pair corrections
# ---------------------------------------------------------------------------
# Two corrections that both change measured rank, and were both missing:
#
#   1. Row centring. Attention applies a softmax over keys, which is invariant to
#      adding any per-query (per-row) constant to the bias. So attention depends on
#      ``U`` only through ``U P_c`` with ``P_c = I - 11^T/n``, and the rank that a
#      factorization actually has to reproduce is ``rank(U P_c)``, not ``rank(U)``.
#      Measuring the raw block answers a question no kernel asks. See
#      ``logs/gauge/`` for the closed-form study of this and for the direction of
#      the error: for full-rank biases, centring pushes the *energy-threshold* rank
#      UP, so choosing a truncation rank from the raw spectrum under-provisions.
#
#   2. The self-pair diagonal. ``U_ii`` is a particle paired with itself, where the
#      pair features degenerate onto the ``eps`` clamp, so the diagonal is an
#      artifact of the feature construction rather than physics. Leaving it in
#      inflates measured rank. It is treated here as *missing data*, not as a value:
#      excluded from the row mean and from the fit residual, never replaced by a
#      number that would itself carry rank.
#
# Both raw and corrected spectra are reported side by side so any change in a
# recorded number can be attributed to one or the other rather than guessed at.

def _zero_diagonal(m: Tensor) -> Tensor:
    n = m.shape[-1]
    eye = torch.eye(n, dtype=torch.bool, device=m.device)
    return m.masked_fill(eye, 0.0)


def row_center(m: Tensor, exclude_diagonal: bool = False) -> Tensor:
    """``m @ P_c``. With *exclude_diagonal*, the row mean is over off-diagonal keys."""
    n = m.shape[-1]
    if not exclude_diagonal:
        return m - m.mean(dim=-1, keepdim=True)
    if n < 2:
        return torch.zeros_like(m)
    off = _zero_diagonal(m)
    row_mean = off.sum(dim=-1, keepdim=True) / (n - 1)
    eye = torch.eye(n, dtype=torch.bool, device=m.device)
    return (off - row_mean).masked_fill(eye, 0.0)


def rho_and_shrink(m: Tensor, exclude_diagonal: bool = True) -> dict:
    """rho and the per-row quantization scale saving, on one (n, n) bias block.

        rho = std(per-row midrange) / mean(within-row half-range)

    plan.md B4 calls this the single measurement that decides whether the gauge buys anything
    for quantization, and until now it had only ever been *swept synthetically* (exp4.py) --
    never measured on a real bias tensor. It predicts the gain because a symmetric per-row
    quantizer must cover `max|row|`, whereas shifting each row by its midrange -- free, since
    softmax cannot see a per-row constant -- only needs the half-range. So

        shrink = mean_i max_j |U_ij|  /  mean_i halfrange_i

    is the realised scale reduction, and `log2(shrink)` is the bits it is worth. Unlike an
    asymmetric quantizer the shift is never stored, so this costs zero zero-points.

    The self-pair diagonal is excluded by default: it sits on the eps clamp of the pair
    features, and being a near-constant `c` per head it would drag every row's midrange
    toward `c` and manufacture a gain that is an artifact of the clamp.
    """
    n = int(m.shape[-1])
    if n < 2:
        return {"rho": float("nan"), "shrink": float("nan"), "equivalent_symmetric_range_bits": float("nan")}
    if exclude_diagonal:
        eye = torch.eye(n, dtype=torch.bool, device=m.device)
        hi = m.masked_fill(eye, float("-inf")).max(dim=-1).values
        lo = m.masked_fill(eye, float("inf")).min(dim=-1).values
        absmax = m.masked_fill(eye, 0.0).abs().max(dim=-1).values
    else:
        hi = m.max(dim=-1).values
        lo = m.min(dim=-1).values
        absmax = m.abs().max(dim=-1).values

    mid = (hi + lo) / 2.0
    half = (hi - lo) / 2.0
    denom = float(half.mean())
    num = float(mid.std(unbiased=False))
    rho = (float("inf") if num > 0 else 0.0) if denom == 0 else num / denom
    mean_absmax = float(absmax.mean())
    shrink = float("inf") if denom == 0 else mean_absmax / denom
    return {
        "rho": rho,
        "shrink": shrink,
        "equivalent_symmetric_range_bits": math.log2(shrink) if 0 < shrink < float("inf") else float("nan"),
        "mean_row_absmax": mean_absmax,
        "mean_row_halfrange": denom,
    }


def self_pair_profile(m: Tensor) -> dict:
    """How much does handling the self-pair diagonal separately cost -- 1 scalar, or n?

    This decides how the B4 result must be *written*, not just what it equals. Row centring is
    unconditionally free; dropping the self-pair is free only if the implementation supplies the
    diagonal some other way. So the question is how expensive "some other way" is, and there are
    three answers, measured here side by side at 90% energy:

    ``rank_centered``
        Row-centred, diagonal left alone. The unconditional number. Costs nothing extra.
    ``rank_minus_constant``
        Row-centred after subtracting ``c I`` with ``c = mean_i U_ii``, i.e. the kernel adds one
        scalar per head and the factorization reproduces the rest.
    ``rank_offdiagonal``
        Row-centred with the diagonal excluded entirely, i.e. the kernel supplies all ``n``
        diagonal entries as a per-particle vector.

    The comparison is decisive because a constant diagonal is the extreme case of cheap-but-costly:
    ``c I`` carries **rank n** while holding **one** degree of freedom. So if
    ``rank_minus_constant`` lands near ``rank_offdiagonal``, the whole gap between the
    unconditional and conditional B4 numbers is bought for one scalar per head and the conditional
    number is the operationally relevant one. If instead it lands near ``rank_centered``, it is the
    diagonal's *variation* that carries the rank, the honest cost is ``n`` values per head, and B4
    must be stated with that attached.
    """
    n = int(m.shape[-1])
    if n < 2:
        return {}
    diag = m.diagonal()
    mean_c = float(diag.mean())
    scale = float(diag.abs().mean())

    def rank90(x: Tensor) -> int:
        return _energy_ranks(torch.linalg.svdvals(x), n)["0.90"]

    eye = torch.eye(n, dtype=m.dtype, device=m.device)
    return {
        "diag_mean": mean_c,
        "diag_std": float(diag.std(unbiased=False)),
        # Relative spread of the diagonal. Near 0 => the diagonal is one number per head.
        "diag_rel_spread": (float(diag.std(unbiased=False)) / scale) if scale > 0 else 0.0,
        "rank_centered": rank90(row_center(m)),
        "rank_minus_constant": rank90(row_center(m - mean_c * eye)),
        "rank_offdiagonal": rank90(row_center(m, exclude_diagonal=True)),
    }


def matrix_variants(block: Tensor, remove_self_pair: bool = True) -> dict[str, Tensor]:
    """The four matrices worth spectra, so the two corrections can be attributed.

    ``raw``       what every audit before 2026-08-30 measured.
    ``no_diag``   self-pair removed only — isolates correction 2.
    ``centered``  row-centred only, over all keys — isolates correction 1.
    ``corrected`` both. **This is the attention-relevant object and the primary result.**
    """
    m = block.to(torch.float64)
    return {
        "raw": m,
        "no_diag": _zero_diagonal(m),
        "centered": row_center(m),
        "corrected": row_center(m, exclude_diagonal=remove_self_pair),
    }


def _energy_ranks(sv: Tensor, n: int) -> dict[str, int]:
    energy = sv ** 2
    total = float(energy.sum())
    if total <= 0.0:
        return {f"{f:.2f}": 0 for f in ENERGY_FRACTIONS}
    cumulative = torch.cumsum(energy, dim=0) / energy.sum()
    out = {}
    for frac in ENERGY_FRACTIONS:
        hits = torch.nonzero(cumulative >= frac, as_tuple=False)
        out[f"{frac:.2f}"] = int(hits[0].item()) + 1 if hits.numel() else int(n)
    return out


_NULL_CACHE: dict[tuple[int, int, int, bool, bool], dict] = {}


def gaussian_null(n: int, trials: int = 64, seed: int = 0,
                  symmetric: bool = True, remove_self_pair: bool = True) -> dict:
    """Energy-threshold rank of a random matrix of the same size — the noise floor.

    **A rank without its null is unusable.** Under iid Gaussian entries the rank needed
    for 90% of Frobenius energy sits near ``n/2``, so at JetClass's mean multiplicity
    (~39) a measured rank in the twenties is indistinguishable from noise. The recorded
    per-head profile ``[22, 21, 26, 13, 9, 12, 6, 4]`` is above this floor for only the
    last few heads.

    ``symmetric=True`` matches the data: weaver's ``PairEmbed`` is symmetric (the audit
    asserts asymmetry < 1e-5), and a symmetric null has a different spectral law
    (semicircle) from an iid one (Marchenko-Pastur), so the correct null is the
    symmetric one. Both are reported because the ``n/2`` rule of thumb on record was
    derived for the iid case.
    """
    key = (int(n), int(trials), int(seed), bool(symmetric), bool(remove_self_pair))
    if key in _NULL_CACHE:
        return _NULL_CACHE[key]

    generator = torch.Generator().manual_seed(seed * 1_000_003 + n)
    names = ("raw", "no_diag", "centered", "corrected")
    draws: dict[str, dict[str, list[int]]] = {
        name: {f"{f:.2f}": [] for f in ENERGY_FRACTIONS} for name in names
    }
    for _ in range(trials):
        a = torch.randn(n, n, generator=generator, dtype=torch.float64)
        if symmetric:
            a = (a + a.T) / math.sqrt(2.0)
        # The null must undergo the SAME transformations as the data. Row centring and
        # diagonal removal change the rank profile by themselves, so comparing a corrected
        # measurement against a raw null is not like-for-like and was wrong before. That is
        # also why `remove_self_pair` is a *parameter* here and not hardcoded: under
        # `--keep-self-pair` the data's `corrected` variant is plain row-centring, and a null
        # that still removed its diagonal would be the same mismatch in a new place.
        for name, m in matrix_variants(a, remove_self_pair=remove_self_pair).items():
            for frac_key, rank in _energy_ranks(torch.linalg.svdvals(m), n).items():
                draws[name][frac_key].append(rank)

    out: dict = {"n": int(n), "trials": int(trials), "symmetric": bool(symmetric),
                 "remove_self_pair": bool(remove_self_pair)}
    for name in names:
        out[name] = {
            k: {
                "median": float(np.median(v)),
                "p90": float(np.percentile(v, 90)),
                "mean": float(np.mean(v)),
                # The per-trial draws are kept, not just their summaries, because
                # `null_matched_to_records` has to rebuild the *batch* statistic trial by trial:
                # a p90 over per-matrix medians is not the null distribution of a batch p90.
                "draws": [int(x) for x in v],
            }
            for k, v in draws[name].items()
        }
    _NULL_CACHE[key] = out
    return out


def null_matched_to_records(records: Sequence[dict], trials: int = 64, seed: int = 0,
                            symmetric: bool = True, remove_self_pair: bool = True) -> dict:
    """The null aggregated over the *same multiset of ``n``* as the real records.

    Matching the size distribution matters: the floor scales with ``n``, so a null computed at a
    single representative ``n`` would not be comparable to a median or a p90 taken over a batch
    with a wide multiplicity spread.

    **The statistic must match too.** The audit's headline number is a p90 over a batch of (jet,
    head) blocks, so the thing to compare it against is the null distribution of *that same batch
    p90* -- not the null's median, and not a p90 taken over per-matrix medians. Those are different
    random variables: at n=40 the per-matrix rank varies by a few units, so a batch p90 sits
    systematically above a batch median, and comparing the data's p90 to the null's median charges
    the data for the spread of the null. An earlier version did exactly that.

    So each trial is replayed as a whole batch: draw one null rank per record from that record's
    own ``n``, form the batch median and batch p90, and summarize those across trials. The returned
    ``median`` and ``p90`` are therefore both *batch* statistics and each is the like-for-like
    partner of the data statistic of the same name. ``batch_p90_draws`` is kept so a one-sided
    p-value can be computed without re-running the null.
    """
    usable = [r for r in records if not r.get("degenerate")]
    if not usable:
        return {}
    names = ("raw", "no_diag", "centered", "corrected")
    nulls = [
        gaussian_null(int(rec["n"]), trials=trials, seed=seed, symmetric=symmetric,
                      remove_self_pair=remove_self_pair)
        for rec in usable
    ]

    out: dict = {"symmetric": bool(symmetric), "trials": int(trials),
                 "remove_self_pair": bool(remove_self_pair), "variants": {}}
    for name in names:
        per_threshold: dict = {}
        for frac in ENERGY_FRACTIONS:
            frac_key = f"{frac:.2f}"
            batch_median, batch_p90 = [], []
            for trial in range(trials):
                sample = [null[name][frac_key]["draws"][trial] for null in nulls]
                batch_median.append(float(np.median(sample)))
                batch_p90.append(float(np.percentile(sample, 90)))
            per_threshold[frac_key] = {
                "median": float(np.median(batch_median)),
                "p90": float(np.median(batch_p90)),
                "batch_p90_p10": float(np.percentile(batch_p90, 10)),
                "batch_p90_p90": float(np.percentile(batch_p90, 90)),
                "batch_p90_draws": batch_p90,
            }
        out["variants"][name] = {"rank_for_energy": per_threshold}
    # Back-compat: `rank_for_energy` at the top level is the RAW null.
    out["rank_for_energy"] = out["variants"]["raw"]["rank_for_energy"]
    return out


def bilinear_residual(matrix: Tensor, phi: Tensor, *,
                      exclude_diagonal: bool = False) -> float:
    """Relative residual of the best fit ``matrix ~= phi @ M @ phi.T``.

    With ``Phi`` fixed, minimizing over ``M`` is linear least squares whose
    solution is ``M = Phi^+ U (Phi^+)^T``; the fitted value is then the
    projection of ``U`` onto ``span(Phi)`` on both sides.  Uses ``pinv`` rather
    than a normal-equation solve because the moment features are strongly
    collinear (``E`` and ``|p|`` nearly coincide for light constituents), so
    ``Phi^T Phi`` is ill-conditioned by construction.

    Parameters
    ----------
    exclude_diagonal : bool
        Fit **and** score on the off-diagonal only. The self-pair entry ``U_ii`` sits on the
        ``eps`` clamp of the pair features and is a construction artifact, so requiring the fit
        to reproduce it charges the factorization for a value no physical pair produces. Both
        halves matter: an earlier version masked only the score while still fitting against an
        artificial zero diagonal, which is not the best off-diagonal fit and reported 0.14 on a
        matrix that is exactly representable off the diagonal. Default ``False`` preserves the
        historical whole-matrix number; the audit reports both.

    Returns
    -------
    float
        ``||U - Phi M Phi^T||_F / ||U||_F`` over the scored entries, in ``[0, 1]``. 0 = exactly
        representable with ``phi.shape[1]`` appended Q/K dimensions -- but see
        :func:`phi_conditioning` for how many of those dimensions the fit could actually use.
    """
    u = matrix.to(torch.float64)
    weight: Optional[Tensor] = None
    if exclude_diagonal and u.shape[-1] > 1:
        weight = ~torch.eye(u.shape[-1], dtype=torch.bool, device=u.device)

    denom = float(u.norm() if weight is None else u[weight].norm())
    if denom == 0.0:
        return 0.0

    # `rtol` is not optional. The degree-3 moment features are catastrophically
    # collinear at real jet momentum scales even after the column normalisation in
    # `moment_features`, and an untruncated pinv amplifies the near-null space until the
    # "fit" is worse than predicting zero -- observed as a relative residual of 2730 at
    # r=34 on real JetClass jets. Truncating makes it a genuine projection, so the
    # residual is bounded by 1 as it must be. Truncation also means the *effective* feature
    # dimension can be smaller than ``phi.shape[1]``, and by a jet-dependent amount, so
    # ``phi_conditioning`` records it alongside every residual rather than leaving
    # "degree 3 / 34 dimensions" to be read as 34 dimensions actually used.
    phi64 = phi.to(torch.float64)
    pinv = torch.linalg.pinv(phi64, rtol=PINV_RTOL)  # (r, N)

    if weight is None:
        m_hat = pinv @ u @ pinv.T                    # (r, r)
        return float((u - phi64 @ m_hat @ phi64.T).norm() / denom)

    # Off-diagonal-only objective. Fitting the full block with an artificial zero diagonal
    # and merely *masking the score* is not the best off-diagonal fit: the zeros are data the
    # fit is pulled toward. On a matrix that is exactly `Phi M Phi^T` off the diagonal, that
    # reported a residual of 0.14 for something exactly representable -- enough to reject a
    # physical feature family that reproduces every entry the audit says matters.
    #
    # The diagonal is *missing data*, so treat it that way: alternate between fitting and
    # imputing the diagonal from the current fit. This is EM for a linear-Gaussian model with
    # a quadratic objective, so it converges to the masked least-squares optimum -- verified
    # against the exact Kronecker solution (`vec(U) = (Phi kron Phi) vec(M)` restricted to
    # off-diagonal rows), which agrees to machine precision but costs a 16384 x 1156 lstsq per
    # (jet, head) block, ~280x slower than this at N=128.
    eye = ~weight
    filled = u.masked_fill(eye, 0.0)
    previous: Optional[float] = None
    residual = denom
    for _ in range(MASKED_FIT_MAX_ITERS):
        fit = phi64 @ (pinv @ filled @ pinv.T) @ phi64.T
        filled = torch.where(eye, fit, u)
        residual = float((u - fit)[weight].norm())
        if previous is not None and abs(previous - residual) < 1e-13 * max(1.0, previous):
            break
        previous = residual
    return residual / denom


def phi_conditioning(phi: Tensor) -> dict:
    """Effective feature dimension and conditioning of ``phi`` at ``PINV_RTOL``.

    Recorded with every residual because ``bilinear_residual`` truncates: the nominal feature
    count (``degree 3 -> 34``) is an upper bound, and the number of directions the fit can
    actually use is data-dependent. A residual read without this is a residual whose model
    complexity is unknown.
    """
    sv = torch.linalg.svdvals(phi.to(torch.float64))
    if sv.numel() == 0 or float(sv[0]) == 0.0:
        return {"nominal_dim": int(phi.shape[-1]), "effective_dim": 0,
                "condition_number": float("inf"), "rtol": PINV_RTOL}
    kept = int((sv > float(sv[0]) * PINV_RTOL).sum())
    smallest_kept = float(sv[kept - 1]) if kept else 0.0
    return {
        "nominal_dim": int(phi.shape[-1]),
        "effective_dim": kept,
        "condition_number": (float(sv[0]) / smallest_kept) if smallest_kept > 0 else float("inf"),
        "condition_number_untruncated": (
            float(sv[0]) / float(sv[-1]) if float(sv[-1]) > 0 else float("inf")
        ),
        "rtol": PINV_RTOL,
    }


# ---------------------------------------------------------------------------
# Driving one batch
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    """Per-(jet, head) records plus the metadata needed to reproduce them."""

    metadata: dict
    records: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"metadata": self.metadata, "records": self.records}


#: A head counts as factorizable when its p90 rank is at or below this.
#: Matches the ``rank 8`` figure in the §8.7a decision rule.
#:
#: **This threshold is now known to be far too loose, and it is kept only because it was
#: pre-registered.** ``ablation/bias_truncation_probe.py`` measured what these energy thresholds
#: actually cost by truncating a trained bias at inference (2026-09-05, 2000 jets, both arms), and
#: calibrated them against accuracy for the first time:
#:
#:     90% energy  -> rank 10 -> **-0.0335 accuracy**   (~10x the largest real gain in the ablation)
#:     95% energy  -> rank 16 -> -0.0198
#:     99% energy  -> rank 33 -> -0.0013               <- the accuracy-neutral gate
#:
#: For scale: the biggest genuine architectural gain the whole 200k ablation ever produced is
#: ``baseline_wide``'s **+0.00339**. So a bias truncated at the 90%-energy rank costs an order of
#: magnitude more accuracy than the best thing the ablation ever gained, and a "BUILD" verdict read
#: off 90% energy is not a green light. **Read the 0.99 row, not the 0.90 row.**
#: See :data:`ACCURACY_CALIBRATION` and ``logs/audit-2026-09-05/README.md`` (result P5).
LOW_RANK_HEAD_MAX = 8

#: Measured accuracy cost of truncating a *trained* bias at the rank each energy threshold picks.
#: From ``bias_truncation_probe.py`` on ``baseline`` @500k, const_diag mode, 2000 jets. Recorded as
#: data rather than prose so the verdict can quote it and no reader has to find the write-up.
ACCURACY_CALIBRATION = {
    "0.90": {"rank": 10.0, "accuracy_cost": -0.0335},
    "0.95": {"rank": 16.5, "accuracy_cost": -0.0198},
    "0.99": {"rank": 33.0, "accuracy_cost": -0.0013},
}

#: The largest genuine architectural gain in the 200k ablation (``baseline_wide``), for scale.
#: ``baseline_ffn2x``'s larger +0.00296 is excluded: that arm never ran (see cluster-audit §4.1).
BEST_ARCH_GAIN = 0.00339


#: Minimum valid particle count for a jet to enter the audit.
#:
#: Rank is bounded by ``n``, so small jets make low rank trivially achievable
#: and drag the median down: an 8-particle jet cannot exceed rank 8 no matter
#: what the pair MLP does.  Mixing them with realistic jets produces a
#: meaningless statistic (observed: median 2, p90 21, max 41 in the same run).
#: The gate asks whether rank ~8 suffices for jets whose full rank could be
#: 50-100, so audit only those.
MIN_VALID_PARTICLES = 32


def audit_batch(
    model: torch.nn.Module,
    v: Tensor,
    mask: Tensor,
    *,
    degrees: Sequence[int] = MOMENT_DEGREES,
    max_heads: Optional[int] = None,
    min_valid: int = MIN_VALID_PARTICLES,
    remove_self_pair: bool = True,
) -> list[dict]:
    """Measure spectrum and moment-fit residuals for every (jet, head) pair.

    Parameters
    ----------
    model : nn.Module
        Weaver ``ParticleTransformer`` in ``eval()`` mode.
    v : Tensor
        ``(B, 4, N)`` raw four-vectors.
    mask : Tensor
        ``(B, 1, N)`` float validity mask.
    degrees : sequence of int
        Moment degrees to fit (see :data:`MOMENT_DEGREES`).
    max_heads : int, optional
        Only audit the first ``max_heads`` heads — the spectrum is measured per
        head and heads are cheap to subsample when scanning.
    min_valid : int
        Skip jets with fewer valid particles (see :data:`MIN_VALID_PARTICLES`).
    remove_self_pair : bool
        Treat the self-pair diagonal as missing data in the corrected spectra and in
        the corrected moment fit. See :func:`matrix_variants`.

    Returns
    -------
    list of dict
        One record per (jet, head), each carrying ``spectral_profile`` output
        plus ``moment_residual`` keyed by appended-dimension count, and
        ``rank_fraction`` = rank-for-90%-energy divided by ``n`` so the measure
        is comparable across jet sizes.

        Top-level spectral keys are the **raw** measurement, preserved so that numbers
        recorded before the 2026-08-30 corrections stay comparable. The corrected
        measurement is under ``spectra["corrected"]`` and ``moment_residual_corrected``;
        ``spectra`` also carries ``no_diag`` and ``centered`` so a change can be
        attributed to one correction or the other.
    """
    u = pair_bias(model, v, mask)                    # (B, H, N, N)
    lengths = mask.squeeze(1).sum(dim=-1).long().tolist()
    blocks = slice_valid(u, lengths)

    records: list[dict] = []
    for b, block in enumerate(blocks):
        n = block.shape[-1]
        if n < max(2, min_valid):
            continue

        p = v[b, :, :n].T.contiguous()               # (n, 4)
        phis = {
            moment_dim(d): moment_features(p, d)
            for d in degrees
            if moment_dim(d) < n  # a fit with r >= n is trivially exact
        }
        # How many of those nominal dimensions the truncated fit can actually use, per jet.
        # Without this a residual at "34 dimensions" is unreadable: PINV_RTOL discards
        # near-null directions, so the effective model size is data-dependent.
        conditioning = {str(r): phi_conditioning(phi) for r, phi in phis.items()}

        heads = block.shape[0] if max_heads is None else min(block.shape[0], max_heads)
        for h in range(heads):
            head_matrix = block[h]
            # Top-level keys stay the RAW measurement so historical numbers remain
            # directly comparable; the corrected measurement lives under `spectra`.
            record = {"jet": b, "head": h, **spectral_profile(head_matrix)}

            variants = matrix_variants(head_matrix, remove_self_pair=remove_self_pair)
            record["spectra"] = {
                name: spectral_profile(m) for name, m in variants.items()
            }
            record["remove_self_pair"] = bool(remove_self_pair)
            # rho on the RAW block: the quantization question is about the tensor a kernel
            # would actually store, and the gauge shift is what we are pricing, so centring
            # first would remove the very thing being measured.
            record["rho"] = rho_and_shrink(
                variants["raw"], exclude_diagonal=remove_self_pair
            )

            record["moment_residual"] = {
                str(r): bilinear_residual(head_matrix, phi)
                for r, phi in phis.items()
            }
            # The fit that matters: same features, but against the attention-relevant
            # matrix and scored off-diagonal. This is the number that answers "can the
            # (B, H, N, N) bias be replaced by r appended per-particle channels?".
            record["moment_residual_corrected"] = {
                str(r): bilinear_residual(
                    variants["corrected"], phi, exclude_diagonal=remove_self_pair
                )
                for r, phi in phis.items()
            }
            record["moment_conditioning"] = conditioning
            # Prices the self-pair assumption that the `corrected` variant makes for free.
            record["self_pair"] = self_pair_profile(head_matrix)
            # Rank as a fraction of the maximum possible rank, so a rank-8 result
            # on a 40-particle jet is not conflated with rank 8 on an 8-particle
            # jet (where it is vacuous).
            record["rank_fraction"] = (
                record["rank_for_energy"]["0.90"] / n if n else 0.0
            )
            records.append(record)

    return records


def summarize(records: Sequence[dict]) -> dict:
    """Aggregate per-(jet, head) records into the numbers the gate reads.

    Medians rather than means: the spectrum of a 12-particle jet and a
    100-particle jet are not comparable quantities and a mean would let the
    long-jet tail dominate a decision about the typical case.
    """
    usable = [r for r in records if not r["degenerate"]]
    if not usable:
        return {"count": 0, "usable": 0}

    def median(values: Sequence[float]) -> float:
        return float(np.median(np.asarray(values, dtype=np.float64)))

    out: dict = {
        "count": len(records),
        "usable": len(usable),
        "median_n": median([r["n"] for r in usable]),
        "min_n": int(min(r["n"] for r in usable)),
        "median_rank_fraction": median(
            [r.get("rank_fraction", 0.0) for r in usable]
        ),
        "max_asymmetry": max(r["asymmetry"] for r in usable),
        "rank_for_energy": {},
        "residual_at_rank": {},
        "moment_residual": {},
    }

    for frac in ENERGY_FRACTIONS:
        key = f"{frac:.2f}"
        ranks = [r["rank_for_energy"][key] for r in usable]
        out["rank_for_energy"][key] = {
            "median": median(ranks),
            "p90": float(np.percentile(ranks, 90)),
            "max": int(max(ranks)),
        }

    for rank in PROBE_RANKS:
        key = str(rank)
        out["residual_at_rank"][key] = median(
            [r["residual_at_rank"][key] for r in usable]
        )

    dims = sorted({int(d) for r in usable for d in r["moment_residual"]})
    for dim in dims:
        vals = [
            r["moment_residual"][str(dim)]
            for r in usable
            if str(dim) in r["moment_residual"]
        ]
        if vals:
            out["moment_residual"][str(dim)] = {
                "median": median(vals),
                "p90": float(np.percentile(vals, 90)),
                "count": len(vals),
            }

    # Per-head breakdown. Heads are not interchangeable: the aggregate can hide a
    # bimodal split where most heads are near rank-1 and a couple carry all the
    # structure, which changes the verdict from all-or-nothing to a hybrid.
    out["per_head"] = {}
    for head in sorted({r["head"] for r in usable}):
        rows = [r for r in usable if r["head"] == head]
        ranks = [r["rank_for_energy"]["0.90"] for r in rows]
        out["per_head"][str(head)] = {
            "count": len(rows),
            "median": median(ranks),
            "p90": float(np.percentile(ranks, 90)),
            "max": int(max(ranks)),
            "median_rank_fraction": median([r.get("rank_fraction", 0.0) for r in rows]),
        }

    out["low_rank_heads"] = sorted(
        int(h) for h, s in out["per_head"].items() if s["p90"] <= LOW_RANK_HEAD_MAX
    )
    out["num_heads_seen"] = len(out["per_head"])

    # ---- corrections (2026-08-30): the attention-relevant measurement -------------
    # Aggregated exactly like the raw block above, so the two are directly comparable.
    if any("spectra" in r for r in usable):
        out["variants"] = {}
        for name in ("raw", "no_diag", "centered", "corrected"):
            rows = [r["spectra"][name] for r in usable if name in r.get("spectra", {})]
            if not rows:
                continue
            entry: dict = {"rank_for_energy": {}}
            for frac in ENERGY_FRACTIONS:
                key = f"{frac:.2f}"
                ranks = [row["rank_for_energy"][key] for row in rows]
                entry["rank_for_energy"][key] = {
                    "median": median(ranks),
                    "p90": float(np.percentile(ranks, 90)),
                    "max": int(max(ranks)),
                }
            entry["per_head"] = {}
            for head in sorted({r["head"] for r in usable}):
                hrows = [
                    r["spectra"][name] for r in usable
                    if r["head"] == head and name in r.get("spectra", {})
                ]
                if hrows:
                    hranks = [row["rank_for_energy"]["0.90"] for row in hrows]
                    entry["per_head"][str(head)] = {
                        "median": median(hranks),
                        "p90": float(np.percentile(hranks, 90)),
                        "max": int(max(hranks)),
                    }
            out["variants"][name] = entry

        def _low_rank_heads(entry: dict) -> list:
            return sorted(
                int(h) for h, s in entry["per_head"].items()
                if s["p90"] <= LOW_RANK_HEAD_MAX
            )

        # Two different claims, kept apart on purpose (see `verdict`):
        #
        #   `centered`  = row centring only. Unconditionally valid -- softmax over keys cannot
        #                 see a per-row constant, full stop. This is what the gate reads.
        #   `corrected` = centring AND dropping the self-pair diagonal. Valid only if the
        #                 implementation handles the diagonal separately, which is an
        #                 engineering choice, not a property of attention.
        gauge = out["variants"].get("centered")
        if gauge:
            out["gauge_rank_for_energy"] = gauge["rank_for_energy"]
            out["gauge_low_rank_heads"] = _low_rank_heads(gauge)

        corrected = out["variants"].get("corrected")
        if corrected:
            out["corrected_rank_for_energy"] = corrected["rank_for_energy"]
            out["corrected_low_rank_heads"] = _low_rank_heads(corrected)

    sp_rows = [r["self_pair"] for r in usable if r.get("self_pair")]
    if sp_rows:
        out["self_pair"] = {
            key: {
                "median": median([r[key] for r in sp_rows]),
                "p90": float(np.percentile([r[key] for r in sp_rows], 90)),
            }
            for key in ("diag_rel_spread", "rank_centered", "rank_minus_constant",
                        "rank_offdiagonal")
        }
        out["self_pair"]["count"] = len(sp_rows)
        # How much of the centred->off-diagonal rank gap a single scalar per head recovers.
        # 1.0 => one scalar buys the whole gap; 0.0 => the diagonal's variation carries the rank.
        cen = out["self_pair"]["rank_centered"]["median"]
        off = out["self_pair"]["rank_offdiagonal"]["median"]
        con = out["self_pair"]["rank_minus_constant"]["median"]
        out["self_pair"]["gap_recovered_by_one_scalar"] = (
            (cen - con) / (cen - off) if cen > off else float("nan")
        )

    rho_rows = [r["rho"] for r in usable if "rho" in r
                and math.isfinite(r["rho"].get("rho", float("nan")))]
    if rho_rows:
        out["rho"] = {
            key: {
                "median": median([r[key] for r in rho_rows]),
                "p10": float(np.percentile([r[key] for r in rho_rows], 10)),
                "p90": float(np.percentile([r[key] for r in rho_rows], 90)),
            }
            for key in ("rho", "shrink", "equivalent_symmetric_range_bits")
        }
        out["rho"]["count"] = len(rho_rows)

    if any("moment_residual_corrected" in r for r in usable):
        out["moment_residual_corrected"] = {}
        dims_c = sorted({
            int(d) for r in usable for d in r.get("moment_residual_corrected", {})
        })
        for dim in dims_c:
            vals = [
                r["moment_residual_corrected"][str(dim)] for r in usable
                if str(dim) in r.get("moment_residual_corrected", {})
            ]
            if vals:
                out["moment_residual_corrected"][str(dim)] = {
                    "median": median(vals),
                    "p90": float(np.percentile(vals, 90)),
                    "count": len(vals),
                }

    return out


def verdict(summary: dict) -> str:
    """Apply the §8.7a decision rule to a summary.

    Reads the **p90** rank, not the median: a factorization that serves the
    typical jet but fails the worst decile degrades exactly the busy,
    high-multiplicity jets that carry the discrimination, so the gate is set on
    the tail.
    """
    if not summary.get("usable"):
        return "INCONCLUSIVE — no usable (jet, head) blocks"

    # Gate on `centered`: row centring ONLY.
    #
    # Two corrections were being applied to the headline number, and only one of them is a
    # property of attention. Row centring is unconditionally free -- softmax over keys is
    # invariant to a per-row constant. Dropping the self-pair diagonal is not: `U_ii` does enter
    # the softmax and does re-weight self-attention, so removing it is free only if the kernel
    # handles the diagonal separately. Gating on the diagonal-dropped variant therefore
    # overstates how factorizable the bias is, and the failure is not subtle -- for a pure
    # self-pair boost `U = c I`, which is a real change to attention, the diagonal-dropped
    # variant is *identically zero* and reports rank 0, i.e. "this bias is free". It is not.
    #
    # So `centered` decides, and `corrected` is reported next to it as the conditional number.
    gauge = summary.get("gauge_rank_for_energy")
    if gauge:
        variant = "centered"
        energy_90 = gauge["0.90"]["p90"]
        low = summary.get("gauge_low_rank_heads", [])
    elif summary.get("corrected_rank_for_energy"):
        # Summaries written before this split recorded only the diagonal-dropped variant.
        variant = "corrected"
        energy_90 = summary["corrected_rank_for_energy"]["0.90"]["p90"]
        low = summary.get("corrected_low_rank_heads", [])
    else:
        variant = "raw"
        energy_90 = summary["rank_for_energy"]["0.90"]["p90"]
        low = summary.get("low_rank_heads", [])
    total = summary.get("num_heads_seen", 0)

    # What the diagonal assumption is worth, stated rather than folded in.
    conditional = ""
    if gauge and summary.get("corrected_rank_for_energy"):
        also = summary["corrected_rank_for_energy"]["0.90"]["p90"]
        conditional = (
            f" [additionally dropping the self-pair diagonal would give rank {also:.0f}, "
            "but only if the kernel handles the diagonal separately]"
        )

    # Compare against the null in the right direction AND with the same statistic.
    #
    # A random matrix needs a rank near N/2 for 90% of its energy, so a measured rank *well
    # below* the null is evidence FOR structure and a rank at or above it is the absence of one.
    # (An earlier version had this inverted and reported "indistinguishable from noise" for
    # exactly the low-rank case it was meant to confirm.) The null must be the null for the same
    # variant, since centring and diagonal removal change the rank profile by themselves -- and
    # for the same *statistic*: `energy_90` is a batch p90, so it is compared against the null
    # distribution of the batch p90, not against the null's median. Comparing a p90 to a median
    # charges the data for the null's spread.
    null_all = summary.get("null") or {}
    null_v = (null_all.get("variants", {}).get(variant, {})
              .get("rank_for_energy", {}).get("0.90", {}))
    null_p90 = null_v.get("p90")
    if null_p90:
        ratio = energy_90 / null_p90
        draws = null_v.get("batch_p90_draws") or []
        # One-sided: how often a matched random bias is at least as low-rank as the data.
        # Add-one (Davison-Hinkley) Monte Carlo p-value: (1 + #{null <= data}) / (1 + #draws).
        # The naive #{...}/#draws prints "p=0.000" when no null draw is as low as the data, which is
        # not a reportable quantity -- with 64 draws the smallest attainable p is 1/65 = 0.0154, and
        # claiming 0.000 asserts precision the simulation cannot supply. Reported as an upper bound
        # when no draw is as extreme, because that is exactly what the simulation licenses.
        if draws:
            hits = sum(1 for d in draws if d <= energy_90)
            p_add_one = (1 + hits) / (1 + len(draws))
            pval = (
                f", Monte Carlo p{'<=' if hits == 0 else '='}{p_add_one:.4f} "
                f"({hits}/{len(draws)} null batches at or below; add-one estimate)"
            )
        else:
            pval = ""
        if energy_90 >= null_p90:
            return _finish(
                f"INCONCLUSIVE — {variant} p90 rank {energy_90:.0f} is at or above the matched "
                f"random-matrix batch p90 {null_p90:.0f} (ratio {ratio:.2f}{pval}). No evidence "
                "of low-rank structure beyond what a random bias of the same jet sizes would "
                "show." + conditional
            )
        structure = (
            f" [structure confirmed: {ratio:.2f}x the matched random-matrix batch p90 "
            f"{null_p90:.0f}{pval}]"
        )
    else:
        structure = ""
    structure += conditional

    # Checked before the aggregate thresholds: heads are independent bias
    # channels, so a split population admits a hybrid the aggregate would miss.
    if total and 0 < len(low) < total:
        return _finish(
            f"HYBRID — {len(low)}/{total} heads factorize at rank "
            f"<= {LOW_RANK_HEAD_MAX} (heads {low}); the rest need up to rank "
            f"{energy_90:.0f}. Factorize the low-rank heads and keep a dense "
            "bias for the remainder rather than deciding all-or-nothing." + structure
        )

    if energy_90 <= LOW_RANK_HEAD_MAX:
        return _finish(
            f"BUILD N8 — rank {energy_90:.0f} carries 90% of Frobenius energy "
            f"(threshold: <= {LOW_RANK_HEAD_MAX})" + structure
        )
    if energy_90 > 40:
        return _finish(
            f"N8 IS DEAD — needs rank {energy_90:.0f} for 90% energy "
            "(threshold: > 40); fall back to N7 sparse execution" + structure
        )
    return _finish(
        f"INTERMEDIATE — rank {energy_90:.0f} for 90% energy; build the "
        "degree-2 (14-dim) variant and re-gate on measured quality" + structure
    )


def _finish(message: str) -> str:
    """Single exit for every substantive verdict, so the calibration cannot be dropped.

    Routing all paths through one function rather than appending at each `return` is deliberate:
    the first version of this appended only on three of six paths, and the two that a real
    checkpoint actually reaches (HYBRID, and INCONCLUSIVE-vs-null) were among the ones missed.
    """
    return message + _calibration_warning()


def _calibration_warning() -> str:
    """Attach the measured accuracy cost to every 90%-energy verdict.

    Appended unconditionally rather than left in a docstring: this string is what gets pasted into
    notes and read months later, so the caveat has to travel *with* the number. A verdict that says
    "BUILD" off a 90%-energy rank, with no indication that the rank it blessed costs ~10x the
    largest real gain the ablation ever found, is an actively misleading artifact -- and this
    repository's whole history is of exactly that failure mode.
    """
    cost = ACCURACY_CALIBRATION["0.90"]["accuracy_cost"]
    safe = ACCURACY_CALIBRATION["0.99"]
    return (
        f"\n  CALIBRATION (measured, not assumed): a bias truncated at the 90%-energy rank costs "
        f"{cost:+.4f} accuracy — about {abs(cost) / BEST_ARCH_GAIN:.0f}x the largest genuine "
        f"architectural gain in the whole 200k ablation (+{BEST_ARCH_GAIN}). The accuracy-neutral "
        f"threshold is 99% energy (rank ~{safe['rank']:.0f}, {safe['accuracy_cost']:+.4f}). So read "
        "the 0.99 row above, not the 0.90 row, and treat any 'BUILD' here as 'worth measuring with "
        "ablation/bias_truncation_probe.py', never as 'safe to replace'."
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_report(result: AuditResult) -> str:
    """Render a human-readable summary of an audit."""
    meta = result.metadata
    summary = meta["summary"]
    lines = [
        "Pair-bias rank audit (hypothesis T0.1)",
        "=" * 62,
        f"  weights        : {meta['weights']}",
        f"  jets source    : {meta['source']}",
        f"  jets / heads   : {meta['num_jets']} / {meta['num_heads']}",
        f"  (jet,head) obs : {summary.get('usable', 0)}"
        f"  (of {summary.get('count', 0)} before the min-N filter)",
        f"  valid N        : median {summary.get('median_n', 0):.1f}, "
        f"min {summary.get('min_n', 0)} (filter: >= {meta['min_valid']})",
        f"  rank/N at 90%  : {summary.get('median_rank_fraction', 0):.3f} median",
        f"  max asymmetry  : {summary.get('max_asymmetry', 0):.2e}",
        "",
    ]

    if not summary.get("usable"):
        lines.append("  no usable blocks — nothing to report")
        return "\n".join(lines)

    variants = summary.get("variants") or {}

    if variants:
        def variant_null(name: str, key: str, which: str = "null") -> Optional[float]:
            """The null p90 for *this* variant. Each column gets its own null, because row
            centring and diagonal removal shift the random-matrix floor by themselves -- so the
            back-compat top-level `null` (which is the RAW null) is not the right comparison for
            any other column."""
            entry = ((summary.get(which) or {}).get("variants", {}).get(name, {})
                     .get("rank_for_energy", {}).get(key))
            return entry.get("p90") if entry else None

        lines.append("  Spectral rank for cumulative Frobenius energy — by correction")
        lines.append("  ------------------------------------------------------------")
        lines.append(
            "    self-pair diagonal removed: "
            f"{meta.get('remove_self_pair', True)}; row-centred: True"
        )
        lines.append(
            "    '-mean' is the number the gate reads: softmax is invariant to a per-row shift,"
        )
        lines.append(
            "    so a factorization must reproduce U@P_c, not U. '+nodiag' additionally drops the"
        )
        lines.append(
            "    self-pair diagonal, which is free ONLY if the kernel handles the diagonal"
        )
        lines.append(
            "    separately — it is not a property of attention, so it does not set the verdict."
        )
        lines.append(
            "    Each column is compared against a null that had the SAME transformation applied,"
        )
        lines.append(
            "    at the SAME statistic (batch p90). A rank at or above its null is noise."
        )
        lines.append("")
        header = (
            f"    {'energy':>7}  {'raw p90':>8}  {'-diag':>8}  {'-mean':>8}  "
            f"{'+nodiag':>8}  {'-mean null':>10}  {'iid':>7}  {'below null?':>11}"
        )
        lines.append(header)
        for frac in ENERGY_FRACTIONS:
            key = f"{frac:.2f}"

            def p90(name: str) -> Optional[float]:
                entry = variants.get(name, {}).get("rank_for_energy", {}).get(key)
                return entry["p90"] if entry else None

            raw, nod, cen, cor = (p90(n) for n in
                                  ("raw", "no_diag", "centered", "corrected"))
            nl = variant_null("centered", key)
            nli = variant_null("centered", key, "null_iid")
            verdict_cell = "—"
            if cen is not None and nl is not None:
                # Below the null is the structured direction.
                verdict_cell = "yes" if cen < nl else "**NO**"

            def fmt(x: Optional[float], width: int) -> str:
                return f"{x:>{width}.1f}" if x is not None else f"{'—':>{width}}"

            lines.append(
                f"    {frac:>7.0%}  {fmt(raw, 8)}  {fmt(nod, 8)}  {fmt(cen, 8)}  "
                f"{fmt(cor, 8)}  {fmt(nl, 10)}  {fmt(nli, 7)}  {verdict_cell:>11}"
            )
        lines.append("")
        lines.append(
            "    raw = pre-2026-08-30 measurement; -diag = self-pair removed only;"
        )
        lines.append(
            "    -mean = row-centred only (THE GATE); +nodiag = both. Comparing the middle two"
        )
        lines.append(
            "    attributes any change to one correction or the other."
        )
        lines.append("")

    lines.append("  Spectral rank needed for cumulative Frobenius energy (raw block)")
    lines.append("  ---------------------------------------------------------------")
    lines.append(f"    {'energy':>8}  {'median':>8}  {'p90':>8}  {'max':>6}")
    for frac in ENERGY_FRACTIONS:
        row = summary["rank_for_energy"][f"{frac:.2f}"]
        lines.append(
            f"    {frac:>8.0%}  {row['median']:>8.1f}  {row['p90']:>8.1f}  "
            f"{row['max']:>6d}"
        )

    lines += [
        "",
        "  Relative residual  ||U - U_r|| / ||U||",
        "  --------------------------------------",
        f"    {'rank':>6}  {'best (SVD)':>12}  {'moment feats':>14}"
        + ("  " + f"{'corrected':>14}" if summary.get("moment_residual_corrected")
           else ""),
    ]
    corrected_fits = summary.get("moment_residual_corrected") or {}
    for rank in PROBE_RANKS:
        best = summary["residual_at_rank"][str(rank)]
        moment = summary["moment_residual"].get(str(rank))
        moment_text = f"{moment['median']:>14.4f}" if moment else f"{'-':>14}"
        corr = corrected_fits.get(str(rank))
        corr_text = f"  {corr['median']:>14.4f}" if corr else ""
        lines.append(f"    {rank:>6}  {best:>12.4f}  {moment_text}{corr_text}")

    lines += [
        "",
        "    best (SVD)   = Eckart-Young optimum, an upper bound on any",
        "                   factorization at that rank",
        "    moment feats = achievable with r appended Q/K dims built from",
        "                   symmetric powers of the four-momenta",
        "    corrected    = the same fit against U@P_c with the self-pair diagonal",
        "                   scored as missing. This is the number that answers",
        "                   'can the (B,H,N,N) bias be replaced by r per-particle",
        "                   channels?'. Note the pair features are LOGS of bilinear",
        "                   quantities (ln m^2 etc.); the bilinear part is exactly",
        "                   low rank, the log is what breaks it, so do not expect r=4.",
    ]

    per_head = summary.get("per_head")
    if per_head:
        lines += [
            "",
            "  Per-head rank for 90% energy  (heads are independent bias channels)",
            "  ------------------------------------------------------------------",
            f"    {'head':>5}  {'median':>7}  {'p90':>7}  {'max':>5}  "
            f"{'rank/N':>7}  factorizable",
        ]
        for head in sorted(per_head, key=int):
            row = per_head[head]
            flag = "yes" if row["p90"] <= LOW_RANK_HEAD_MAX else "no"
            lines.append(
                f"    {head:>5}  {row['median']:>7.1f}  {row['p90']:>7.1f}  "
                f"{row['max']:>5d}  {row['median_rank_fraction']:>7.3f}  {flag}"
            )

    # The row-centred per-head table is the actual result; the raw one above is kept only for
    # comparison. verdict() reads these numbers, so printing only the raw table made the verdict
    # look inconsistent with its own evidence. `centered` and not `corrected`: centring is free
    # by softmax invariance, whereas dropping the self-pair diagonal is free only if the kernel
    # handles the diagonal separately, and treating that as given reports rank 0 for a pure
    # self-pair boost -- a bias that does change attention.
    gauge_heads = (variants.get("centered") or {}).get("per_head") or {}
    if gauge_heads:
        null_g = ((summary.get("null") or {}).get("variants", {}).get("centered", {})
                  .get("rank_for_energy", {}).get("0.90", {}))
        # p90 against the null's batch p90: like-for-like statistics, not p90 against a median.
        np90 = null_g.get("p90")
        lines += [
            "",
            "  ROW-CENTRED per-head rank for 90% energy  (the attention-relevant number)",
            "  ----------------------------------------------------------------------",
            f"    {'head':>5}  {'median':>7}  {'p90':>7}  {'max':>5}  {'vs null':>8}  "
            f"factorizable",
        ]
        for head in sorted(gauge_heads, key=int):
            row = gauge_heads[head]
            flag = "yes" if row["p90"] <= LOW_RANK_HEAD_MAX else "no"
            ratio = f"{row['p90'] / np90:.2f}x" if np90 else "—"
            lines.append(
                f"    {head:>5}  {row['median']:>7.1f}  {row['p90']:>7.1f}  "
                f"{row['max']:>5d}  {ratio:>8}  {flag}"
            )
        if np90:
            lines.append(
                f"    matched random-matrix batch p90 for the same jet sizes: {np90:.1f}"
            )
        cor_agg = (variants.get("corrected") or {}).get("rank_for_energy") or {}
        if cor_agg:
            lines.append(
                f"    conditional: additionally dropping the self-pair diagonal gives p90 "
                f"{cor_agg['0.90']['p90']:.1f}, valid only if the diagonal is handled separately."
            )

    sp = summary.get("self_pair")
    if sp:
        lines += [
            "",
            "  What the self-pair assumption costs  (1 scalar per head, or n?)",
            "  --------------------------------------------------------------",
            "  Row centring is free unconditionally. Dropping the self-pair diagonal is free only",
            "  if the kernel supplies the diagonal some other way, so the question is how much",
            "  that costs. `c*I` is the extreme case: it carries rank n while holding ONE degree",
            "  of freedom, so if subtracting a single per-head constant recovers most of the gap,",
            "  the conditional number is the operationally relevant one -- for one scalar/head.",
            "",
            f"    {'treatment':<38}{'median':>8}{'p90':>8}  extra storage per head",
            f"    {'row-centred, diagonal left alone':<38}"
            f"{sp['rank_centered']['median']:>8.1f}{sp['rank_centered']['p90']:>8.1f}  none",
            f"    {'row-centred after subtracting c*I':<38}"
            f"{sp['rank_minus_constant']['median']:>8.1f}"
            f"{sp['rank_minus_constant']['p90']:>8.1f}  1 scalar",
            f"    {'row-centred, diagonal excluded':<38}"
            f"{sp['rank_offdiagonal']['median']:>8.1f}"
            f"{sp['rank_offdiagonal']['p90']:>8.1f}  n values",
            "",
            f"    diagonal relative spread std/mean|U_ii|: median "
            f"{sp['diag_rel_spread']['median']:.4f}  (near 0 => the diagonal IS one number)",
        ]
        recovered = sp.get("gap_recovered_by_one_scalar")
        if recovered is not None and math.isfinite(recovered):
            lines.append(
                f"    one scalar per head recovers {recovered:.0%} of the centred->off-diagonal "
                "rank gap"
            )
            if recovered >= 0.8:
                lines.append(
                    "    => STATE B4 UNCONDITIONALLY, with a named '+1 scalar per head' term."
                )
            elif recovered <= 0.3:
                lines.append(
                    "    => the diagonal's VARIATION carries the rank. The honest cost is n values"
                )
                lines.append(
                    "       per head, and B4 must be written with that attached."
                )
            else:
                lines.append(
                    "    => intermediate; report all three numbers rather than picking one."
                )

    rho_s = summary.get("rho")
    if rho_s:
        lines += [
            "",
            f"  Gauge quantization gain  (rho on {meta['weights']} weights, "
            f"{'real jets' if meta['source'] != 'synthetic' else 'SYNTHETIC jets'})",
            "  ---------------------------------------------------------------",
            f"    {'quantity':<14}{'p10':>9}{'median':>9}{'p90':>9}",
            f"    {'rho':<14}{rho_s['rho']['p10']:>9.4f}{rho_s['rho']['median']:>9.4f}"
            f"{rho_s['rho']['p90']:>9.4f}",
            f"    {'shrink':<14}{rho_s['shrink']['p10']:>9.4f}{rho_s['shrink']['median']:>9.4f}"
            f"{rho_s['shrink']['p90']:>9.4f}",
            f"    {'equiv. range bits':<14}{rho_s['equivalent_symmetric_range_bits']['p10']:>9.4f}"
            f"{rho_s['equivalent_symmetric_range_bits']['median']:>9.4f}{rho_s['equivalent_symmetric_range_bits']['p90']:>9.4f}",
            "",
            "    rho    = std(per-row midrange) / mean(within-row half-range). A HETEROGENEITY",
            "             statistic, and NOT an explanation of shrink. shrink = 1 + mean|mid|/mean(half),",
            "             and std(mid) is not mean|mid|: rows all spanning [9, 11] give rho = 0.0000",
            "             while shrink = 11.0x (verified). So a low rho does NOT imply a low shrink;",
            "             report them side by side, never one as the mechanism for the other.",
            "    shrink = mean_i(half_i + |mid_i|) / mean_i(half_i), the range reduction from shifting",
            "             each row by its own midrange. exp4.py frames",
            "             this as the predictor of the gain; it had only ever been swept",
            "             synthetically, never measured on a real bias.",
            "    shrink = mean_i max_j|U| / mean_i halfrange_i -- the per-row quantization scale",
            "             saved by shifting each row by its midrange, which is free because",
            "             softmax cannot see a per-row constant and the shift is never stored",
            "             (so, unlike an asymmetric quantizer, zero zero-points).",
            f"    {'':<6}n = {rho_s['count']} (jet, head) blocks; self-pair diagonal excluded.",
        ]
        if meta["weights"] == "untrained" or meta["source"] == "synthetic":
            lines.append(
                "    NOT A RESULT: rho is only meaningful on trained weights and real jets. "
                "This row is a null baseline."
            )

    lines += ["", f"  VERDICT: {meta['verdict']}"]

    if meta["weights"] == "untrained":
        lines += [
            "",
            "  WARNING: random weights. The pair MLP is an arbitrary smooth",
            "  function of the four pair features here, so this spectrum is a",
            "  NULL BASELINE only. The §8.7a gate applies to trained weights;",
            "  re-run with --checkpoint once an arm has trained.",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _synthetic_batch(num_jets: int, num_particles: int, seed: int):
    """Physically valid synthetic jets, reusing the property-test builders.

    Keeps one definition of "valid four-vector batch" in the repo rather than
    growing a second, subtly different generator here.
    """
    from variants.tests.strategies import _make_four_vectors, _make_lengths_masks

    generator = torch.Generator()
    generator.manual_seed(seed)
    lengths = torch.randint(
        max(2, num_particles // 4),
        num_particles + 1,
        (num_jets,),
        generator=generator,
    ).tolist()

    v = _make_four_vectors(lengths, num_particles, generator)
    mask, _ = _make_lengths_masks(lengths, num_particles)
    return v, mask


def _real_batch(data_dir: str, num_jets: int, num_particles: int,
                shards: Optional[Sequence[int]] = None, num_classes: int = 10):
    """Load real jets from ragged CSR ``.pt`` shards via the canonical loader.

    Uses :func:`dataloader.ragged_loader.load_bench_batch`, which performs one
    vectorised CSR gather, rather than a Python loop over the loader's per-jet
    flat ``__getitem__``. Beyond speed that also avoids depending on
    ``RaggedShardDataset.__len__``, which is an *estimate*
    (``events_per_shard x num_files``) until shards have actually been loaded.

    Clipping is reported rather than silent: this audit measures the rank of the
    pair bias, and rank is bounded by the number of valid particles, so quietly
    truncating high-multiplicity jets would bias exactly the jets the
    factorisation decision hinges on (see plan section 8.7a).
    """
    from dataloader.ragged_loader import load_bench_batch

    # One .pt shard is one ROOT file is ONE CLASS (see preprocessing/convert_*.py).
    # `load_bench_batch` defaults to `shard_idx=0`, so calling it once — which every
    # audit before 2026-08-30 did — measures a single-class batch and calls it "real
    # jets". Pair-bias structure is exactly what differs between a QCD jet and a top
    # jet, so a single-class audit is not a measurement of the model's bias, it is a
    # measurement of the model's bias *on QCD*. Draw from one shard per class instead.
    # Shards are grouped by class, NOT interleaved: val_5M has 50 shards for 10 classes,
    # so indices 0..9 land inside the first two classes only (measured: 2/10 coverage).
    # Auto-select evenly spaced indices instead, which is stride = n_shards // num_classes.
    if shards is None:
        available = sorted(Path(data_dir).glob("*.pt"))
        n_shards = len(available)
        if n_shards == 0:
            raise FileNotFoundError(f"no .pt shards in {data_dir}")
        stride = max(1, n_shards // num_classes)
        shards = list(range(0, n_shards, stride))[:num_classes]
        print(f"auto-selected {len(shards)} of {n_shards} shards with stride {stride}: "
              f"{shards}", flush=True)
    shards = list(dict.fromkeys(int(s) for s in shards)) or [0]
    per_shard = max(1, num_jets // len(shards))

    vs, masks, ys = [], [], []
    for shard_idx in shards:
        _x, v_s, mask_s, y_s = load_bench_batch(
            data_dir, batch_size=per_shard, shard_idx=shard_idx
        )
        vs.append(v_s)
        masks.append(mask_s)
        ys.append(y_s)

    # Shards differ in max-in-batch width, so pad to the widest before concatenating.
    width = max(int(t.shape[-1]) for t in vs)
    v = torch.cat([torch.nn.functional.pad(t, (0, width - t.shape[-1])) for t in vs], 0)
    mask = torch.cat(
        [torch.nn.functional.pad(t, (0, width - t.shape[-1])) for t in masks], 0
    )
    y = torch.cat(ys, 0)

    # Report class coverage rather than assuming it. `y` was previously discarded, so
    # nothing could have caught the single-class bug.
    present = y.argmax(dim=-1).bincount(minlength=y.shape[-1]).tolist()
    covered = sum(1 for c in present if c > 0)
    print(
        f"class coverage: {covered}/{y.shape[-1]} classes over {len(shards)} shard(s) "
        f"{shards}; per-class jet counts {present}",
        flush=True,
    )
    if covered < 2:
        print(
            "WARNING: this batch is single-class. One .pt shard is one ROOT file is one "
            "class, so pass --shards with one index per class (e.g. --shards 0 1 2 3 4 "
            "5 6 7 8 9). A single-class rank number does not generalise.",
            flush=True,
        )

    counts = mask.sum(dim=-1).squeeze(-1)  # (B,) valid particles per jet
    natural = int(v.shape[-1])

    if natural > num_particles:
        n_clipped = int((counts > num_particles).sum())
        print(
            f"WARNING: batch max multiplicity is {natural} but --num-particles "
            f"is {num_particles}; {n_clipped} of {v.shape[0]} jets will be "
            f"truncated. Rank is bounded by the valid-particle count, so raise "
            f"--num-particles (JetClass global max is 183) to avoid biasing the "
            f"high-multiplicity tail.",
            flush=True,
        )
        v = v[..., :num_particles]
        mask = mask[..., :num_particles]
    elif natural < num_particles:
        # Pad up so the reported --num-particles matches the tensor width, and
        # so a --data-dir run is directly comparable with a synthetic one.
        width = num_particles - natural
        v = torch.nn.functional.pad(v, (0, width))
        mask = torch.nn.functional.pad(mask, (0, width))

    return v, mask, shards


def _strip_module_prefixes(state: dict) -> dict:
    """Strip DDP ``module.`` / torch.compile ``_orig_mod.`` key prefixes.

    ``train.save_checkpoint`` unwraps DDP before saving, but a state dict that
    passed through ``torch.compile`` (or an older tool) can still carry these
    prefixes — and ``load_state_dict`` is strict by default, so the audit must
    normalize them rather than fail on a checkpoint it can otherwise read.
    """
    stripped = {}
    for key, value in state.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        stripped[key] = value
    return stripped


def _load_model_from_checkpoint(payload: dict):
    """Rebuild the audit model from a checkpoint's stored config.

    The audit measures what the *trained* bias does, so the model must be
    reconstructed the way training built it, not the way this script's
    defaults would: the serialized ``AblationConfig`` is authoritative for
    the arm and its hyperparameters, and the part-kernel fusion is re-applied
    when the checkpoint was trained with it (``ablation.train``'s
    ``_apply_part_kernels``).  Config keys that no longer exist in the
    dataclass are dropped rather than crashing the load, so old checkpoints
    stay readable.

    Returns
    -------
    (model, config)
        The reconstructed model with the checkpoint weights loaded, and the
        config it was built from.
    """
    import dataclasses as _dataclasses

    from ablation.config import AblationConfig
    from variants import build_variant_part

    raw = payload.get("config") or {}
    known = {f.name for f in _dataclasses.fields(AblationConfig)}
    config = AblationConfig(**{k: v for k, v in raw.items() if k in known})

    model = build_variant_part(config.arm, **config.model_kwargs())

    if config.use_part_kernels:
        # Same wrap as ablation/train.py `_apply_part_kernels`: the fused
        # PairEmbed dispatch is the code path that produced the trained
        # activations, so the audit must read the bias through it too.
        from ml4sci_26.part_kernels import optimize_part_model

        target = model.part if hasattr(model, "part") else model
        optimize_part_model(
            target, use_attention_patch=config.part_kernels_attention
        )

    model.load_state_dict(_strip_module_prefixes(payload["model"]))
    return model, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure whether the ParT pairwise interaction bias U is low rank "
            "(hypothesis T0.1 / plan section 8.7a)."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Path to an ablation checkpoint (last.pt / best.pt). The model is "
            "rebuilt from the checkpoint's stored config (arm, pair_embed_dims, "
            "part-kernel fusion) and DDP/compile key prefixes are stripped. "
            "Omit to audit randomly-initialized weights as a null baseline."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Directory of ragged CSR .pt shards (as produced by "
            "preprocessing.convert_jetclass_ragged_pt), read through "
            "dataloader.ragged_loader. Omit to use synthetic physically-valid "
            "jets, which is sufficient for the geometry of the pair features."
        ),
    )
    parser.add_argument("--num-jets", type=int, default=256)
    parser.add_argument(
        "--num-particles",
        type=int,
        default=128,
        help=(
            "Particle-axis width. With --data-dir, jets above this are "
            "truncated and the count is reported; JetClass's global max is 183 "
            "(only ~0.001%% of jets exceed 128, so the default rarely clips). "
            "Raise it to 183 for a no-clipping run (default: 128)."
        ),
    )
    parser.add_argument(
        "--min-particles",
        type=int,
        default=MIN_VALID_PARTICLES,
        help=(
            "Skip jets with fewer valid particles. Rank is bounded by N, so "
            f"small jets make low rank vacuous (default: {MIN_VALID_PARTICLES})."
        ),
    )
    parser.add_argument(
        "--max-heads",
        type=int,
        default=None,
        help="Audit only the first N heads (default: all).",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shards",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Which .pt shards to draw from, with --data-dir. One shard is one ROOT file "
            "is ONE CLASS, and shards are GROUPED by class rather than interleaved, so "
            "0..9 covers only the first one or two classes. Default (unset) auto-selects "
            "one shard per class at stride n_shards//num_classes and reports the class "
            "coverage it achieved."
        ),
    )
    parser.add_argument(
        "--keep-self-pair",
        action="store_true",
        help=(
            "Keep the self-pair diagonal in the corrected spectra. The diagonal sits on "
            "the eps clamp of the pair features, so it is an artifact that inflates "
            "measured rank; excluding it is the default."
        ),
    )
    parser.add_argument(
        "--null-trials",
        type=int,
        default=64,
        help=(
            "Random-matrix trials per distinct jet size for the noise floor. A rank "
            "without its null is unusable: under a Gaussian null the 90%%-energy rank "
            "sits near N/2, so at JetClass's mean multiplicity (~39) a rank in the "
            "twenties is indistinguishable from noise. 0 disables (default: 64)."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the full per-(jet,head) records here as JSON.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)

    from ablation.config import AblationConfig
    from variants import build_variant_part

    # The bias is produced by weaver's stock PairEmbed, which every ParT-family
    # arm shares, so the baseline arm is the right subject regardless of which
    # arm eventually trains.  With --checkpoint the model is instead rebuilt
    # from the checkpoint's own stored config, so the audit measures the exact
    # architecture (and kernel fusion) that produced the weights.
    weights = "untrained"
    ckpt_config: Optional[AblationConfig] = None
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model, ckpt_config = _load_model_from_checkpoint(payload)
        weights = f"{args.checkpoint} @ step {payload.get('step', '?')}"
    else:
        ckpt_config = AblationConfig(arm="baseline")
        model = build_variant_part("baseline", **ckpt_config.model_kwargs())
    model.eval()

    if args.data_dir:
        v, mask, used_shards = _real_batch(
            args.data_dir, args.num_jets, args.num_particles, shards=args.shards
        )
        source = args.data_dir
    else:
        v, mask = _synthetic_batch(args.num_jets, args.num_particles, args.seed)
        source = "synthetic"
        used_shards = None

    remove_self_pair = not args.keep_self_pair
    records: list[dict] = []
    for start in range(0, v.shape[0], args.batch_size):
        stop = start + args.batch_size
        batch_records = audit_batch(
            model,
            v[start:stop],
            mask[start:stop],
            max_heads=args.max_heads,
            min_valid=args.min_particles,
            remove_self_pair=remove_self_pair,
        )
        for record in batch_records:
            record["jet"] += start
        records.extend(batch_records)

    summary = summarize(records)
    if args.null_trials > 0:
        # Attached to the summary before verdict() so the gate can refuse to call a
        # rank "low" when it sits at the random-matrix floor.
        # `remove_self_pair` is threaded through so the null undergoes the same transformation
        # as the data. Under --keep-self-pair the data's `corrected` variant is plain
        # row-centring, and a null that still dropped its diagonal would reintroduce exactly
        # the mismatch this argument exists to prevent.
        summary["null"] = null_matched_to_records(
            records, trials=args.null_trials, seed=args.seed, symmetric=True,
            remove_self_pair=remove_self_pair,
        )
        summary["null_iid"] = null_matched_to_records(
            records, trials=args.null_trials, seed=args.seed, symmetric=False,
            remove_self_pair=remove_self_pair,
        )
    result = AuditResult(
        metadata={
            "hypothesis": "T0.1",
            "weights": weights,
            "source": source,
            "num_jets": int(v.shape[0]),
            "num_heads": int(ckpt_config.num_heads),
            "num_particles": int(args.num_particles),
            "min_valid": int(args.min_particles),
            "seed": args.seed,
            "shards": list(used_shards) if used_shards else None,
            "remove_self_pair": bool(remove_self_pair),
            "null_trials": int(args.null_trials),
            "row_centered": True,
            "energy_fractions": list(ENERGY_FRACTIONS),
            "probe_ranks": list(PROBE_RANKS),
            "moment_degrees": list(MOMENT_DEGREES),
            # How the audited model was reconstructed, so a verdict can be
            # traced back to the exact checkpoint config that produced it.
            "checkpoint_config": {
                "arm": ckpt_config.arm,
                "use_part_kernels": ckpt_config.use_part_kernels,
                "part_kernels_attention": ckpt_config.part_kernels_attention,
                "compile_model": ckpt_config.compile_model,
                "precision": ckpt_config.precision,
                "num_layers": ckpt_config.num_layers,
                "pair_embed_dims": list(ckpt_config.pair_embed_dims)
                if ckpt_config.pair_embed_dims
                else None,
            },
            "summary": summary,
            "verdict": verdict(summary),
        },
        records=records,
    )

    print(format_report(result))

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nwrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
