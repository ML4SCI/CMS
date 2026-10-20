"""Rank-r factorized pair bias — the arm the rank audit licensed.

Replaces weaver's ``PairEmbed`` (an MLP over all ``N^2`` pairs) with per-particle factors whose
outer product *is* the bias::

    U[b, h, i, j] = sum_k s[h, k] * f[b, h, k, i] * f[b, h, k, j]   +   c[h] * delta_ij

so the ``N^2`` MLP is never evaluated: ``N`` per-particle forward passes replace ``N^2`` pair
passes. At ``N = 128`` that is **128x fewer MLP evaluations**.

Why this shape, and not another
-------------------------------
Every choice here is pinned by a measurement in ``logs/audit-2026-09-05/``:

* **rank r.** The trained bias needs p90 rank 25 row-centred, 12 with a per-head scalar handled
  separately (result B4/P4). Truncating a *trained* bias to rank r and measuring accuracy (P5) gives
  the operating curve: r=34 is accuracy-neutral (-0.0005), r=16 costs -0.0205, r=8 costs -0.0385.
* **the ``c[h] * delta_ij`` term.** P4 found one constant per head recovers 90 % of the self-pair
  diagonal's rank cost, because ``c*I`` carries rank ``n`` on a *single* degree of freedom. P5 then
  confirmed it behaviourally: it buys a **factor of 2 in rank** for the same accuracy, on both arms.
  So it is not an optional extra, it is the cheapest rank in the model.
* **symmetric, and indefinite.** The audit asserts weaver's ``PairEmbed`` output is symmetric to
  <1e-5, so the factorization uses the *same* factors on both sides. But ``F^T F`` alone would be
  positive semi-definite, and the measured bias is not: the learnable per-component sign/scale
  ``s[h, k]`` is what keeps the family indefinite. Dropping it would silently restrict the arm to a
  strict subset of the biases it is meant to represent.
* **learned factors, not polynomial ones.** B4a is the reason this arm exists at all: hand-built
  symmetric momentum monomials reach a relative residual of 0.360 at r=34 where the SVD optimum is
  0.034 — a **10x gap**. The rank was never the obstacle, the features were. ParT's pair features are
  *logs* of bilinear quantities; the bilinear part is exactly low rank and the logarithm breaks it.

What this arm does and does not deliver
---------------------------------------
It tests **whether a rank-r bias trains to baseline accuracy**. It does **not** make the model
faster: weaver needs a materialized ``(B, H, N, N)`` ``attn_mask``, so the outer product is still
formed here. The speedup needs a fused kernel that consumes the factors directly and never
materializes ``U`` — which is what ``part_kernels`` exists for. :func:`lowrank_cost_report` reports
the FLOP reduction and the realized-tensor cost *separately*, because conflating "fewer FLOPs" with
"measured speedup" is the exact error this repository has already published once.

Constraint honoured: **no weaver source is modified.** Weaver builds the bias in one line
(``attn_mask = self.pair_embed(v, uu=uu, mask=mask)``) and passes the same tensor to every block, so
replacing ``model.pair_embed`` with this module is the whole integration.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "PARTICLE_FEATURE_NAMES",
    "particle_features",
    "LowRankPairEmbed",
    "lowrank_cost_report",
]

#: Per-particle inputs, chosen so that the *relative* quantities the pair bias actually depends on
#: are cheaply reachable by an outer product rather than having to be approximated:
#:
#: * ``cos phi`` / ``sin phi`` instead of ``phi``, because ``cos(phi_i - phi_j) = cos_i cos_j +
#:   sin_i sin_j`` is **exactly rank 2** in these features — and because a raw periodic ``phi`` puts
#:   a discontinuity in the input space at the wrap point.
#: * ``eta`` and ``eta^2``, because ``(eta_i - eta_j)^2 = eta_i^2 - 2 eta_i eta_j + eta_j^2`` is
#:   **exactly rank 3**. Together with the above, ``Delta R^2`` — the geometric core of every one of
#:   weaver's pair features — is rank 5 by construction rather than learned from scratch.
#: * ``ln pT`` and ``ln E``, the scale variables that ``ln kT`` and ``ln z`` are built from.
#:
#: These are *ingredients*, not the answer: B4a showed hand-built polynomials in these quantities
#: plateau far from the spectral optimum, so the net is free to compose them however it likes.
PARTICLE_FEATURE_NAMES = ("ln_pt", "eta", "eta_sq", "cos_phi", "sin_phi", "ln_e")

_EPS = 1e-8

#: Rough per-feature scales, applied as fixed divisors rather than a BatchNorm.
#:
#: Deliberately *not* BatchNorm, even though weaver's ``PairEmbed`` uses one. A BatchNorm here would
#: reduce over padded slots as well as real particles, so a batch whose ``N_max`` is set by one long
#: jet is mostly padding and the real features get squashed relative to a batch of uniformly short
#: jets — the same jet would embed differently depending on its batch-mates, and those polluted
#: running statistics carry into eval. That is a known live defect in this repository's training
#: path; there is no reason to reproduce it in new code. Fixed divisors are deterministic,
#: batch-composition-independent, and good enough given the features are already O(1)-O(5).
_FEATURE_SCALE = (4.0, 3.0, 9.0, 1.0, 1.0, 4.0)


def particle_features(v: Tensor) -> Tensor:
    """``(B, 4, N)`` four-vectors ``[px, py, pz, E]`` -> ``(B, F, N)`` per-particle features.

    All logs are clamped: padded slots are exact zeros, so ``pT`` and ``E`` are ``0`` there and an
    unclamped ``log`` would produce ``-inf`` that then contaminates the outer product. The caller
    masks the factors afterwards regardless, but ``-inf * 0`` is ``nan``, so the clamp is load-bearing
    rather than cosmetic.
    """
    px, py, pz, energy = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
    pt = torch.sqrt(px * px + py * py).clamp_min(_EPS)
    p_abs = torch.sqrt(px * px + py * py + pz * pz).clamp_min(_EPS)
    # eta = atanh(pz / |p|), guarded away from the +-1 poles.
    eta = torch.atanh((pz / p_abs).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    feats = torch.stack(
        [
            torch.log(pt),
            eta,
            eta * eta,
            px / pt,                      # cos phi
            py / pt,                      # sin phi
            torch.log(energy.clamp_min(_EPS)),
        ],
        dim=1,
    )
    scale = torch.as_tensor(_FEATURE_SCALE, dtype=feats.dtype, device=feats.device)
    return feats / scale[None, :, None]


class LowRankPairEmbed(nn.Module):
    """Drop-in replacement for weaver's ``PairEmbed`` producing a rank-``r`` symmetric bias.

    Signature and output shape match weaver's exactly — ``forward(x, uu=None, mask=None)`` ->
    ``(B, H, N, N)`` — because weaver calls it in one place and hands the result to every block.

    Parameters
    ----------
    rank : int
        Components per head. The P5 operating curve: 34 accuracy-neutral, 16 costs -0.02, 8 costs
        -0.04 *when truncating an already-trained bias*. Training with the constraint should do
        better, since the network adapts to it rather than being deprived of structure it had
        learned to use, so those are a lower bound.
    num_heads : int
        Must equal the model's head count; this module's output *is* the per-head bias.
    hidden : sequence of int
        Per-particle MLP widths.
    self_pair_scalar : bool
        Add the learnable ``c[h] * delta_ij`` term (P4/P5). On by default: it is the cheapest rank in
        the model, worth a factor of 2 in effective rank for one number per head.
    """

    def __init__(self, rank: int, num_heads: int, hidden: Sequence[int] = (64, 64),
                 self_pair_scalar: bool = True, input_dim: Optional[int] = None,
                 init_bias_std: float = 0.096):
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        self.rank = int(rank)
        self.num_heads = int(num_heads)
        self.out_dim = int(num_heads)     # weaver reads `.out_dim` when wiring the bias
        self.feature_dim = len(PARTICLE_FEATURE_NAMES) if input_dim is None else int(input_dim)

        widths = [self.feature_dim, *hidden, num_heads * rank]
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Conv1d(a, b, 1))       # Conv1d over particles == per-particle MLP
            if i < len(widths) - 2:
                layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

        # Per-(head, component) sign and scale. WITHOUT this the bias would be `F^T F`, i.e.
        # positive semi-definite, and the measured bias is indefinite -- the arm would be silently
        # restricted to a strict subset of what it is supposed to represent.
        #
        # Initialised from `init_bias_std` so that the bias this module emits at init matches the
        # `PairEmbed` it replaces (~0.096 measured). Getting this wrong is not cosmetic: `U` is
        # *quadratic* in the factors, so an innocuous-looking 0.1 shrink on the final layer shrinks
        # `U` by 100x. A first attempt did exactly that and produced |U| std ~ 5e-5, i.e. numerically
        # no pair bias at all -- which starts the arm at P5's r=0 point (0.30 accuracy) with a
        # vanishing gradient, since dU/dF = 2 s F goes to zero with F. The arm would then have
        # failed for reasons having nothing to do with rank. Because the factors are RMS-normalised
        # in `forward`, `std(U) ~ |s| * sqrt(rank)`, so the scale is analytically predictable and
        # data-independent -- which is what makes this assertable in a test.
        # Alternating signs, not all-positive. With every scale positive, `F^T diag(s) F` is
        # positive semi-definite, so the arm would start on the BOUNDARY of the indefinite family it
        # is supposed to explore and would have to learn its way off it. The measured bias is
        # indefinite (verified: both eigenvalue signs present), so half the components start
        # negative and the arm begins in the interior. Deterministic alternation rather than random
        # signs, so two builds at the same seed are identical.
        signs = torch.where(
            torch.arange(rank) % 2 == 0, 1.0, -1.0
        ).expand(num_heads, rank).contiguous()
        self.component_scale = nn.Parameter(signs * (init_bias_std / math.sqrt(rank)))
        self.self_pair = nn.Parameter(torch.zeros(num_heads)) if self_pair_scalar else None

    def forward(self, x: Tensor, uu: Optional[Tensor] = None,
                mask: Optional[Tensor] = None) -> Tensor:
        """``x`` is weaver's ``v``: ``(B, 4, N)`` four-vectors. ``uu`` is accepted and ignored."""
        feats = particle_features(x)                       # (B, F, N)
        factors = self.net(feats)                          # (B, H*r, N)
        b, _, n = factors.shape
        factors = factors.view(b, self.num_heads, self.rank, n)

        if mask is not None:
            valid = (mask > 0).to(factors.dtype)
            valid = valid.view(b, 1, 1, n)
            # Padded particles must contribute exactly nothing, so that U is exactly 0 on padded
            # pairs. `rank_audit.slice_valid` and every downstream measurement depend on that
            # contract ("padded pairs in U are exactly 0.0"), and attention's key masking assumes
            # it too.
            factors = factors * valid

        # RMS-normalise the factors per (jet, head) over valid particles, so the bias magnitude is
        # carried entirely by `component_scale`. Two reasons, both load-bearing:
        #   1. it makes the init scale analytically predictable (`std(U) ~ |s| sqrt(rank)`) and
        #      therefore assertable, instead of depending on how the MLP's activations happen to
        #      come out;
        #   2. `U` is quadratic in the factors, so without it the bias scale drifts as the square of
        #      whatever the factor net does, which is a poor thing to leave to chance early in
        #      training.
        # Normalised over (rank, particle) jointly rather than per particle, so *relative*
        # magnitudes between particles survive -- a per-particle normalisation would discard exactly
        # the information that makes one constituent matter more than another.
        if mask is not None:
            denom = valid.sum(dim=-1).clamp_min(1.0).view(b, 1, 1, 1) * self.rank
        else:
            denom = float(n * self.rank)
        rms = (factors.pow(2).sum(dim=(-2, -1), keepdim=True) / denom).clamp_min(1e-12).sqrt()
        factors = factors / rms

        scaled = factors * self.component_scale[None, :, :, None]
        bias = torch.einsum("bhki,bhkj->bhij", scaled, factors)

        if self.self_pair is not None:
            eye = torch.eye(n, dtype=bias.dtype, device=bias.device)
            bias = bias + self.self_pair[None, :, None, None] * eye
            if mask is not None:
                # The diagonal term must also respect padding.
                pair_valid = valid.view(b, 1, 1, n) * valid.view(b, 1, n, 1)
                bias = bias * pair_valid
        return bias


def lowrank_cost_report(rank: int, num_heads: int = 8, hidden: Sequence[int] = (64, 64),
                        num_particles: int = 128, pair_hidden: Sequence[int] = (64, 64, 64),
                        pair_input_dim: int = 4) -> dict:
    """FLOPs and parameters of this arm against the ``PairEmbed`` it replaces.

    Reports the MLP saving and the still-materialized ``(B, H, N, N)`` tensor **separately and
    explicitly**. The FLOP reduction is real and large; the wall-clock speedup is *not* delivered by
    this arm, because weaver requires a dense ``attn_mask`` and the outer product is therefore still
    formed. Quoting the FLOP number as a speedup is the error this repository has already made once
    (``baseline_ffn2x``), so the two live in different keys with different names.
    """
    def mlp_params(widths: Sequence[int]) -> int:
        return sum(a * b + b for a, b in zip(widths, widths[1:]))

    feature_dim = len(PARTICLE_FEATURE_NAMES)
    lr_widths = [feature_dim, *hidden, num_heads * rank]
    lr_params = mlp_params(lr_widths) + num_heads * rank + num_heads   # + scales + self-pair
    pair_widths = [pair_input_dim, *pair_hidden, num_heads]
    pair_params = mlp_params(pair_widths)

    n = num_particles
    lr_mlp_flops = 2 * n * mlp_params(lr_widths)
    pair_mlp_flops = 2 * n * n * mlp_params(pair_widths)
    outer_flops = 2 * num_heads * rank * n * n

    return {
        "rank": int(rank),
        "num_heads": int(num_heads),
        "num_particles": int(n),
        "pair_embed_params": pair_params,
        "lowrank_params": lr_params,
        "param_ratio_lowrank_over_pair": lr_params / pair_params,
        # The load-bearing number: N per-particle evaluations instead of N^2 pair evaluations.
        "mlp_evaluations_pair": n * n,
        "mlp_evaluations_lowrank": n,
        "mlp_evaluation_reduction": (n * n) / n,
        "pair_mlp_flops": pair_mlp_flops,
        "lowrank_mlp_flops": lr_mlp_flops,
        "lowrank_outer_product_flops": outer_flops,
        "lowrank_total_flops": lr_mlp_flops + outer_flops,
        "flop_reduction": pair_mlp_flops / max(lr_mlp_flops + outer_flops, 1),
        # Stated so nobody reads the above as a measured speedup.
        "materializes_dense_bias": True,
        # MEASURED 2026-09-06: part_kernels' Patch A fires on `isinstance(module, PairEmbed)`, so
        # the baseline arm gets its dominant kernel fused (pair_embed_instances=1, 4.54x: 9,227us ->
        # 2,034us) and this arm does NOT (pair_embed_instances=0) -- LowRankPairEmbed is not a
        # weaver PairEmbed subclass, so it is skipped rather than clobbered. Verified: the module
        # survives patching intact, logits are bit-identical, and the rank bound still holds.
        #
        # The consequence runs OPPOSITE to the usual worry. Accuracy stays comparable (the fused
        # kernel is parity-tested as behaviourally equivalent), but WALL-CLOCK is biased AGAINST
        # this arm: baseline's most expensive kernel is fused and this one's is not. So a
        # jets_per_sec comparison between them would show the low-rank arm as slower while it does
        # ~40x fewer FLOPs. Do not quote throughput across these arms until a fused kernel consuming
        # the factors exists.
        "fused_pair_kernel_applies": False,
        "fused_pair_kernel_note": (
            "part_kernels Patch A matches isinstance(module, PairEmbed); this module is skipped, so "
            "baseline gets a 4.54x fused pair kernel and this arm does not. Accuracy is comparable; "
            "wall-clock is NOT, and is biased against this arm."
        ),
        "speedup_delivered": None,
        "speedup_note": (
            "FLOP reduction only. Weaver requires a dense (B, H, N, N) attn_mask, so this arm still "
            "forms the outer product and will NOT be wall-clock faster. Realizing the speedup needs "
            "a fused kernel consuming the factors directly (part_kernels). Measure, do not infer."
        ),
    }
