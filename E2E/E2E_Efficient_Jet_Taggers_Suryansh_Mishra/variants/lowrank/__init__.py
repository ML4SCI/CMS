"""Rank-r factorized pair bias — the arm the 2026-09-05 rank audit licensed.

Weaver's ``PairEmbed`` runs an MLP over all ``N^2`` particle pairs and is, by
``docs/kernels.md``, the **most expensive kernel in the model** (9,227 us stock, 2,034 us fused).
This arm replaces it with per-particle factors whose outer product is the bias, so ``N``
per-particle evaluations do the work of ``N^2`` pair evaluations.

The measurements that specify it, all in ``logs/audit-2026-09-05/README.md``:

===========  ==========================================================================
result       what it fixed about the design
===========  ==========================================================================
**B4**       the trained bias really is low rank: p90 25 row-centred, 0.78x/0.62x the
             matched random-matrix p90 at p=0.000, replicated on two arms
**P4**       one scalar per head recovers 90 % of the self-pair diagonal's rank cost,
             because ``c*I`` carries rank ``n`` on a single degree of freedom
**P5**       and it is worth a **factor of 2 in rank** behaviourally. Operating curve
             from truncating a trained bias: r=34 -0.0005, r=16 -0.0205, r=8 -0.0385
**B4a**      hand-built polynomial features plateau at 0.360 where the SVD optimum is
             0.034, so the factors must be **learned** -- a 10x gap, and the reason
             this arm is not simply N8
**B2b/P5**   pick the rank from the **99 %** energy row, not 90 %: the 90 % rank costs
             -0.034 accuracy, ~10x the largest genuine gain the ablation ever found
===========  ==========================================================================

Read :mod:`variants.lowrank.pair` for why each piece of the module is shaped the way it is.

**This arm is not a speedup.** Weaver requires a materialized ``(B, H, N, N)`` ``attn_mask``, so the
outer product is still formed and wall-clock time will not improve. It answers "does a rank-r bias
train to baseline accuracy?", which has to be answered first because a fused kernel for a bias that
does not train is worthless. :func:`lowrank_cost_report` keeps the FLOP reduction and the
(undelivered) speedup in separate keys for that reason.
"""

from __future__ import annotations

from .pair import (
    PARTICLE_FEATURE_NAMES,
    LowRankPairEmbed,
    lowrank_cost_report,
    particle_features,
)

__all__ = [
    "PARTICLE_FEATURE_NAMES",
    "LowRankPairEmbed",
    "lowrank_cost_report",
    "particle_features",
]
