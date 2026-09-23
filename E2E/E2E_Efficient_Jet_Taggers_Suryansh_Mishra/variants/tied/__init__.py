"""Weight-tied (recursive / looped) encoder stacks for ParT — arms R1-R4.

Encoder blocks are **74.5% of ParT's parameters** (1,597,504 of 2,143,354; 199,688 per block
x 8), so sharing them is the largest available parameter reduction: 8 -> 1 gives 745,538 params,
**2.87x smaller**. The scientific question underneath is not compression though, it is

    does ParT's depth compute eight different functions, or the same function eight times?

Run ``ablation/layer_redundancy.py`` (R0) on an existing checkpoint *before* training any arm
here: it measures block redundancy for free and predicts whether tying can work at all.

Naming
------
Call this **weight tying** or **looped**, never "recursive". In HEP a "recursive network" already
means a tree-structured network over the jet clustering history (arXiv:1711.02633). The
general-ML lineage is ALBERT-style cross-layer sharing, Universal Transformer, Loop-ViT
(arXiv:2602.02156), and Mixture-of-Recursions (arXiv:2507.10524).

What is here
------------
- :class:`TiedSchedule` — which unique block runs at each depth, and the two strategies MoR
  distinguishes (``cycle`` and ``sequence``). Grouped tying (8 -> k) interpolates between full
  tying and stock ParT, so an arm sweep produces a *curve* rather than a single yes/no point.
- :class:`DepthModulation` — optional per-depth affine on a tied block's output (R3). Separates
  "each depth needs different weights" from "each depth needs a different *scale*", at a cost of
  ``2 * embed_dim`` per depth (2,048 params for d=128, depth 8) against 199,688 for a real block.
- :class:`MoRRouter` — expert-choice routing with **fixed capacity** (R4), i.e.
  Mixture-of-Recursions. Fixed capacity is deliberate: it keeps FLOPs and worst-case latency
  *deterministic*, which matters because the one place jet tagging is compute-bound is the
  trigger, where the budget is worst-case rather than average.

FLOP and parameter matching is not optional
-------------------------------------------
SMELT (arXiv:2609.01343) matches per-token FLOPs and non-embedding parameters exactly in order to
isolate the effect of depth reuse, and that is the standard to hold to here. Every headline
comparison in this repository turned out to be confounded — by unmatched training steps, by
unmatched commits, or by a config key that was silently ignored — so :func:`tied_cost_report`
exists to make the confound visible before a run rather than after. Tying at fixed depth holds
FLOPs constant and reduces parameters; it does **not** make the model faster.
"""

from __future__ import annotations

from .stack import (
    DepthModulation,
    apply_residual_scale,
    MoRRouter,
    TiedSchedule,
    assert_shared_blocks_consistent,
    build_tied_blocks,
    shared_block_groups,
    tied_cost_report,
)

__all__ = [
    "DepthModulation",
    "apply_residual_scale",
    "MoRRouter",
    "TiedSchedule",
    "assert_shared_blocks_consistent",
    "build_tied_blocks",
    "shared_block_groups",
    "tied_cost_report",
]
