#!/usr/bin/env python3
"""R0: how redundant are ParT's encoder blocks? — the free pre-screen for weight tying.

Encoder blocks are **74.5% of ParT's parameters** (1,597,504 of 2,143,354; 199,688 per block
x 8). Tying them into one would give a 2.87x smaller model. Whether that costs accuracy depends
on a question you can answer from a checkpoint you already have, with no training and no GPU:

    do the eight blocks compute eight different functions, or the same function eight times?

If training has already driven them toward each other, tying is nearly free and the arm is worth
building. If they are strongly differentiated, expect a large drop. Either answer is a result --
nobody has measured layer redundancy in a jet tagger -- and it is decided before any GPU-hours
are spent. This is the same role `rank_audit.py` played for N8: measure first, build second.

**Naming.** Call this weight *tying*, never "recursion". In HEP "recursive network" already means
a tree-structured network over the jet clustering history (arXiv:1711.02633), which is an
unrelated idea. The general-ML reference for what is measured here is weight-tied recurrence,
e.g. Loop-ViT (arXiv:2602.02156).

Why an untrained null is mandatory
---------------------------------
Two independently initialised blocks are *not* dissimilar in a useful sense: with 199,688
parameters drawn i.i.d., their cosine similarity concentrates near 0 with a spread of order
1/sqrt(n), and any two *trained* blocks share a task, a data distribution and a schedule, so some
similarity is guaranteed by construction rather than by redundancy. A raw similarity number is
therefore uninterpretable. Every quantity here is reported against the same quantity computed on
a freshly initialised model of identical shape, exactly as `rank_audit.py` reports a
random-matrix floor. A trained similarity that does not exceed the untrained floor is not
evidence of redundancy.

What is measured
----------------
1. **Weight-space similarity** (no data needed). Per parameter group (attention projections,
   fc1/fc2, norms), cosine similarity and relative Frobenius distance between every pair of
   blocks, as an 8x8 matrix. Cheap, and enough to predict whether tying can work at all.
2. **Adjacent-block drift.** ||W_{i+1} - W_i|| / ||W_i|| along the depth. A model doing iterative
   refinement should show small, roughly constant drift; a specialising model should not.
3. **Block-swap and block-drop** (needs `--data-dir`). Replace block i's weights with block j's,
   or skip a block entirely, and measure the change in accuracy on real jets **without
   retraining**. This is the most direct evidence: if swapping two blocks barely moves accuracy,
   they are interchangeable in fact and not merely in weight space.

Usage
-----
    # weight-space only, no data required
    python -m ablation.layer_redundancy --checkpoint runs/baseline/best.pt

    # plus swap/drop on real jets
    python -m ablation.layer_redundancy --checkpoint runs/baseline/best.pt \
        --data-dir $PSCRATCH/jetclass/pt_ragged/val_5M --num-jets 2000

    # untrained: reports the null against itself, which is the sanity check
    python -m ablation.layer_redundancy
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

__all__ = [
    "PARAM_GROUPS",
    "gauge_invariant_composites",
    "gauge_invariant_similarity",
    "block_similarity",
    "adjacent_drift",
    "swap_drop_scan",
    "format_report",
]

#: Coarse parameter groups. Weaver's ``Block`` names its submodules ``attn`` (in/out projections),
#: ``fc1``/``fc2`` (FFN) and various norms; grouping keeps the report readable and lets a
#: differentiated FFN show up separately from differentiated attention.
PARAM_GROUPS = {
    "attn": ("attn.", "q_proj", "k_proj", "v_proj", "out_proj", "in_proj"),
    "ffn": ("fc1", "fc2"),
    "norm": ("norm", "ln", "layer_scale"),
}


def _group_of(name: str) -> str:
    for group, needles in PARAM_GROUPS.items():
        if any(n in name for n in needles):
            return group
    return "other"


def _flat_groups(block: nn.Module) -> dict[str, Tensor]:
    """Flatten a block's parameters into one vector per group, in a stable order."""
    buckets: dict[str, list[Tensor]] = {}
    for name, param in sorted(block.named_parameters()):
        buckets.setdefault(_group_of(name), []).append(param.detach().reshape(-1).double())
    out = {g: torch.cat(v) for g, v in buckets.items()}
    out["all"] = torch.cat([out[g] for g in sorted(out)])
    return out


def gauge_invariant_composites(block: nn.Module, num_heads: int = 8) -> dict[str, Tensor]:
    """The parts of a block that attention and the FFN *actually* compute.

    This exists because :func:`block_similarity` — cosine similarity of raw flattened weights — is
    **blind to the symmetries that leave a block's function unchanged**, which makes it unable to
    distinguish eight blocks computing the identical function from eight independent random inits
    (measured: such blocks score ``cos attn = -0.0151``). Every quantity here is invariant under
    those symmetries by construction:

    ``qk``
        ``Wq^(h)T Wk^(h)`` per head. The attention logit is ``x^T Wq^T Wk y``, so this bilinear form
        *is* what attention computes. Invariant under ``Wq -> M Wq, Wk -> M^-T Wk`` for any invertible
        ``M`` acting on a head's 16-dim subspace, because ``M^T M^-T = I``.
    ``vo``
        ``O^(h) Wv^(h)`` per head. The attention output is ``O (Wv y)``, so this product is what is
        computed. Invariant under ``Wv -> M Wv, O -> O M^-1``.
    ``ffn``
        ``fc2 @ fc1``. Invariant under any permutation of the 512 hidden units (``fc1 -> P fc1,
        fc2 -> fc2 P^T``), which is the FFN's exact symmetry group.

    Heads are concatenated **in head order**, deliberately. Head permutation ``S_8`` is *not* a free
    symmetry in ParT: the pair bias ``U[b,h,i,j]`` is shared across blocks and labels heads globally,
    so permuting heads genuinely changes the model (measured: 3.2 output change with a per-head bias,
    versus 2.9e-06 without). Head identity is therefore meaningful across depth, and a
    head-order-respecting invariant is the correct one.

    Caveat to state alongside any result: ``fc2 @ fc1`` is the **GELU-linearised** composite. It is
    exactly permutation-invariant, but two blocks with the same ``fc2 @ fc1`` need not compute the
    same nonlinear function, so it is a necessary and not a sufficient condition.
    """
    params = dict(block.named_parameters())
    embed = params["out_proj.weight"].shape[0] if "out_proj.weight" in params else None
    in_w = params.get("attn.in_proj.weight", params.get("in_proj.weight"))
    out_w = params.get("attn.out_proj.weight", params.get("out_proj.weight"))
    fc1 = params.get("fc1.weight")
    fc2 = params.get("fc2.weight")
    if in_w is None or out_w is None:
        raise AttributeError(
            "block has no attn.in_proj/out_proj; gauge invariants are defined for weaver Blocks"
        )

    embed = out_w.shape[0]
    head_dim = embed // num_heads
    w = in_w.detach().double()
    wq, wk, wv = w[:embed], w[embed:2 * embed], w[2 * embed:]
    o = out_w.detach().double()

    qk, vo = [], []
    for h in range(num_heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        qk.append((wq[sl].T @ wk[sl]).reshape(-1))     # (embed, embed) -> flat
        vo.append((o[:, sl] @ wv[sl]).reshape(-1))
    out: dict[str, Tensor] = {
        "qk": torch.cat(qk),
        "vo": torch.cat(vo),
    }
    if fc1 is not None and fc2 is not None:
        out["ffn"] = (fc2.detach().double() @ fc1.detach().double()).reshape(-1)
    out["all"] = torch.cat([out[k] for k in sorted(out)])
    return out


def gauge_invariant_similarity(blocks: Sequence[nn.Module], num_heads: int = 8) -> dict:
    """:func:`block_similarity`, but on quantities a symmetry cannot move.

    Same output shape, so it drops into the same report. Verified to have the discriminating power
    the raw-weight version lacks: **1.0000 for blocks computing the identical function, ~0.0000 for
    independent initialisations**.
    """
    comps = [gauge_invariant_composites(b, num_heads) for b in blocks]
    groups = sorted(set().union(*(set(c) for c in comps)))
    n = len(blocks)
    out: dict = {"num_blocks": n, "num_heads": num_heads, "groups": {}}
    for group in groups:
        if not all(group in c for c in comps):
            continue
        vs = [c[group] for c in comps]
        if len({v.numel() for v in vs}) != 1:
            continue
        cos = np.full((n, n), np.nan)
        rel = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                a, b = vs[i], vs[j]
                na, nb = float(a.norm()), float(b.norm())
                cos[i, j] = float(a @ b) / (na * nb) if na and nb else np.nan
                rel[i, j] = float((a - b).norm()) / na if na else np.nan
        off = ~np.eye(n, dtype=bool)
        out["groups"][group] = {
            "numel_per_block": int(vs[0].numel()),
            "cosine": cos.tolist(),
            "relative_distance": rel.tolist(),
            "mean_offdiag_cosine": float(np.nanmean(cos[off])),
            "max_offdiag_cosine": float(np.nanmax(cos[off])),
            "mean_offdiag_relative_distance": float(np.nanmean(rel[off])),
            "min_offdiag_relative_distance": float(np.nanmin(rel[off])),
        }
    return out


def block_similarity(blocks: Sequence[nn.Module]) -> dict:
    """Pairwise cosine similarity and relative distance between blocks, per parameter group."""
    flats = [_flat_groups(b) for b in blocks]
    groups = sorted(set().union(*(set(f) for f in flats)))
    n = len(blocks)
    out: dict = {"num_blocks": n, "groups": {}}
    for group in groups:
        if not all(group in f for f in flats):
            continue
        vs = [f[group] for f in flats]
        if len({v.numel() for v in vs}) != 1:
            continue  # shapes differ across blocks; not comparable
        cos = np.full((n, n), np.nan)
        rel = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                a, b = vs[i], vs[j]
                na, nb = float(a.norm()), float(b.norm())
                cos[i, j] = float(a @ b) / (na * nb) if na and nb else np.nan
                rel[i, j] = float((a - b).norm()) / na if na else np.nan
        off = ~np.eye(n, dtype=bool)
        out["groups"][group] = {
            "numel_per_block": int(vs[0].numel()),
            "cosine": cos.tolist(),
            "relative_distance": rel.tolist(),
            "mean_offdiag_cosine": float(np.nanmean(cos[off])),
            "max_offdiag_cosine": float(np.nanmax(cos[off])),
            "mean_offdiag_relative_distance": float(np.nanmean(rel[off])),
            "min_offdiag_relative_distance": float(np.nanmin(rel[off])),
        }
    return out


def adjacent_drift(blocks: Sequence[nn.Module]) -> dict:
    """``||W_{i+1} - W_i|| / ||W_i||`` along depth, per group.

    Distinguishes iterative refinement (small, roughly constant drift) from specialisation.
    Reported alongside the untrained null, where consecutive blocks are independent draws and the
    drift is therefore ~sqrt(2) regardless of depth.
    """
    flats = [_flat_groups(b) for b in blocks]
    out: dict = {}
    for group in sorted(flats[0]):
        seq = []
        for i in range(len(blocks) - 1):
            a, b = flats[i][group], flats[i + 1][group]
            na = float(a.norm())
            seq.append(float((b - a).norm()) / na if na else float("nan"))
        out[group] = seq
    return out


# --------------------------------------------------------------------------------------------
# behavioural probes (need data)
# --------------------------------------------------------------------------------------------
def _accuracy(model: nn.Module, x: Tensor, v: Tensor, mask: Tensor, y: Tensor,
              batch: int = 256) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for s in range(0, x.shape[0], batch):
            e = s + batch
            logits = model(x[s:e], v=v[s:e], mask=mask[s:e])
            correct += int((logits.argmax(-1) == y[s:e]).sum())
            total += int(logits.shape[0])
    return correct / total if total else float("nan")


class _Skip(nn.Module):
    """Identity stand-in for a dropped block, matching weaver's Block call signature."""

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        return x


def swap_drop_scan(model: nn.Module, x: Tensor, v: Tensor, mask: Tensor, y: Tensor,
                   max_pairs: Optional[int] = None) -> dict:
    """Accuracy under (a) dropping each block and (b) overwriting block i with block j.

    No retraining: this measures whether the *trained* blocks are interchangeable in fact.
    Weights are restored after every probe, so the model is left exactly as found.
    """
    blocks = model.blocks
    n = len(blocks)
    base = _accuracy(model, x, v, mask, y)
    out: dict = {"baseline_accuracy": base, "drop": {}, "swap": {}}

    for i in range(n):
        original = blocks[i]
        blocks[i] = _Skip()
        out["drop"][str(i)] = _accuracy(model, x, v, mask, y) - base
        blocks[i] = original

    # Overwrite i with j's weights. Only same-shape state dicts, which all encoder blocks share.
    #
    # Ordered by *offset* (all `i <- i+1`, then all `i <- i+2`, ...) rather than by destination,
    # because `--max-swap-pairs` truncates this list. Destination-major order (`for i: for j:`)
    # put every pair with i=0 first, so a cap of 7 measured only substitutions into block 0 and
    # the reported "median swap delta" described one destination while claiming to describe eight.
    # Offset-major means any prefix of length k*n covers all n destinations k times, so a
    # truncated scan stays a fair sample of the depth.
    pairs = [(i, (i + d) % n) for d in range(1, n) for i in range(n)]
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    for i, j in pairs:
        saved = {k: t.detach().clone() for k, t in blocks[i].state_dict().items()}
        blocks[i].load_state_dict(blocks[j].state_dict())
        out["swap"][f"{i}<-{j}"] = _accuracy(model, x, v, mask, y) - base
        blocks[i].load_state_dict(saved)
    return out


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------
def format_report(res: dict) -> str:
    meta = res["meta"]
    L: list[str] = []
    A = L.append
    A("Encoder-block redundancy (R0 pre-screen for weight tying)")
    A("=" * 66)
    A(f"  weights        : {meta['weights']}")
    A(f"  jets source    : {meta['source']}")
    A(f"  blocks         : {meta['num_blocks']}  ({meta['params_per_block']:,} params each, "
      f"{meta['block_share']:.1%} of {meta['total_params']:,} total)")
    if meta.get("already_tied"):
        A("")
        A("  NOT A REDUNDANCY RESULT: this checkpoint is ALREADY TIED.")
        A(f"  {meta['num_blocks']} depths share {meta['distinct_blocks']} distinct block(s); "
          f"tied groups {meta['shared_block_groups']}.")
        A("  Every similarity below is 1.000 and every distance 0.000 by construction -- the same")
        A("  tensor compared with itself. This tool asks whether tying WOULD work, which is only")
        A("  a question about an untied model. Re-run it on the untied baseline checkpoint.")
    else:
        A(f"  tying 8->1     : {meta['tied_params']:,} params "
          f"({meta['total_params'] / meta['tied_params']:.2f}x smaller)")
    A("")
    gauge = res.get("gauge_similarity")
    null_gauge = res.get("null_gauge_similarity")
    if gauge:
        A("  GAUGE-INVARIANT block similarity, trained vs UNTRAINED NULL  <-- READ THIS ONE")
        A("  --------------------------------------------------------------------------")
        A("  Cosine between the quantities attention and the FFN ACTUALLY compute: qk = Wq^T Wk per")
        A("  head (the bilinear form in the logit), vo = O Wv per head (the output map), ffn =")
        A("  fc2 @ fc1. These are invariant under the symmetries that leave a block's function")
        A("  unchanged, so unlike the raw-weight table below they can tell a functionally identical")
        A("  pair of blocks from two independent random inits. Verified: 1.0000 for the former,")
        A("  ~0.000 for the latter.")
        A("")
        A(f"    {'group':<8}{'numel':>10}{'cos (trained)':>15}{'cos (null)':>12}{'excess':>10}"
          f"{'rel (trained)':>15}{'rel (null)':>12}")
        for group, entry in gauge["groups"].items():
            null = (null_gauge or {}).get("groups", {}).get(group)
            if not null:
                continue
            A(f"    {group:<8}{entry['numel_per_block']:>10,}"
              f"{entry['mean_offdiag_cosine']:>15.4f}{null['mean_offdiag_cosine']:>12.4f}"
              f"{entry['mean_offdiag_cosine'] - null['mean_offdiag_cosine']:>+10.4f}"
              f"{entry['mean_offdiag_relative_distance']:>15.4f}"
              f"{null['mean_offdiag_relative_distance']:>12.4f}")
        A("")
        A("  Caveat: ffn = fc2 @ fc1 is the GELU-LINEARISED composite. Exactly permutation-invariant,")
        A("  but two blocks with equal fc2@fc1 need not compute the same nonlinear function, so it is")
        A("  a necessary and not a sufficient condition for functional equality.")
        A("")
    A("  RAW-WEIGHT similarity below is RETRACTED as a gate (2026-09-06) and kept only so the")
    A("  original R0 numbers stay regenerable. It is blind to the symmetries above: a block moved")
    A("  by a symmetry element scores ffn cosine -0.0005 where the true value is 1.0, because")
    A("  permuting the 512 FFN hidden units is a DISCRETE symmetry and destroys the cosine")
    A("  unconditionally. The FFN is 66% of a block, so this is the group R0 weighted most heavily")
    A("  and the one where its metric fails hardest. See logs/looped-survey-2026-09-06/.")
    A("")
    A("  Pairwise block similarity, trained vs UNTRAINED NULL")
    A("  ----------------------------------------------------")
    A("  Two independently initialised blocks already share some similarity by construction, so")
    A("  a raw number means nothing. `excess` is what exceeds the null; only that is evidence of")
    A("  redundancy. cos -> 1 and rel -> 0 would mean the blocks have converged together.")
    A("")
    A("  READ THE PER-GROUP ROWS, NOT `all`. LayerNorm weights initialise to exactly 1.0, so at")
    A("  init they are identical across blocks AND carry a large share of the concatenated")
    A("  vector's norm (1,792 entries at 1.0 give norm ~42, against ~23 for 66k attention")
    A("  weights at std ~1/sqrt(128)). That inflates `all` to ~0.62 on a model where attention")
    A("  and FFN are provably uncorrelated. `all` is a norm-weighted average dominated by the")
    A("  smallest, most trivially-similar group -- it is reported only for continuity.")
    A("")
    A(f"    {'group':<8}{'numel':>10}{'cos (trained)':>15}{'cos (null)':>12}{'excess':>10}"
      f"{'rel (trained)':>15}{'rel (null)':>12}")
    for group, entry in res["similarity"]["groups"].items():
        null = res["null_similarity"]["groups"].get(group)
        if not null:
            continue
        exc = entry["mean_offdiag_cosine"] - null["mean_offdiag_cosine"]
        A(f"    {group:<8}{entry['numel_per_block']:>10,}"
          f"{entry['mean_offdiag_cosine']:>15.4f}{null['mean_offdiag_cosine']:>12.4f}"
          f"{exc:>+10.4f}{entry['mean_offdiag_relative_distance']:>15.4f}"
          f"{null['mean_offdiag_relative_distance']:>12.4f}")
    A("")
    A("  Adjacent-block drift  ||W_i+1 - W_i|| / ||W_i||   (null ~ sqrt(2) = 1.414)")
    A("  -----------------------------------------------------------------------")
    for group in ("all", "attn", "ffn"):
        seq = res["drift"].get(group)
        if seq:
            A(f"    {group:<6}" + "  ".join(f"{d:5.3f}" for d in seq))
    A("")
    if res.get("probes"):
        p = res["probes"]
        A("  Behavioural probes on real jets (NO retraining)")
        A("  ----------------------------------------------")
        A(f"    baseline accuracy on this batch: {p['baseline_accuracy']:.5f}")
        A("")
        A("    drop block i -> delta accuracy")
        A("      " + "  ".join(f"{k}:{v:+.4f}" for k, v in sorted(p["drop"].items(),
                                                                  key=lambda kv: int(kv[0]))))
        worst = min(p["swap"].items(), key=lambda kv: kv[1]) if p["swap"] else None
        best = max(p["swap"].items(), key=lambda kv: kv[1]) if p["swap"] else None
        if worst and best:
            A("")
            A(f"    swap i<-j: most harmful {worst[0]} {worst[1]:+.4f};  "
              f"least harmful {best[0]} {best[1]:+.4f}")
            deltas = np.array(list(p["swap"].values()))
            n_blocks = meta["num_blocks"]
            full = n_blocks * (n_blocks - 1)
            A(f"    swap deltas: median {np.median(deltas):+.4f}, "
              f"p10 {np.percentile(deltas, 10):+.4f}, p90 {np.percentile(deltas, 90):+.4f}")
            if len(deltas) < full:
                dests = len({k.split("<-")[0] for k in p["swap"]})
                A(f"    (over {len(deltas)} of {full} pairs, covering {dests} of {n_blocks} "
                  "destinations — pairs are ordered by offset so a cap stays a fair sample)")
            A("")
            A("    A median swap delta near zero means the blocks are interchangeable in fact,")
            A("    which is the strongest available evidence that tying will be nearly free.")
    else:
        A("  Behavioural probes skipped (pass --data-dir to run swap/drop on real jets).")
    A("")
    if meta["weights"] == "untrained":
        A("  NOT A RESULT: untrained weights. Every 'excess' is zero by construction here;")
        A("  this mode exists to validate the tool and to generate the null.")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=None,
                    help="ablation checkpoint; omit to audit untrained weights")
    ap.add_argument("--data-dir", default=None,
                    help="ragged CSR .pt shards, for the swap/drop probes")
    ap.add_argument("--num-jets", type=int, default=2000)
    ap.add_argument("--num-particles", type=int, default=128)
    ap.add_argument("--max-swap-pairs", type=int, default=None,
                    help="cap the i<-j scan (default: all n*(n-1) pairs)")
    ap.add_argument("--norm-stats", default=None,
                    help="norm_stats.json. REQUIRED with --data-dir: `load_bench_batch` returns "
                         "RAW features while training normalized them, and an out-of-distribution "
                         "model shows ~0 swap/drop delta for every block, which falsely reads as "
                         "'the blocks are interchangeable'. Defaults to "
                         "`<data-dir>/../norm_stats.json` when that exists.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None, help="write the full record here as JSON")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    from ablation.config import AblationConfig
    from ablation.rank_audit import _load_model_from_checkpoint, _real_batch
    from variants import build_variant_part

    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model, config = _load_model_from_checkpoint(payload)
        weights = f"{args.checkpoint} @ step {payload.get('step', '?')}"
    else:
        config = AblationConfig(arm="baseline")
        model = build_variant_part("baseline", **config.model_kwargs())
        weights = "untrained"
    model.eval()

    # Null: identical architecture, fresh init. Different seed so it is an independent draw.
    torch.manual_seed(args.seed + 1)
    null_model = build_variant_part(config.arm, **config.model_kwargs()).eval()

    blocks = list(model.blocks)
    per_block = sum(p.numel() for p in blocks[0].parameters())
    total = sum(p.numel() for p in model.parameters())

    # An already-tied checkpoint holds the same module object at several `blocks` indices, and
    # every quantity below then measures that identity rather than anything about training:
    # cosine similarity is exactly 1, relative distance exactly 0, and the projected saving is
    # nonsense (`per_block * len(blocks)` double-counts shared parameters, giving a 214% block
    # share and a *negative* tied-parameter count at k=1). This tool answers "would tying work?",
    # which is only a question about an untied model, so the degenerate case is named rather than
    # silently reported as a spectacular redundancy result.
    from variants.tied import shared_block_groups

    shared = shared_block_groups(model)
    distinct = len({id(getattr(b, "block", b)) for b in blocks})

    res: dict = {
        "meta": {
            "weights": weights,
            "source": args.data_dir or "none (weight-space only)",
            "arm": config.arm,
            "num_blocks": len(blocks),
            "distinct_blocks": distinct,
            "shared_block_groups": shared,
            "already_tied": bool(shared),
            "params_per_block": per_block,
            "total_params": total,
            # Counted over *distinct* blocks, so these stay meaningful on a tied checkpoint.
            "block_share": per_block * distinct / total,
            "tied_params": total - per_block * (distinct - 1),
            "seed": args.seed,
        },
        "similarity": block_similarity(blocks),
        "null_similarity": block_similarity(list(null_model.blocks)),
        # The gauge-invariant metric is the one to READ. The raw-weight metric above is kept only
        # so the retracted R0 numbers remain regenerable.
        "gauge_similarity": gauge_invariant_similarity(blocks, config.num_heads),
        "null_gauge_similarity": gauge_invariant_similarity(
            list(null_model.blocks), config.num_heads
        ),
        "drift": adjacent_drift(blocks),
        "null_drift": adjacent_drift(list(null_model.blocks)),
    }

    if args.data_dir:
        v, mask, shards = _real_batch(args.data_dir, args.num_jets, args.num_particles)
        from dataloader.ragged_loader import load_bench_batch  # noqa: F401  (documented dep)
        # `_real_batch` drops x and y; re-fetch them the same way for the probes.
        x_all, v_all, mask_all, y_all = [], [], [], []
        for shard in shards:
            xs, vs, ms, ys = load_bench_batch(
                args.data_dir, batch_size=max(1, args.num_jets // len(shards)),
                shard_idx=shard, pad_to=args.num_particles,
            )
            x_all.append(xs); v_all.append(vs); mask_all.append(ms); y_all.append(ys)
        x = torch.cat(x_all); v = torch.cat(v_all)
        mask = torch.cat(mask_all); y = torch.cat(y_all).argmax(-1)

        # Normalize `x` exactly as training did, or these probes measure nothing.
        # `load_bench_batch` returns RAW features while `ablation/data.py` applies NormStats
        # (pT/E divided by their mean, the rest z-scored; `v` stays raw). The failure mode here is
        # not a small bias, it is a **false positive**: a model fed out-of-distribution features
        # sits near chance, so dropping or swapping a block moves accuracy by ~0 and every block
        # looks interchangeable -- which is exactly the conclusion this tool exists to test.
        # Measured on the truncation probe with raw features: 0.17 accuracy at loss 4.99, on a
        # checkpoint that scores 0.8624.
        stats_path = args.norm_stats
        if stats_path is None:
            candidate = Path(args.data_dir).parent / "norm_stats.json"
            stats_path = str(candidate) if candidate.exists() else None
        if stats_path:
            from ablation.data import NormStats
            x = NormStats.load(Path(stats_path)).apply(x, mask)
        res["meta"]["normalized_from"] = stats_path
        res["meta"]["shards"] = list(shards)
        res["meta"]["probe_jets"] = int(x.shape[0])
        if not stats_path:
            print("WARNING: no norm_stats.json found or given; features are RAW, so the "
                  "swap/drop deltas below are meaningless (an out-of-distribution model shows "
                  "~0 delta for everything, which falsely reads as 'blocks are interchangeable'). "
                  "Pass --norm-stats.")
        res["probes"] = swap_drop_scan(model, x, v, mask, y, max_pairs=args.max_swap_pairs)

    print(format_report(res))
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=2))
        print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
