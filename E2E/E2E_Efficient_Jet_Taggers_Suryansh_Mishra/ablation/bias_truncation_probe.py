#!/usr/bin/env python3
"""Does a rank-r pair bias still tag jets? — the probe that closes B4's open question.

The rank audit says the trained `(B, H, N, N)` pair bias is low rank: p90 rank 25 row-centred,
**12** once one scalar per head is handled separately, against a matched random-matrix p90 of 32
(p = 0.000, two arms). And the required rank is **flat in jet multiplicity** -- median 5 at every N,
with rank/N falling from 0.147 to 0.041 -- so a *fixed* per-particle budget works and gets
relatively cheaper on exactly the busy jets that cost the most.

None of that says the model still *works*. Rank 16 captures 97.8% of the bias's Frobenius energy,
and 97.8% of energy is not 97.8% of accuracy: the audit's own caveat is that rank 8 at 90% energy
is a 22% relative error on the bias, and `LOW_RANK_HEAD_MAX = 8` was pre-registered before anyone
knew that. So the load-bearing question -- *does tagging survive it?* -- has never been asked.

It can be asked without training anything. Truncate the bias of an already-trained checkpoint at
inference and measure the accuracy you get. Same idea as `layer_redundancy.py`'s block-swap probe:
perturb a trained model in the exact way the proposed architecture would, and read the damage
directly. If accuracy holds at r = 16 the factorized design is de-risked before a single GPU-hour;
if it collapses at r = 34 the compression story is dead and that cost one login-node session.

What is measured
----------------
For each rank r, the bias each attention block sees is replaced by its best rank-r approximation,
and top-1 accuracy is measured on real jets. Two modes, matching the two things the audit priced:

``centered``
    Row-centre (over *valid keys only*), then truncate. This is the honest configuration: softmax
    over keys cannot see a per-row constant, so centring is free and the truncation is spending its
    entire budget on structure that actually reaches attention.
``const_diag``
    Additionally subtract one constant per head (``c = mean_i U_ii``) before centring and add it
    back after. This is the "+1 scalar per head" design the audit's P4 measurement recommends: it
    recovered 90% of the diagonal's rank cost for one stored number, because ``c*I`` carries rank n
    on a single degree of freedom.

Two controls make the numbers readable, and both are mandatory:

* ``r = full`` must reproduce the unmodified model **exactly**. Row centring is a provable softmax
  no-op, so any deviation here is a bug in this file, not a result -- it is asserted, not assumed.
* ``r = 0`` deletes the bias entirely at inference. That is the *lower* bound: it says how much of
  the trained model's accuracy the pair bias is carrying at all, and every truncation number should
  be read as a fraction of the span between it and ``full``. A rank that looks "cheap" against
  ``full`` alone means nothing if ``r = 0`` is also cheap.

Usage
-----
    python -m ablation.bias_truncation_probe \
        --checkpoint $SCRATCH/part_ablation/runs/baseline/best.pt \
        --data-dir   $PSCRATCH/jetclass/pt_ragged/val_5M \
        --num-jets 2000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

__all__ = ["TruncatedPairEmbed", "truncate_batch", "evaluate_ranks"]

#: Ranks to probe. 4/14/34 are the moment-feature dimensions the audit compares against; 8 is the
#: pre-registered §8.7a gate; 12 is the measured p90 under the "+1 scalar per head" design; 16 and
#: 24 bracket it. ``None`` means no truncation.
DEFAULT_RANKS = (0, 4, 8, 12, 16, 24, 34, None)


def truncate_batch(u: Tensor, valid: Tensor, rank: Optional[int],
                   mode: str = "centered", global_const: Optional[Tensor] = None) -> Tensor:
    """Best rank-``rank`` approximation of every ``(n, n)`` block in a ``(B, H, N, N)`` bias.

    Fully batched: one ``torch.linalg.svd`` over ``(B*H, N, N)`` rather than a Python loop over
    jets and heads, which matters because this is called once per probed rank.

    Padded rows and columns are forced to exact zero *first*. That is what makes operating on the
    padded ``N x N`` matrix equivalent to operating on the valid ``n x n`` block: zeros contribute
    only zero singular values, so the truncation keeps the same directions either way. Centring,
    by contrast, is **not** padding-invariant -- a mean taken over all ``N`` keys instead of the
    jet's own ``n`` would be the wrong constant -- so the row mean is computed over valid keys
    only, from ``valid``.

    Parameters
    ----------
    u : Tensor
        ``(B, H, N, N)`` bias as produced by weaver's ``PairEmbed``.
    valid : Tensor
        ``(B, N)`` bool, ``True`` = real particle.
    rank : int or None
        ``None`` leaves the (centred) bias alone; ``0`` deletes it.
    mode : {"centered", "const_diag", "global_diag"}
        See the module docstring. ``const_diag`` subtracts an ORACLE per-(jet, head) constant;
        ``global_diag`` subtracts a single per-head constant supplied by the caller, which is what a
        trainable module can actually store. The gap between them is the load-bearing measurement.
    global_const : Tensor, optional
        ``(H,)`` per-head constants for ``mode="global_diag"``.
    """
    if mode not in ("centered", "const_diag", "global_diag"):
        raise ValueError(
            f"mode must be 'centered', 'const_diag' or 'global_diag', got {mode!r}"
        )
    if mode == "global_diag" and global_const is None:
        raise ValueError("mode='global_diag' needs global_const, a (H,) tensor")

    b, h, n, _ = u.shape
    work = u.to(torch.float32)
    row = valid[:, None, :, None]          # (B, 1, N, 1) -- query side
    col = valid[:, None, None, :]          # (B, 1, 1, N) -- key side
    keep = row & col
    work = torch.where(keep, work, torch.zeros((), dtype=work.dtype, device=work.device))

    if rank == 0:
        # The rank-0 approximation of anything is the zero matrix, and a zero bias is no bias --
        # which is the point of this control. Returning `work` here (the padding-masked but
        # otherwise untouched bias) was a bug: it made the r=0 lower bound report that deleting
        # the bias costs nothing, which would have made every truncation number unreadable.
        return torch.zeros_like(u)

    counts = valid.sum(dim=1).clamp(min=1).to(work.dtype)[:, None, None]   # (B, 1, 1)

    const = None
    if mode == "global_diag":
        # ONE number per head for the whole dataset -- what `LowRankPairEmbed.self_pair` actually
        # stores. `const_diag` below instead uses a per-(jet, head) mean, which is an ORACLE: it
        # depends on the very matrix being approximated. The spectral and accuracy results were all
        # measured with the oracle, so they license a jet-adaptive constant and NOT the global
        # parameter the trainable arm implements. This mode exists to price that gap.
        eye = torch.eye(n, dtype=torch.bool, device=u.device)
        const = global_const.to(work.dtype).to(u.device).view(1, h, 1).expand(b, h, 1)
        work = work - const[..., None] * eye
    elif mode == "const_diag":
        # One number per (jet, head): the mean self-pair value over valid particles. Subtracted
        # before truncation and added back after, so the truncation budget is spent on everything
        # except a term the kernel would store as a single scalar.
        eye = torch.eye(n, dtype=torch.bool, device=u.device)
        diag = work.diagonal(dim1=-2, dim2=-1)                            # (B, H, N)
        const = (diag * valid[:, None, :]).sum(-1, keepdim=True) / counts  # (B, H, 1)
        work = work - const[..., None] * eye

    # Row-centre over valid keys only. Free by softmax invariance, so it costs the budget nothing.
    row_mean = (work * col).sum(dim=-1, keepdim=True) / counts[..., None]
    work = torch.where(keep, work - row_mean, torch.zeros((), dtype=work.dtype, device=work.device))

    if rank is not None and rank < n:
        flat = work.reshape(b * h, n, n)
        u_s, s, vh = torch.linalg.svd(flat, full_matrices=False)
        s = s.clone()
        s[:, rank:] = 0.0
        work = (u_s @ torch.diag_embed(s) @ vh).reshape(b, h, n, n)

    if const is not None:
        eye = torch.eye(n, dtype=torch.bool, device=u.device)
        work = work + const[..., None] * eye

    # Re-zero the padded region: truncation smears energy into it, and while attention masks those
    # keys anyway, leaving them nonzero would make this function's output depend on padding width.
    work = torch.where(keep, work, torch.zeros((), dtype=work.dtype, device=work.device))
    return work.to(u.dtype)


class TruncatedPairEmbed(nn.Module):
    """Wraps weaver's ``PairEmbed`` and truncates its output in place.

    Wrapping ``pair_embed`` rather than hooking the encoder blocks is deliberate and is the same
    reasoning ``rank_audit.pair_bias`` gives: weaver builds the bias **once** outside the block
    loop and hands the same tensor to all eight blocks, so the module call is the authoritative
    value and one wrapper covers the whole model. It also receives ``mask``, which is what makes
    valid-key-only centring possible without threading state in from the eval loop.
    """

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self.rank: Optional[int] = None
        self.mode: str = "centered"
        self.enabled: bool = False
        self.global_const: Optional[Tensor] = None
        self.collect_diagonals: bool = False
        self._diag_sum: Optional[Tensor] = None
        self._diag_count: int = 0

    def fitted_global_const(self) -> Optional[Tensor]:
        """Mean per-head diagonal over everything seen while ``collect_diagonals`` was on."""
        if self._diag_sum is None or self._diag_count == 0:
            return None
        return (self._diag_sum / self._diag_count).float()

    def forward(self, x, uu=None, mask=None):
        u = self.inner(x, uu=uu, mask=mask)
        if not self.enabled or mask is None or u.dim() != 4:
            return u
        valid = mask.squeeze(1) > 0 if mask.dim() == 3 else mask > 0
        if self.collect_diagonals:
            # Accumulate the per-(jet, head) diagonal mean so a GLOBAL per-head constant can be
            # fitted on one split and frozen for evaluation on another.
            with torch.no_grad():
                d = u.diagonal(dim1=-2, dim2=-1)                    # (B, H, N)
                v = valid[:, None, :].to(d.dtype)
                per = (d * v).sum(-1) / v.sum(-1).clamp_min(1.0)    # (B, H)
                self._diag_sum = self._diag_sum + per.sum(0).double() if self._diag_sum is not None else per.sum(0).double()
                self._diag_count += per.shape[0]
        return truncate_batch(u, valid, self.rank, self.mode, self.global_const)


@torch.no_grad()
def _accuracy(model: nn.Module, x: Tensor, v: Tensor, mask: Tensor, y: Tensor,
              batch: int = 128) -> tuple[float, float]:
    """Top-1 accuracy, mean cross-entropy, and the per-jet predictions.

    The predictions are returned so callers can count how many jets *changed* class relative to the
    unmodified model. That paired count is a far sharper instrument than the accuracy difference:
    accuracy is a difference of two rates over the same jets, so its naive binomial error bar
    (+-0.0078 at 2000 jets) badly overstates the uncertainty on a paired comparison, while "3 of
    2000 predictions moved" is exact and needs no error bar at all.
    """
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    preds = []
    for start in range(0, x.shape[0], batch):
        stop = start + batch
        logits = model(x[start:stop], v=v[start:stop], mask=mask[start:stop])
        target = y[start:stop]
        pred = logits.argmax(-1)
        preds.append(pred)
        correct += int((pred == target).sum())
        loss_sum += float(torch.nn.functional.cross_entropy(
            logits, target, reduction="sum"))
        total += int(logits.shape[0])
    return (correct / total if total else math.nan,
            loss_sum / total if total else math.nan,
            torch.cat(preds) if preds else torch.zeros(0, dtype=torch.long))


def evaluate_ranks(model: nn.Module, x: Tensor, v: Tensor, mask: Tensor, y: Tensor,
                   ranks: Sequence[Optional[int]] = DEFAULT_RANKS,
                   modes: Sequence[str] = ("centered", "const_diag", "global_diag"),
                   batch: int = 128) -> dict:
    """Accuracy as a function of bias rank, for each mode, plus the two mandatory controls."""
    inner = getattr(model, "pair_embed", None)
    if inner is None:
        raise AttributeError("model has no .pair_embed; this probe needs a pair-bias arm")

    x_count = x.shape[0]
    wrapper = TruncatedPairEmbed(inner)
    model.pair_embed = wrapper

    out: dict = {"modes": {}}
    try:
        wrapper.enabled = False
        t0 = time.time()
        # Baseline on the SAME held-out half every mode is scored on.
        half_ = max(1, x_count // 2)
        base_acc, base_loss, base_pred = _accuracy(
            model, x[half_:], v[half_:], mask[half_:], y[half_:], batch
        )
        out["unmodified"] = {"accuracy": base_acc, "loss": base_loss,
                             "seconds": time.time() - t0}

        # Fit ONE per-head constant on the first half of the jets, then freeze it. This is the
        # honest counterpart to `const_diag`'s oracle: a trainable module stores exactly this, and
        # it must be fitted on data it is not then scored on.
        half = max(1, x_count // 2)
        wrapper.enabled, wrapper.rank, wrapper.mode = True, None, "centered"
        wrapper.collect_diagonals = True
        _accuracy(model, x[:half], v[:half], mask[:half], y[:half], batch)
        wrapper.collect_diagonals = False
        wrapper.global_const = wrapper.fitted_global_const()
        out["global_const"] = (
            wrapper.global_const.tolist() if wrapper.global_const is not None else None
        )
        out["global_const_fitted_on_jets"] = half

        for mode in modes:
            rows = []
            for rank in ranks:
                wrapper.enabled, wrapper.rank, wrapper.mode = True, rank, mode
                acc, loss, pred = _accuracy(
                    model, x[half:], v[half:], mask[half:], y[half:], batch
                )
                flipped = int((pred != base_pred).sum())
                rows.append({"rank": rank, "accuracy": acc, "loss": loss,
                             "delta_vs_unmodified": acc - base_acc,
                             "predictions_changed": flipped,
                             # Denominator is the SCORED set, not the whole batch. Dividing by
                             # x_count understated every reported percentage 2x once the
                             # held-out-half split was introduced, because predictions come from
                             # x[half:] while x_count is the full batch.
                             "predictions_changed_frac": flipped / max(pred.shape[0], 1),
                             "scored_jets": int(pred.shape[0])})
            out["modes"][mode] = rows
    finally:
        # Leave the model exactly as found, whatever happened.
        model.pair_embed = inner
        wrapper.enabled = False

    # Control 1: r = full must reproduce the unmodified model, because row centring is a provable
    # softmax no-op. Checked rather than trusted.
    #
    # The tolerance is one prediction flip, not zero. Centring is exact in real arithmetic but the
    # model runs in float32, where subtracting a row mean perturbs logits at the 1e-8 level -- and
    # a jet whose top two classes are that close can flip. One flip in `n_jets` is therefore
    # expected roundoff, whereas a genuine bug in the masking or the gauge moves accuracy by
    # percent. The loss gap is reported alongside because it is continuous and so a far more
    # sensitive witness than a discrete accuracy.
    n_jets = int(x_count)
    for mode, rows in out["modes"].items():
        full = next((r for r in rows if r["rank"] is None), None)
        if full is not None:
            gap = abs(full["accuracy"] - base_acc)
            loss_gap = abs(full["loss"] - base_loss)
            full["control_gap_vs_unmodified"] = gap
            full["control_loss_gap"] = loss_gap
            full["control_passed"] = (gap <= 1.0 / max(n_jets, 1) + 1e-12
                                      and loss_gap < 1e-4
                                      and full.get("predictions_changed", 0) <= 1)
    # Control 2: r = 0 is the lower bound -- how much accuracy the bias carries at all.
    zero = next((r for r in out["modes"][modes[0]] if r["rank"] == 0), None)
    if zero is not None:
        out["bias_worth"] = base_acc - zero["accuracy"]
    return out


def format_report(res: dict) -> str:
    meta, L = res["meta"], []
    A = L.append
    A("Pair-bias rank truncation probe — does a rank-r bias still tag?")
    A("=" * 68)
    A(f"  weights   : {meta['weights']}")
    A(f"  jets      : {meta['num_jets']} from {meta['source']}")
    A(f"  shards    : {meta.get('shards')}  (class coverage {meta.get('class_coverage', '?')})")
    A("")
    if meta.get("normalized_from"):
        how = ("mask-aware (pad stays zero)" if meta.get("normalization_masked_padding")
               else "unmasked (pad carries `offset`, as this checkpoint was TRAINED)")
        A(f"  features   : normalized from {meta['normalized_from']} — {how}")
    else:
        A("  features   : RAW — accuracy below is MEANINGLESS, pass --norm-stats")
    A("")
    base = res["unmodified"]["accuracy"]
    A(f"  unmodified accuracy on this batch: {base:.5f}"
      + ("   <- sanity: should be near the run's recorded val accuracy" if meta.get("normalized_from") else ""))
    if "bias_worth" in res:
        A(f"  deleting the bias entirely (r=0) costs {res['bias_worth']:+.5f}"
          "  <- the span every number below sits inside")
    A("")
    A("  Row centring is free (softmax cannot see a per-row constant), so `full` MUST equal the")
    A("  unmodified accuracy exactly. `const_diag` additionally stores one scalar per head, which")
    A("  the audit's P4 measurement found recovers 90% of the self-pair's rank cost.")
    A("")
    for mode, rows in res["modes"].items():
        A(f"  mode = {mode}")
        A(f"    {'rank':>6}{'accuracy':>11}{'delta':>10}{'loss':>9}"
              f"{'preds moved':>13}{'% of bias kept':>16}")
        worth = res.get("bias_worth") or float("nan")
        for row in rows:
            label = "full" if row["rank"] is None else str(row["rank"])
            frac = ("—" if not math.isfinite(worth) or worth == 0
                    else f"{100 * (1 + row['delta_vs_unmodified'] / worth):.1f}%")
            flag = ""
            if row["rank"] is None:
                flag = "  <- CONTROL " + ("PASS" if row.get("control_passed") else "**FAIL**")
            moved = (f"{row.get('predictions_changed', 0)}"
                     f" ({100*row.get('predictions_changed_frac', 0):.1f}%)")
            A(f"    {label:>6}{row['accuracy']:>11.5f}{row['delta_vs_unmodified']:>+10.5f}"
              f"{row['loss']:>9.4f}{moved:>13}{frac:>16}{flag}")
        A("")
    A("  '% of bias kept' = how much of the accuracy the bias contributes that survives at this")
    A("  rank, i.e. measured against the r=0 lower bound rather than against zero. A rank that")
    A("  looks cheap against `full` alone is meaningless if r=0 is also cheap.")
    if meta["weights"] == "untrained":
        A("")
        A("  NOT A RESULT: untrained weights. The bias is an arbitrary smooth function here, so")
        A("  this run only exercises the machinery and the two controls.")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default=None,
                    help="ablation checkpoint; omit to exercise the probe on untrained weights")
    ap.add_argument("--data-dir", required=True, help="ragged CSR .pt shards")
    ap.add_argument("--num-jets", type=int, default=2000)
    ap.add_argument("--num-particles", type=int, default=128)
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ranks", type=str, default=None,
                    help="comma-separated ranks; 'full' for no truncation (default: 0,4,8,12,16,24,34,full)")
    ap.add_argument("--norm-stats", default=None,
                    help="norm_stats.json. REQUIRED for a meaningful accuracy: `load_bench_batch` "
                         "returns RAW features while training normalized them, so omitting this "
                         "feeds the model out-of-distribution inputs (observed: 0.17 accuracy and "
                         "a loss of 4.99, worse than the 2.30 of random guessing). Defaults to "
                         "`<data-dir>/../norm_stats.json` when that exists.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    ranks: Sequence[Optional[int]] = DEFAULT_RANKS
    if args.ranks:
        ranks = tuple(None if tok.strip() == "full" else int(tok)
                      for tok in args.ranks.split(","))

    from ablation.config import AblationConfig
    from ablation.rank_audit import _load_model_from_checkpoint
    from dataloader.ragged_loader import load_bench_batch
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

    # Same shard-stride selection as the audit: val_5M's shards are grouped by class, not
    # interleaved, so a contiguous range reaches only a couple of classes.
    available = sorted(Path(args.data_dir).glob("*.pt"))
    stride = max(1, len(available) // args.num_classes)
    shards = list(range(0, len(available), stride))[: args.num_classes]
    per_shard = max(1, args.num_jets // max(1, len(shards)))

    xs, vs, ms, ys = [], [], [], []
    for shard in shards:
        x_s, v_s, m_s, y_s = load_bench_batch(
            args.data_dir, batch_size=per_shard, shard_idx=shard,
            pad_to=args.num_particles,
        )
        xs.append(x_s); vs.append(v_s); ms.append(m_s); ys.append(y_s)
    x = torch.cat(xs); v = torch.cat(vs); mask = torch.cat(ms)
    y = torch.cat(ys)
    y = y.argmax(-1) if y.dim() > 1 else y

    # Normalize `x` exactly as training did. `load_bench_batch` is the *benchmark* loader and
    # returns raw features, whereas `ablation/data.py` applies NormStats (pT/E divided by their
    # mean, the rest z-scored) while leaving `v` raw. Feeding raw features to a model trained on
    # normalized ones is not a small perturbation: measured 0.17 accuracy at a loss of 4.99, i.e.
    # worse than random, on a checkpoint that scores 0.8624. The unmodified-accuracy line in the
    # report exists to catch exactly this, and did.
    stats_path = args.norm_stats
    if stats_path is None:
        candidate = Path(args.data_dir).parent / "norm_stats.json"
        stats_path = str(candidate) if candidate.exists() else None
    normalized_masked: Optional[bool] = None
    if stats_path:
        from ablation.data import NormStats
        stats = NormStats.load(Path(stats_path))
        # Call whichever signature this checkout's NormStats has, and record which.
        #
        # Newer `apply(x, mask)` keeps padded slots at exact zero; the older `apply(x)` applies the
        # affine everywhere, so padded slots become `offset` and those constants reach weaver's
        # `Embed.input_bn`. That difference is not this probe's to choose: the checkpoint was
        # trained with whatever its own tree did (`_normalizing_collate` calls `stats.apply(x)`), so
        # faithfulness to training is what makes the unmodified accuracy match the recorded value.
        # Overriding it with the "better" masked version would be measuring a different model.
        try:
            x = stats.apply(x, mask)
            normalized_masked = True
        except TypeError:
            x = stats.apply(x)
            normalized_masked = False
        normalized_from: Optional[str] = stats_path
    else:
        normalized_from = None
        print("WARNING: no norm_stats.json found or given, so features are RAW. The accuracy "
              "below will be near-chance and meaningless. Pass --norm-stats.")

    res = evaluate_ranks(model, x, v, mask, y, ranks=ranks, batch=args.batch_size)
    res["meta"] = {
        "weights": weights,
        "source": args.data_dir,
        "num_jets": int(x.shape[0]),
        "num_particles": args.num_particles,
        "shards": shards,
        "class_coverage": f"{len(set(int(c) for c in y.tolist()))}/{args.num_classes}",
        "ranks": [("full" if r is None else r) for r in ranks],
        "seed": args.seed,
        "arm": config.arm,
        "normalized_from": normalized_from,
        "normalization_masked_padding": normalized_masked,
    }

    print(format_report(res))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(res, indent=2))
        print(f"\nwrote {path}")

    failed = [m for m, rows in res["modes"].items()
              for r in rows if r["rank"] is None and not r.get("control_passed", True)]
    if failed:
        print(f"\nCONTROL FAILED for mode(s) {failed}: `full` must equal the unmodified "
              "accuracy exactly, because row centring is a softmax no-op. Treat every number "
              "above as invalid until this passes.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
