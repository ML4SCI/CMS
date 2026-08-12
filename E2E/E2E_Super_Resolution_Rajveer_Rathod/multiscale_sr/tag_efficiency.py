"""Tagging efficiency — does super-resolution preserve jet-classification info?

This is the physics-facing metric for the SR task. A jet tagger is trained on the
class label `y`, then its AUC is compared across image sources:

    HR  (ground truth)        -> ceiling: best taggability available
    LR  (bicubic upsampled)   -> floor: taggability of the degraded input
    SR  (generator output)    -> recovery: how much SR restores

Two analyses (both requested):

  PRIMARY  — fixed HR-trained tagger.
      Train one tagger on HR, evaluate that *same frozen tagger* on HR/LR/SR.
      Answers "do SR images look like HR to a tagger trained on real data?"
      Headline efficiency = AUC_SR / AUC_HR.

  SECONDARY — per-source taggers.
      Train an independent tagger on each of HR/LR/SR and report each one's AUC.
      Answers "how much class-discriminative information does each source contain,
      regardless of HR-compatibility?"

Parquet only: the HDF5/CaloChallenge data has no class label.

Images are drawn from the "test" split — the held-out row-half never used for
training or for best.pt checkpoint selection (that's "val", see train.py).
--val-ratio must match the checkpoint's training value so "test" refers to
the same held-out file group.

Usage
-----
    python tag_efficiency.py \
        --checkpoint experiments/<run>/checkpoints/best.pt \
        --data-dir ../datasets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from multiscale_sr.data import get_dataloader
from multiscale_sr.data.normalization import ChannelStats
from multiscale_sr.engine import collect_tagging_tensors
from multiscale_sr.models import Generator
from multiscale_sr.tagger import eval_tagger_auc, roc_auc, tagger_scores, train_tagger
from multiscale_sr.utils import resolve_env, seed_everything

PACKAGE_ROOT = Path(__file__).resolve().parent
_SOURCES = ("hr", "lr", "sr")
_SOURCE_LABELS = {"hr": "HR (ground truth)", "lr": "LR (bicubic)", "sr": "SR (generator)"}
_SOURCE_COLORS = {"hr": "tab:green", "lr": "tab:red", "sr": "tab:blue"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Tagging efficiency for a multi-scale SR checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--dataset-format", type=str, default=None, choices=["parquet", "hdf5"])
    p.add_argument("--scale", type=int, default=None, help="Override scale (else from checkpoint)")
    p.add_argument("--hr-size", type=int, default=None, help="Override HR size (else from checkpoint)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.33,
                   help="Must match the checkpoint's training --val-ratio; determines "
                        "which held-out file group 'test' is drawn from")
    p.add_argument("--max-samples", type=int, default=4000,
                   help="Cap images materialized from the held-out test split "
                        "(the tagger's own train/test split is drawn from these)")
    p.add_argument("--test-frac", type=float, default=0.3,
                   help="Fraction of materialized samples held out for the tagger's own "
                        "eval split (independent of the SR generator's val/test split)")
    p.add_argument("--tagger-epochs", type=int, default=15)
    p.add_argument("--tagger-width", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to write json + figures (default: <run>/figures/tagging)")
    p.add_argument("--skip-per-source", action="store_true",
                   help="Run only the primary fixed-HR-tagger analysis")
    p.add_argument("--save-hr-tagger", type=str, default=None,
                   help="Save the trained HR tagger here for reuse as train.py --tagger-ckpt")
    return p


def _split_indices(n: int, test_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = max(1, int(round(n * test_frac)))
    return perm[n_test:], perm[:n_test]  # train_idx, test_idx


def _save_roc_figure(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for src in _SOURCES:
        if src not in curves:
            continue
        fpr, tpr, auc = curves[src]
        ax.plot(fpr, tpr, color=_SOURCE_COLORS[src], lw=2,
                label=f"{_SOURCE_LABELS[src]}  (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_bar_figure(primary: dict[str, float], efficiency: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    srcs = [s for s in _SOURCES if s in primary]
    aucs = [primary[s] for s in srcs]
    colors = [_SOURCE_COLORS[s] for s in srcs]
    bars = ax.bar([_SOURCE_LABELS[s] for s in srcs], aucs, color=colors, alpha=0.85,
                  edgecolor="k", linewidth=0.8)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="Random (0.50)")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Tagging AUC (fixed HR-trained tagger)")
    ax.set_title(f"Tagging Efficiency = AUC_SR / AUC_HR = {efficiency:.1%}")
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=10)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    env = resolve_env()
    print(f"[env] {env}")

    ckpt = torch.load(args.checkpoint, map_location=env.device)
    ckpt_args = ckpt.get("args", {})
    scale = args.scale or ckpt_args.get("scale")
    hr_size = args.hr_size or ckpt_args.get("hr_size", 125)
    if scale is None:
        raise SystemExit("scale not found in checkpoint; pass --scale explicitly.")

    stats = ChannelStats.from_dict(ckpt["stats"])
    gen = Generator(
        base_channels=ckpt_args.get("gen_channels", 64),
        num_blocks=ckpt_args.get("gen_blocks", 8),
    ).to(env.device)
    gen.load_state_dict(ckpt["generator"])

    cache = Path(args.checkpoint).parent.parent / "normalization.json"
    if not cache.exists():
        stats.save(cache)

    test_loader, _ = get_dataloader(
        path=Path(args.data_dir), split="test", env=env, batch_size=args.batch_size,
        scale=scale, hr_size=hr_size, dataset_type=args.dataset_format,
        val_ratio=args.val_ratio, stats_cache_path=cache,
    )

    print(f"[tag] materializing up to {args.max_samples} samples (scale={scale})...")
    try:
        data = collect_tagging_tensors(gen, test_loader, stats, env.device, max_samples=args.max_samples)
    except ValueError as exc:
        raise SystemExit(f"[tag] {exc}")

    y = data["y"]
    n = y.shape[0]
    # Tagger's own train/test split, drawn from the SR generator's held-out
    # "test" split — distinct from and unrelated to the generator's val/test.
    train_idx, test_idx = _split_indices(n, args.test_frac, args.seed)
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"[tag] n={n} (tagger-train={len(train_idx)} tagger-test={len(test_idx)}) "
          f"class balance: {torch.bincount(y).tolist()}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(args.checkpoint).parent.parent / "figures" / "tagging"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {"scale": scale, "n_train": len(train_idx), "n_test": len(test_idx)}

    # ---- PRIMARY: fixed HR-trained tagger evaluated on HR/LR/SR ----
    print("[tag] PRIMARY: training tagger on HR...")
    hr_tagger = train_tagger(
        data["hr"][train_idx], y_train, env.device,
        width=args.tagger_width, epochs=args.tagger_epochs, seed=args.seed,
    )
    if args.save_hr_tagger:
        save_path = Path(args.save_hr_tagger)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tagger": hr_tagger.state_dict(), "width": args.tagger_width}, save_path)
        print(f"[tag] saved HR tagger -> {save_path} (reuse via train.py --tagger-ckpt)")
    primary_auc: dict[str, float] = {}
    primary_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for src in _SOURCES:
        scores = tagger_scores(hr_tagger, data[src][test_idx], env.device)
        fpr, tpr, auc = roc_auc(scores, y_test.numpy())
        primary_auc[src] = auc
        primary_curves[src] = (fpr, tpr, auc)
        print(f"  fixed-HR-tagger on {src.upper():2s}: AUC={auc:.4f}")

    auc_hr = primary_auc["hr"]
    efficiency = (primary_auc["sr"] / auc_hr) if auc_hr and not np.isnan(auc_hr) else float("nan")
    # Recovery fraction relative to the LR->HR gap (0 = no better than LR, 1 = full HR).
    gap = auc_hr - primary_auc["lr"]
    recovery = ((primary_auc["sr"] - primary_auc["lr"]) / gap) if abs(gap) > 1e-6 else float("nan")
    results["primary_fixed_hr_tagger"] = {
        "auc": primary_auc,
        "tagging_efficiency_sr_over_hr": efficiency,
        "recovery_fraction_lr_to_hr": recovery,
    }
    print(f"[tag] tagging efficiency (AUC_SR/AUC_HR) = {efficiency:.1%}")
    print(f"[tag] recovery fraction (LR->HR gap closed by SR) = {recovery:.1%}")

    _save_roc_figure(primary_curves, "Tagging ROC — fixed HR-trained tagger", out_dir / "tagging_roc.png")
    _save_bar_figure(primary_auc, efficiency, out_dir / "tagging_auc_bar.png")

    # ---- SECONDARY: per-source taggers ----
    if not args.skip_per_source:
        print("[tag] SECONDARY: training a tagger per source...")
        per_source_auc: dict[str, float] = {}
        for src in _SOURCES:
            t = train_tagger(
                data[src][train_idx], y_train, env.device,
                width=args.tagger_width, epochs=args.tagger_epochs, seed=args.seed,
            )
            auc = eval_tagger_auc(t, data[src][test_idx], y_test, env.device)
            per_source_auc[src] = auc
            print(f"  tagger-on-{src.upper():2s} eval-on-{src.upper():2s}: AUC={auc:.4f}")
        results["secondary_per_source_tagger"] = {"auc": per_source_auc}

    out_json = out_dir / "tag_efficiency.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[tag] wrote {out_json}")
    print(f"[tag] figures -> {out_dir}/tagging_roc.png, tagging_auc_bar.png")


if __name__ == "__main__":
    main()
