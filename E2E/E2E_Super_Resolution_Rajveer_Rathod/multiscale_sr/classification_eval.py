"""Classification-based evaluation pipeline for a multi-scale SR checkpoint.

Why this exists
---------------
PSNR/SSIM/L1 measure whether the super-resolved image *looks* like the ground
truth. They say nothing about whether SR preserves the **physics-classification**
information the downstream jet tagger relies on. This pipeline makes that the
headline metric: it trains a jet tagger and measures, with a battery of
classification diagnostics and figures, how taggable HR / LR / SR images are.

Three image sources, all at HR resolution (LR is bicubic-upsampled):
    HR  ground truth        -> ceiling
    LR  bicubic upsample    -> floor (degraded input)
    SR  generator output    -> recovery

Headline = tagging efficiency = AUC_SR / AUC_HR (fixed HR-trained tagger).

Figures produced (in <out-dir>/):
    roc_overlay.png              ROC for HR/LR/SR (fixed HR tagger) + per-source
    auc_summary_bar.png          AUC bars with the efficiency headline
    score_distributions.png      tagger score histograms per class, per source
    confusion_matrices.png       confusion at the optimal (Youden-J) threshold
    efficiency_vs_threshold.png  signal eff & bkg rejection vs threshold + ROC50
    calibration.png              reliability curves + ECE per source
    score_agreement.png          SR-vs-HR per-sample score scatter (HR tagger)
    classification_eval.json     every metric as machine-readable JSON
    EXPLANATION.md               what each figure/metric means

Parquet only — the HDF5/CaloChallenge data has no class label.

Images are drawn from the "test" split: the held-out file(s) not used for
training are further split row-wise into val (used during training for
checkpoint/best.pt selection — see train.py) and test (never touched until
this evaluation). --val-ratio here must match the value the checkpoint was
trained with, or "test" here will not line up with the held-out file group
that checkpoint actually used.

Usage
-----
    python classification_eval.py \
        --checkpoint experiments/<run>/checkpoints/best.pt \
        --data-dir ../datasets
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from multiscale_sr.classification_metrics import (
    background_rejection_at_signal_eff,
    best_threshold,
    calibration_curve,
    confusion_at_threshold,
    roc_points,
    score_agreement,
    working_points,
)
from multiscale_sr.data import get_dataloader
from multiscale_sr.data.normalization import ChannelStats, denormalize
from multiscale_sr.engine import collect_tagging_tensors
from multiscale_sr.models import Generator
from multiscale_sr.tagger import tagger_scores, train_tagger
from multiscale_sr.utils import resolve_env, seed_everything

PACKAGE_ROOT = Path(__file__).resolve().parent
_SOURCES = ("hr", "lr", "sr")
_LABELS = {"hr": "HR (ground truth)", "lr": "LR (bicubic)", "sr": "SR (generator)"}
_COLORS = {"hr": "tab:green", "lr": "tab:red", "sr": "tab:blue"}
_ROC50_EFF = 0.5  # signal efficiency at which background rejection is quoted


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classification-based SR evaluation")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--dataset-format", type=str, default=None, choices=["parquet", "hdf5"])
    p.add_argument("--scale", type=int, default=None)
    p.add_argument("--hr-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-ratio", type=float, default=0.33,
                   help="Fraction of files held out from training (split into val/test "
                        "row-halves); must match the checkpoint's training --val-ratio")
    p.add_argument("--max-samples", type=int, default=4000,
                   help="Cap images materialized from the held-out test split for tagging "
                        "(the tagger's own train/test split is drawn from these)")
    p.add_argument("--test-frac", type=float, default=0.3,
                   help="Fraction of materialized samples held out for the tagger's own "
                        "eval split (independent of the SR generator's val/test split)")
    p.add_argument("--tagger-epochs", type=int, default=15)
    p.add_argument("--tagger-width", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Default: <run>/figures/classification")
    p.add_argument("--skip-per-source", action="store_true",
                   help="Only train the fixed-HR tagger (skip per-source taggers)")
    return p


def _split_indices(n: int, test_frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = max(1, int(round(n * test_frac)))
    return perm[n_test:], perm[:n_test]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _fig_roc_overlay(roc: dict[str, dict], per_source_roc: dict[str, dict] | None, path: Path) -> None:
    plt = _plt()
    ncols = 2 if per_source_roc else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6), squeeze=False)

    def _panel(ax, curves, title):
        for src in _SOURCES:
            if src not in curves:
                continue
            c = curves[src]
            ax.plot(c["fpr"], c["tpr"], color=_COLORS[src], lw=2,
                    label=f"{_LABELS[src]}  (AUC={c['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.50)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    _panel(axes[0][0], roc, "Fixed HR-trained tagger")
    if per_source_roc:
        _panel(axes[0][1], per_source_roc, "Per-source taggers")
    fig.suptitle("Tagging ROC — does SR preserve class-discriminative structure?", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_auc_bar(primary_auc: dict[str, float], efficiency: float, recovery: float, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    srcs = [s for s in _SOURCES if s in primary_auc]
    aucs = [primary_auc[s] for s in srcs]
    bars = ax.bar([_LABELS[s] for s in srcs], aucs,
                  color=[_COLORS[s] for s in srcs], alpha=0.85, edgecolor="k", linewidth=0.8)
    ax.axhline(0.5, color="k", ls="--", lw=1, label="Random (0.50)")
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Tagging AUC (fixed HR-trained tagger)")
    ax.set_title(f"Efficiency AUC_SR/AUC_HR = {efficiency:.1%}   |   "
                 f"LR→HR gap recovered = {recovery:.1%}")
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=10)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_score_distributions(scores: dict[str, np.ndarray], y_test: np.ndarray, path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(_SOURCES), figsize=(5 * len(_SOURCES), 4.5), squeeze=False)
    bins = np.linspace(0, 1, 31)
    for ax, src in zip(axes[0], _SOURCES):
        s = scores[src]
        ax.hist(s[y_test == 0], bins=bins, alpha=0.6, color="tab:gray",
                label="class 0", density=True)
        ax.hist(s[y_test == 1], bins=bins, alpha=0.6, color=_COLORS[src],
                label="class 1", density=True)
        ax.set_title(f"{_LABELS[src]}")
        ax.set_xlabel("tagger score (P[class 1])")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    fig.suptitle("Tagger score distributions (fixed HR tagger) — separability by source", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_confusion(conf: dict[str, object], path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, len(_SOURCES), figsize=(4.5 * len(_SOURCES), 4.2), squeeze=False)
    for ax, src in zip(axes[0], _SOURCES):
        c = conf[src]
        m = c.matrix
        im = ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(m[i, j]), ha="center", va="center",
                        color="white" if m[i, j] > m.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
        ax.set_yticks([0, 1], labels=["true 0", "true 1"])
        ax.set_title(f"{_LABELS[src]}\nacc={c.accuracy:.3f}  F1={c.f1:.3f}  thr={c.threshold:.2f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion at Youden-J optimal threshold (fixed HR tagger)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_efficiency_vs_threshold(roc: dict[str, dict], scores: dict[str, np.ndarray],
                                 y_test: np.ndarray, path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: signal eff & background eff vs threshold.
    thr_grid = np.linspace(0, 1, 101)
    for src in _SOURCES:
        s = scores[src]
        sig = [float((s[y_test == 1] >= t).mean()) for t in thr_grid]
        bkg = [float((s[y_test == 0] >= t).mean()) for t in thr_grid]
        axes[0].plot(thr_grid, sig, color=_COLORS[src], lw=2, label=f"{src.upper()} signal eff")
        axes[0].plot(thr_grid, bkg, color=_COLORS[src], lw=1.3, ls="--", label=f"{src.upper()} bkg eff")
    axes[0].set_xlabel("threshold")
    axes[0].set_ylabel("efficiency")
    axes[0].set_title("Signal (solid) vs background (dashed) efficiency")
    axes[0].legend(fontsize=8, ncol=3)
    axes[0].grid(True, alpha=0.3)

    # Right: background rejection (1/eps_B) vs signal efficiency, the HEP ROC view.
    for src in _SOURCES:
        c = roc[src]
        fpr, tpr = np.asarray(c["fpr"]), np.asarray(c["tpr"])
        mask = fpr > 0
        axes[1].plot(tpr[mask], 1.0 / fpr[mask], color=_COLORS[src], lw=2,
                     label=f"{src.upper()}  1/eps_B@{_ROC50_EFF:.0%}={c['bkg_rej_at_50']:.1f}")
    axes[1].axvline(_ROC50_EFF, color="k", ls=":", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("signal efficiency (eps_S)")
    axes[1].set_ylabel("background rejection (1 / eps_B)")
    axes[1].set_title("Background rejection vs signal efficiency")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle("Working-point analysis (fixed HR tagger)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_calibration(cal: dict[str, object], path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for src in _SOURCES:
        c = cal[src]
        m = np.isfinite(c.bin_confidence)
        ax.plot(c.bin_confidence[m], c.bin_accuracy[m], "o-", color=_COLORS[src], lw=2,
                label=f"{_LABELS[src]}  (ECE={c.ece:.3f})")
    ax.set_xlabel("mean predicted score (confidence)")
    ax.set_ylabel("empirical fraction positive (accuracy)")
    ax.set_title("Reliability curves (fixed HR tagger)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _image_energy(t: "torch.Tensor", stats: ChannelStats) -> np.ndarray:
    """Per-image total raw energy = sum over (C,H,W) after denormalizing."""
    raw = denormalize(t.float(), stats)
    return raw.sum(dim=(1, 2, 3)).cpu().numpy()


def _corr_stats(x: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Correlation of a per-source quantity x against the HR reference ref."""
    mask = np.isfinite(x) & np.isfinite(ref) & (ref > 0)
    x, ref = x[mask], ref[mask]
    r = float(np.corrcoef(x, ref)[0, 1]) if x.size > 1 else float("nan")
    ratio = float(np.mean(x / ref)) if x.size else float("nan")
    mad = float(np.mean(np.abs(x - ref) / ref)) if x.size else float("nan")
    return {"pearson_r": r, "mean_ratio": ratio, "mean_abs_frac_diff": mad}


def _fig_energy_correlation(energy: dict[str, np.ndarray], corr: dict[str, dict], path: Path) -> None:
    """2D number-density of per-image total energy: SR-vs-HR and LR-vs-HR."""
    plt = _plt()
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    hr = energy["hr"]
    for ax, src in zip(axes, ("sr", "lr")):
        s = energy[src]
        lo = float(min(hr.min(), s.min()))
        hi = float(max(hr.max(), s.max()))
        h = ax.hist2d(hr, s, bins=60, range=[[lo, hi], [lo, hi]],
                      norm=LogNorm(), cmap="viridis")
        fig.colorbar(h[3], ax=ax, label="number density (events)")
        ax.plot([lo, hi], [lo, hi], "w--", lw=1.2)
        c = corr[src]
        ax.set_xlabel("HR total energy (raw)")
        ax.set_ylabel(f"{src.upper()} total energy (raw)")
        ax.set_title(f"{src.upper()} vs HR  (r={c['pearson_r']:.3f}, "
                     f"mean ratio={c['mean_ratio']:.3f})")
    fig.suptitle("Per-image energy correlation (number density) — is total energy conserved?",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fig_pt_correlation(energy: dict[str, np.ndarray], pt: np.ndarray,
                        pt_corr: dict[str, float], path: Path) -> None:
    """Per-source 2D number-density (hist2d) of jet pt (x) vs image total energy (y).

    hist2d (not scatter) so the dense population shows structure; axes are clipped
    to robust percentiles so a single high-energy jet doesn't flatten everything.
    """
    plt = _plt()
    from matplotlib.colors import LogNorm

    # Robust shared ranges across sources (ignore the extreme tail for the view).
    pt_hi = float(np.nanpercentile(pt, 99))
    pt_lo = float(np.nanmin(pt))
    all_e = np.concatenate([energy[s] for s in _SOURCES])
    e_hi = float(np.nanpercentile(all_e, 99))
    e_lo = float(max(0.0, np.nanmin(all_e)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    last_mesh = None
    for ax, src in zip(axes, _SOURCES):
        h = ax.hist2d(pt, energy[src], bins=60,
                      range=[[pt_lo, pt_hi], [e_lo, e_hi]],
                      norm=LogNorm(), cmap="viridis")
        last_mesh = h[3]
        ax.set_xlabel("jet pt")
        ax.set_ylabel("image total energy (raw)")
        ax.set_title(f"{_LABELS[src]}  (r(pt,E)={pt_corr[src]:.3f})")
    fig.colorbar(last_mesh, ax=axes, label="number density (events)", shrink=0.85)
    fig.suptitle("pt vs image-energy number density by source — does SR preserve the pt–energy band?",
                 fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_score_agreement(scores: dict[str, np.ndarray], agreement: dict[str, dict], path: Path) -> None:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, src in zip(axes, ("sr", "lr")):
        ag = agreement[src]
        ax.scatter(scores["hr"], scores[src], s=6, alpha=0.3, color=_COLORS[src])
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("HR tagger score on HR")
        ax.set_ylabel(f"HR tagger score on {src.upper()}")
        ax.set_title(f"{src.upper()} vs HR  (r={ag['pearson_r']:.3f}, "
                     f"mean|Δ|={ag['mean_abs_score_diff']:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-sample score agreement — does SR fool the HR tagger like HR does?", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# EXPLANATION.md
# --------------------------------------------------------------------------- #
def _write_explanation(results: dict, path: Path) -> None:
    prim = results["primary_fixed_hr_tagger"]
    eff = prim["tagging_efficiency_sr_over_hr"]
    rec = prim["recovery_fraction_lr_to_hr"]
    auc = prim["auc"]
    md = f"""# Classification-Based Evaluation

Pixel metrics (PSNR/SSIM/L1) answer *"does it look right?"*. This pipeline answers
the question that matters for physics: **does super-resolution preserve the
class-discriminative information a jet tagger needs?** A jet tagger (small CNN) is
trained on the real class label, then evaluated on three image sources at HR
resolution — HR (ground truth, the ceiling), LR (bicubic upsample, the floor), and
SR (generator output, the recovery).

## Headline numbers (scale={results['scale']}x, n_test={results['n_test']})

| source | AUC (fixed HR tagger) |
|--------|----------------------:|
| HR     | {auc['hr']:.4f} |
| LR     | {auc['lr']:.4f} |
| SR     | {auc['sr']:.4f} |

- **Tagging efficiency (AUC_SR / AUC_HR) = {eff:.1%}** — how close SR taggability
  gets to the HR ceiling. Close to 100% means SR images are essentially as
  taggable as real HR.
- **LR→HR gap recovered = {rec:.1%}** — of the taggability lost by downscaling,
  how much SR restores. 0% = no better than LR; 100% = fully restored to HR.
- **Background rejection 1/eps_B @ {_ROC50_EFF:.0%} signal eff** —
  HR={prim['bkg_rej_at_50']['hr']:.1f}, LR={prim['bkg_rej_at_50']['lr']:.1f},
  SR={prim['bkg_rej_at_50']['sr']:.1f}. The HEP-standard operating point.

## Figures

| file | what it shows |
|------|---------------|
| `roc_overlay.png` | ROC curves for HR/LR/SR. SR sitting near HR (above LR) is the win condition. Left: one fixed HR-trained tagger on all sources. Right: an independent tagger per source. |
| `auc_summary_bar.png` | The three AUCs as bars with the efficiency / recovery headline. |
| `score_distributions.png` | Tagger score histograms split by true class. Wider class separation = more taggable. SR should resemble HR, not LR. |
| `confusion_matrices.png` | Confusion matrix + accuracy/F1 at the Youden-J optimal threshold, per source. |
| `efficiency_vs_threshold.png` | Left: signal/background efficiency vs threshold. Right: background rejection vs signal efficiency (log scale) — the curve physicists read, marked at {_ROC50_EFF:.0%}. |
| `calibration.png` | Reliability curves + ECE: does the tagger's confidence stay calibrated on SR images, or does SR shift it? |
| `score_agreement.png` | Per-sample scatter of the HR tagger's score on SR (and LR) vs its score on HR. Points on the diagonal mean SR fools the tagger *the same way* HR does — a stricter test than matching aggregate AUC. |

## Method notes

- **Fixed-HR tagger (primary):** one tagger trained on HR, evaluated frozen on all
  three sources. Answers *"do SR images look like HR to a tagger trained on real
  data?"* This is the headline.
- **Per-source taggers (secondary):** an independent tagger trained on each source,
  measuring the class information each source contains regardless of HR
  compatibility.
- All sources share HR spatial size (LR is bicubic-upsampled) so one tagger
  architecture consumes any of them and comparisons are fair.
- Parquet only — the HDF5/CaloChallenge data has no class label.
"""
    path.write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
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

    print(f"[cls] materializing up to {args.max_samples} samples (scale={scale})...")
    try:
        data = collect_tagging_tensors(gen, test_loader, stats, env.device, max_samples=args.max_samples)
    except ValueError as exc:
        raise SystemExit(f"[cls] {exc}")

    y = data["y"]
    n = y.shape[0]
    # Tagger's own train/test split, drawn from the SR generator's held-out
    # "test" split — distinct from and unrelated to the generator's val/test.
    train_idx, test_idx = _split_indices(n, args.test_frac, args.seed)
    y_test = y[test_idx]
    y_test_np = y_test.numpy()
    print(f"[cls] n={n} (tagger-train={len(train_idx)} tagger-test={len(test_idx)}) "
          f"class balance: {torch.bincount(y).tolist()}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(args.checkpoint).parent.parent / "figures" / "classification"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, object] = {"scale": scale, "n_train": len(train_idx), "n_test": len(test_idx)}

    # ---- Train the fixed HR tagger, score every source ----
    print("[cls] training fixed HR tagger...")
    hr_tagger = train_tagger(
        data["hr"][train_idx], y[train_idx], env.device,
        width=args.tagger_width, epochs=args.tagger_epochs, seed=args.seed,
    )
    scores = {src: tagger_scores(hr_tagger, data[src][test_idx], env.device) for src in _SOURCES}

    roc: dict[str, dict] = {}
    for src in _SOURCES:
        fpr, tpr, thr, auc = roc_points(scores[src], y_test_np)
        roc[src] = {
            "fpr": fpr, "tpr": tpr, "thr": thr, "auc": auc,
            "bkg_rej_at_50": background_rejection_at_signal_eff(fpr, tpr, _ROC50_EFF),
        }
        print(f"  fixed-HR-tagger on {src.upper():2s}: AUC={auc:.4f}  "
              f"1/eps_B@50%={roc[src]['bkg_rej_at_50']:.1f}")

    auc_hr = roc["hr"]["auc"]
    efficiency = (roc["sr"]["auc"] / auc_hr) if auc_hr and not np.isnan(auc_hr) else float("nan")
    gap = auc_hr - roc["lr"]["auc"]
    recovery = ((roc["sr"]["auc"] - roc["lr"]["auc"]) / gap) if abs(gap) > 1e-6 else float("nan")

    # Threshold-dependent metrics: confusion at each source's Youden-J threshold.
    conf = {src: confusion_at_threshold(
        scores[src], y_test_np, best_threshold(roc[src]["fpr"], roc[src]["tpr"], roc[src]["thr"]))
        for src in _SOURCES}
    cal = {src: calibration_curve(scores[src], y_test_np) for src in _SOURCES}
    agreement = {src: score_agreement(scores["hr"], scores[src]) for src in ("lr", "sr")}

    results["primary_fixed_hr_tagger"] = {
        "auc": {s: roc[s]["auc"] for s in _SOURCES},
        "bkg_rej_at_50": {s: roc[s]["bkg_rej_at_50"] for s in _SOURCES},
        "tagging_efficiency_sr_over_hr": efficiency,
        "recovery_fraction_lr_to_hr": recovery,
        "confusion": {s: {k: v for k, v in asdict(conf[s]).items()} for s in _SOURCES},
        "working_points": {s: [asdict(w) for w in working_points(
            roc[s]["fpr"], roc[s]["tpr"], roc[s]["thr"])] for s in _SOURCES},
        "calibration_ece": {s: cal[s].ece for s in _SOURCES},
        "score_agreement_vs_hr": agreement,
    }
    print(f"[cls] tagging efficiency (AUC_SR/AUC_HR) = {efficiency:.1%}")
    print(f"[cls] recovery fraction (LR->HR gap closed by SR) = {recovery:.1%}")

    # ---- Per-source taggers (secondary) ----
    per_source_roc: dict[str, dict] | None = None
    if not args.skip_per_source:
        print("[cls] training a tagger per source...")
        per_source_roc = {}
        per_source_auc: dict[str, float] = {}
        for src in _SOURCES:
            t = train_tagger(
                data[src][train_idx], y[train_idx], env.device,
                width=args.tagger_width, epochs=args.tagger_epochs, seed=args.seed,
            )
            s = tagger_scores(t, data[src][test_idx], env.device)
            fpr, tpr, _thr, auc = roc_points(s, y_test_np)
            per_source_roc[src] = {"fpr": fpr, "tpr": tpr, "auc": auc}
            per_source_auc[src] = auc
            print(f"  tagger-on-{src.upper():2s} eval-on-{src.upper():2s}: AUC={auc:.4f}")
        results["secondary_per_source_tagger"] = {"auc": per_source_auc}

    # ---- Figures ----
    print("[cls] rendering figures...")
    _fig_roc_overlay(roc, per_source_roc, out_dir / "roc_overlay.png")
    _fig_auc_bar({s: roc[s]["auc"] for s in _SOURCES}, efficiency, recovery, out_dir / "auc_summary_bar.png")
    _fig_score_distributions(scores, y_test_np, out_dir / "score_distributions.png")
    _fig_confusion(conf, out_dir / "confusion_matrices.png")
    _fig_efficiency_vs_threshold(roc, scores, y_test_np, out_dir / "efficiency_vs_threshold.png")
    _fig_calibration(cal, out_dir / "calibration.png")
    _fig_score_agreement(scores, agreement, out_dir / "score_agreement.png")

    # ---- Physics correlation: per-image total energy + pt relationship ----
    print("[cls] rendering physics-correlation figures...")
    energy = {src: _image_energy(data[src], stats) for src in _SOURCES}
    energy_corr = {src: _corr_stats(energy[src], energy["hr"]) for src in ("sr", "lr")}
    _fig_energy_correlation(energy, energy_corr, out_dir / "energy_correlation.png")
    physics: dict[str, object] = {"energy_correlation_vs_hr": energy_corr}

    if "pt" in data:
        pt_np = data["pt"].numpy()
        pt_corr = {}
        for src in _SOURCES:
            m = np.isfinite(pt_np) & np.isfinite(energy[src])
            pt_corr[src] = (float(np.corrcoef(pt_np[m], energy[src][m])[0, 1])
                            if m.sum() > 1 else float("nan"))
        _fig_pt_correlation(energy, pt_np, pt_corr, out_dir / "pt_correlation.png")
        physics["pt_energy_pearson_r"] = pt_corr
    else:
        print("[cls] no pt in batch (HDF5?) — skipping pt_correlation.png")
    results["physics_correlation"] = physics

    # ---- JSON + EXPLANATION (drop numpy arrays from the JSON payload) ----
    out_json = out_dir / "classification_eval.json"
    out_json.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    _write_explanation(results, out_dir / "EXPLANATION.md")

    print(f"[cls] wrote {out_json}")
    print(f"[cls] figures + EXPLANATION.md -> {out_dir}")


if __name__ == "__main__":
    main()
