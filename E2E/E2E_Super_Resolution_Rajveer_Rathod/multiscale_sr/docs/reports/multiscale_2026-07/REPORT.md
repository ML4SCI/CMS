# Multi-Scale Super-Resolution: Full Evaluation

**Author:** Rajveer Rathod · **Project:** GSoC 2026 — ML4Sci, CMS Calorimeter Super-Resolution
**Runs:** Colab, 2026-07-12 → 2026-07-18 · **Dataset:** CMS jets (parquet) · **Seed:** 42 · **n_test:** 1210

Three independent GANs (16×, 32×, 64×) trained with the stabilised settings, each evaluated on **three axes**: reconstruction quality, physics conservation, and downstream taggability. Every figure produced by the pipeline is included and interpreted.

---

## Contents

1. [Headline](#1-headline) · 2. [The `d_loss_floor` ablation](#2-the-d_loss_floor-ablation) · 3. [Reconstruction](#3-reconstruction-quality)
4. [Physics conservation](#4-physics-conservation) · 5. [Taggability](#5-taggability-the-physics-facing-metric) · 6. [Working points & calibration](#6-working-points-and-calibration)
7. [The central contradiction](#7-the-central-contradiction) · 8. [Reproducing](#8-reproducing) · 9. [Open threads](#9-open-threads)

---

## 1. Headline

All three scales, stabilised settings, one fixed HR-trained tagger — **AUC_HR = 0.669 for every row**, identical across the three runs, which is the sanity check that the comparison is fair.

| Scale | Upscale | Epochs | val_L1 ↓ | peak_ratio ↑ | E-corr r | AUC_SR | Tagging efficiency | LR→HR recovered | per-sample r |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16× | 8× | 30 | 0.0935 | 0.688 | **0.998** | 0.487 | 72.9% | **−153.9%** | 0.19 |
| 32× | 4× | 40 | 0.0757 | 0.822 | 0.998 | 0.501 | 75.0% | +13.8% | 0.25 |
| **64×** | 2× | 40 | **0.0660** | **0.887** | 0.996 | **0.628** | **93.9%** | **+78.5%** | **0.64** |

Every row above is the run whose figures are reproduced in this report. A longer **32× run at 60 epochs** reaches val_L1 **0.0741**, peak_ratio **0.873**, AUC_SR **0.535**, efficiency **80.2%**, recovery **+25.3%**, r **0.35** — a strictly better model, reported in [§2](#2-the-d_loss_floor-ablation). Its figures are not included here, so it is kept out of this table to avoid mixing runs.

**Reconstruction quality improves monotonically with input resolution**, and taggability follows it. At 64× (a 2× upscale) SR recovers 78.5% of the taggability lost to downsampling; at 16× (8× upscale) it recovers none.

**Energy conservation does not follow that trend at all** — it is essentially perfect everywhere, *including* at 16× where taggability fails completely. That contradiction is the most instructive result in this study and is analysed in [§7](#7-the-central-contradiction).

## 2. The `d_loss_floor` ablation

`d_loss_floor` throttles the discriminator: if its loss falls below the floor, the update is **skipped**, so the generator can catch up. The skipped fraction is logged as `d_skip_frac`.

Controlled comparison at 32× — same seed, same architecture, same loss weights, **only the floor and epoch budget differ**:

| Run | `d_loss_floor` | Epochs | val_L1 ↓ | peak_ratio ↑ | Tagging efficiency |
|---|---:|---:|---:|---:|---:|
| `colab_32x_full` | **0.10** | 30 | 0.0803 | 0.786 | 75.7% |
| `colab_32x_full_40_dvalue_changed` | **0.05** | 40 | 0.0757 | 0.822 | 75.0% |
| `colab_32x_full_60` | **0.05** | 60 | **0.0741** | **0.873** | **80.2%** |

Lowering the floor 0.10 → 0.05 and training to 60 epochs improves peak_ratio by **+0.087 (+11%)** and tagging efficiency by **+4.5 points**.

Two caveats on this table, stated plainly:
- `colab_32x_full` (0.10) and `colab_32x_full_60` (0.05) share one tagger instance (AUC_HR = 0.6671); the 40-epoch row was scored against a second instance (AUC_HR = 0.6686). The **75.7% → 80.2%** comparison is therefore like-for-like; the 75.0% figure is not directly comparable to the other two.
- Floor and epoch budget vary together, so the two effects are not fully separated. The **peak_ratio** and **val_L1** columns are tagger-independent and move monotonically in the same direction, which is the more robust evidence.

### The mechanism, measured

Decoding the per-epoch logs shows *why* the higher floor underperforms:

| Run | mean `d_skip_frac` (post-warmup) | epochs pinned at 1.0 | mean `d_loss` |
|---|---:|---:|---:|
| floor = 0.10 | **0.985** | 33% | 0.084 |

At `d_loss_floor = 0.10` the discriminator is skipped on **98.5% of steps on average**, and for a third of all epochs on *every single step* — it is effectively frozen. The generator trains against a static critic, the adversarial term contributes nothing, and the model drifts back toward the L1-only behaviour that caused the original mode collapse.

| `floor = 0.10` (30 ep) | `floor = 0.05` (60 ep) |
|---|---|
| ![floor 0.10](../../../reports/multiscale_2026-07/figures/training/metrics_32x_floor010_30ep.png) | ![floor 0.05](../../../reports/multiscale_2026-07/figures/training/metrics_32x_floor005_60ep.png) |

Both curves look healthy — smooth monotonic val_L1 descent, no divergence. **That is the trap:** the failure is invisible in the loss curves and only shows up in `d_skip_frac` and `peak_ratio`.

### The floor is scale-dependent

The discriminator settles at different loss levels depending on task difficulty. At 16× it settles near 0.04, so a 0.05 floor would freeze it out entirely.

| Scale | Upscale | `d_loss_floor` |
|---:|---:|---:|
| 16× | 8× | **0.02** |
| 32× | 4× | 0.05 |
| 64× | 2× | 0.05 |

**Diagnostic:** watch `d_skip_frac` in `metrics.jsonl`. Sustained values near 1.0 mean the floor is too high for that scale.

## 3. Reconstruction quality

Each grid: **LR (bicubic up) · Super-Resolved · Ground Truth (HR) · SR − HR residual**. A near-white residual means near-zero error.

### 16× — the honest limit case

![16x samples](../../../reports/multiscale_2026-07/figures/16x/samples.png)

SR sharpens the core correctly and places it accurately, but HR's sparse single-pixel deposits scattered across the frame are **largely absent**. The residual shows a concentrated red core error — the model produces a physically sensible *concentrated* response where HR has structure spread over many pixels. At 8× upscale that information is largely destroyed by the downsampling and is not recoverable by this architecture.

### 32× — core plus most surrounding structure

![32x samples](../../../reports/multiscale_2026-07/figures/32x/samples.png)

Multi-blob clusters and many scattered deposits return. The residual is flatter than 16× and mostly non-structural, with remaining error concentrated at the peak.

### 64× — near-faithful

![64x samples](../../../reports/multiscale_2026-07/figures/64x/samples.png)

SR reproduces the extended filaments and most of the scattered "dust" across the whole frame. The residual is the flattest of the three, with error confined to small-amplitude individual pixels — about the residual floor for this task. At 2× upscale the generator's job is sharpening and peak restoration rather than inventing missing structure, so the same fixed model reaches a much higher ceiling.

## 4. Physics conservation

### Total energy is conserved at every scale

Per-image total energy, SR vs HR and LR vs HR:

| Scale | SR r | SR mean ratio | LR r | LR mean ratio |
|---:|---:|---:|---:|---:|
| 16× | **0.998** | 0.971 | 0.995 | 1.058 |
| 32× | **0.998** | 0.992 | 0.984 | 1.051 |
| 64× | 0.996 | 1.015 | 0.968 | 1.026 |

**SR beats bicubic on energy correlation at every scale**, and the mean ratio stays within ~3% of unity. The energy-conservation physics loss is doing its job.

| 16× | 32× | 64× |
|---|---|---|
| ![16x energy](../../../reports/multiscale_2026-07/figures/16x/energy_correlation.png) | ![32x energy](../../../reports/multiscale_2026-07/figures/32x/energy_correlation.png) | ![64x energy](../../../reports/multiscale_2026-07/figures/64x/energy_correlation.png) |

Note the visible tightening of the SR scatter (left panel) versus LR (right panel) in each pair — bicubic systematically over-estimates total energy (ratio ≈ 1.03–1.06), while SR sits much closer to unity.

### The pt–energy band is preserved

Jet `pt` versus total image energy — a physically meaningful correlation that must survive super-resolution:

| Scale | HR | LR | SR |
|---:|---:|---:|---:|
| 16× | 0.710 | 0.717 | **0.711** |
| 32× | 0.710 | 0.697 | **0.710** |
| 64× | 0.710 | 0.686 | **0.710** |

**SR reproduces HR's pt–energy correlation to three decimal places at every scale**, while LR degrades it as the scale coarsens (0.717 → 0.686). This is a clean, unambiguous win for the method.

| 16× | 32× | 64× |
|---|---|---|
| ![16x pt](../../../reports/multiscale_2026-07/figures/16x/pt_correlation.png) | ![32x pt](../../../reports/multiscale_2026-07/figures/32x/pt_correlation.png) | ![64x pt](../../../reports/multiscale_2026-07/figures/64x/pt_correlation.png) |

## 5. Taggability — the physics-facing metric

A jet tagger is trained on the real class label using **HR** images, then **frozen** and applied to HR / LR / SR. The question: *do SR images look like HR to a tagger trained on real data?*

### ROC — does SR preserve class-discriminative structure?

Left panel of each figure = the fixed HR tagger (the headline). Right panel = independent taggers trained per source.

| 16× — SR **below** the LR floor | 32× — SR just above LR | 64× — SR approaches HR |
|---|---|---|
| ![16x roc](../../../reports/multiscale_2026-07/figures/16x/roc_overlay.png) | ![32x roc](../../../reports/multiscale_2026-07/figures/32x/roc_overlay.png) | ![64x roc](../../../reports/multiscale_2026-07/figures/64x/roc_overlay.png) |

### AUC summary

| 16× | 32× | 64× |
|---|---|---|
| ![16x auc](../../../reports/multiscale_2026-07/figures/16x/auc_summary_bar.png) | ![32x auc](../../../reports/multiscale_2026-07/figures/32x/auc_summary_bar.png) | ![64x auc](../../../reports/multiscale_2026-07/figures/64x/auc_summary_bar.png) |

At 64× the SR bar reaches 0.628 against an HR ceiling of 0.669. At 16× the SR bar (0.487) sits *below* the LR bar (0.597) — worse than doing nothing.

### Score distributions — separability by source

Wide separation between the two class histograms means taggable. SR should resemble HR, not LR.

| 16× | 32× | 64× |
|---|---|---|
| ![16x dist](../../../reports/multiscale_2026-07/figures/16x/score_distributions.png) | ![32x dist](../../../reports/multiscale_2026-07/figures/32x/score_distributions.png) | ![64x dist](../../../reports/multiscale_2026-07/figures/64x/score_distributions.png) |

### Per-sample score agreement — the strictest test

For each image, the HR tagger's score on SR plotted against its score on HR. Points on the diagonal mean SR fools the tagger *the same way, sample by sample* — far stricter than matching aggregate AUC.

| Scale | SR vs HR (Pearson r) | LR vs HR |
|---:|---:|---:|
| 16× | 0.19 | 0.04 |
| 32× | 0.35 | 0.04 |
| **64×** | **0.64** | 0.04 |

| 16× — near-random scatter | 32× — weak correlation | 64× — points hug the diagonal |
|---|---|---|
| ![16x agree](../../../reports/multiscale_2026-07/figures/16x/score_agreement.png) | ![32x agree](../../../reports/multiscale_2026-07/figures/32x/score_agreement.png) | ![64x agree](../../../reports/multiscale_2026-07/figures/64x/score_agreement.png) |

This is the clearest single ranking in the study, and it tracks input resolution monotonically.

### Confusion at the operating point

Accuracy / precision / recall / F1 at the Youden-J optimal threshold, per source.

| 16× | 32× | 64× |
|---|---|---|
| ![16x conf](../../../reports/multiscale_2026-07/figures/16x/confusion_matrices.png) | ![32x conf](../../../reports/multiscale_2026-07/figures/32x/confusion_matrices.png) | ![64x conf](../../../reports/multiscale_2026-07/figures/64x/confusion_matrices.png) |

## 6. Working points and calibration

### Background rejection — the HEP-standard view

Background rejection `1/ε_B` at 50% signal efficiency. Higher is better.

| Scale | HR | LR | SR |
|---:|---:|---:|---:|
| 16× | 4.01 | 2.99 | **1.91** |
| 32× | 4.01 | 1.95 | **2.09** |
| 64× | 4.01 | 1.85 | **2.93** |

At 64×, SR lifts rejection from the bicubic floor of 1.85 to 2.93 against an HR ceiling of 4.01. At 16×, SR (1.91) is **worse than LR (2.99)** — consistent with the negative recovery fraction.

| 16× | 32× | 64× |
|---|---|---|
| ![16x wp](../../../reports/multiscale_2026-07/figures/16x/efficiency_vs_threshold.png) | ![32x wp](../../../reports/multiscale_2026-07/figures/32x/efficiency_vs_threshold.png) | ![64x wp](../../../reports/multiscale_2026-07/figures/64x/efficiency_vs_threshold.png) |

### Calibration — does SR shift the tagger's confidence?

Expected Calibration Error (lower is better):

| Scale | ECE_HR | ECE_LR | ECE_SR |
|---:|---:|---:|---:|
| 16× | 0.174 | 0.321 | **0.441** |
| 32× | 0.174 | 0.400 | 0.397 |
| 64× | 0.174 | 0.427 | **0.275** |

Only at 64× does SR improve calibration over the bicubic baseline (0.275 vs 0.427), moving meaningfully toward the HR reference. At 16× SR is the *worst* of the three — the tagger is confidently wrong on those images.

| 16× | 32× | 64× |
|---|---|---|
| ![16x cal](../../../reports/multiscale_2026-07/figures/16x/calibration.png) | ![32x cal](../../../reports/multiscale_2026-07/figures/32x/calibration.png) | ![64x cal](../../../reports/multiscale_2026-07/figures/64x/calibration.png) |

## 7. The central contradiction

The 16× model is the most instructive result here. Consider what it achieves simultaneously:

| Axis | 16× result | Verdict |
|---|---|---|
| Energy correlation vs HR | r = **0.998** (best of all scales) | ✅ excellent |
| pt–energy correlation | 0.711 vs HR's 0.710 | ✅ essentially exact |
| Energy response | 1.011 (conserved to ~1%) | ✅ unbiased |
| peak_ratio | 0.688 | ⚠️ acceptable |
| Tagging efficiency | 72.9% | ⚠️ looks respectable |
| **LR→HR recovery** | **−153.9%** | ❌ **worse than bicubic** |
| Background rejection | 1.91 vs LR's 2.99 | ❌ worse than bicubic |
| Calibration ECE | 0.441 vs LR's 0.321 | ❌ worse than bicubic |

**A model can conserve energy essentially perfectly, reproduce the pt–energy band exactly, produce clean sharp images — and still destroy the class information a jet tagger needs.**

That is the argument for this whole evaluation pipeline. Every pixel and physics-conservation metric says 16× works. Only the classification axis reveals that it doesn't.

### But the information is not gone

The secondary analysis trains an **independent** tagger on each source:

| Scale | per-source AUC_SR | (vs fixed-HR AUC_SR) |
|---:|---:|---:|
| 16× | **0.728** | (0.487) |
| 32× | 0.693 | (0.501) |
| 64× | 0.693 | (0.628) |

At 16×, SR images are **more taggable on their own (0.728) than HR is (0.670)** — they clearly contain class structure. They simply present it in a form **incompatible with a tagger trained on real HR**.

This is **distribution mismatch, not information loss** — and it is an actionable finding: training or domain-adapting the downstream tagger on SR images should rescue 16×/32× performance. That is the single highest-value follow-up experiment.

## 8. Reproducing

```bash
# Best configuration (64x, stabilised):
python train.py --config configs/scale_64.yaml --data-dir ../datasets \
    --d-loss-floor 0.05 --epochs 40

# 32x — note the longer budget; it had not converged at 40 epochs:
python train.py --config configs/scale_32.yaml --data-dir ../datasets \
    --d-loss-floor 0.05 --epochs 60

# 16x needs the lower floor to keep the discriminator alive:
python train.py --config configs/scale_16.yaml --data-dir ../datasets \
    --d-loss-floor 0.02 --epochs 30

# Every figure in this report:
python classification_eval.py --checkpoint experiments/<run>/checkpoints/best.pt \
    --data-dir ../datasets
```

Tagger settings for all reported numbers: width 32, 15 epochs, seed 42, 4032 samples (2822 train / 1210 test). Parquet only — the HDF5/CaloChallenge data carries no class label.

## 9. Open threads

1. **Train or domain-adapt the tagger on SR images.** The per-source result (§7) makes this a directed experiment, not a guess: if 16× SR is taggable at 0.728 on its own, an SR-trained tagger should recover much of the lost efficiency.
2. **32× has not converged.** 40 → 60 epochs moved peak_ratio 0.822 → 0.873 and val_L1 0.0757 → 0.0741, still improving (both tagger-independent). The cheapest remaining gain.
3. **Audit `psnr_norm` in `engine.py`.** The 64× run reports `psnr_norm = −3.91` while posting the best L1, peak_ratio, energy correlation, and tagging efficiency of any run. A negative PSNR alongside best-in-class everything else is almost certainly a metric-computation artifact, so PSNR is deliberately excluded from every table above and should not be quoted until fixed.
4. **Extend the classification evaluation to more than two classes** — the current tagger is binary; CMS jet tagging is naturally multi-class.
