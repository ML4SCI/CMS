# Physics-Aware Multi-Scale Super-Resolution for CMS Calorimeter Images

**GSoC 2026 · ML4Sci (Machine Learning for Science)**

**Contributor:** Rajveer Rathod ([@rajveer43](https://github.com/rajveer43) · [LinkedIn](https://linkedin.com/in/rajveer-rathod) · rathodrajveer1311@gmail.com)
**Mentors:** Pranath Reddy, Diptarko Choudhury, Resham Lal Sohal Jr
**Organization:** [Machine Learning for Science (ML4SCI)](https://ml4sci.org/)

---

## Overview

Calorimeter images from the CMS detector are sparse, high-dynamic-range, and physically constrained — the total deposited energy means something. Standard image super-resolution (SR) treats them like photographs and optimizes PSNR/SSIM, which rewards a generator for producing smooth, plausible-looking output. For physics data that is the wrong target: a model can score well on pixels while destroying exactly the fine structure a downstream jet tagger depends on.

This project trains an **independent GAN per downsampling scale** (16×, 32×, 64×) to answer a specific question:

> As detector input resolution degrades, how much *physics-classification* information can super-resolution actually recover — and at what point does it stop working?

The central methodological contribution is a **classification-based evaluation** that makes taggability the headline metric instead of pixel fidelity.

## Headline result

**Tagging efficiency = AUC_SR / AUC_HR**, using a single tagger trained on real HR images and then frozen. 100% means SR output is as taggable as ground truth.

Latest runs, after the training-stability fix (seed 42, n_test = 1210, AUC_HR = 0.669 for every row):

| Scale | Epochs | val_L1 ↓ | peak_ratio ↑ | AUC_SR | Efficiency (SR/HR) | LR→HR gap recovered | per-sample r |
|------:|-------:|---------:|-------------:|-------:|-------------------:|--------------------:|-------------:|
| 16× | 30 | 0.0935 | 0.688 | 0.487 | 72.9% | **−153.9%** | 0.19 |
| 32× | 40 | 0.0757 | 0.822 | 0.501 | 75.0% | +13.8% | 0.25 |
| **64×** | 40 | **0.0660** | **0.887** | **0.628** | **93.9%** | **+78.5%** | **0.64** |

A longer **32× run at 60 epochs** does better still — val_L1 0.0741, peak_ratio 0.873, efficiency **80.2%**, recovery **+25.3%** — but was scored against a separate tagger instance, so it is reported in the ablation rather than mixed into this table.

**Reconstruction quality improves monotonically with input resolution**, and the physics metric follows it: at 64× (a 2× upscale) SR recovers 78.5% of the taggability lost to downsampling; at 16× (an 8× upscale) it recovers none of it.

**The 16× case is the important negative result.** Its 72.9% efficiency looks respectable in isolation, but the *recovery fraction is strongly negative* — SR output is **less** HR-taggable than a plain bicubic upsample, despite clean images, a sharp core, and energy conserved to ~1%. The generator produces detail that carries no usable jet-class information.

This is a failure mode **PSNR and SSIM cannot see**, and it is the reason the evaluation pipeline in this repo exists.

**Nuance:** taggers trained *independently on each source* reach AUC ≈ 0.69–0.73 on 16×/32× SR images — at 16×, SR is *more* taggable on its own (0.728) than HR is (0.670). The class information is present; it is simply presented in a form incompatible with a tagger trained on real HR. This is **distribution mismatch**, not information loss, which points at a concrete follow-up (train or domain-adapt the tagger on SR).

### Physics conservation — a clean win at every scale

Unlike taggability, the physics-conservation metrics are excellent everywhere:

| Scale | Energy corr. r (SR) | (LR) | pt–energy r (SR) | (HR ref) | (LR) |
|------:|--------------------:|-----:|-----------------:|---------:|-----:|
| 16× | **0.998** | 0.995 | 0.711 | 0.710 | 0.717 |
| 32× | **0.998** | 0.984 | 0.710 | 0.710 | 0.697 |
| 64× | 0.996 | 0.968 | 0.710 | 0.710 | 0.686 |

**SR beats bicubic on energy correlation at every scale and reproduces HR's pt–energy correlation to three decimals.** Note that 16× posts the *best* energy correlation of any scale while simultaneously being the worst on taggability — the contradiction that motivates the whole evaluation approach.

Full evaluation, 32 figures across all three scales: [`reports/multiscale_2026-07/REPORT.md`](reports/multiscale_2026-07/REPORT.md).

Earlier per-scale analysis with the full figure suite: [`reports/classification_eval/REPORT.md`](reports/classification_eval/REPORT.md).

## Training stability — the failure that shaped the design

The first multi-scale runs **collapsed**: SR output was a dim, smeared blob with the energy core flattened, worst at coarse scales. Two causes compounded — the discriminator won outright (`d_loss` → ~0.004, dead adversarial gradient), and an L1-heavy objective (λ=50) rewarded "predict small everywhere" on images where most pixels are zero.

The fix is five coordinated changes: adversarial warmup and ramp, discriminator throttling with a per-scale loss floor, instance noise, energy-weighted L1 at λ=10, and nearest-neighbour upsampling to preserve peaks.

| | Before | After |
|---|---|---|
| Discriminator | dead (`d_loss` ≈ 0.004) | active & stable (~0.04–0.09) |
| `val_psnr` | decreasing | increasing / stable |
| `peak_ratio` | dim blob | 0.89 at 64× |
| Energy response | — | ≈ 1.0 (conserved to <1%) |

**The `d_loss_floor` throttle is scale-dependent and can silently un-fix itself.** Set too high, the discriminator is skipped so often it freezes, and the generator quietly reverts to L1-only behaviour — while training *looks* healthy. At `d_loss_floor = 0.10` on 32×, the discriminator was skipped on 98.5% of steps on average and on 100% of steps for a third of all epochs. Lowering it to 0.05 (and training to 60 epochs) improved peak_ratio 0.786 → 0.873 and tagging efficiency 75.7% → 80.2% against the same tagger. 16× needs 0.02, because its discriminator settles lower.

Diagnostic: watch `d_skip_frac` in `metrics.jsonl` — sustained values near 1.0 mean the floor is too high for that scale.

Full write-up: [`TRAINING_STABILITY.md`](TRAINING_STABILITY.md) · measured ablation: [`reports/multiscale_2026-07/REPORT.md`](reports/multiscale_2026-07/REPORT.md).

## Method

The high-resolution image is the fixed ground truth; the low-resolution input is produced by **area-downsampling** HR to the target scale. A separate model is trained per scale.

**Progressive-residual generator.** Upsampling is *learned*, not interpolated: the LR input passes through a stack of 2× sub-pixel convolution stages (`Conv → PixelShuffle(2) → ReLU → ResidualBlock`), one stage per power of two between input and target size, followed by 8 residual blocks at full resolution. A bicubic-upsampled copy of the LR input is added to the output as a global residual skip (`lr_skip`, on by default, disable with `--no-lr-skip`), so the head learns a correction on top of the low-frequency prior.

The architecture is resolution-agnostic: `forward(lr, target_size)` infers the stage count per call, so **one checkpoint serves every scale** — 16× runs 3 stages, 32× runs 2, 64× runs 1.

An earlier design bicubic-upsampled straight to HR and learned a single residual on top. On sparse deposits that pre-blurred the peaks before any learned layer ran, and the residual head could not recover them: 16×/32× posted near-zero or negative `val_psnr_norm` against 64×'s ~9. Staged learned upsampling replaced it — see the docstring in [`multiscale_sr/models/generator.py`](multiscale_sr/models/generator.py).

**Conditional spectral-norm PatchGAN discriminator.** It consumes the `(LR, HR)` pair, so it judges whether an image is a plausible super-resolution *of its specific input*, not merely a plausible calorimeter image.

**Objective** — LSGAN adversarial term, a heavy L1 reconstruction term, and a direct energy-conservation physics term:

```
L_G   = L_adv + 10·L_l1 + 10·L_phys
L_adv = 0.5·mean[(D(fake) − 1)²]                   (LSGAN)
L_D   = 0.5·mean[(D(real) − 0.9)² + D(fake)²]      (one-sided label smoothing)
L_l1  = mean|G(lr) − hr|                            (on normalized tensors)
L_phys= mean|sum(E_pred)/sum(E_true) − 1|          (on denormalized energy)
```

The L1 term is **energy-weighted** (`l1_weighting: energy`, `alpha: 5.0`) so high-deposit pixels dominate the reconstruction loss. Uniform L1 at λ=50 was the original cause of mode collapse — see [`TRAINING_STABILITY.md`](TRAINING_STABILITY.md).

Normalization is `log1p` + channel-wise z-score, with statistics computed once on **HR** and cached to `normalization.json` per run — HR is the common reference scale for every model regardless of input resolution.

Design rationale, line-referenced against the source, is in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Which script does what

Five entrypoints. Run them in this order.

| # | Script | Purpose | Needs |
|---|---|---|---|
| 1 | `train.py` | Train one GAN at one scale. Writes checkpoints, per-epoch metrics, sample grids. | `--data-dir` |
| 2 | `evaluate.py` | Pixel + physics metrics for one checkpoint (L1, peak_ratio, energy response). Fast. | a checkpoint |
| 3 | `classification_eval.py` | **The headline metric.** Trains a jet tagger, measures tagging efficiency, writes the full 9-figure diagnostic suite. | a checkpoint, parquet |
| 4 | `tag_efficiency.py` | Lightweight sibling of #3 — ROC + AUC bar only, when you just want the number. | a checkpoint, parquet |
| 5 | `run_evaluations.py` | Batch-runs #3 over *every* checkpoint under `experiments/`, then writes a cross-run comparison table. | `--data-dir` |

Supporting modules (not run directly): `engine.py` holds the losses and metrics, `experiment.py` manages run directories, `tagger.py` is the jet tagger used by #3, `data/` handles both dataset formats, `utils/env.py` resolves the device automatically.

## Setup

Requires Python 3.10+ and PyTorch. Runs on CUDA, Apple MPS, or CPU — the device, worker count, and AMP settings are resolved automatically in `utils/env.py`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Weights & Biases logging is **optional**. To enable it, create a `.env` file in this directory:

```bash
# .env  (gitignored — never commit this file)
WANDB_API_KEY=your_key_here
# WANDB_ENTITY=your-entity      # optional, overrides --wandb-entity
```

Without a key, pass `--no-wandb` and everything still runs and logs locally.

### Data

Both formats are auto-detected from `--data-dir`:

- **Parquet** (CMS jets) — `*.parquet` with columns `X_jets_LR`, `X_jets`, `pt`, `m0`, `y`. At scale 64 you may pass `--use-native-lr` to feed the detector's native LR instead of a downsampled HR.
- **HDF5** (CaloChallenge Dataset 2) — `dataset_2_*.hdf5`. HR is resized to a square `--hr-size` (default 125, zero-padded to 128) so both datasets share one HR resolution.

Datasets are **not** committed to this repo. Point `--data-dir` at a local copy.

Note: classification-based evaluation is **parquet only** — the CaloChallenge data carries no class label.

## Quickstart

With the environment installed and `--data-dir` pointing at a local dataset, verify the pipeline end to end in about a minute, then run a real experiment:

```bash
# 1. Smoke test — tiny, no W&B, confirms data loading + training + eval all work
python train.py --data-dir ../datasets --scale 32 --epochs 1 \
    --max-train-batches 3 --max-val-batches 2 --max-stats-batches 2 --no-wandb

# 2. Train the best-performing configuration (64x)
python train.py --config configs/scale_64.yaml --data-dir ../datasets \
    --d-loss-floor 0.05 --epochs 40 --run-name my_64x --cache

# 3. Score it on the headline physics metric
python classification_eval.py \
    --checkpoint experiments/<run>/checkpoints/best.pt --data-dir ../datasets
```

Step 3 prints the tagging efficiency and writes all nine figures to `experiments/<run>/figures/classification/`.

**Per-scale `d_loss_floor` is not optional** — use **0.02** at 16×, **0.05** at 32× and 64×. The wrong value silently freezes the discriminator and degrades results while the loss curves still look healthy; see [`TRAINING_STABILITY.md`](TRAINING_STABILITY.md).

## Repository layout

```
multiscale_sr/
├── multiscale_sr/                # the package
│   ├── models/
│   │   ├── generator.py          # progressive-residual generator (sub-pixel upsampling)
│   │   └── discriminator.py      # conditional spectral-norm PatchGAN
│   ├── data/
│   │   ├── parquet_dataset.py    # CMS jet parquet loader
│   │   ├── hdf5_dataset.py       # CaloChallenge Dataset 2 loader
│   │   ├── multiscale.py         # multi-scale collate / downsampling
│   │   ├── normalization.py      # log1p + z-score, HR-referenced
│   │   ├── cache.py              # decode cache for repeated epochs
│   │   └── factory.py            # format auto-detection
│   ├── utils/
│   │   ├── env.py                # device/worker resolution (CUDA/MPS/CPU)
│   │   └── seed.py               # deterministic seeding
│   ├── engine.py                 # losses, metrics, eval loop, figures
│   ├── experiment.py             # run-dir layout + YAML config
│   ├── tagger.py                 # jet tagger used by the evaluation
│   ├── classification_metrics.py # AUC, ECE, working points, agreement
│   └── wandb_logger.py           # W&B wrapper (no-op when disabled)
├── train.py                      # training entrypoint
├── evaluate.py                   # pixel/physics metrics for one checkpoint
├── classification_eval.py        # full tagging-efficiency diagnostic suite
├── tag_efficiency.py             # lightweight sibling (ROC + AUC bar only)
├── run_evaluations.py            # batch-evaluate every checkpoint, cross-run tables
├── configs/                      # scale_16 / scale_32 / scale_64 YAML
├── ARCHITECTURE.md               # line-referenced design rationale
├── TRAINING_STABILITY.md         # the mode-collapse failure and its fix
├── reports/
│   ├── multiscale_2026-07/       # full evaluation: 32 figures, all 3 scales
│   └── classification_eval/      # earlier per-scale figure suite
├── evaluations/<date>/           # dated batch-evaluation output
└── experiments/<run>/            # per-run training output (gitignored)
```

Each training run writes to:

```
experiments/{YYYY-MM-DD}_{dataset}_{scale}x_{run_name}/
├── checkpoints/       best.pt, latest.pt
├── figures/           sample_epoch_{N}.png, metrics.png
├── config.yaml        frozen run config
├── eval.json          final metrics
├── normalization.json cached HR statistics
└── metrics.jsonl      per-epoch log
```


## Usage

### Train

```bash
# Scale 32 via config file (recommended):
python train.py --config configs/scale_32.yaml --data-dir ../datasets --run-name baseline

# Scale 16, pure CLI:
python train.py --data-dir ../datasets --scale 16 --epochs 50

# Quick smoke test (tiny, no W&B) — verifies the pipeline end to end:
python train.py --data-dir ../datasets --scale 32 --epochs 1 \
    --max-train-batches 3 --max-val-batches 2 --max-stats-batches 2 --no-wandb
```

CLI flags override values from `--config`. Add `--cache` to build a decode cache — a large speedup when running many epochs over parquet.

### Evaluate pixel + physics metrics

```bash
python evaluate.py --checkpoint experiments/<run>/checkpoints/best.pt \
    --data-dir ../datasets --save-grid /tmp/grid.png
```

### Evaluate tagging efficiency (the headline metric)

```bash
python classification_eval.py --checkpoint experiments/<run>/checkpoints/best.pt \
    --data-dir ../datasets
```

Writes to `experiments/<run>/figures/classification/`:

| Output | What it shows |
|---|---|
| `roc_overlay.png` | ROC for HR/LR/SR — fixed HR tagger *and* per-source taggers |
| `auc_summary_bar.png` | AUC bars with the efficiency / recovery headline |
| `score_distributions.png` | Tagger score histograms per class, per source |
| `confusion_matrices.png` | Confusion at the Youden-J optimal threshold |
| `efficiency_vs_threshold.png` | Background rejection `1/ε_B` vs signal efficiency (HEP view) |
| `calibration.png` | Reliability curves + ECE per source |
| `score_agreement.png` | Per-sample HR-vs-SR score scatter (the strictest test) |
| `energy_correlation.png` | Per-image total energy, SR vs HR — is energy conserved? |
| `pt_correlation.png` | Jet `pt` vs image energy — is the physical band preserved? (parquet only) |
| `classification_eval.json` | Every metric, machine-readable |
| `EXPLANATION.md` | What each figure and metric means |

`tag_efficiency.py` is the lighter sibling (ROC + AUC bar only).

### Batch-evaluate every checkpoint

```bash
python run_evaluations.py --data-dir ../datasets
```

Evaluates all runs under `experiments/` with one fixed HR tagger and a common seed, then writes a dated cross-run comparison to `evaluations/<date>/` — master table, per-scale best checkpoint, epoch-effect trend, and an AUC_HR consistency sanity check.

## Metrics

| Metric | Meaning |
|---|---|
| `val_l1` | L1 on normalized tensors — the primary model-selection metric |
| `val_peak_ratio` | SR peak brightness / HR peak brightness; the mode-collapse detector |
| `val_energy_response` | `sum(E_pred)/sum(E_true)` on denormalized energy; 1.0 is unbiased |
| `d_skip_frac` | Fraction of steps the discriminator was throttled; ~1.0 means it is frozen |
| **tagging efficiency** | `AUC_SR / AUC_HR` — the physics-facing headline |
| **recovery fraction** | How much of the LR→HR gap SR closes; negative means worse than bicubic |

> `val_psnr_norm` is currently **not trustworthy**: the 64× run reports −3.91 while posting the best L1, peak_ratio, and tagging efficiency of any run. That combination points to a metric-computation artifact in `engine.py`, so PSNR is excluded from the tables above pending an audit.

## Reproducing the reported results

The numbers in the headline table come from a fixed HR-trained tagger at seed 42, 4032 samples (2822 train / 1210 test), tagger width 32, 15 epochs, on the parquet dataset. AUC_HR = 0.669 across every run, which is the sanity check that the comparison is fair.

```bash
# Train the best configuration (64x, stabilized):
python train.py --config configs/scale_64.yaml --data-dir ../datasets \
    --d-loss-floor 0.05 --epochs 40 --run-name colab_64x_full_40_dvalue_changed

# Score it on the physics metric:
python classification_eval.py \
    --checkpoint experiments/<run>/checkpoints/best.pt \
    --data-dir ../datasets
```

Per-scale `d_loss_floor`: **0.02** at 16×, **0.05** at 32× and 64×. See [`TRAINING_STABILITY.md`](TRAINING_STABILITY.md) for why it differs.

Training checkpoints and per-run artifacts are gitignored (they are large and reproducible); the curated figures and analysis are committed under `reports/`.

## Status and next steps

The 64× result is complete and defensible: 93.9% tagging efficiency, 78.5% of the LR→HR gap recovered, per-sample agreement r = 0.64, and the flattest residual of any scale. The mode-collapse failure is diagnosed, fixed, and the fix is measured.

Open threads, in priority order:

1. **16× still fails on the physics axis** despite clean images and conserved energy. The per-source diagnostic (AUC ≈ 0.72–0.76 on SR alone) says the class information is there but HR-incompatible. Training or domain-adapting the downstream tagger on SR images tests that hypothesis directly and is the highest-value next experiment.
2. **32× has not converged.** Going 40 → 60 epochs moved peak_ratio 0.822 → 0.873 and val_L1 0.0757 → 0.0741, still improving (both tagger-independent). A longer budget is the cheapest remaining gain.
3. **Audit `psnr_norm` in `engine.py`** — see the note under Metrics.
4. Progressive and stabilized training variants at 128-padded resolution are in progress.
