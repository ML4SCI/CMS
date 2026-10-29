# VAR-SR: Next-Scale Prediction for Physics Super-Resolution of Calorimeter Data

**GSoC 2026 · ML4Sci** — E2E Super-Resolution

**Contributor:** Rajveer Rathod ([@rajveer43](https://github.com/rajveer43))
**Project:** End-to-end super-resolution of CMS calorimeter data, judged by physics observables and downstream taggers
**Method:** [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905) (Tian et al., 2024), adapted from natural images to calorimeter jets and showers
**Compute:** NERSC Perlmutter (A100-40GB), SLURM array jobs
**Sibling work:** the multiscale GAN super-resolution in [`../multiscale_sr/`](../multiscale_sr/) is the other approach to the same task

> **Status:** Two complete multi-seed experiments: quark/gluon jets (`qg`, 3 LR levels, 3 tagger seeds) and CaloChallenge showers (`calo`, 3 LR levels, 3 regressor seeds). The pipeline runs end to end: download, tokenizer, transformer, generation, evaluation, taggers, summary. **The headline result is mixed and reported as such below:** the VAR output recovers many distributions at the finest level and in `calo`, but it does not yet beat the plain low-resolution input as a tagger input, and a classifier tells it apart from truth almost perfectly.

---

## Overview

Detectors with larger cells record less information. The question here is whether a generative model can map coarse calorimeter data back to the native high-resolution grid well enough that **physics observables and analyses** (not SSIM or PSNR) still work on the output.

Two datasets share one code path (2-D for jets, 3-D for showers):

| Dataset | Content | HR grid (native) | LR levels (sum-pooled) |
|---|---|---|---|
| `qg` | CMS quark/gluon jet images, channels tracks / ECAL / HCAL, 40k jets | 3 x 125 x 125 (padded to 128) | `pool2x2` (64²), `pool4x4` (32²), `pool8x8` (16²) |
| `calo` | CaloChallenge 2022 Dataset 2, e⁻ showers, 1 GeV–1 TeV | 45 layers x 16 angle x 9 radius (padded to 48 layers) | `pool1x2x1`, `pool1x4x3`, `pool3x8x3` |

Level names give the pooling factor per axis. Each LR cell is the **sum** of the HR cells it covers, as a detector with larger cells would record, so total energy is conserved exactly.

**Why only up to native resolution:** there is no ground truth above 125 x 125 (qg) or 45 x 16 x 9 (calo). Anything beyond could not be checked, so the SR target is always the native HR grid: 2x, 4x and 8x for jets.

## What is adapted from the VAR paper, and what is new

VAR replaces raster-order token prediction with **coarse-to-fine, next-scale** prediction: an image becomes K token maps of growing size, and a transformer predicts each whole map conditioned on the coarser ones. That is a natural fit for super-resolution, because the LR measurement *is* a coarse scale.

Carried over from the paper:
- the multi-scale residual VQ-VAE (Algorithms 1 and 2) with a codebook shared across scales
- next-scale prediction with a block-causal mask, AdaLN, and L2-normalised q/k

Original to this project:
1. **Physics losses replace LPIPS and GAN losses.** The decoder has two heads per channel: *hit or not* (cross-entropy against E > readout threshold) and *log-energy if hit* (MSE on hit cells only). Cells without a predicted hit are exactly zero, as after detector zero suppression, so there is no diffuse background. A **per-channel energy loss** stops energy moving between sub-detectors (for example tracks into ECAL).
2. **LR image into the decoder.** Layout the measurement already fixes does not have to pass through the tokens. The tokenizer is trained with every LR level in turn.
3. **LR measurement replaces the class-label start token.** An LR encoder supplies the start token and the AdaLN condition, and adds a spatial feature map to every scale's tokens. One transformer per LR level.
4. **Energy-preserving down-sampling** and **periodic padding on the calorimeter angular axis**.
5. **Evaluation by physics.** Observable distributions, per-event bias and resolution, LR closure, a classifier two-sample test, and downstream taggers trained on HR, LR and SR.

## What is compared

The result (`sr`) is the model output exactly as generated (greedy decoding), set against the HR truth and the LR input. The training target is always HR. Diagnostics are kept out of the main tables:

| Name | What it is | Why it is there |
|---|---|---|
| `vqrec` | HR → tokens → HR, tokenizer only | Best any token-predicting model could do with this tokenizer |
| `sr-sample` | Tokens sampled from the predicted distribution (T = 0.8, top-k 50) | Greedy vs sampled |
| `uniform` | Each coarse cell's energy spread evenly over its fine cells | A no-learning reference |
| `srproj` | SR rescaled so every coarse cell matches its measured energy | Post-processing, never used in training |

## Headline results (multi-seed)

All numbers are test-set values from the saved summaries in [`results/`](results/). Tagger seeds are 42, 43, 44. "±" is the spread over seeds.

### Quark/gluon jets: ResNet-18 tagger ROC AUC (trained and tested on each input)

| Input | 2x (pool2x2) | 4x (pool4x4) | 8x (pool8x8) |
|---|---|---|---|
| **HR** (truth) | 0.8086 ± 0.0020 | 0.8086 ± 0.0020 | 0.8086 ± 0.0020 |
| **LR** (coarse input) | 0.8075 ± 0.0019 | 0.8036 ± 0.0024 | 0.8033 ± 0.0024 |
| **SR** (VAR output) | 0.7871 ± 0.0022 | 0.7887 ± 0.0003 | 0.7771 ± 0.0018 |

The tagger loses very little from coarse-graining alone (LR is within 0.005 of HR even at 8x), and **the SR output is below LR at every level**. Background rejection at 50% signal efficiency shows the same ordering (HR 9.47, LR 8.59–9.71, SR 7.17–7.84).

### CaloChallenge showers: CNN energy-resolution regressor (lower is better)

| Input | pool1x2x1 | pool1x4x3 | pool3x8x3 |
|---|---|---|---|
| **HR** (truth) | 0.0712 ± 0.0005 | 0.0712 ± 0.0005 | 0.0712 ± 0.0005 |
| **LR** | 0.0735 ± 0.0006 | 0.0754 ± 0.0001 | 0.0801 ± 0.0022 |
| **SR** | 0.0726 ± 0.0003 | 0.0775 ± 0.0005 | 0.0882 ± 0.0017 |

SR matches or marginally beats LR only at the finest level; at the two coarser levels it is worse.

### Do the generated distributions look like truth? (W1 / σ_HR, 0 = identical)

| Observable | qg 2x | qg 4x | qg 8x |
|---|---|---|---|
| hit multiplicity | 0.37 | 0.45 | 0.86 |
| girth | 0.40 | 0.11 | 0.70 |
| τ₂₁ | 0.06 | 0.08 | 0.30 |
| jet mass | 1.07 | 1.36 | 1.92 |
| jet pT | 1.12 | 1.28 | 1.47 |
| total energy | 1.14 | 1.32 | 1.54 |

| Observable | calo pool1x2x1 | calo pool1x4x3 | calo pool3x8x3 |
|---|---|---|---|
| E_dep / E_inc | 0.09 | 0.24 | 0.42 |
| depth centroid | 0.03 | 0.14 | 0.24 |
| depth width | 0.04 | 0.55 | 0.84 |
| radial centroid | 0.14 | 0.73 | 1.11 |
| radial width | 0.06 | 0.92 | 1.54 |

In `calo` at the finest level the distributions are close (all W1/σ ≤ 0.14, event-by-event Pearson r ≥ 0.96). Quality falls steadily as the grid gets coarser.

## What the diagnostics say about the failures

Reported as measured, not smoothed over:

- **Energy is under-reconstructed in the tracks channel.** In `qg` at 2x, mean jet pT is 145 for SR against 234 for HR, and mean `sum_tracks` is 36 against 123. Jet mass and total energy move with it (W1/σ above 1).
- **The SR output does not respect what was measured.** Median LR closure (Σ|pool(SR) − LR| / Σ LR) is 0.48 / 0.51 / 0.58 for `qg` and 0.14 / 0.15 / 0.11 for `calo`. The `srproj` diagnostic brings closure to about 1e-8, as designed.
- **The bottleneck is the tokenizer, not the transformer.** The tokenizer-only diagnostic (`vqrec`) already has closure 0.45–0.47 in `qg`, nearly equal to SR. The transformer adds little loss on top of what the tokens cannot represent.
- **Energy projection is not a free fix.** `SR + energy constraint` recovers the HR-trained tagger at 2x (0.8057 vs 0.7834 without), but at 8x it falls to 0.6272 (SR alone 0.7627), because redistributing energy on a poorly placed layout hurts the shape.
- **A classifier can tell SR from truth almost perfectly.** C2ST AUC is 0.9987 / 0.9992 / 1.0000 (`qg`) and about 0.9999 at all `calo` levels (0.5 = indistinguishable). Agreement in means and 1-D distributions does not imply the events look real.
- **Transformer fit worsens with coarseness.** Best validation cross-entropy: `qg` 0.94 (2x) → 1.56 (8x); `calo` 3.46 (pool1x2x1) → 4.57 (pool3x8x3).

## Open threads

Directions suggested by the diagnostics above, in priority order:

1. **Fix the tokenizer bottleneck first** (more or larger scales, larger codebook, a tracks-specific term). `vqrec` bounds what the transformer can reach.
2. **Build LR consistency into generation** rather than as post-processing, so closure is satisfied by construction without the shape damage that `srproj` causes at 8x.
3. **Tracks channel:** its hit count and energy are the largest deficits. A dedicated head or a sparse-aware loss is the obvious next step.
4. **Close the AUC gap to LR** by training the tagger on SR (or jointly) once 1–3 improve; today SR < LR means the SR output is *removing* information the tagger could use.
5. **`calo` at coarse levels:** radial and depth widths are the weak observables (W1/σ near or above 1).

## Reported honestly, not oversold

- SR is **worse than the unprocessed LR input** as a tagger input in `qg` at every level, and in `calo` at the two coarser levels. The model does not yet help an analysis; it is a measured baseline.
- Seed spread is small relative to the level-to-level and SR-vs-LR gaps in `qg`. In `calo` the SR-vs-LR difference at pool1x2x1 (0.0726 vs 0.0735) is within the seed spread and should not be read as a win.
- Evaluation uses 4,000 events (`qg`) and 10,000 events (`calo`) for observables and C2ST (the `eval.max_events` setting in `superres/config.py`).
- One tokenizer and one transformer per level were trained once (the three seeds are tagger seeds, not generator seeds), so generator-to-generator variance is not measured.
- Hyper-parameters were tuned with Optuna on small subsets and short trials (4 epochs, 10k samples); the final runs use the tuned values but were not re-tuned per level.

## Repository structure

```
VAR_Next_Scale_Prediction/
├── README.md
├── README_PERLMUTTER.md             # run the full pipeline on NERSC Perlmutter
├── requirements.txt
├── docs/
│   └── PIPELINE.md                  # stages, Colab usage, result layout, disk and time budget
├── superres/
│   ├── config.py                    # all settings, per-dataset defaults, smoke-test overrides
│   ├── data.py  download.py         # CERNBox (qg) and Zenodo (calo) loaders, sum-pooling, splits
│   ├── physics_ops.py               # energy-preserving pooling, hit/energy transforms
│   ├── models/
│   │   ├── vqvae.py                 # multi-scale residual VQ-VAE with hit/energy heads
│   │   ├── var.py                   # conditional next-scale transformer (AdaLN, block-causal)
│   │   ├── tagger.py                # ResNet-18 / CNN taggers and the calo regressor
│   │   └── nn_utils.py
│   ├── train.py  tuning.py          # training loops, Optuna studies
│   ├── pipeline.py                  # stage runner: python -m superres.pipeline
│   ├── evaluate.py  observables.py  # physics observables, W1/KS, bias/resolution, C2ST, closure
│   ├── metrics.py  io_utils.py
├── tests/
│   ├── smoke_test.py                # synthetic end-to-end run, no download
│   └── test_observables.py
├── slurm/                           # Perlmutter sbatch scripts, one per stage; settings.sh
└── results/
    ├── qg/
    │   ├── combined/                # per-level figures side by side, evaluation_all.pdf
    │   └── qg_v2_2026-10-13_multiseed/
    └── calo/
        ├── combined/
        └── calo_v2_2026-10-23_multiseed/
            ├── summary/             # report.md, tables (CSV), figures
            ├── models/              # config, history, metrics, training curves (no weights)
            ├── evaluation/          # per-level metrics and observables
            └── run_info.json  data_meta.json  README_layout.md
```

## Setup

```bash
git clone https://github.com/ML4SCI/CMS.git
cd CMS/E2E/E2E_Super_Resolution_Rajveer_Rathod/VAR_Next_Scale_Prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On NERSC Perlmutter, edit `slurm/settings.sh` and follow [`README_PERLMUTTER.md`](README_PERLMUTTER.md).

## Dataset

| Dataset | Source | Notes |
|---|---|---|
| `qg` | CMS quark/gluon jet images, CERNBox share (WebDAV) | tracks / ECAL / HCAL, 125 x 125, 40k jets used |
| `calo` | CaloChallenge 2022 Dataset 2, Zenodo [10.5281/zenodo.6366271](https://doi.org/10.5281/zenodo.6366271) | e⁻ showers, 45 x 16 x 9, md5-checked download |

Preprocessing: pad to 128 (qg) or 48 layers (calo), clip negatives, sum-pool to every LR level, split into train / val / test, per-channel energy scale. The tokenizer works on `log1p(E / s_c)` images.

**Settings (all experiments):** codebook 1024 shared across scales, VQ-VAE 30 epochs, VAR transformer depth 6 / width 384 / 6 heads for 40 epochs, taggers 20 epochs with seeds 42/43/44, mixed precision on A100.

## Usage

Quick synthetic check (no data, no GPU needed):
```bash
python tests/smoke_test.py --root /tmp/superres_smoke
```

Run the whole pipeline for one dataset:
```bash
python -m superres.pipeline --dataset qg --stages all \
  --set paths.drive_root=/path/experiments run_date=2026-10-13 run_tag=multiseed
```

Run selected stages and levels:
```bash
python -m superres.pipeline --dataset qg --stages train_var,generate --levels pool8x8 --set var.epochs=20
```

Submit on Perlmutter (see [`README_PERLMUTTER.md`](README_PERLMUTTER.md) and `slurm/`):
```bash
bash slurm/submit_all.sh
```

Stages: `download`, `prepare`, `tune_vqvae`, `train_vqvae`, `tune_var`, `train_var`, `generate`, `eval_sr`, `tune_tagger`, `train_taggers`, `summarize`. Training stages resume from `last.pt`. The full stage table, result layout and naming rules are in [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Reproducing the tables above

The numbers come from the saved summaries:

- `results/qg/qg_v2_2026-10-13_multiseed/summary/qg_v2_2026-10-13_multiseed_report.md`
- `results/calo/calo_v2_2026-10-23_multiseed/summary/calo_v2_2026-10-23_multiseed_report.md`
- C2ST and closure: `…_c2st-closure.csv` in the same folders
- Side-by-side figures: `results/<dataset>/combined/`

## Acknowledgments

This work is part of Google Summer of Code 2026 under **ML4Sci** (Machine Learning for Science), with experiments run on NERSC's Perlmutter supercomputer. The method follows Tian et al., *Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction*.

## Contact

**Rajveer Rathod** — [GitHub](https://github.com/rajveer43) · [LinkedIn](https://www.linkedin.com/in/rajveer-rathod/)
