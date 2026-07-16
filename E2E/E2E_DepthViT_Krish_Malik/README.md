# DepthViT: Channel-Asymmetric Vision Transformers for LHC Jet Classification

**GSoC 2026 · ML4Sci** — Midterm Submission

**Contributor:** Krish Malik ([@krishoncloud](https://github.com/krishoncloud))
**Mentors:** Eric Reinhardt (University of Alabama), Diptarko Choudhury
**Project:** [Linear Attention Vision Transformers for End-to-End Mass Regression and Classification](https://ml4sci.org/gsoc/2026/proposal_E2E5.html)
**Compute:** NERSC Perlmutter (4× A100), SLURM account `m4392`
**Full write-up:** ["Small Models, Big Egos: Scaling DepthViT from 165K to 22M Parameters"](https://medium.com/@krishmalikus/small-models-big-egos-scaling-depthvit-from-165k-to-22m-parameters-43dbfc17f0e1)

> **Status:** This is a midterm snapshot, not a final state. The repo will continue to be updated past GSoC as the project targets a workshop paper (ML4PS @ NeurIPS 2026) and, as a stretch goal, ICLR 2027 — see [Roadmap](#roadmap-remainder-of-gsoc) below.

---

## Overview

Every 25 nanoseconds, the LHC produces a new collision, and a real-time FPGA trigger system has to decide, on a microsecond budget, whether the event is worth keeping. That makes model size and compute hard physical constraints, not just efficiency metrics. This project asks: **how small can a vision transformer get on a real LHC jet-classification benchmark while staying competitive with — and eventually beating — a much larger baseline?**

Standard ViTs sum across image channels at the embedding layer, an assumption that holds for RGB natural images but breaks down for calorimeter data: the ECAL and HCAL channels here are two physically separate subdetectors, at different radii, built to capture different particle types. Summing them away throws out exactly the structure that makes jet classification possible.

**DepthViT** addresses this by embedding each channel independently and computing attention *across channels* rather than across spatial patches. The channel-wise attention mechanism is adapted from an unsupervised anomaly-detection architecture (Julson et al., Cerium Laboratories / University of Alabama / CMS HCAL Collaboration) into a fully supervised, five-class jet classifier — a substantial change in kind, not just domain. Only the depth-wise embedding and channel-attention operators are carried over from the original; the classification head, and the mechanism to recover cross-patch spatial communication (Hierarchical Attention Pooling, or HAP), are new to this project.

## Why this matters

Channel-wise attention buys physically meaningful structure but gives up something: it never lets one spatial patch talk to another. HAP blocks, interleaved between channel-attention blocks, close that gap cheaply — mixing information within local windows and across window summaries, without the quadratic cost of full spatial self-attention.

## Headline result

At the 22M-parameter tier, matched against ViT-Small on both parameters and FLOPs:

| Metric | DepthViT | ViT-Small | Δ |
|---|---|---|---|
| Params | 22,046,777 | 22,012,613 | +0.16% |
| FLOPs (forward) | 4.2945G | 4.3113G | −0.39% |
| Top-1 accuracy | 75.12% | 74.14% | **+0.98 pts** |
| Macro AUC | 0.9366 | 0.9326 | **+0.0040** |
| Train/val gap | 1.1 pts | 3.7 pts | less overfit |

DepthViT wins **all five jet classes individually** (gluon, light quark, W, Z, top) — not just on macro average — at matched parameters and matched FLOPs, which rules out the usual objection that efficiency gains are just squeezing more out of fewer parameters. This is a stronger claim than "efficient": it's "the architecture is better."

### Per-class AUC at 22M (150p split)

| Jet class | DepthViT | ViT-Small | Δ |
|---|---|---|---|
| Gluon (g) | 0.9323 | 0.9305 | +0.0018 |
| Light quark (q) | 0.9180 | 0.9152 | +0.0028 |
| W boson | 0.9449 | 0.9394 | +0.0055 |
| Z boson | 0.9324 | 0.9264 | +0.0060 |
| Top | 0.9553 | 0.9515 | +0.0038 |

DepthViT wins every class, ruling out a macro-average result driven by one easy class. Combined with the smaller train/val gap (1.1 vs 3.7 pts), the model at this scale generalizes slightly better rather than just fitting the training set harder.

## Full results across the scaling frontier

All numbers below are on the densest (150p) HLS4ML jet split, 90-epoch training, single seed, seed 42. FLOPs are measured with `calflops` (exact, not estimated).

### DepthViT across four parameter tiers

| Tier | Params | FLOPs | Top-1 | Macro AUC |
|---|---|---|---|---|
| 165K | 164,417 | 0.0316G | 70.14% | 0.9175 |
| 1M | 1,008,226 | 0.197G | 72.31% | 0.9270 |
| 5.4M | 5,389,691 | 1.039G | **74.99%** | 0.9361 |
| 22M | 22,046,777 | 4.2945G | **75.12%** | 0.9366 |

### Two baseline framings

This project compares DepthViT against ViT baselines two ways, and reports both because they tell different stories.

**Framing A — vs. standard full-size ViT-Tiny (5.4M).** The efficiency claim: at 165K parameters, DepthViT recovers 95.7% of a full ViT-Tiny's accuracy (70.14% vs 73.30%) at **32.8× fewer parameters and 34.3× fewer FLOPs** — a compelling FPGA/trigger operating point. At 5.4M it beats full ViT-Tiny outright (+1.69 pts Top-1, +0.0063 AUC, ~4% fewer FLOPs). This framing is honest and strong: it's the baseline a practitioner would actually reach for.

**Framing B — matched-budget control (ViT-Tiny shrunk to each tier's size).** The more rigorous test: shrink ViT-Tiny to the *same* parameter budget as DepthViT at each tier, rather than leaving it full-size.

| Tier | DepthViT | ViT-Tiny (shrunk to match) | Winner | Margin |
|---|---|---|---|---|
| 165K | 164,417 · 70.14% · 0.9175 | 162,373 · 71.70% · 0.9238 | **ViT-Tiny** | −1.56 pts Top-1 / −0.0063 AUC |
| 1M | 1,008,226 · 72.31% · 0.9270 | 1,006,473 · 72.30% · 0.9260 | **Tie** | +0.01 pts / +0.0010 AUC (single-seed noise) |
| 5.4M | 5,389,691 · 74.99% · 0.9361 | 5,397,893 · 73.30% · 0.9298 | **DepthViT** | +1.69 pts / +0.0063 AUC |
| 22M | 22,046,777 · 75.12% · 0.9366 | 22,012,613 · 74.14% · 0.9326 (ViT-Small) | **DepthViT** | +0.98 pts / +0.0040 AUC |

Under this control, the picture is **scale-dependent**: DepthViT loses at its own 165K headline budget, ties at 1M, and pulls ahead from 5.4M onward. The FLOPs advantage compresses too — 0.0316G vs 0.0316G (identical) at 165K, 0.197G vs 0.200G at 1M, 1.039G vs 1.083G at 5.4M — so the large FLOPs ratios in Framing A exist specifically against *standard-config* ViT-Tiny, and shrink to near-parity once ViT-Tiny is matched. The crossover is monotonic and crosses zero: DepthViT's relative accuracy improves steadily with scale, and its structural O(LC²) FLOPs advantage only matters in the regime where the MLP isn't dominating the budget.

Both framings are reported here in full; the headline 22M result (matched params *and* matched FLOPs against ViT-Small, winning on every class) is the strongest single point and belongs to both.

A ResNet-9 CNN trained at the same parameter count as the 164K DepthViT (164,671, within 0.15%) reached only 60.40% accuracy — 9.74 points below DepthViT at that budget — confirming channel-wise attention beats convolution at matched parameters, independent of the ViT comparison above.

## A negative result worth reporting

Self-supervised pretraining (iJEPA and an energy-weighted variant) did **not** help at this scale. Cold-start training beat both pretraining variants across every particle-multiplicity split, including under a matched-compute check. This narrows rather than contradicts prior work (HEP-JEPA found pretraining gains concentrated in low-label regimes at larger scale); at this parameter budget and full label budget, capacity — not initialization — appears to be the binding constraint.

## Results across all particle-multiplicity splits (165K tier)

The 164,417-parameter model was evaluated on all four HLS4ML splits (30p / 50p / 100p / 150p), not just the densest one. Two findings come out of this that don't show up if you only look at 150p.

### Accuracy scales with training length, unevenly across splits

| Split | 30 epochs | 90 epochs | Δ |
|---|---|---|---|
| 30p | 57.1% | 58.1% | +1.0 |
| 50p | 62.9% | 63.9% | +1.0 |
| 100p | 68.1% | 69.0% | +0.9 |
| 150p | 62.2% | 70.14% | **+7.9** |

50p and 100p are near convergence by 30 epochs — extra training barely moves them. 150p is the outlier: at 30 epochs it actually *underperforms* 100p (62.2% vs 68.1%), which looks like a ceiling but isn't — it's underfitting. Given the full 90 epochs, 150p jumps 7.9 points and becomes the best-performing split, consistent with richer splits (more particles, more substructure) needing more training to fully exploit that structure. This is a defining property of the channel-asymmetric architecture at this parameter budget worth keeping in mind when comparing splits at a fixed epoch count.

### The gap to full-size ViT-Tiny narrows as splits get richer (Framing A)

Against standard full-size ViT-Tiny (5.4M), the 165K DepthViT's accuracy gap shrinks as jets get richer in substructure:

| Split | DepthViT (164K) | ViT-Tiny (5.4M) | Gap |
|---|---|---|---|
| 30p | 58.43% | 68.06% | −9.63 pts |
| 150p | 70.14% | 73.30% | −3.16 pts |

The macro AUC gap tells the same story, narrowing from 0.048 to 0.012. The 164K model isn't fundamentally capacity-starved for this task — it does comparatively better the more structure there is to work with.

## HAP interleaving ratio ablation

An architecture sanity check on ImageNet-1K (16-block budget) tested how often to interleave HAP blocks between channel-attention blocks:

| HAP ratio | Config | Top-1 (ImageNet-1K) |
|---|---|---|
| 0% | 12×D (no HAP) | 49.0% |
| 25% | (DDDH)×4 — **adopted** | 52.3% |
| 50% | (DH)×12 | 45.5% |

25% interleaving (one HAP block after every three channel-attention blocks) gave the best result; over-mixing at 50% actively hurt, consistent with over-smoothing local structure. This ratio was chosen on ImageNet as a sanity check and carried unchanged into every jet-tier config — it was not re-ablated directly on the HLS4ML jet data.

## Depth sweep at fixed 22M parameter budget

Naive single-knob scaling (raising k alone to reach 22M parameters) produces a technically valid but lopsided model, with HAP alone reaching nearly 40% of parameters. A depth sweep at a fixed HAP ratio determined how much of a second knob (depth) to spend correcting this:

| L (depth) | k | hidden_dim | Channel-attn | FFN | HAP |
|---|---|---|---|---|---|
| 12 | 233 | 466 | 20.8% | 39.0% | 39.7% |
| 16 | 207 | 414 | 21.9% | 46.2% | 31.3% |
| **18** | **196** | **392** | **22.1%** | **49.3%** | **28.1%** |
| 20 | 186 | 372 | 22.1% | 52.0% | 25.4% |
| 24 | 169 | 338 | 21.9% | 56.6% | 20.9% |

Channel-attention's share stays essentially flat (~21–22%) regardless of depth — depth only trades HAP against FFN, never against the mechanism itself. **L=18 is the shallowest depth that pulls HAP out of co-dominance with FFN** (under 30%), while channel-attention holds its share throughout. This is the tier-4 config (L=18, k=196) used in the headline 22M result above. As a bonus, the resulting hidden width (≈392) lands almost exactly on ViT-Small's own width (384).

## Self-supervised pretraining ablation (all splits, 165K tier)

Standard iJEPA (10 epochs pretraining + 60 finetuning) and an energy-weighted variant (masks only non-zero energy deposits, motivated by jet-image sparsity) were tested against cold-start across all four splits:

| Split | Cold-start (90ep) | Standard iJEPA | Energy-weighted iJEPA |
|---|---|---|---|
| 30p | 58.1% | 57.9% | 59.0% |
| 50p | 63.9% | 62.2% | 62.4% |
| 100p | 69.0% | 64.3% | 65.0% |
| 150p | 70.1% | 62.6% | 64.2% |

iJEPA is at or below cold-start on every split except a marginal +0.8 pt on the sparsest (30p); the worst case is a 4.7-point drop on 150p. Energy-weighted masking is directionally correct — it recovers some of the gap versus standard iJEPA on every split — but never closes it. Likely explanation: jet images are sparse (most pixels are zero), so random masking wastes predictor capacity reconstructing empty patches, and at ~165K–200K parameters the model may be too small to retain pretrained representations through SGD finetuning. This is consistent with HEP-JEPA's finding that pretraining gains concentrate in low-label regimes at larger scale — a genuinely different regime from what's tested here.

## Repository structure

```
E2E_DepthViT_Krish_Malik/
├── README.md
├── requirements.txt
├── configs/
│   ├── tier1_165k.json
│   ├── tier2_1M.json
│   ├── tier3_5M.json
│   ├── tier4_22M.json            # L=18, k=196 — headline config
│   └── baselines/
│       ├── vit_tiny_165k.json
│       ├── vit_tiny_1M.json
│       ├── vit_tiny_5M.json
│       ├── vit_small.json
│       └── resnet9_control.json
├── models/
│   ├── depthvit.py                # HAP block defined inside this file
│   ├── vit_tiny.py / vit_tiny_trainer.py
│   ├── vit_small.py / vit_small_trainer.py
│   └── resnet9.py / resnet9_trainer.py
├── data/
│   ├── dataset.py                 # HLS4ML jet loader + preprocessing (log1p + per-channel z-score), from lhc_jets.py
│   └── wds_data.py                # webdataset formatter/loader
├── training/
│   └── train.py                   # imagenet_trainer.py — single trainer used across jets and ImageNet runs
├── eval/
│   ├── evaluate.py
│   ├── count_params.py
│   └── compute_flops.py          # calflops wrapper
├── slurm/
│   ├── run_tier1_165k.slurm
│   ├── run_tier2_1M.slurm
│   ├── run_tier3_5M.slurm
│   ├── run_tier4_22M.slurm
│   ├── vit_tiny_165k.slurm
│   ├── vit_tiny_1M.slurm
│   ├── vit_tiny_5M.slurm
│   ├── vit_small.slurm
│   └── resnet9.slurm
├── results/
│   ├── tables/                   # locked calflops numbers: all 4 DepthViT tiers + matched baselines + all-split ablations
│   └── figures/                  # extracted blog charts + architecture/physics diagrams
```

## Setup

```bash
git clone https://github.com/ML4SCI/CMS.git
cd CMS/E2E/E2E_DepthViT_Krish_Malik
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On NERSC Perlmutter specifically:
```bash
module load pytorch/2.6.0
source /path/to/your/venv/bin/activate
```

## Dataset

This project uses the **HLS4ML LHC Jet dataset** (Pierini, Duarte, Tran, Freytsis — CERN/UCSD/Fermilab/Rutgers), publicly available on Zenodo, no access request required:

| Split | Zenodo DOI |
|---|---|
| 30 particles | [10.5281/zenodo.3601436](https://doi.org/10.5281/zenodo.3601436) |
| 50 particles | [10.5281/zenodo.3601443](https://doi.org/10.5281/zenodo.3601443) |
| 100 particles | [10.5281/zenodo.3602254](https://doi.org/10.5281/zenodo.3602254) |
| 150 particles | [10.5281/zenodo.3602260](https://doi.org/10.5281/zenodo.3602260) |

Each image has two channels (ECAL, HCAL) at 100×100 px, five jet classes (gluon, light quark, W, Z, top), and roughly 610K–640K training samples with 240K–270K held out for validation depending on split. Preprocessing: `log(1+x)` transform followed by per-channel, per-split standardization (see `data/dataset.py`).

**Training protocol:** SGD (lr 0.1, momentum 0.9), cosine schedule with 5,000 warmup steps, global batch size 512, bf16-mixed precision, 4× A100, 90 epochs, seed 42.

## Usage

Train a config:
```bash
python training/train.py --config configs/tier4_22M.json
```

Check parameter count before launching a run:
```bash
python eval/count_params.py --config configs/tier4_22M.json --k 196
```

Example SLURM submission (see `slurm/` for the full set):
```bash
sbatch slurm/run_tier4_22M.slurm
```

Evaluate:
```bash
python eval/evaluate.py --config configs/tier4_22M.json --checkpoint <path>
```

## Roadmap (remainder of GSoC)

- [ ] Multi-seed evaluation (3 seeds) across the 1M and 22M tiers
- [ ] Second dataset, to check generalization beyond HLS4ML jets
- [ ] Channel-symmetric ViT control at matched parameters
- [ ] Custom fused Triton kernel for small-channel attention
- [ ] FPGA synthesis and measured latency (longer-term target)

## Acknowledgments

This work is part of Google Summer of Code 2026 under **ML4Sci** (Machine Learning for Science), mentored by **Eric Reinhardt** (University of Alabama) and **Diptarko Choudhury**, with all experiments run on NERSC's Perlmutter supercomputer.

## Contact

**Krish Malik** — [krishmalikus@gmail.com](mailto:krishmalikus@gmail.com) · [GitHub](https://github.com/krishoncloud) · [LinkedIn](https://www.linkedin.com/in/krish-malik-0933822b3/)
