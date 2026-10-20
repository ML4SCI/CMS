# DepthViT: Channel-Asymmetric Vision Transformers for LHC Jet Classification

**GSoC 2026 · ML4Sci** — Endterm Submission

**Contributor:** Krish Malik ([@krishoncloud](https://github.com/krishoncloud))
**Mentors:** Eric Reinhardt (University of Alabama), Diptarko Choudhury
**Project:** [Linear Attention Vision Transformers for End-to-End Mass Regression and Classification](https://ml4sci.org/gsoc/2026/proposal_E2E5.html)
**Compute:** NERSC Perlmutter (4× A100), SLURM account `m4392`
**Endterm write-up:** ["Small Models, Big Egos: Scaling DepthViT from 165K to 22M Parameters"](https://medium.com/@krishmalikus/small-models-big-egos-scaling-depthvit-from-165k-to-22m-parameters-43dbfc17f0e1)

> **Status:** This is the endterm snapshot. Since midterm, the project moved from a single-seed efficiency comparison to a matched-budget scaling study with multi-seed statistics, a same-scaffold symmetric control isolating what the architecture's win is actually attributable to, generalization checks on two non-calorimeter domains, a systems-level diagnosis of a wall-clock/FLOPs paradox, and a fused kernel demonstrating the fix.

---

## Overview

Every 25 nanoseconds, the LHC produces a new collision, and a real-time FPGA trigger system has to decide, on a microsecond budget, whether the event is worth keeping. That makes model size and compute hard physical constraints, not just efficiency metrics. This project asks: **at equal parameter budget, does a channel-asymmetric vision transformer actually beat a channel-symmetric one — and if so, why?**

Standard ViTs sum across image channels at the embedding layer, an assumption that holds for RGB natural images but breaks down for calorimeter data: the ECAL and HCAL channels here are two physically separate subdetectors, at different radii, built to capture different particle types. Summing them away throws out exactly the structure that makes jet classification possible.

**DepthViT** addresses this by embedding each channel independently and computing attention *across channels* rather than across spatial patches. The channel-wise attention mechanism is adapted from an unsupervised anomaly-detection architecture (Julson et al., Cerium Laboratories / University of Alabama / CMS HCAL Collaboration) into a fully supervised, five-class jet classifier. Only the depth-wise embedding and channel-attention operators are carried over from the original; the classification head, the mechanism to recover cross-patch spatial communication (Hierarchical Attention Pooling, or HAP), and — new since midterm — a scale-adjusted positional encoding for resolution independence, are original to this project.

## Why this matters, and the question this repo answers

Channel-wise attention buys physically meaningful structure but gives up something: it never lets one spatial patch talk to another. HAP blocks close that gap cheaply. But the harder question, and the one the endterm work is built around, is: reported efficiency gains for architectures like this are usually measured against a *standard-configuration* transformer — a comparison at unequal budget that credits a small model for being small, not for being channel-asymmetric. **Does the complete design still win once the baseline is shrunk to match?** The answer turns out to depend on the budget, and the gap doesn't just narrow as the budget grows — it changes sign.

## Headline result (multi-seed, 22M tier)

Matched against ViT-Small on both parameters and FLOPs, three seeds (42, 123, 1234):

| Metric | DepthViT | ViT-Small | Δ |
|---|---|---|---|
| Params | 22,046,777 | 22,012,613 | +0.16% |
| FLOPs (forward) | 4.2945G | 4.3113G | −0.39% |
| Top-1 accuracy | **75.34 ± 0.20%** | 73.98 ± 0.14% | **+1.36 pts** |
| Macro AUC | **0.9374 ± 0.0008** | 0.9319 ± 0.0006 | **+0.0055** |
| Train/val gap | 1.1 pts | 3.7 pts | less overfit |

DepthViT wins **all five jet classes individually** at matched parameters and matched FLOPs — a stronger claim than "efficient." The margin survives replication with non-overlapping standard-deviation bands.

### Per-class AUC at 22M (150p split, single-seed breakdown)

| Jet class | DepthViT | ViT-Small | Δ |
|---|---|---|---|
| Gluon (g) | 0.9323 | 0.9305 | +0.0018 |
| Light quark (q) | 0.9180 | 0.9152 | +0.0028 |
| W boson | 0.9449 | 0.9394 | +0.0055 |
| Z boson | 0.9324 | 0.9264 | +0.0060 |
| Top | 0.9553 | 0.9515 | +0.0038 |

## The four-tier scaling ladder — the sign changes, not just the gap

All numbers on the 150p HLS4ML split, 90-epoch training, identical protocol (SGD lr 0.1, momentum 0.9, cosine schedule, 5,000 warmup steps, batch 512, bf16, 4× A100). Multi-seed means (42/123/1234) shown where available; the 164K tier is single-seed.

| Tier | Model | Params | FLOPs | Top-1 | Macro AUC |
|---|---|---|---|---|---|
| 164K | DepthViT (k=4) | 164,417 | 0.032G | 70.14% | 0.9175 |
| 164K | ViT-Tiny (shrunk to match) | 162,373 | 0.032G | **71.70%** | **0.9238** |
| 1M | DepthViT (k=23) | 1,008,226 | 0.197G | 72.37 ± 0.41% | 0.9271 ± 0.0019 |
| 1M | ViT-Tiny (shrunk to match) | 1,006,473 | 0.200G | 72.30 ± 0.20% | 0.9259 ± 0.0005 |
| 5.4M | DepthViT (k=90) | 5,389,691 | 1.039G | **75.12 ± 0.13%** | **0.9367 ± 0.0008** |
| 5.4M | ViT-Tiny (native config) | 5,397,893 | 1.083G | 73.30 ± 0.11% | 0.9298 ± 0.0009 |
| 22M | DepthViT (L=18, k=196) | 22,046,777 | 4.2945G | **75.34 ± 0.20%** | **0.9374 ± 0.0008** |
| 22M | ViT-Small | 22,012,613 | 4.3113G | 73.98 ± 0.14% | 0.9319 ± 0.0006 |

**The ladder is a sign change, not a narrowing gap.** DepthViT loses at its own headline 164K budget (−1.56 pts), ties at 1M (+0.07 pts, within either arm's seed spread), and wins at 5.4M (+1.82 pts) and 22M (+1.36 pts). A gap shrinking toward zero would be consistent with both models converging to one ceiling; a gap crossing zero is not.

A ResNet-9 CNN matched to 164,671 parameters (within 0.15%) reaches only 60.40% — 9.74 points below DepthViT and 11.30 below matched ViT-Tiny at that budget — confirming both attention architectures beat convolution here, independent of the channel-symmetric question above.

## Same-scaffold symmetric control — what the win is actually attributable to

The scaling-ladder baselines above lack HAP and the structure-preserving head, so that comparison alone can't say whether *channel asymmetry itself* drives the win, or just having more structure in that slot. This control holds everything fixed (HAP placement, head, depth, training protocol, parameter budget, ~matched FLOPs) and changes **only** the channel treatment, at the 22M tier:

| Arm | Params | FLOPs | Top-1 / Macro AUC |
|---|---|---|---|
| `asym` (DepthViT, 3-seed mean) | 22,046,777 | 4.2945G | 75.34 ± 0.20% / 0.9374 |
| `chansum` (channel-summing embedding) | 22,085,977 | 4.3023G | 75.09% / 0.9360 |
| `symmix` (permutation-equivariant, matched params) | 22,046,777 | 4.7023G | 75.53% / 0.9382 |

**This is a careful null result, reported honestly rather than buried.** Turning off physical channel asymmetry (`chansum`) leaves accuracy essentially unchanged. Replacing the operator entirely with a symmetric one of matched budget (`symmix`) doesn't hurt either — it lands *inside* the asymmetric arm's seed band. At this tier, channel asymmetry per se is not what separates DepthViT from a symmetric transformer; the win is carried by the scaffold (HAP + structure-preserving head) and where the parameters get allocated (below), not by the asymmetric operator itself. The asymmetric operator remains the cheapest of the three in FLOPs at equal accuracy, so it's still the efficient default — just not, on this evidence, the mechanistic explanation.

## Where the parameters go — resolving the "dilution" intuition

The naive expectation is that the channel-attention mechanism gets *diluted* as the model scales up. Tracing the actual parameter tensors shows the opposite: channel-attention's share **rises** from 1.0% at k=4 to 20.8% at k=233, because the channel operator scales quadratically in the capacity knob k while the pinned feed-forward block (mlp_dim=768) is only linear in k.

| k | Channel-attn share | FFN share | HAP share |
|---|---|---|---|
| 4 | 1.0% | 95.3% | 2.2% |
| 23 | 4.6% | 85.1% | 9.0% |
| 90 | 12.8% | 61.8% | 24.5% |
| 233 | 20.8% | 39.0% | 39.7% |

At k=4, the model is 95.3% feed-forward network with a channel-attention operator attached — consistent with it losing there. The mechanism only crosses into double-digit share above ~5M parameters, coinciding with where the ladder's sign flips.

**Depth sweep at the fixed 22M budget.** Reaching 22M by raising k alone (at L=12) balloons HAP to nearly 40% of parameters — a lopsided architecture. A depth sweep at fixed HAP ratio shows channel-attention's share stays essentially flat (~21–22%) regardless of depth; depth only trades HAP against FFN:

| L (depth) | k | hidden_dim | Channel-attn | FFN | HAP |
|---|---|---|---|---|---|
| 12 | 233 | 466 | 20.8% | 39.0% | 39.7% |
| 16 | 207 | 414 | 21.9% | 46.2% | 31.3% |
| **18** | **196** | **392** | **22.1%** | **49.3%** | **28.1%** |
| 20 | 186 | 372 | 22.1% | 52.0% | 25.4% |
| 24 | 169 | 338 | 21.9% | 56.6% | 20.9% |

L=18, k=196 is the shallowest depth that pulls HAP below co-dominance with FFN, and is the config used in the headline 22M result. Bonus: the resulting hidden width (≈392) lands almost exactly on ViT-Small's own width (384).

## Generalization beyond calorimetry

**CIFAR-100 (3 seeds).** A natural worry is that this is a calorimeter-specific artifact. Repeated at ~5.3M on CIFAR-100 — deliberately adversarial, since RGB channels are ordinary color planes with no physical asymmetry motivating the design:

| Model | Params | FLOPs | Top-1 | Macro AUC |
|---|---|---|---|---|
| DepthViT (k=61) | 5,323,575 | 0.6454G | **62.76 ± 0.18%** | **0.9847 ± 0.0003** |
| Matched ViT (D=192) | 5,380,132 | 0.6935G | 52.92 ± 0.32% | 0.9697 ± 0.0005 |

+9.84 points Top-1 at 1.06% fewer parameters and 6.9% fewer FLOPs. The comparison doesn't invert outside calorimetry, which — combined with the symmetric-control null above — points toward scaffold and parameter allocation as the real driver, not physical channel asymmetry.

**Imagewoof resolution study (3 seeds, ~5.5M tier).** DepthViT's linear-in-tokens cost (vs. standard attention's quadratic) means one trained model can accept many input shapes via a scale-adjusted positional encoding, tested against a fixed-crop ViT baseline that must discard part of every image:

| Model | Params | Top-1 | Macro AUC | Top-5 |
|---|---|---|---|---|
| DepthViT (bucket routing, k=62) | 5,551,220 | **57.40 ± 0.29%** | **0.9093 ± 0.0014** | 93.40 ± 0.14% |
| ViT-384 (fixed-crop baseline) | 5,599,306 | 55.90 ± 1.14% | 0.9048 ± 0.0031 | **93.87 ± 0.29%** |

The bucketed DepthViT retains ~100% of each image's area at native aspect ratio; the fixed-square ViT baseline discards 44% of the average image and upscales the rest 1.20×. DepthViT wins Top-1 and macro AUC while being the *smaller* model, so the win isn't extra capacity.

**Reported honestly, not oversold:**
- Augmentation asymmetry: the ViT arm's random-resized-crop supplies implicit augmentation the bucket pipeline doesn't — a real confound, not controlled for
- Top-5 is essentially tied, marginally favoring ViT
- Per-class picture softens under replication (seed 42 alone: 8/10 classes; seeds 123/1234: 5/10 each; pooled 18/30)
- **Wall-clock is *not* a win here** — bucketed DepthViT trains ~3.5–4× slower per epoch, consistent with the systems finding below

## Wall-clock vs. FLOPs paradox — diagnosed, and a fix demonstrated

At the 22M tier, matched FLOPs do not translate into matched wall-clock: DepthViT-22M trains at ~13.2 min/epoch against ViT-Small's ~3.3 min/epoch (~4×), with GPU utilization 91–95% — ruling out data-loading stalls.

**Root cause:** channel attention runs over only C=2 channels at every spatial location, independently across k=196 heads (~19,600 tiny 2×2 products per block, across 18 blocks), executed as an unfused `matmul → softmax → einsum` sequence that never dispatches to `F.scaled_dot_product_attention` — fused attention kernels are built for large-N token attention, precisely the shape channel attention lacks.

| Component | Eager wall-clock share | Wall-clock/FLOPs ratio |
|---|---|---|
| Channel attention | 83.9% | 3.23× |
| Feed-forward | 6.3% | 0.12× |
| HAP | 9.8% | 0.42× |

Compiling barely moves this (ratio 3.23× → 3.27×) — the disproportion is structural, not a `torch.compile` artifact.

**Fix demonstrated on the isolated operator.** A fused Triton kernel specialized to C=2 collapses the 2×2 attention to elementwise operations in a single launch:

| Implementation | Time/op | Bandwidth |
|---|---|---|
| Eager (unfused) | 3.895 ms | 2.6 GB/s |
| `torch.compile` | 1.626 ms | — |
| Triton (fused) | **0.055 ms** | **183.0 GB/s** |
| A100 HBM2e peak | — | ~1,555 GB/s |

**71× speedup**, fp32-parity verified (max abs. error 4.8×10⁻⁷) before any timing is reported. This is validated on the isolated operator only — not yet integrated into the training path, and no accuracy number in this repo was produced with it.

## Repository structure

```
E2E_DepthViT_Krish_Malik/
├── README.md
├── models/
│   ├── DepthViT.py                  # HAP block + channel attention + structure-preserving head
│   ├── ResNet9.py                   # matched-budget CNN control
│   ├── vit_small.py                 # ViT-Small baseline
│   ├── vit_tiny.py                  # ViT-Tiny baseline
│   └── channel_treatments.py        # symmetric-control arms: chansum, symmix
├── training/
│   ├── imagenet_trainer.py          # shared trainer, jets + ImageNet + CIFAR/Imagewoof
│   ├── vit_small_trainer.py
│   ├── vit_tiny_trainer.py
│   └── resnet9_trainer.py
├── data/
│   ├── __init__.py                  # HLS4ML jet loader + preprocessing (log1p + per-channel z-score)
│   ├── imagewoof.py                 # bucket-routing loader, 5 aspect buckets
│   ├── data_cifar100.py
│   └── bucket_data.py
├── scripts/
│   └── prepare_cifar100.py
├── eval/
│   ├── compute_flops.py             # calflops wrapper — exact, not estimated
│   ├── eval_roc_flops.py            # per-class ROC-AUC + Top-1
│   ├── count_params.py
│   └── count_params_symctrl.py
├── perf/                            # wall-clock diagnosis + fused kernel
│   ├── bench_chanattn_triton.py     # parity gate + timing, isolated operator
│   ├── profile_tier4.py             # component wall-clock/FLOPs attribution
│   ├── run_track4_A.sh / run_track4_B.sh
│   └── track4_tier1_*.json, track4_tier2_*.json, track4_trace_eager.json
├── configs/                         # 4 DepthViT tiers × seeds, symmetric control, CIFAR-100, Imagewoof
├── slurm/                           # Perlmutter sbatch scripts, matches configs/
└── results/                         # locked eval JSONs, incl. results/roc_auc/
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

**HLS4ML LHC Jet dataset** (Pierini, Duarte, Tran, Freytsis), publicly available on Zenodo, no access request required:

| Split | Zenodo DOI |
|---|---|
| 30 particles | [10.5281/zenodo.3601436](https://doi.org/10.5281/zenodo.3601436) |
| 50 particles | [10.5281/zenodo.3601443](https://doi.org/10.5281/zenodo.3601443) |
| 100 particles | [10.5281/zenodo.3602254](https://doi.org/10.5281/zenodo.3602254) |
| 150 particles | [10.5281/zenodo.3602260](https://doi.org/10.5281/zenodo.3602260) |

Two channels (ECAL, HCAL) at 100×100 px, five jet classes (gluon, light quark, W, Z, top), ~610K–640K training samples with 240K–270K held out for validation depending on split. Preprocessing: `log(1+x)` followed by per-channel, per-split standardization.

**CIFAR-100** and **Imagewoof** are used for the generalization checks above; both are standard public benchmarks, loaded via `scripts/prepare_cifar100.py` and `data/imagewoof.py` respectively.

**Training protocol (all experiments):** SGD (lr 0.1, momentum 0.9), cosine schedule with 5,000 warmup steps, global batch size 512, bf16-mixed precision, 4× A100, 90 epochs, seeds 42/123/1234 where multi-seed.

## Usage

Train a config:
```bash
python training/imagenet_trainer.py --config configs/jets_150p_22M_90epoch.json
```

Check parameter count before launching:
```bash
python eval/count_params.py --config configs/jets_150p_22M_90epoch.json
```

Submit on Perlmutter (see `slurm/` for the full set):
```bash
sbatch slurm/jets_150p_22M_90ep.slurm
```

Evaluate:
```bash
python eval/eval_roc_flops.py --config configs/jets_150p_22M_90epoch.json --checkpoint <path>
```

Benchmark the fused kernel (no dataset or checkpoint required, synthetic inputs):
```bash
python perf/bench_chanattn_triton.py
```

## Acknowledgments

This work is part of Google Summer of Code 2026 under **ML4Sci** (Machine Learning for Science), mentored by **Eric Reinhardt** (University of Alabama) and **Diptarko Choudhury**, with all experiments run on NERSC's Perlmutter supercomputer.

## Contact

**Krish Malik** — [krishmalikus@gmail.com](mailto:krishmalikus@gmail.com) · [GitHub](https://github.com/krishoncloud) · [LinkedIn](https://www.linkedin.com/in/krish-malik-0933822b3/)
