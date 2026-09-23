# Which parts of the Particle Transformer earn their cost? Controlled ablations, weight tying and fused kernels for LGATr and ParT backbones

**GSoC 2026 · ML4Sci** — Endterm Submission
**Contributor:** [Suryansh Mishra](https://github.com/omasho-codes)
**Mentors:** Diptarko Choudhury, Eric Reinhardt, Guillermo Fidalgo
**Project:** [Foundation models for End-to-End event reconstruction](https://ml4sci.org/gsoc/2026/proposal_E2E7.html)
**Write-up:** [Finding the Right Architecture for Jet Taggers (endterm)](https://medium.com/@omasho.works/finding-the-right-architecture-for-jet-taggers-84c5dad467ee) · [Making Jet Taggers Fast: custom GPU kernels (midterm)](https://medium.com/@omasho.works/making-jet-taggers-fast-a-tutorial-on-custom-gpu-kernels-for-l-gatr-and-part-799f9a88df0a)
**Compute:** NERSC Perlmutter (4× A100), SLURM account m4392
> **Status:** The project set out to build a foundation model for end-to-end event
> reconstruction. Before pre-training anything at scale, we ran controlled architecture
> ablations of the Particle Transformer to find the encoder worth scaling: which components
> carry the accuracy, which can be removed or tied, and what each costs in parameters and
> latency. Alongside, we wrote fused Triton kernels for the pairwise path of ParT and for
> L-GATr's equivariant layers. The pre-training objective is the next
> stage and is not part of this work.

---

## Contents

- [Overview](#overview)
- [Why this matters, and the question this repo answers](#why-this-matters-and-the-question-this-repo-answers)
- [What was compared](#what-was-compared)
- [Headline result: the floor, and what survives it](#headline-result-the-floor-and-what-survives-it)
- [Weight tying: 2.87× fewer parameters for about 0.4 points](#weight-tying-287-fewer-parameters-for-about-04-points)
- [Fused kernels: 2 to 7× per kernel, 2× end to end](#fused-kernels-2-to-7-per-kernel-2-end-to-end-in-training-at-matched-accuracy)
- [Relation to prior ML4SCI work](#relation-to-prior-ml4sci-work)
- [Repository structure](#repository-structure)
- [Setup](#setup) · [Dataset](#dataset) · [Data pipeline](#data-pipeline) · [Usage](#usage)
- [Conclusions](#conclusions)
- [Known limitations](#known-limitations) · [What's next](#whats-next)
- [References](#references) · [Acknowledgments](#acknowledgments)

## Overview

Every 25 nanoseconds the LHC produces a collision, and a trigger system has to decide within
microseconds whether to keep it. Jet tagging, deciding whether a spray of hadrons came from a
top quark, a Higgs boson, a W or Z, or ordinary QCD, sits inside that budget, so the cost of a
tagger is a physical constraint, not a convenience.

This project asks: **which parts of ParT actually earn their cost, and can the model be made
smaller or faster without losing what works?** To answer it, ten variants of ParT were trained
on JetClass under one fixed recipe and compared against seed-repeated baselines at matched
training steps, so that noise could be told apart from effect. On top of that fused GPU kernels cut the
wall-clock  by 2 to 7× per kernel.

## Why this matters, and the question this repo answers

Published jet-tagging comparisons rarely carry error bars. JetClass papers, including ParT,
L-GATr, LLoCa and MIParT, report accuracies to four decimals with no seed repeats, and the
improvements they report over each other are typically +0.001 to +0.004.

So before asking which architecture is best, this project measured the floor: three
architecturally identical baselines, same recipe, same data, differing only in seed. At 200k
steps they span **0.00085** in accuracy. The two runs that share a seed differ by as much as the
two that do not. **A single-run difference below about 0.001 at matched steps cannot be
attributed to the architecture with any confidence.**

## What was compared

Every arm is stock ParT with one thing changed, trained under the same recipe on the same data
(details under [Dataset](#dataset)). The baseline has 8 encoder blocks of width 128, a
pair-embedding MLP that turns four pairwise kinematic features into an attention bias, and two
class-attention blocks that read the jet out; 2,143,354 parameters.

**Architecture arms**

| arm | what changes | idea |
|---|---|---|
| `tied_k1` | one encoder block placed at all eight depths (the k = 1 case of `tied`) | share weights across depth, as in ALBERT and Universal Transformers; compute unchanged, parameters ÷ 2.87 |
| `tied_k1_mor` | `tied_k1` plus a learned router: at each of the eight depths, half of each jet's particles (top-k by router score) take the block's update and the rest pass through unchanged | per-particle adaptive depth, after Mixture-of-Recursions (Bae et al. 2025); +129 parameters, compute unchanged in this implementation |
| `lowrank_r16`, `lowrank_r32` | the pair bias is factorised to rank 16 or 32 plus a learned diagonal, instead of a dense per-pair MLP output | if the trained bias is low rank, the O(P²) pair path can be made cheaper |
| `n8_k6_v2` | the learned pair-embedding MLP is replaced by closed-form pair features (Minkowski product and rotary phases) | test whether analytic pairwise physics can stand in for the learned bias |
| `lloca` | particles are expressed in Lorentz local frames before attention | Lorentz Local Canonicalisation (Spinner et al. 2025); trained in fp32 |
| `moe`, `moe_e8_top2` | the FFN becomes a mixture of experts (FLOP-matched, or 8 experts with top-2 routing) | more parameters at similar compute |
| `sparsemax` | sparsemax replaces softmax in attention | sparse attention weights |
| `diff_v1`, `diff_v2` | differential attention: two softmax maps subtracted | cancel attention noise (Ye et al. 2024) |
| `urot`, `urot_rope` | queries and keys rotated by a shared pairwise angle; `_rope` pools it into a rotary embedding | relative geometry inside QK instead of an additive bias |

**Controls and screens**

| run | what changes | why |
|---|---|---|
| `baseline_s42_wave0`, `baseline_s43_wave0` | nothing; seed repeats of `baseline` | measure the run-to-run floor |
| `baseline_nopair` | the pair bias is removed | how much the pair path is worth |
| `baseline_pair2x` | pair-embedding MLP widened 64³ → 128³ | is the pair path capacity-limited? |
| `baseline_wide` | embedding 128 → 160, FFN 512 → 640 (+55 % parameters) | is the model width-limited? |
| `baseline_noca` | the two class-attention blocks are removed; mean pooling reads the jet out | is the class-attention readout doing anything? |
| `baseline_ca` | training-time Cambridge–Aachen augmentation: with some probability a jet is re-clustered at a random radius and its constituents replaced by the surviving pseudojets; validation jets are untouched | a physics-motivated data augmentation (random jet resolution) |

## Headline result: the floor, and what survives it

Delta against the mean of the three baselines (0.85499) at step 200k, one run per arm.
Parameter counts are the built models'. Bold marks effects more than twice the floor.

| arm | params | accuracy @200k | Δ | reading |
|---|---|---|---|---|
| `baseline_nopair` (pair bias off) | 2,133,786 | 0.83696 | **−0.0180** | the pair bias is worth 1.8 points, the single most valuable component |
| `n8_k6_v2` (closed-form pair features) | 2,133,274 | 0.83823 | **−0.0168** | analytic Minkowski/rotary features do not replace the learned pair MLP |
| `lowrank_r16` | 2,146,850 | 0.84756 | **−0.0074** | factorising the bias to rank 16 is not free |
| `lowrank_r32` | 2,155,298 | 0.84842 | **−0.0066** | rank 32 barely better than 16 |
| `tied_k1_mor` (tied + adaptive depth, capacity 0.5) | 746,435 | 0.84849 | **−0.0065** | the router costs a further 0.4 points on top of tying (§ below) |
| `baseline_ca` (declustering augmentation) | 2,143,354 | 0.85014 | **−0.0048** | hurts at 200k, recovers by 1M |
| `urot_rope` | 2,142,330 | 0.85228 | **−0.0027** | pooled rotary variant costs a quarter point |
| `tied_k1` (one block × 8) | 746,306 | 0.85264 | **−0.0023** | 2.87× fewer parameters for a quarter point (§ below) |
| `baseline_noca` (class attention off) | 2,143,354 | 0.85449 | −0.0005 | class attention does nothing measurable |
| `urot` | 2,142,330 | 0.85478 | −0.0002 | null |
| `moe` (FLOP-matched) | 5,338,170 | 0.85459 | −0.0004 | null |
| `sparsemax` | 2,142,266 | 0.85502 | +0.0000 | null |
| `moe_e8_top2` | 4,293,690 | 0.85509 | +0.0001 | null |
| `baseline_pair2x` (pair MLP 128³) | 2,169,274 | 0.85615 | +0.0012 | at the floor |
| `diff_v2` (differential attention) | 2,282,618 | 0.85642 | +0.0014 | at the floor |
| `diff_v1` (differential attention) | 2,406,522 | 0.85654 | +0.0016 | at the floor |
| `baseline_wide` (+55 % params) | 3,332,122 | 0.85792 | **+0.0029** | a small real gain, bought with parameters |
| `lloca` (Lorentz local frames, fp32) | 2,145,021 | 0.85837 | **+0.0034** | a small real gain |

Three readings. The only components whose removal costs something are the pair bias and, by a
much smaller margin, the encoder's untied depth. Every attention variant tried (sparsemax,
differential, rotary, mixture-of-experts FFN) is a null at this resolution. The two positive
effects that clear the floor, width and Lorentz local frames, are each worth about a third of a
point.

## Weight tying: 2.87× fewer parameters for about 0.4 points

One encoder block placed at all eight depths: 746,306 parameters against 2,143,354, compute
per jet unchanged. Reusing a block makes its residual updates add coherently and inflates the
residual stream (9.2× versus 3.7× untied), so each residual branch is scaled by
ε = λ/(N√L) with N = 8 applications and L = 1 unique block, i.e. ε = 1/8.

| step | tied k=1 | baseline | Δ |
|---|---|---|---|
| 200k | 0.85264 | 0.85453 | −0.0019 |
| 300k | 0.85465 | 0.85982 | −0.0052 |
| 400k | 0.85571 | 0.85970 | −0.0040 |
| 425k | 0.85729 | 0.86107 | −0.0038 |

The gap is stable at about −0.004 rather than closing, in the direction ALBERT and Takase &
Kiyono predict for a model this small. The tied model carries the ε residual scale and the
untied baselines do not, so this delta mixes tying with residual scaling; the matched control
(`baseline_wave2`, same ε) is configured but not yet trained. Per-class background rejection
for the tied model is in [`logs/results_tables.md`](logs/results_tables.md).



## Fused kernels: 2 to 7× per kernel, 2× end to end in training at matched accuracy

`part_kernels/` and `lgatr_kernels/` replace the hot paths of ParT and L-GATr with Triton
kernels. Correctness is gated by parity, gradcheck and equivariance tests before any timing is
reported. One A100-SXM4-80GB, PyTorch 2.10, Triton 3.6, bf16, batch 128, dense 128-particle layout.
Source: raw logs in [`logs/kernel_benchmarks/`](logs/kernel_benchmarks/) (copied from the midterm repo's `nohup/`; see its README for which script produced each file).
QuarkGluon jets average ~39 particles; the benchmarks pad to 128:

| kernel | model | stock | fused | speedup |
|---|---|---|---|---|
| Pairwise four-vector features | ParT | 766 µs | 114 µs | **6.69×** |
| Attention with additive bias | ParT | 452 µs | 189 µs | **2.39×** |
| Pair-embedding MLP | ParT | 9,227 µs | 2,034 µs | **4.54×** |
| Equivariant linear | L-GATr | 459 µs | 152 µs | **3.03×** |
| Geometric product | L-GATr | 305 µs | 136 µs | **2.25×** |

End to end: ParT 3.4× forward / 2.0× forward+backward, peak memory 8.2 → 5.1 GB; L-GATr
2.5× / 2.9×, 5.3 → 3.4 GB. Full 10-epoch QuarkGluon training runs, the load-bearing check
because they hold accuracy fixed:

| model | params | val acc stock → fused | val loss stock → fused | peak mem stock → fused | epoch time | training speedup |
|---|---|---|---|---|---|---|
| L-GATr | 708,992 | 0.7731 → 0.7732 | 0.4857 → 0.4860 | 20.2 → 13.1 GB | 91.3 s → 33.6 s | **2.72×** |
| Hybrid LorentzParT | 2,270,088 | 0.7735 → 0.7721 | 0.4864 → 0.4865 | 30.4 → 18.8 GB | 45.1 s → 22.5 s | **2.00×** |

Both packages patch a model in place, so a training script needs no other change:

```python
from part_kernels import optimize_part_model, unpatch_part_model

model, stats = optimize_part_model(model, compile_mode="reduce-overhead")
stats["patches"]             # ['pair_embed', 'attention']
stats["device_dispatch"]     # 'triton' on CUDA, 'cpu-stub' otherwise
stats["pairwise_fallbacks"]  # live counter; >0 means a config fell back at call time
model = unpatch_part_model(model)   # restore the pristine weaver forwards
```

```python
from lgatr_kernels import optimize_lgatr_model

model, stats = optimize_lgatr_model(
    model, use_compile_patches=True, compile_mode="reduce-overhead",
)
```

The individual kernels are callable directly, which is how the parity tests use them:

```python
from part_kernels import fused_pairwise_lv_fts, fused_attention_with_bias

feats = fused_pairwise_lv_fts(v)        # (N,4,P) 4-vectors -> (N,4,P,P) [lnkt, lnz, lndelta, lnm2]
out = fused_attention_with_bias(        # Q,K,V: (N*H,P,D); bias: (N*H,P,P)
    Q, K, V, bias=bias, pad_mask=pad_mask, scale=head_dim ** -0.5, num_heads=H,
)
```

`optimize_part_model` imports without a GPU and falls back to weaver's reference math; the Triton
kernels and `optimize_lgatr_model` import `triton` eagerly and need CUDA. The kernels themselves are in
[`part_kernels/triton/`](part_kernels/triton/) and [`lgatr_kernels/triton/`](lgatr_kernels/triton/),
the parity, gradcheck and equivariance gates in each package's `tests/`, and the reproduction commands
in each package's `benchmarks/bench_e2e.py`.

**Reported honestly:** these are QuarkGluon numbers on a dense 128-particle layout. The
benchmarks now read JetClass ragged shards, where jets average 39 particles and the O(P²) pair
path is proportionally smaller, so the speedups will not reproduce as-is and have not yet been
re-measured there. One kernel is in the training path: every arm with a pair path trains with
the fused pairwise-feature kernel on (`use_part_kernels=true` in the launcher), while the
pair-embedding MLP runs stock and the attention and eval-only MLP kernels stay off. The kernel
is parity-tested against weaver's math, but a GPU training-trajectory comparison against the
stock path has not been recorded; since every arm shares the same setting, the arm-to-arm
comparisons are unaffected.

## Relation to prior ML4SCI work

The kernel work started from **Thanh Nguyen's Hybrid Transformer** (GSoC 2025 with ML4Sci;
[code](https://github.com/ML4SCI/CMS/tree/main/MAEs/Hybrid_Transformer_Thanh_Nguyen),
[blog post](https://medium.com/@thanhnguyen14401/gsoc-2025-with-ml4sci-event-classification-with-masked-transformer-autoencoders-6da369d42140)):
a LorentzParT encoder, a ParT-style transformer with Lorentz-equivariant layers, pre-trained
with a masked autoencoder. The Hybrid LorentzParT row above is that architecture with the fused
kernels, and the original proposal was to speed up its pipeline and replace the MAE objective
with a Lorentz-JEPA one. The hybrid tree itself was retired from this repository when the
benchmarks moved to the JetClass ragged loader, so it is not included here; the JEPA objective
remains future work.

## Repository structure

```
.
├── dataloader/          ragged_loader.py: the single data entry point for trainer, probes and benchmarks
├── preprocessing/       ROOT -> CSR .pt shard converter, normalisation stats, loader benchmark
├── ablation/            train.py, config.py, metrics.py, report.py, Slurm launcher
│   ├── README.md        trainer, environment, and every config field
│   └── configs/         one YAML per run (33), all inheriting base.yaml
├── variants/            the ten arms, tied/ (weight tying), lgatr_model.py, tests/
├── part_kernels/        Triton kernels for ParT + benchmarks
├── lgatr_kernels/       Triton kernels for L-GATr + benchmarks
├── logs/                per-arm configs, provenance, metrics; frozen results tables
├── requirements-cluster.txt         training environment (Perlmutter)
└── requirements-preprocessing.txt   ROOT conversion / plotting extras
```

`ablation/README.md` documents the trainer and every config field; the Data pipeline
section below documents the data contract.

## Setup

Python 3.10 or newer, PyTorch 2.x. A GPU is needed only for training and for the kernel
packages (Triton); everything else, including the whole test suite, runs on CPU.

```bash
git clone https://github.com/ML4SCI/CMS.git && cd CMS/E2E/E2E_Efficient_Jet_Taggers_Suryansh_Mishra
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-cluster.txt

# weaver-core must be this git pin: PyPI releases through v0.4.17 lack
# PairEmbed._forward_sparse, which the variants and the kernels rely on.
pip install --no-deps "weaver-core @ git+https://github.com/hqucms/weaver-core.git@154db69"

export PYTHONPATH=$PWD
```

Two checks that need no data, then the test suite (about 15 s on a laptop; it builds every
configured model and asserts its parameter count and residual scales):

```bash
python dataloader/ragged_loader.py      # loader self-check on synthetic shards
python -c "
from variants import VARIANTS, build_variant_part
m = build_variant_part('baseline', input_dim=16, num_classes=10)
print(sorted(VARIANTS), sum(p.numel() for p in m.parameters()))"   # -> 2143354
python -m pytest -q
```

## Dataset

**JetClass** (Qu, Li, Qian 2022): 100M training jets, 5M validation; 10 classes
(QCD, H→bb̄, H→cc̄, H→gg, H→4q, H→ℓνqq′, Z→qq̄, W→qq̄, t→bqq′, t→bℓν); up to 128 particles
per jet with per-particle kinematics, impact parameters and particle ID.
[Zenodo 6619768](https://zenodo.org/record/6619768), no access request required.

The training package never reads ROOT files directly. Convert each split once into CSR-packed
`.pt` shards (about 100k jets and 310 MB per shard, with structural verification gates) and
compute the normalisation statistics from the training shards. Paying that cost once is what makes
the rest cheap: ROOT decompression and the jagged→dense conversion happen at convert time instead of
every epoch, CSR stores only real particles rather than padding every jet to 128 slots (they average
39, so a padded shard would be roughly 3× the bytes), and a shard is then a single `torch.load`
whose batches come out of a vectorised gather — no per-jet Python in the training loop.

```bash
python ml4sci_26/preprocessing/convert_jetclass_ragged_pt.py --help
python ml4sci_26/preprocessing/compute_norm_stats.py --help
```

One loader then serves every consumer with dynamically padded ragged batches:

```
x     (B, 16, P)   per-particle features, channel-first, float32
v     (B,  4, P)   four-momenta [px, py, pz, E]
mask  (B,  1, P)   1 = real particle, 0 = padding
y     (B, 10)      one-hot label
```

`P` is the largest multiplicity in the batch, not a fixed width; with jets averaging 39
particles this roughly halves the pair-path work compared with padding to 128.

**Training protocol (all experiments):** 8 encoder blocks, embedding 128, 8 heads, FFN 512,
pair-embedding MLP 64-64-64, 2 class-attention blocks; global batch 1024 (4 GPUs × 256);
Lookahead(RAdam), learning rate 1e-3 constant for 70 % of a 1M-step schedule then exponential
decay; gradient clipping 1.0; bf16 autocast; class-balanced sampling; validation on the full
4.9M-jet set every 25k steps with float32 prediction archives. Most arms ran 200k steps (about
15 A100-hours each); nine ran to 225k–1M. Inputs are the 16 raw per-particle channels, not the
official 17 derived features, which puts absolute accuracies about 0.01 below the paper's
0.861 without affecting any comparison within the repository.

## Data pipeline

`dataloader/ragged_loader.py` is the single canonical loader. Every run path goes
through it: the trainer, the rank audit, the loader benchmarks, and both kernel
benchmarks in `part_kernels/` and `lgatr_kernels/`. There is deliberately no
second loader.

**Batch contract** — identical on every path, channel-first:

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `x` | `(B, 16, P)` | float32 | particle features, `ALL_PARTICLE_FEATURES` order |
| `v` | `(B, 4, P)` | float32 | raw four-vectors `[px, py, pz, E]` |
| `mask` | `(B, 1, P)` | float32 | `1` = real particle, `0` = padding |
| `y` | `(B, 10)` | float32 | one-hot label |

`P` is the max multiplicity **in that batch**, not a fixed width — padding is
dynamic. Real JetClass jets average ~39 particles against a global max of 183,
so a fixed 128-wide pad would waste most of the tensor.

Every arm is called the same way:

```python
logits = model(x, v=v, mask=mask)
batch  = tuple(t.to(device, non_blocking=True) for t in batch)
```

`variants.lgatr_model.LGATrJetClassifier` accepts that signature too but
**ignores `x`** — it is a four-vector-only equivariant model. That is a real
asymmetry against the ParT arms, which see all 16 features; state it when
comparing numbers.

### The public API

| use | call |
|---|---|
| DDP training | `create_ragged_train_loader(pt_dir, batch_size, rank, world_size, ...)` |
| DDP validation | `create_ragged_val_loader(..., max_jets=...)` |
| flat-shuffle baseline (bench mode A) | `create_ragged_dataloader(...)` |
| one batch for a benchmark | `load_bench_batch(pt_dir, batch_size, pad_to=...)` |

Training and validation use `ShardCoherentBatchSampler`, which gives each global
rank a **disjoint** round-robin partition of shards and emits whole batches from
one shard at a time. That is what keeps the LRU shard cache at `cache_size=1` and
stops every node opening the same hot file on Lustre.

**Callers must advance the epoch.** The sampler seeds its shuffle on
`(epoch, rank)`, so a stale epoch replays an identical permutation:

```python
loader.batch_sampler.set_epoch(epoch)
```

`ablation/data.py::InfiniteLoader` does this automatically at each wrap-around,
and `ablation/train.py` checkpoints the loader epoch so a resumed job does not
restart the data order. Chained Slurm jobs depend on that — see below.

`load_bench_batch(..., pad_to=P)` is the exception that pads to a fixed width:
CUDA-graph capture and `torch.compile` need a stable shape.

---

### Pipeline order

Conversion is a **single-writer** step and is never done by the trainer — with
four ranks per node they would race to write the same files, so `ablation/data.py`
raises with instructions if the shard directories are missing.

```bash
export JETCLASS_ROOT=$SCRATCH/jetclass
export JETCLASS_PT_DIR=$PSCRATCH/jetclass/pt_ragged

# 1. ROOT → ragged CSR .pt shards, once. ~310 MB/shard, 0% particle clipping.
python -m preprocessing.convert_jetclass_ragged_pt \
    --root-dir $JETCLASS_ROOT --out-dir $JETCLASS_PT_DIR \
    --splits train_100M val_5M --max-files 20 --num-workers 32

# 2. Optional: normalization stats, from the TRAIN split only.
#    Deriving them from validation would leak into training.
python -m preprocessing.compute_norm_stats \
    --pt-dir $JETCLASS_PT_DIR/train_100M \
    --output $JETCLASS_PT_DIR/norm_stats.json --num-shards 10

# 3. Verify the pipeline before spending queue time.
python dataloader/ragged_loader.py                        # self-check, no data needed
sbatch -A ${NERSC_ACCOUNT}_g ablation/slurm/smoke.sh      # smoke arms, minutes

# 4. Launch, then read.
./ablation/slurm/submit_all.sh
python -m ablation.report --runs $SCRATCH/part_ablation/runs
```

Normalization is applied **on the fly in the collate**, never baked into the
shards, so the raw shards stay reusable. `v` is always left raw — the pair-feature
MLP is defined on raw four-momenta.

Dataset layout, measured multiplicity statistics, the on-disk shard schema, and
the storage-tier layout are covered in the Dataset section above
and the preprocessing scripts in `preprocessing/`.

---

### Resume semantics

`ablation/slurm/submit_all.sh` chains dependent 12-hour jobs because 1e6 steps
does not fit one allocation. Each link resumes from `last.pt`, which carries
`step`, `best_accuracy`, **and the loader epoch**. Without that last field every
link would replay epoch 0's shard order and within-shard permutation — the model
would revisit the same jets in the same order on every restart. Checkpoints
written before this was recorded default to epoch 0 and load fine.


## Usage

Every run is one YAML in `ml4sci_26/ablation/configs/`, inheriting from `base.yaml` through
`_base:`; unknown keys are rejected at load time, and any field can be overridden with
`--set key=value`.

```bash
cd ml4sci_26
P=/path/to/pt_ragged

# 60-step, 2-layer smoke that exercises training, evaluation, checkpoint and archive in ~1 min;
# a real arm is the same command with baseline.yaml, tied_k1_wave2.yaml, lowrank_r16.yaml, ...
python -m ablation.train --config ablation/configs/smoke.yaml \
    --set train_pt_dir=$P/train_100M --set val_pt_dir=$P/val_5M \
    --set norm_stats_path=$P/norm_stats.json

# one arm on a 4-GPU Slurm node, one process per GPU
sbatch -A <account> -q <qos> -t 12:00:00 -J part-baseline \
  --export "ALL,ARM=baseline,RUN_NAME=baseline,CONFIG=ablation/configs/baseline.yaml,TRAIN_PT_DIR=$P/train_100M,VAL_PT_DIR=$P/val_5M,NORM_STATS=$P/norm_stats.json,OUTPUT_DIR=/path/to/runs,PACKAGE_DIR=$PWD" \
  ablation/slurm/train_arm.sh
```

Runs write `metrics.jsonl`, `config.json`, `provenance.json` (git commit and dirty flag),
`last.pt`, `best.pt` and `predictions/step_*.npz` under `<output_dir>/<run_name>/`; relaunching
the same `run_name` resumes from `last.pt` including the loader position (RNG state is not
restored, so a resumed trajectory is not bit-identical). 200k steps take about 3 h 40 min on
four A100s (about 17k jets/s), the full 1M schedule about 18 h. Read runs back with
`python -m ablation.report --runs /path/to/runs`, and re-score archived predictions at any
working point with `python -m ablation.recompute_metrics`.


## Conclusions

- **Differential attention is a direction worth pursuing; hard sparsity, as tried here, is
  not.** Both differential arms landed on the positive side (+0.0014 / +0.0016) — consistent
  with a sharper, denoised attention distribution helping — but under twice the seed floor, so
  at this resolution it is a hint, not an established gain. Sparsemax, the hard version of the
  same intuition, was an exact null. Its likely handicap is not undefined gradients (sparsemax
  is differentiable almost everywhere) but dead ones: a particle pushed out of a head's support
  receives exactly zero gradient through that head, so support chosen early in training
  freezes. **IAFormer** (Esmail, Hammad & Nojiri, SciPost 2026) is the strongest existing
  evidence that the soft form is the one that works in collider physics: it uses differential
  attention *as* its sparsity mechanism — two subtracted softmax maps with a learnable per-layer
  β clipped to [0, 1], which the authors call implicit sparsity, with no top-k, no threshold and
  no token dropping anywhere in the model — and reports 510 ± 6 background rejection at
  ε_s = 0.5 on top tagging against ParT's 413 ± 6, at 211k parameters against 2.14M. The form to
  try next is sparsity annealed in during the run — α-entmax scheduled from softmax toward
  sparsemax, or an annealed penalty on attention mass — rather than hard sparsemax from step 0;
  the objective is differentiable end to end, so a penalty term is the right tool (a reward-style
  formulation would only add gradient variance). Both ideas deserve a re-test at larger width,
  where attention noise should matter more.
- **Recursion holds more accuracy per parameter than anything else tried.** One encoder block
  applied eight times stays within about 0.4 points of the untied baseline at 2.87× fewer
  parameters. That eight copies of one computation nearly match eight different computations
  suggests the task rewards iterated refinement — depth of processing — more than per-depth
  parameter diversity, which is the regime where recursive and adaptive-depth models have room
  to grow (caveat: the residual-scale-matched control is configured but not yet trained, so
  the −0.004 still mixes tying with residual scaling).
- **Width beat experts.** Mixture-of-experts was a null in both configurations at up to 2.5×
  the parameters, while +55 % dense width bought a small real gain (+0.0029). Parameters
  helped only when they widened the path every particle actually passes through; capacity
  parked in conditionally-routed experts did not. At this data scale the binding constraint is
  per-particle representation width, not total parameter count — future capacity should go
  into width (or into the pair path, worth 1.8 points), not expert count.

## Known limitations

1. The 0.00085 floor is a lower bound: the seed changes initialisation and dropout but not the
   data order, and only three baselines exist.
2. Every non-baseline arm is a single run, and the grid is not yet dense enough to commit to a
   final encoder; the deciding ablations (particle truncation, iso-latency shape, cheaper pair
   bias) are still ahead.
3. The long-run leaders are compared against a plain baseline that stopped at 650k, in a dip.
  
4.  `tied_k1_mor` is a hard top-k gate on a tied stack, not the Mixture-of-Recursions algorithm
   (shared router, constant capacity, no auxiliary loss, no hierarchical filtering), it delivers
   no speedup by construction, and it has no random-routing control at the same capacity.
6. No large-scale pre-training was done. The plan was JEPA-style pre-training on top of the
   selected encoder; time and compute ran out before that stage, so everything here is
   supervised classification.

## What's next

Ordered for a latency-first goal, since the pair path dominates runtime and the width result
says parameters are not the binding constraint at this data scale:

1. **Particle truncation** (top-k by pT, k ∈ {24, 32, 48, 64}), trained per k: how many
   constituents does the pair path need? The largest single latency lever.
2. **Shape at iso-latency**: a depth × width grid timed on one GPU, choosing the fastest shape
   at a target accuracy rather than the smallest parameter count.
3. **A cheaper pair bias** that keeps the 1.8 points it buys: bias inside QK, kT-sparse pairs,
   variable-length attention.

5. Finish `tied_k1_1M`, train `baseline_wave2` as its control, and run the free inference-depth
   sweep on the tied checkpoint (same weights at 2 to 16 applications).
6. Read the trained router of `tied_k1_mor` (kept fraction against pT rank and class, depths
   used per particle) from its checkpoint, a short single-GPU pass. Any further adaptive-depth
   arm should use the paper's recipe (per-depth routers, auxiliary loss, decreasing capacity
   with hierarchical filtering) and carry a random-routing control at matched capacity.
7. **Quantization-aware training.** Since the goal is a trigger-budget tagger, latency and not
   accuracy is the binding constraint, so once the encoder is fixed, QAT (INT8, and INT4 on
   the pair path) is the natural next lever: fold quantization into the schedule rather than
   quantizing post-hoc, so the model learns weights robust to it and the accuracy cost is paid
   during training instead of at deployment.

## References

- H. Qu, C. Li, S. Qian. *Particle Transformer for Jet Tagging.* ICML 2022. [arXiv:2202.03772](https://arxiv.org/abs/2202.03772)
- H. Qu, C. Li, S. Qian. *JetClass.* [Zenodo 6619768](https://zenodo.org/record/6619768)
- J. Spinner et al. *Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics.* NeurIPS 2024. [arXiv:2405.14806](https://arxiv.org/abs/2405.14806)
- J. Spinner et al. *Lorentz Local Canonicalization.* 2025.
- Z. Lan et al. *ALBERT: A Lite BERT.* ICLR 2020. [arXiv:1909.11942](https://arxiv.org/abs/1909.11942)
- S. Takase, S. Kiyono. *Lessons on Parameter Sharing across Layers in Transformers.* 2021. [arXiv:2104.06022](https://arxiv.org/abs/2104.06022)
- S. Bae et al. *Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation.* 2025. [arXiv:2507.10524](https://arxiv.org/abs/2507.10524)
- T. Ye et al. *Differential Transformer.* 2024. [arXiv:2410.05258](https://arxiv.org/abs/2410.05258)
- W. Esmail, A. Hammad, M. Nojiri. *IAFormer: Interaction-Aware Transformer network for collider data analysis.* SciPost Phys. 20, 108 (2026). [arXiv:2505.03258](https://arxiv.org/abs/2505.03258)
- A. Martins, R. Astudillo. *From Softmax to Sparsemax.* ICML 2016. [arXiv:1602.02068](https://arxiv.org/abs/1602.02068)
- M. Vigl, N. Hartman, M. Kagan, L. Heinrich. *Neural scaling laws for jet tagging.* 2026. [arXiv:2602.15781](https://arxiv.org/abs/2602.15781)
- `weaver-core`: https://github.com/hqucms/weaver-core · `lgatr`: https://github.com/heidelberg-hepml/lgatr

## Acknowledgments

This work is part of Google Summer of Code 2026 under **ML4Sci** (Machine Learning for
Science), mentored by Eric Reinhardt, Diptarko Choudhury and Guillermo Fidalgo. Training runs used a NERSC allocation.