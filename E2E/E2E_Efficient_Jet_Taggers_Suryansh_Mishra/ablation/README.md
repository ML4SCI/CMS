# ParT architecture ablation on NERSC Perlmutter

Six arms, one architectural delta each, trained on JetClass under an identical
optimizer, schedule, data pipeline and evaluation harness.

| arm | delta vs baseline | params | active FLOPs |
|---|---|---|---|
| `baseline` | stock weaver `ParticleTransformer` | 2.14 M | 1.00x |
| `lloca` | learned local Lorentz frames + tensorial messages ([arXiv:2505.20280](https://arxiv.org/abs/2505.20280)) | 1.003x | ~1.1-1.5x |
| `moe` | Mixture-of-Experts FFN, FLOP-matched | 2.49x | 1.00x |
| `sparsemax` | sparsemax instead of softmax attention | 1.000x | 1.00x |
| `diff_v1` | Differential Attention ([arXiv:2410.05258](https://arxiv.org/abs/2410.05258)) | 1.07x | ~1.1x |
| `diff_v2` | Differential Attention V2 | 1.04x | ~1.05x |

Everything else is held fixed on purpose. All arms use a global batch of 512,
1e6 steps, Lookahead(RAdam) at peak LR 1e-3 held for 70% of training then decayed
exponentially, and the same validation jets. If the arms differed in optimizer or
batch size, an apparent architecture effect could not be separated from a tuning
effect.

---

## Layout

```
ablation/                    training harness (this package)
├── config.py                AblationConfig; YAML with `_base:` inheritance + CLI overrides
├── data.py                  shard-coherent ragged CSR loaders for DDP training
├── distributed.py           process topology from Slurm / torchrun / single process
├── metrics.py               accuracy, AUC, background rejection at fixed efficiency
├── schedule.py              the official ParT optimizer and LR schedule
├── train.py                 the training entrypoint  (python -m ablation.train)
├── report.py                cross-arm comparison table  (python -m ablation.report)
├── rank_audit.py            pair-bias rank study (separate tool, not part of the runs)
├── configs/                 base.yaml + one file per arm + smoke.yaml
└── slurm/                   Perlmutter job scripts
    ├── train_arm.sh         one arm on 4 GPUs
    ├── submit_all.sh        all six arms, chained
    └── smoke.sh             pre-flight validation

variants/                    the models themselves
├── weaver_adapter.py        build_variant_part(): the factory for every arm
├── blocks/                  interchangeable encoder sub-blocks (one swap per arm)
│   ├── feedforward.py             dense pre-LN FFN (the baseline)
│   ├── moe_feedforward.py         Mixture-of-Experts FFN
│   ├── attention_block.py         softmax attention; hosts the MoE arm
│   ├── sparsemax_attention.py     sparsemax attention
│   ├── differential_attention.py      Differential Attention V1
│   └── differential_attention_v2.py   Differential Attention V2
├── lloca/                   the LLoCa arm (needs local frames, so not a plain block)
│   ├── frames.py                  Minkowski algebra + frame construction (Alg. 1)
│   ├── attention.py               frame-aware attention (Eq. 11)
│   └── part.py                    LLoCaParT: the assembled model
├── optim.py                 Lion / Lookahead / EMA
├── lgatr_model.py           the separate L-GATr classifier arm
└── tests/                   property-based suite (Hypothesis) + loader contract
```

`blocks/` and `lloca/` are split along a real boundary: everything in `blocks/`
is interchangeable behind `forward(x, padding_mask, U)`, whereas LLoCa
additionally needs per-particle reference frames and so cannot be swapped in
without them. `frames.py` is separated from the layer because it is pure
mathematics and therefore exactly testable — a subtly wrong frame still trains
normally, it just silently stops being equivariant.

Every public symbol is re-exported from the `variants` root, so
`from variants import X` works regardless of which subpackage owns it.

---

## 1. Environment

```bash
ssh perlmutter.nersc.gov
cd $HOME/ml4sci_26              # wherever this package lives

module load pytorch                 # `module avail pytorch` to pin a version

# weaver-core must come from the verified git pin: the PyPI releases through
# v0.4.17 lack PairEmbed._forward_sparse, which this package relies on.
pip install --user --no-deps \
  "weaver-core @ git+https://github.com/hqucms/weaver-core.git@154db69"
pip install --user uproot awkward vector pyyaml
```

Confirm the arms build before going further:

```bash
python -c "
from variants import VARIANTS, build_variant_part
m = build_variant_part('baseline', input_dim=16, num_classes=10)
print(sorted(VARIANTS), sum(p.numel() for p in m.parameters()))
"
# -> ['baseline','diff_v1','diff_v2','lloca','moe','n8','sparsemax','urot'] 2143354
```

## 2. Stage the data

Download [JetClass](https://zenodo.org/records/6619768) to `$SCRATCH` (not
`$HOME`, which is small and not meant for datasets), then convert ROOT to
ragged CSR `.pt` shards **once**:

```bash
export JETCLASS_ROOT=$SCRATCH/jetclass
export JETCLASS_PT_DIR=$PSCRATCH/jetclass/pt_ragged

python -m preprocessing.convert_jetclass_ragged_pt \
    --root-dir  $JETCLASS_ROOT \
    --out-dir   $JETCLASS_PT_DIR \
    --splits    train_100M val_5M \
    --max-files 20 \
    --num-workers 32
```

Optionally, compute normalization statistics:

```bash
python -m preprocessing.compute_norm_stats \
    --pt-dir $JETCLASS_PT_DIR/train_100M \
    --output $JETCLASS_PT_DIR/norm_stats.json \
    --num-shards 10
```

### Sizing this decision

Ragged CSR shards are ~310 MB each (100k jets, zero padding waste). So:

| training jets | `--max-files` | disk |
|---|---|---|
| 10 M | 10 | ~3.1 GB |
| 20 M | 20 | ~6.2 GB |
| 100 M (full) | omit the flag | **~310 GB** |

The full split fits easily in a default 20 TB scratch quota. For an architecture
ablation a subset is usually the better budget: what matters is that all six arms
see *identical* data, not that the data is maximal. Start with `--max-files 20`;
the same command without the flag converts everything later, and the arms stay
comparable as long as you do not mix.

Two properties worth knowing about:

- Normalization statistics come from the **training** split only. Deriving them
  from validation would leak information into training.
- The trainer never converts data. With four ranks per node they would race to
  write the same files, so `ablation.data` raises with instructions if the
  directories are missing.

## 3. Smoke test first

This exercises the real code paths — data loading, 4-GPU DDP, AMP, evaluation,
checkpointing, resume — at a size where a mistake costs minutes:

```bash
export NERSC_ACCOUNT=m1234           # your project, without the _g suffix
export JETCLASS_PT_DIR=$PSCRATCH/jetclass/pt_ragged
sbatch -A ${NERSC_ACCOUNT}_g ablation/slurm/smoke.sh
```

Or interactively, which is usually faster to iterate on:

```bash
salloc -A ${NERSC_ACCOUNT}_g -C gpu -q interactive -t 00:30:00 -N 1 --gpus 4
./ablation/slurm/smoke.sh
```

It must print `all six arms completed`. Do not skip it — a wrong `data_dir` or a
missing package surfaces here in two minutes instead of after a day in the queue.

The smoke also runs the fused Triton `PairEmbed` path by default
(`EXTRA_SET="--set use_part_kernels=true"`), so a green light covers the
production kernel path, not just stock weaver math.

**Kernel-parity smoke.** Before trusting the optimized kernels for real, verify
they train as close to the pristine baseline as floating-point noise allows:

```bash
sbatch -A ${NERSC_ACCOUNT}_g ablation/slurm/smoke_parity.sh
```

It runs `arm=baseline` twice — `use_part_kernels=false` and `=true` — from the
same seed and data order, then `python -m ablation.parity_check` diffs the
per-step loss/accuracy trajectories and exits non-zero on divergence.

## 4. Launch

```bash
export NERSC_ACCOUNT=m1234
export JETCLASS_PT_DIR=$PSCRATCH/jetclass/pt_ragged

./ablation/slurm/submit_all.sh -n          # dry run: prints the sbatch commands
./ablation/slurm/submit_all.sh             # submit
```

Each arm gets a chain of dependent 12-hour jobs (6 by default). Chaining is
necessary because 1e6 steps does not fit in one allocation — roughly 2-3 days per
arm on 4x A100. Every job resumes from `last.pt`, and the chain uses
`--dependency=afterany` so a walltime kill (non-zero exit, but `last.pt` written)
does not break it.

Every arm runs with the fused Triton `PairEmbed` (`--set use_part_kernels=true`)
by default — the accelerated version of the baseline, not the stock one. The
`smoke_parity.sh` gate above is what makes that default safe. Disable or extend
with `EXTRA_SET="--set use_part_kernels=false"` or
`EXTRA_SET="--set use_part_kernels=true --set compile_model=true"`.

```bash
./ablation/slurm/submit_all.sh -a "baseline lloca"   # subset of arms
./ablation/slurm/submit_all.sh -c 10 -t 06:00:00     # more, shorter links
squeue --me
```

Budget: about 15 node-days for all six arms at 1e6 steps, so plan on
**~1,400 A100-hours**. Cut `total_steps` if that exceeds your allocation — an
ablation at 200k steps per arm is still internally valid, it just is not
comparable to published ParT numbers.

## 5. Read the results

```bash
python -m ablation.report --runs $SCRATCH/part_ablation/runs
```

Output shape (values below are illustrative, not measured):

```
run                params  x base      step      acc    d acc       auc    rej50    rej99    prog
-------------------------------------------------------------------------------------------------
baseline        2,143,354   1.000 1,000,000   0.8612  +0.0000   0.98553   1841.2    142.7    100%
lloca           2,149,837   1.003 1,000,000   0.8698  +0.0086   0.98701   2043.5    158.1    100%
...

Background rejection at 50% signal efficiency (1/eps_B vs QCD):
  run              H4q      Hbb      Hcc      Hgg     Hqql      Tbl     Tbqq      Wqq      Zqq
  baseline      2412.1   5482.0    997.3    923.4   1802.5   9251.0    3247.9   1394.2   1032.1
```

`rej50` / `rej99` are background rejection `1/eps_B` against QCD at 50% / 99%
signal efficiency, averaged over signal classes, with a per-class breakdown
underneath. That is the number ParT, ParticleNet, L-GATr and LLoCa all report, so
it is the column to compare against the literature.

The report flags arms still training. **Arms evaluated at different steps are not
comparable** — wait for `prog` to read 100% before drawing conclusions.

Raw per-run artifacts live in `$SCRATCH/part_ablation/runs/<arm>/`:

| file | contents |
|---|---|
| `config.json` | fully resolved hyperparameters |
| `provenance.json` | `git_commit`, branch, dirty flag at job start |
| `metrics.jsonl` | train / eval / finish events (see below) |
| `predictions/step_XXXXXXX.npz` | fp16 `probs` + uint8 `labels` per eval (~8 MB / 400k jets) |
| `last.pt` / `best.pt` | resume / best-val checkpoints |

Recompute metrics at other efficiencies without retraining:

```bash
python -m ablation.recompute_metrics --run-dir $SCRATCH/part_ablation/runs/baseline \
    --efficiencies 0.3,0.5,0.95,0.99
```

### `metrics.jsonl` events

| `event` | when | key fields |
|---|---|---|
| `start` | once | `experiment`, `arm`, `git_commit`, `git_dirty`, `params`, `global_batch` |
| `train` | every `log_every` steps | `loss`, `accuracy` (train), `num_jets` in window, `jets_per_sec` |
| `eval` | every `eval_every` steps | `accuracy`, `auc`, `per_class` (rej@50/99, auc vs QCD) |
| `finish` | once | `best_eval`, `final_eval` (full metric dicts), `best_step`, git provenance |

**Train `accuracy` is not validation** — it is top-1 over the last `log_every`
optimizer steps, pooled across all jets seen in that window
(`num_jets` = `log_every × batch_size × grad_accum_steps × world_size`, e.g.
200 × 128 × 1 × 4 = **102,400 jets** on the full recipe, not 100).

**Checkpoints:** `last.pt` every `checkpoint_every` (10k) steps; `best.pt` when
val accuracy improves at an eval step.

### Current experiment: `part_ablation_v1`

Set in `configs/base.yaml` as `experiment: part_ablation_v1`. Six arms, one
architectural delta each, identical optimizer/data/eval harness:

| arm | delta |
|---|---|
| `baseline` | stock ParT + fused PairEmbed (`use_part_kernels=true`) |
| `lloca` | Lorentz local frames (fp32) |
| `moe` | FLOP-matched MoE FFN |
| `sparsemax` | sparsemax attention |
| `diff_v1` / `diff_v2` | Differential Attention variants |

Output root: `$SCRATCH/part_ablation/runs/<arm>/`. Smoke runs use
`$SCRATCH/part_ablation/smoke/` and `experiment` is unchanged unless overridden.

## 6. Configuration

Per-arm YAML lives in `ablation/configs/`, each inheriting shared settings via
`_base: base.yaml`. Anything can be overridden from the command line:

```bash
python -m ablation.train --config ablation/configs/moe.yaml \
    --set train_pt_dir=$PSCRATCH/jetclass/pt_ragged/train_100M \
    --set val_pt_dir=$PSCRATCH/jetclass/pt_ragged/val_5M \
    --set moe_config=param_matched \
    --set total_steps=200000
```

### MoE presets

Your framing allowed two matchings, so both are available
(`variants.MOE_PRESETS`, selected with `--set moe_config=<name>`). `H` is the
dense ParT FFN hidden width, 512:

| preset | experts | top-k | per-expert hidden | FLOPs | FFN params |
|---|---|---|---|---|---|
| `flop_matched_full` **(default)** | 4 | 1 | `H` | 1.00x | 4.00x |
| `flop_matched_top2` | 4 | 2 | `H/2` | 1.00x | 2.00x |
| `shared_flop_matched` | 3 + 1 shared | 1 | `H/2` | 1.00x | 2.00x |
| `param_matched` | 4 | 2 | `H/4` | 0.50x | 1.00x |

The default matches your first description: every expert is exactly the size of
the original ParT FFN, and one fires per token, so per-token compute equals the
baseline while the arm holds 4x the FFN capacity. `param_matched` is the opposite
corner (identical parameters, half the compute) and was the previous default in
this repo.

Two things the code enforces rather than leaving to chance:

- `top_k=1` requires Switch-style gating (softmax over *all* expert logits, then
  gather). A softmax over the single selected logit is identically 1.0, so the
  router would receive **no gradient** and the arm would silently degenerate to a
  fixed random partition of tokens. The constructor rejects that combination.
- The load-balancing auxiliary loss (`moe_aux_alpha`, default 0.01) must be added
  to the task loss or the router collapses onto one expert. The trainer does
  this; padded tokens are excluded from the balance statistics.

### LLoCa

`precision: fp32` is **required**, not a preference — bf16/fp16 destroy the exact
equivariance that is the point of the arm, and the paper runs LLoCa-ParT in single
precision throughout for the same reason. The config layer raises rather than
silently correcting it, because a warning buried in a 24-hour log is easy to miss
and a non-equivariant "equivariant" arm is a wrong result, not a slow one.

fp32 costs activation memory, so the arm uses batch 64 with 2 accumulation steps
to hold the global batch at 512.

`lloca_frames_hidden_dim` (default 64) is the width of the frame-prediction MLP.
The paper uses 128; its appendix E reports 16 is nearly as good. It allocates a
`(B, P, P+3, hidden)` activation, so it dominates this arm's extra memory — lower
it first if a node runs out.

## 7. Deviations from the papers

Recorded because they affect how the results should be read.

**LLoCa pair bias stays global.** The paper also recomputes ParT's pairwise edge
features from four-momenta *in local frames*. That makes the bias asymmetric (pair
`(i,j)` evaluated in the receiver's frame) and needs an `O(N^2)` four-vector
transform replacing weaver's `PairEmbed`. This implementation keeps the stock
global `pair_embed`, so the arm differs from the baseline in exactly the frame
construction and message transport, and the bias is bit-identical to the
baseline's. Consequence: of ParT's four pair features only `m^2` is
Lorentz-invariant, so the assembled model is **not** exactly invariant — this is a
third symmetry-breaking channel alongside the paper's two. Since the paper already
breaks the symmetry deliberately down to the beam-axis `SO(2)` subgroup (its
Table 6 shows that is worth a large amount of background rejection), this is a
difference of degree rather than kind.

The frame machinery itself *is* exact, and the tests prove it: with all
symmetry-breaking channels disabled the model is Lorentz-invariant to 3e-14 in
float64 (`variants/tests/test_lloca_properties.py`).

**LLoCa local kinematics use 6 channels, not 7.** This project's loader emits 16
features where the paper's uses 17. Channels 0-5 are replaced with local-frame
`[log pT, eta, phi, log E, dEta_jet, dPhi_jet]`; channels 6-15 (impact
parameters, charge, PID) are genuine scalars and pass through untransformed, as in
the paper. Keeping `input_dim=16` for every arm matters: it makes the input
embedding identical across the ablation.

**The mass regulator is frame-dependent.** `lloca_frames_min_mass` raises energies
to `sqrt(m_eps^2 + E^2)`, which touches the time component only and is therefore
not covariant. The paper applies it as preprocessing in a fixed frame; it
introduces a ~4e-6 deviation. Set it to 0 for equivariance checks, keep it at
5e-3 for training (it prevents numerically-massless particles from producing a
null `v0` and a divergent boost).

**Warmup added.** The published recipe starts at peak LR. A 2,000-step linear
warmup is added for all arms because several (differential attention's subtracted
maps, MoE's untrained router) are noticeably less stable in the first few hundred
steps. Warming up uniformly is a smaller intervention than per-arm LR tuning.

## 8. What was verified, and what was not

Verified by `python -m pytest ml4sci_26/variants/tests` from the repo root (needs
`PYTHONPATH=$PWD:$PWD/ml4sci_26`):

- LLoCa frame algebra: `L g L^T = g`, `L^-1 L = I`, and the equivariance rule
  `L(Lambda v) = L(v) Lambda^-1`, all to ~1e-14 in float64.
- End-to-end Lorentz invariance of the assembled arm, and that each
  symmetry-breaking channel measurably breaks it.
- MoE preset FLOPs and parameters against **measured** values (within 0.4% and
  1%), and that the router receives gradient under every preset.
- All six arms forward, backward, and give every parameter a gradient.
- Every arm forwards on **real loader output** — `tests/test_loader_contract.py`
  writes a genuine CSR `.pt` shard, reads it back through `RaggedShardDataset`,
  and asserts the loader's own batches satisfy the same contract the synthetic
  Hypothesis fixtures do. Previously the fixtures matched the loader only by
  coincidence; nothing failed if the loader changed shape.
- The full trainer on synthetic memmaps: all six arms, checkpoint resume, and
  2-rank DDP including MoE's unused-parameter path.

**Not verified**, because it needs the machine and the data:

- Any run on real JetClass — no numbers here are physics results.
- NCCL multi-node behaviour and achieved throughput on A100s (the DDP path was
  exercised with the gloo/CPU backend).
- Per-arm memory headroom at batch 128 on 40 GB vs 80 GB A100s. Perlmutter has
  both; if an arm OOMs, halve `batch_size` and double `grad_accum_steps` to hold
  the global batch at 512.
- Whether `module load pytorch` on your allocation is new enough. Run the smoke
  test, which is exactly what it is for.

## 9. Troubleshooting

| symptom | cause and fix |
|---|---|
| `train_pt_dir is not set` | set `--set train_pt_dir=...` or `TRAIN_PT_DIR` env var; the message contains the exact command |
| `the lloca arm requires precision='fp32'` | working as intended; leave `precision: fp32` |
| `top_k=1 with gate_mode='topk_softmax'` | use `gate_mode=full_softmax`, or a preset |
| CUDA OOM | halve `batch_size`, double `grad_accum_steps`; for `lloca` also lower `lloca_frames_hidden_dim` |
| DDP hangs at startup | a stale `MASTER_PORT`; `train_arm.sh` keys it to the job ID, so check nothing else exported it |
| chain link ran but made no progress | it resumed and hit the walltime; check `jets_per_sec` in `metrics.jsonl` and raise `-t` |
| `find_unused_parameters` warning on non-MoE arms | should not happen — it is enabled only for `moe` |
