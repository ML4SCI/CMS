# ml4sci_26 — JetClass training package

Six-arm Particle Transformer architecture ablation on JetClass, with the ragged
CSR data pipeline it runs on. Targets NERSC Perlmutter (multi-node DDP).

```
ml4sci_26/
├── dataloader/       ragged_loader.py — the ONLY data entry point (see below)
├── preprocessing/    ROOT → CSR .pt converter, norm stats, loader benchmarks
├── ablation/         DDP training harness + Slurm scripts for the six arms
├── variants/         the models: ParT arms, LLoCa, MoE, L-GATr
└── requirements.txt
```

Start with [`ablation/README.md`](ablation/README.md) — it covers environment
setup, data staging, the smoke test, launching, and the recorded deviations from
the source papers. This file covers the data contract that everything shares.

---

## One data entry point

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

## Pipeline order

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
sbatch -A ${NERSC_ACCOUNT}_g ablation/slurm/smoke.sh      # all six arms, minutes

# 4. Launch, then read.
./ablation/slurm/submit_all.sh
python -m ablation.report --runs $SCRATCH/part_ablation/runs
```

Normalization is applied **on the fly in the collate**, never baked into the
shards, so the raw shards stay reusable. `v` is always left raw — the pair-feature
MLP is defined on raw four-momenta.

Dataset layout, measured multiplicity statistics, the on-disk shard schema, and
the storage-tier layout are documented in the [project write-up](https://medium.com/@omasho.works/finding-the-right-architecture-for-jet-taggers-84c5dad467ee)
and the preprocessing scripts in `preprocessing/`.

---

## Resume semantics

`ablation/slurm/submit_all.sh` chains dependent 12-hour jobs because 1e6 steps
does not fit one allocation. Each link resumes from `last.pt`, which carries
`step`, `best_accuracy`, **and the loader epoch**. Without that last field every
link would replay epoch 0's shard order and within-shard permutation — the model
would revisit the same jets in the same order on every restart. Checkpoints
written before this was recorded default to epoch 0 and load fine.
