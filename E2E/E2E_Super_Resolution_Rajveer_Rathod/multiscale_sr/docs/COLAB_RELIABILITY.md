# Colab reliability review and operating procedure

## What the evidence establishes

The reported failure occurred after about 20 minutes, but no exception, resource
trace, or final training phase was provided. Its specific cause remains unknown.
These defects were confirmed in the code:

| Finding | Consequence | Change |
| --- | --- | --- |
| SEMD allocated a dense overlap array with `(K+1)^4` entries per sample | At K=128, **one** float32 intermediate was 1,107,691,524 bytes; several intermediates were live, even with chunk=1 | Cumulative integrals/searches use O(K²) storage, including backward |
| Drive copy happened only after training and evaluation | A terminated VM could lose the entire in-flight seed | Timed checkpoints, verified rotating Drive copies, and checkpoint before validation |
| Resume reset warmup/noise schedules and global step | Resumed optimization did not follow the original schedule | Global epoch schedules; optimizer, RNG, sampler replay, counters, and history restored |
| Completion used an old evaluation filename | Old architectures suppressed fresh training | Source/config/data identity, isolated variants, explicit trained/evaluated markers |
| Save helper removed the previous final directory before copying | Failed copy could destroy the last saved run | Unique attempts/staging; checksum validation; previous checkpoint slot retained |
| Prefetch threads blocked on abandoned queues and swallowed producer exceptions | Streaming/debug early exits could retain buffers or hide decode failures | Cancellation and error propagation; queue depth 2. The cached notebook training path does not use this wrapper |
| Cache rebuild truncated mmap before invalidating old metadata | Interrupted rebuild could be accepted as a complete cache | Invalidate marker first; cache format version 2; validate scalars and sizes |
| Arrow fast path ignored offsets and assumed float64 storage | Some slices/dtypes/layouts decoded incorrectly | Offset/dtype-aware conversion and CHW layout |
| Evaluation loaded full optimizer checkpoints onto GPU | Unnecessary GPU memory use | CPU checkpoint load; release unused states |
| Final evaluation ignored the configured SEMD settings | Final metrics could use different settings | Forward identical settings |
| Reports searched old names/all Drive experiments | New results missing or experiments mixed | Reports use only the current summary's completed seeds |

This is a new code fingerprint and therefore a new experiment family. No historical
Drive runs were accessed or deleted during this local review.

## Scientific and numerical effects

- Generator architecture, learnable skip gate, objective weights, production epoch
  budget, top-K, and production sample caps are unchanged by default.
- SEMD computes the same interval-overlap expression using prefix integrals. CPU
  and CUDA accumulate in float64 to reduce cancellation; MPS uses its supported
  dtype. Values and gradients were compared with the dense expression in double
  precision. At tied interval endpoints, subgradients can differ; floating-point
  results are not promised to be bitwise identical to historical SEMD.
- Training loaders now include their final partial batch (`drop_last=False`).
  This intentionally includes samples the old implementation discarded.
- Single-file parquet input now fails explicitly because the old fallback used
  that file for both training and held-out data. Supply at least two files.
- Cache version 2 rebuilds old caches; corrected decoding can change results if
  the old Arrow fast path interpreted the input incorrectly.
- No AMP, automatic batch reduction, gradient accumulation, or production dataset
  subsampling has been enabled. `EnvConfig.use_amp` remains a capability hint;
  the training implementation executes its existing FP32 path.
- Evaluation honors the checkpoint's disabled LR skip and native-LR setting.
  Legacy evaluation permits only the known missing `lr_skip_alpha` key; training
  resume requires the current recovery schema and strict architecture/config.

## Storage and recovery

```text
MyDrive/multiscale_sr_runs/
  learnable_lr_skip_v1_64x_<training-fingerprint>/
    provenance.json
    summary_<evaluation-id>.json
    taggers/hr_<evaluation-id>/tagger-0.pt + checksum record
    seed_123/attempt_<unique-id>/
      run_identity.json
      status.json
      train-<timestamp>.log
      eval-<timestamp>.log
      resources-<session>.jsonl
      checkpoints/latest-{0,1}.pt + checksum records
      checkpoints/best-{0,1}.pt + checksum records
      artifacts/                 # completed local training output
      trained.json
      evaluations/<evaluation-id>/
        classification_eval.json
        complete.json
        resources-<session>.jsonl
```

Each checkpoint publication overwrites only the older of two slots. A slot is
usable only when its contents match its checksum record; an interrupted write
leaves the other valid slot. This does not assume POSIX atomic rename on Drive.
Checksums detect corruption, not malicious changes. Drive availability and eventual
remote synchronization still depend on the mounted service.

Checkpoints are saved at the next completed batch after **300 seconds**, and at
epoch training/validation boundaries. A long batch or failed Drive upload can
exceed five minutes. A hard kill loses work since the last successful durable
checkpoint. Normalization/cache construction has no trained model progress to save.

Recovery is supported for **cached parquet with zero loader workers**. The loader
is replayed from its saved epoch RNG, already processed batches are read/skipped,
and model RNG resumes from the checkpoint. Replaying may take time, but does not
apply their optimizer updates again. Normalization comes from the checkpoint.
Tests establish exact CPU continuation with logging disabled on synthetic data;
cross-device, cross-version, CUDA nondeterminism, and asynchronous W&B effects
are not claimed to be bitwise reproducible. Streaming/HDF5 resume fails explicitly
until a compatible sampler recovery protocol is implemented and tested.

If both copies of a historical best checkpoint are corrupt and cannot be recovered
from latest, the sweep reports an error instead of substituting different weights.
Use a new variant to retrain while preserving the damaged attempt.

The dataset fingerprint records file names, sizes and modification timestamps,
not full dataset content hashes. Same-size modifications preserving timestamps
are not detected. Change the variant for such replacements. Run only one sweep
controller per experiment at a time; cross-VM concurrent writers are unsupported.

## Local verification

From the `multiscale_sr` project directory:

```bash
python -m pytest -q tests
python -m compileall -q .
```

The suite includes dense-versus-integral SEMD values/gradients, ties/zero images,
default K=128 on 128² images, model forward/backward and gate updates at input
sizes 16/32/64, optimizer/checkpoint recovery, partial batches, prefetch cleanup,
Arrow offsets/dtypes/layouts, cache rebuild interruption, checksum corruption,
Drive-copy failures, identity isolation, CLI precedence, honest partial summaries,
and a real subprocess smoke sweep with synthetic parquet and classification eval.

The end-to-end fixture uses a small model, one epoch, bounded samples and CPU. It
tests fresh training, completed-run reuse, evaluation-only retry and recovery
after training finished but final artifact publication was interrupted. It is **not**
a benchmark or a validation of the full CMS dataset or Colab GPU memory capacity.

Local verification on 2026-09-17: **33 passed, 1 skipped** (CUDA allocation-growth
test skipped because CUDA is unavailable). All 13 notebook code cells parse;
repository Python compilation and whitespace checks pass. Model/optimizer states
match exactly in the synthetic CPU interruption tests. The temporary test runtime
used Python 3.13 and Torch 2.14; rerun the smoke procedure on Colab's installed stack.

## Colab validation before the full sweep

1. Make these files available on the branch cloned by Cell 2. Updating only the
   notebook is insufficient: the controller and package changes are required.
   Restart the runtime and run setup cells; check the branch, CUDA GPU and versions.
   Dependencies now include `psutil`. Do not replace Colab's Torch/CUDA stack.
2. Keep `RESUME=None`. Cell 7 handles separate recovery for each seed. Do not pass
   one checkpoint globally across seeds.
3. First run a deliberately isolated smoke experiment from the project directory:

```python
import subprocess, sys
subprocess.run([
    sys.executable, '-u', 'colab_sweep.py',
    '--config', CONFIG, '--data-dir', str(DATA_DIR),
    '--drive-root', '/content/drive/MyDrive/multiscale_sr_runs',
    '--scale', str(SCALE), '--epochs', '1', '--seeds', '123',
    '--variant', 'colab_verification', '--smoke',
    '--semd-topk', '8', '--semd-max-samples', '8', '--no-wandb',
], check=True)
```

`--smoke` explicitly uses a small generator, batch size 2, two training batches,
one validation/statistics batch, and a short tagger evaluation. It appends `_smoke`
to the variant and uses a separate fingerprint. Building the full decode cache can
still take time/disk; smoke mode does not silently redefine the dataset cache.
Run the same command again and verify `action: reused`.

4. Run one seed at the intended production settings beyond the previous failure
   point, through at least one validation/SEMD boundary. Keep the intended final
   epoch budget fixed if this run should later resume. Review the resource traces
   and confirm a usable Drive checkpoint appears within the stated cadence.
5. Stop after a verified checkpoint, confirm the previous process has stopped,
   then rerun the same settings. Verify `action: resume`, continuing global steps,
   unchanged seed/fingerprint and non-restarted warmup. A Colab restart also loses
   the local cache, so rebuilding it is expected.
6. Run the full seed list only after this check. Use scale 64, then 32, then 16.
   Preserve the frozen tagger files. Report completed seeds and failures explicitly.

Full-duration CUDA training, real Drive failure injection, and the representative
run beyond 20 minutes have **not** been executed in the local environment.

## If the session dies again

Retrieve the affected attempt's `train-*.log`, `eval-*.log`, `resources-*.jsonl`,
and the sweep summary. Evaluation resource traces live in its evaluation directory,
including a `.partial` directory if it did not finish.

- `CUDA out of memory`: inspect the last phase and allocated/reserved/peak bytes.
  Distinguish training activation pressure from SEMD/evaluation before changing settings.
- Negative subprocess return code: it was terminated by a signal; `-9` alone does
  not establish host OOM. Check the preceding host RSS/available RAM and Colab message.
- Output stops without a return code: the controller/VM may also have terminated.
  Use the final resource timestamp and Colab runtime message; resource telemetry
  alone cannot distinguish quota eviction, browser disconnect, and VM failure.
- `disk full` or copy failure: local checkpoints remain where logged and the prior
  verified Drive slot is retained. Resolve storage availability before rerunning.
- Non-finite loss: the code raises before applying that generator update. Inspect
  normalization, data and preceding losses; reuse the previous verified checkpoint.

Do not infer that `empty_cache()` repairs a live-tensor leak or that a browser
disconnect means the training process stopped. Include the last phase, traceback,
GPU model, and resource trace when diagnosing the next failure.
