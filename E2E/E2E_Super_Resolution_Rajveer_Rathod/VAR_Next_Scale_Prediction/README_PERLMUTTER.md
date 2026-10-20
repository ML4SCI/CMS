# Running the pipeline on Perlmutter (NERSC)

This explains how to run the whole physics super-resolution pipeline on Perlmutter GPU nodes and send
the results back. You need a NERSC account with access to a project that has GPU hours. You do not need
to know the project itself. Everything is set in **one file**, `slurm/settings.sh`, and run with
**three commands**. Every job can be re-submitted safely, because finished work is skipped.

What runs (one A100 per job, independent work in parallel):

```
download + prepare ──> tokenizer (VQ-VAE) ──> transformer [per level] ──> super-resolve [per level] ──> evaluation [per level] ──┐
        │                                                                     │                                                    ├──> summary + pack
        └──> taggers on HR / LR [per seed] ───────────────────────────────────┴──> taggers on SR [per level x seed] ───────────────┘
```

## 1. Get the code (login node)

Clone the **`perlmutter-slurm` branch** (only this branch has the Slurm scripts; `main` does not):

```bash
cd $HOME            # or anywhere; the code is small. Data and results go to $SCRATCH.
git clone --branch perlmutter-slurm --single-branch https://github.com/rajveer43/physics-superres-var.git
cd physics-superres-var
git branch --show-current     # must print: perlmutter-slurm
```

If you already have a clone of the repository, switch it to this branch instead:

```bash
git fetch origin perlmutter-slurm
git checkout perlmutter-slurm
git pull
```

To pick up later fixes, run `git pull` in this folder, preferably when none of your jobs are queued: a job
uses the code that is in this folder when it starts (its settings stay as they were at submission).

## 2. Edit `slurm/settings.sh`

Normally only these:

| setting | what to put |
|---|---|
| `ACCOUNT` | the NERSC project to charge (`iris` or https://iris.nersc.gov lists yours; some GPU allocations are named `mXXXX_g`) |
| `CPU_ACCOUNT` | optional: a project with CPU hours, so download + prepare don't use a GPU |
| `LEVELS` | the down-sampling levels to train, e.g. `"pool2x2"` or `"pool2x2 pool4x4 pool8x8"` |
| `SEEDS` | tagger seeds, default `"42 43 44"` |
| `CFS_COPY` | optional: a CFS directory the result tarball is copied to (e.g. `/global/cfs/cdirs/mXXXX/superres`) |

Data, caches, results and logs go to `ROOT=$SCRATCH/superres`. Leave `RUN_TAG`, `RUN_DATE`, `VERSION`
and `EXTRA_SET` as they are unless you were asked to change them.

## 3. One-time setup (login node)

```bash
bash slurm/setup_env.sh
```

This loads the NERSC `pytorch` module, installs the few missing packages (optuna, tabulate, ...) into
that module's user site (kept between sessions), and checks that the code imports. It should end with
`superres imports ok` and `setup done`.

## 4. Smoke test (up to 30 minutes, debug queue)

```bash
bash slurm/submit_all.sh --smoke
bash slurm/status.sh --smoke         # repeat until the job is gone from the queue
```

Every stage runs on a few hundred synthetic events. The log (path printed by the submit command) must end with
**`SMOKE OK`**. If it doesn't, send me the log before going on.

## 5. The real run

```bash
bash slurm/submit_all.sh --dry-run   # optional: shows the sbatch commands, submits nothing
bash slurm/submit_all.sh             # submits the whole chain with dependencies
bash slurm/status.sh                 # queue, what is finished, end of each log; run it any time
```

- The experiment is named `<date>_<version>_<tag>`, e.g. `2026-10-06_v2_multiseed`. The submit command fixes
  the name and freezes the settings into `$SCRATCH/superres/logs/<experiment>/run.env`, so changing `settings.sh`
  afterwards does not affect jobs that are already queued.
- Logs: `$SCRATCH/superres/logs/<experiment>/<stage>-<jobid>[_<task>].out`. The list of submitted jobs is in `jobs.tsv` there.
- Results: `$SCRATCH/superres/experiments/qg/<experiment>/`:

  ```
  2026-10-06_v2_multiseed/
    run_info.json                         every job: id, node, GPU, code version, full config
    models/vqvae_tokenizer/               the tokenizer
    models/var_transformer_pool2x2/       the transformer for each level
    models/resnet18_tagger_sr-pool2x2_seed42/  the taggers: input (hr, lr-<level>, sr-<level>) and seed
    evaluation/sr_pool2x2/                physics comparison of each level
    summary/qg_v2_2026-10-06_multiseed_report.md   the report, with tables and figures next to it
  ```
- If a job fails, the jobs that depend on it are cancelled automatically (`--kill-on-invalid-dep`).
- The qg tagger is a ResNet-18 (`models/resnet18_tagger_*`; summary files and figures say `resnet18-tagger`).
  For the small CNN instead, set `EXTRA_SET="tagger.arch=cnn"` (`models/cnn_tagger_*`, `cnn-tagger` files).
  To add ResNet taggers to an experiment trained with the CNN: `bash slurm/submit_all.sh --from taggers`.
  `<name>_report.md` is rewritten for the tagger of the latest summary job.

## 6. If something fails or times out

1. Look at the failed job's log (`bash slurm/status.sh` lists logs with a traceback or time-limit message).
2. Fix the cause if it is on our side (see Troubleshooting), then re-submit **from that stage**:

   ```bash
   bash slurm/submit_all.sh --from var        # stages: prepare vqvae var generate eval taggers summary
   ```

   A restart with `--from` continues the most recent experiment with the same `RUN_TAG`, even on a later day
   (it prints which one). To continue a different one, set `RUN_DATE` in `settings.sh` to its date
   (e.g. `2026-10-06`). Without `--from`, an empty `RUN_DATE` starts a new experiment dated today.
   `bash slurm/status.sh <experiment>` shows any experiment, e.g. `bash slurm/status.sh 2026-10-06_v2_multiseed`.

What happens to unfinished work on a re-submit:

| stage | after a crash or time limit |
|---|---|
| download | `wget -c` continues partial files |
| prepare | redone until it finishes (the cache counts as done only once `meta.json` is written) |
| tokenizer, transformer | continue from the last finished epoch (`last.pt`) |
| super-resolve | finished splits are skipped; an unfinished split is redone |
| evaluation | redone (it is short) |
| taggers | finished ones are skipped; an unfinished one starts again |
| summary | redone |

## 7. Send the results back

The summary job packs everything automatically at the end. To pack again by hand:

```bash
bash slurm/pack_results.sh                     # tables, figures, metrics, configs, logs (no model weights)
bash slurm/pack_results.sh --with-checkpoints  # also the trained models (*.pt), much larger
```

The tarball is `$SCRATCH/superres/results_<dataset>_<version>_<date>_<tag>.tar.gz`, e.g. `results_qg_v2_2026-10-06_multiseed.tar.gz` (and in `CFS_COPY` if set).
Send it with Globus, or tell me the CFS path. **`$SCRATCH` deletes files that haven't been used for 8 weeks**,
so copy anything worth keeping.

## Resources and expected use

- Each job: 1 A100 (`--gpus=1`), 32 CPU threads (16 cores) and, in the `shared` QOS, 64 GB RAM. Only that GPU is charged.
- `QOS=regular` reserves (and charges) a whole node per job, so keep `shared` unless shared jobs don't start.
  `QOS=preempt` is cheaper. Jobs can be stopped after 2 hours, and are then re-queued and continue from their last epoch.
- `GPU_MEM=80` asks for 80 GB A100s. Use it only after an out-of-memory error. There are fewer of them, so they may queue longer.
- Disk with the default settings (40 000 jets): HR + LR cache about 10 GB, super-resolved images about 4 GB per level,
  results under 1 GB without model weights.
- Time: there are no measured Perlmutter timings yet. The limits in `settings.sh` (`TIME_*`) are generous guesses.
  Each log prints `-- <stage> finished in N min`. Please send those numbers back along with the results.

| job | tasks (default settings) | limit |
|---|---|---|
| download + prepare | 1 | 4 h |
| tokenizer | 1 | 16 h |
| transformer | 1 per level | 24 h |
| super-resolve | 1 per level | 8 h |
| evaluation | 1 per level | 4 h |
| taggers | (1 + levels) x seeds on HR/LR, levels x seeds on SR | 6 h |
| summary | 1 | 3 h |

## Troubleshooting

| problem | what to do |
|---|---|
| `module: command not found` / `pytorch` module missing | `module avail pytorch`. If the default name differs, change `module load pytorch` in `slurm/common.sh` |
| torch older than 2.6 (setup warns) | load a newer module version, e.g. `module load pytorch/2.6.0`, then re-run setup |
| `Invalid account or account/partition combination` | wrong `ACCOUNT`: try the `_g` variant, or check `iris` |
| `CUDA out of memory` | set `GPU_MEM=80`, or add e.g. `var.batch_size=16` to `EXTRA_SET`; re-submit `--from` that stage |
| download fails / blocked | the log shows the URL. Copy the raw files into `$SCRATCH/superres/raw/qg/` by hand, then `--from prepare` |
| a job sits in the queue for long | `bash slurm/status.sh` shows the reason. `Dependency` is normal (it waits for the previous stage) |
| `DependencyNeverSatisfied` / cancelled jobs | an earlier job failed: fix it and re-submit `--from` that stage |
| results vanished | `$SCRATCH` purge (8 weeks unused). Re-run, or use `CFS_COPY` next time |

## Files

```
slurm/settings.sh          the only file to edit
slurm/setup_env.sh         one-time environment setup
slurm/submit_all.sh        submit the chain (--smoke, --from <stage>, --dry-run)
slurm/status.sh            progress of the latest (or a given) experiment
slurm/pack_results.sh      tarball of the results
slurm/common.sh            shared helpers (paths, environment, the python command line)
slurm/00_smoke.slurm ... 07_xeval_summary.slurm, tune.slurm   the jobs (submitted by submit_all.sh, not by hand)
```
