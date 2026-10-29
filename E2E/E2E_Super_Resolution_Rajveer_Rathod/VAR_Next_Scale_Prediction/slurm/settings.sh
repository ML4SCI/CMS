# Settings for running the pipeline on Perlmutter. This is the ONLY file to edit.
# submit_all.sh freezes these values into $ROOT/logs/<experiment>/run.env at submit time, so editing this
# file later does not change jobs that are already queued.

# ---- account and resources ----------------------------------------------------------------------
ACCOUNT=mXXXX                 # NERSC project with GPU hours (some GPU allocations are named mXXXX_g)
CPU_ACCOUNT=                  # optional project with CPU hours for download + prepare; empty = use a GPU job
QOS=shared                    # shared (1 GPU, charged per GPU: recommended) | regular (whole node) | preempt
GPU_MEM=40                    # 40 | 80 (80 GB A100s are fewer, so they may queue longer)

# ---- what to run ---------------------------------------------------------------------------------
DATASET=qg                    # qg (CMS quark/gluon jets) | calo (CaloChallenge DS2)
VERSION=v2
RUN_TAG=multiseed             # what this experiment tests, in words joined by '-' (letters, digits, '-', '.')
RUN_DATE=                     # empty = today (Asia/Kolkata) -> new experiment; e.g. 2026-10-06 to resume one
                              # -> experiment folder <RUN_DATE>_<VERSION>_<RUN_TAG>, e.g. 2026-10-06_v2_multiseed
LEVELS="pool2x2"              # space separated, e.g. "pool2x2 pool4x4 pool8x8"
SEEDS="42 43 44"              # tagger seeds
TUNE=0                        # 1 = Optuna studies before the tokenizer, transformer and tagger
EXTRA_SET=""                  # extra config overrides, space separated key=value (JSON values),
                              # e.g. "var.epochs=40 var.gen_max_per_split=20000"

# ---- where things go -----------------------------------------------------------------------------
ROOT="$SCRATCH/superres"      # raw/ cache/ experiments/ logs/ live here ($SCRATCH: fast, purged after 8 weeks unused)
CFS_COPY=                     # optional, e.g. /global/cfs/cdirs/mXXXX/superres: results are copied here at the end

# ---- wall-time limits (HH:MM:SS, max 48:00:00). Estimates: raise them if a job times out ----------
TIME_PREPARE=04:00:00
TIME_TUNE=12:00:00
TIME_VQVAE=16:00:00
TIME_VAR=24:00:00
TIME_GENERATE=08:00:00
TIME_EVAL=04:00:00
TIME_TAGGER=06:00:00
TIME_SUMMARY=03:00:00
