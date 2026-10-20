#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Optuna study (only with TUNE=1). TUNE_STAGE = tune_vqvae | tune_var | tune_tagger.
# Studies are stored in SQLite and resume where they stopped.
#SBATCH -J superres-tune
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
: "${TUNE_STAGE:?}"
job_header "$TUNE_STAGE"
run_pipeline "$TUNE_STAGE"
