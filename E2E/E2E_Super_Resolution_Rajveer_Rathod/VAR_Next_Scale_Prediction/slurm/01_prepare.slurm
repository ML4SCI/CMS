#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Download the raw data (resumable) and build the HR / LR cache (skipped if it already exists).
#SBATCH -J superres-prepare
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
job_header "download + prepare"
run_pipeline download,prepare
