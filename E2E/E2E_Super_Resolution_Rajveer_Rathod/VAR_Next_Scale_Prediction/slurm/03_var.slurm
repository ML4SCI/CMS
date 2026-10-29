#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Conditional VAR transformer, one array task per LR level (task i = i-th entry of LEVELS).
# Resumes from last.pt after a timeout or preemption.
#SBATCH -J superres-var
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
level="${LEVEL_LIST[$SLURM_ARRAY_TASK_ID]}"
job_header "train_var $level"
run_pipeline train_var --levels "$level"
