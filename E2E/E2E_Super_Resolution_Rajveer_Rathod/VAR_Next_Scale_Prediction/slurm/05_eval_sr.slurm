#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Physics observables, event-by-event comparison and the two-sample test for one level.
#SBATCH -J superres-eval
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
level="${LEVEL_LIST[$SLURM_ARRAY_TASK_ID]}"
job_header "eval_sr $level"
run_pipeline eval_sr --levels "$level"
