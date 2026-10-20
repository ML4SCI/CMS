#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Super-resolve the train / val / test LR events of one level. Finished outputs are skipped.
#SBATCH -J superres-generate
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
level="${LEVEL_LIST[$SLURM_ARRAY_TASK_ID]}"
job_header "generate $level"
run_pipeline generate --levels "$level"
