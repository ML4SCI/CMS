#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Tagger training curves, the 2x3 tagger table, the summary report; then pack (and copy) the results.
#SBATCH -J superres-summary
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
job_header "tagger_xeval + summarize"
run_pipeline tagger_xeval,summarize --levels "$(IFS=,; echo "${LEVEL_LIST[*]}")" --seeds "$(IFS=,; echo "${SEED_LIST[*]}")"
bash "$REPO/slurm/pack_results.sh"
