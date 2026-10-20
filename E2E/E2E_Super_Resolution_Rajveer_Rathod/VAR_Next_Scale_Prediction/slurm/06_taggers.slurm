#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# One tagger per array task: TAGGER_GROUP=early -> hr and lr:<level>, late -> sr:<level>, each x SEEDS.
# Finished taggers are skipped, so a re-run only trains what is missing.
#SBATCH -J superres-tagger
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
: "${TAGGER_GROUP:?}"
read -r kind seed <<< "$(tagger_task "$TAGGER_GROUP" "$SLURM_ARRAY_TASK_ID")"
[[ -n "$kind" ]] || die "no tagger for task $SLURM_ARRAY_TASK_ID in group $TAGGER_GROUP"
job_header "train_taggers $kind seed $seed"
run_pipeline train_taggers --kinds "$kind" --seeds "$seed"
