#!/bin/bash
# Submitted by slurm/submit_all.sh, which passes the account, QOS, GPU, time and log file.
# Multi-scale VQ-VAE tokenizer (trained on HR, decoded with the LR image of every level).
# Resumes from last.pt after a timeout or preemption.
#SBATCH -J superres-vqvae
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
job_header "train_vqvae"
run_pipeline train_vqvae
