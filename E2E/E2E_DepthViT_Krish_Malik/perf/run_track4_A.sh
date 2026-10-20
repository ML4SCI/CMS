#!/bin/bash
# JOB A -- the one that matters. Safe on debug QOS (~10 min of a 30 min cap).
#SBATCH --account=m4392
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=track4_a
#SBATCH --output=/pscratch/sd/k/krish_m/depthvit/runs/logs/track4_a_%j.out
#SBATCH --error=/pscratch/sd/k/krish_m/depthvit/runs/logs/track4_a_%j.err

set -euo pipefail

module load pytorch/2.6.0
source /pscratch/sd/k/krish_m/venvs/depthvit/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
cd /pscratch/sd/k/krish_m/depthvit/repo

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Commit:  $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo

echo "########## TIER 1 (eager): component + full-model breakdown ##########"
time python3 profile_tier4.py \
  --config configs/jets_150p_22M_90epoch.json \
  --batch-size 32 \
  --json-out track4_tier1_eager.json \
  --trace-out track4_trace_eager.json

echo
echo "########## TIER 2: fused Triton kernel, bf16 ##########"
time python3 bench_chanattn_triton.py \
  --batch-size 32 --tokens 100 --channels 2 --k 196 --blocks 18 \
  --dtype bf16 --json-out track4_tier2_bf16.json

echo
echo "########## TIER 2: fused Triton kernel, fp32 ##########"
time python3 bench_chanattn_triton.py \
  --batch-size 32 --tokens 100 --channels 2 --k 196 --blocks 18 \
  --dtype fp32 --json-out track4_tier2_fp32.json

echo
echo "JOB A DONE"
