#!/bin/bash
# JOB B -- OPTIONAL. Compiled-component comparison only.
# Uses --skip-full-model so torch.compile touches 3 modules, not 22 blocks.
# Submit only after Job A has landed successfully.
#SBATCH --account=m4392
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=track4_b
#SBATCH --output=/pscratch/sd/k/krish_m/depthvit/runs/logs/track4_b_%j.out
#SBATCH --error=/pscratch/sd/k/krish_m/depthvit/runs/logs/track4_b_%j.err

set -euo pipefail

module load pytorch/2.6.0
source /pscratch/sd/k/krish_m/venvs/depthvit/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
cd /pscratch/sd/k/krish_m/depthvit/repo

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo

echo "########## TIER 1 (compiled components, no full model) ##########"
time python3 profile_tier4.py \
  --config configs/jets_150p_22M_90epoch.json \
  --batch-size 32 \
  --compile \
  --skip-full-model \
  --json-out track4_tier1_compiled.json

echo
echo "JOB B DONE"
