#!/bin/bash
#SBATCH --account=m4392
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=vit_tiny_1M_150p
#SBATCH --output=/pscratch/sd/k/krish_m/depthvit/runs/logs/vit_tiny_1M_jets_150p_%j.out
#SBATCH --error=/pscratch/sd/k/krish_m/depthvit/runs/logs/vit_tiny_1M_jets_150p_%j.err
#SBATCH --mail-user=krish.01214815624@cseaiml.mait.ac.in
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

module load pytorch/2.6.0
source /pscratch/sd/k/krish_m/venvs/depthvit/bin/activate

export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HDF5_USE_FILE_LOCKING=FALSE

cd /pscratch/sd/k/krish_m/depthvit/repo

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Branch:    $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Commit:    $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "Config:    configs/jets_vit_tiny_1M_150p.json"
echo "Model:     ViT-Tiny ~1M (D=82, L=12, heads=2) — 1,006,473 params — matched to DepthViT k=23"
echo "Epochs:    90 (cold start — convergence ceiling)"
echo

python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  vit_tiny_trainer.py \
  --config configs/jets_vit_tiny_1M_150p.json
