#!/bin/bash
# Matched ViT baseline for CIFAR-100 (D=192, L=12, h=3 -> 5,380,132 params).
# Pairs with slurm/cifar100_dvit_5M_90ep.slurm. Identical protocol; width is
# the only free variable. Entry point is vit_small_trainer.py, not
# imagenet_trainer.py -- ViT has its own LightningModule.
#SBATCH --account=m4392
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=cifar100_vit192
#SBATCH --output=/pscratch/sd/k/krish_m/depthvit/runs/logs/cifar100_vit192_5M_90epoch_%j.out
#SBATCH --error=/pscratch/sd/k/krish_m/depthvit/runs/logs/cifar100_vit192_5M_90epoch_%j.err
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

CFG=configs/cifar100_vit192_5M_90epoch.json

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Branch:    $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Commit:    $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "Config:    $CFG"
echo "Model:     ViT (embed_dim=192, L=12, heads=3, 5,380,132 params)"
echo "Data:      CIFAR-100, 50k train / 10k val, 100 classes"
echo "Epochs:    90"
echo

python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  vit_small_trainer.py \
  --config "$CFG"
