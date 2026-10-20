#!/bin/bash
# DepthViT ~5.3M on CIFAR-100, 90 epochs, cold start (no pretraining).
# General-image-dataset evidence for the ICLR generalization claim.
#SBATCH --account=m4392
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=cifar100_dvit_5M
#SBATCH --output=/pscratch/sd/k/krish_m/depthvit/runs/logs/cifar100_dvit_5M_90epoch_%j.out
#SBATCH --error=/pscratch/sd/k/krish_m/depthvit/runs/logs/cifar100_dvit_5M_90epoch_%j.err
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

CFG=configs/cifar100_dvit_5M_90epoch.json

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Branch:    $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Commit:    $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "Config:    $CFG"
echo "Model:     DepthViT (L=12, k_factor=61, in_channels=3, hidden_dim=183, 5,323,575 params)"
echo "Data:      CIFAR-100, 50k train / 10k val, 100 classes"
echo "Epochs:    90"
echo

python3 count_params_symctrl.py --config "$CFG" --arms asym

python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  imagenet_trainer.py \
  --config "$CFG"
