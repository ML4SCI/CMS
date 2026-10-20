#!/bin/bash
# ViT-Tiny 1M-tier, seed 123. Standalone script — invokes vit_tiny_trainer.py
# directly, NOT imagenet_trainer.py. Jobs 56399325/56399327 failed with
# KeyError: 'k_factor' because a prior script routed a ViT-Tiny config into
# DepthViTModule (imagenet_trainer.py), which requires k_factor and ViT-Tiny
# configs never have it. This script cannot hit that bug: it never imports
# DepthViTModule at all.
#SBATCH --account=m4392
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=vit_tiny_1M_seed123
#SBATCH --output=/pscratch/sd/k/krish_m/depthvit/runs/logs/vit_tiny_1M_150p_seed123_%j.out
#SBATCH --error=/pscratch/sd/k/krish_m/depthvit/runs/logs/vit_tiny_1M_150p_seed123_%j.err
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

CFG=configs/jets_vit_tiny_1M_150p_seed123.json

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Branch:    $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Commit:    $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "Config:    $CFG"
echo "Entry:     vit_tiny_trainer.py (NOT imagenet_trainer.py)"
echo "Model:     ViT-Tiny, 1M tier, seed 123"
echo

# Pre-flight: fail fast and loud if config accidentally has k_factor
# (would mean it's a DepthViT config misfiled under a vit_tiny name).
python3 -c "
import json
c = json.load(open('$CFG'))
assert 'k_factor' not in c.get('model', {}), \
    'FATAL: this config has k_factor but is being run as ViT-Tiny -- wrong config file?'
assert 'embed_dim' in c.get('model', {}), \
    'FATAL: no embed_dim in model config -- not a ViT-Tiny config?'
print('Pre-flight OK: embed_dim present, k_factor absent, as expected for ViT-Tiny.')
"

python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  vit_tiny_trainer.py \
  --config "$CFG"
