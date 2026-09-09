#!/bin/bash
# ViT-Small jets multi-seed, seed 1234. Standalone -- invokes
# vit_small_trainer.py directly, NOT imagenet_trainer.py. Same fix as seed123.
#SBATCH --account=m4392
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=28:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=vit_small_jets_seed1234
#SBATCH --output=/pscratch/sd/k/krish_m/depthvit/runs/logs/jets_150p_vit_small_90epoch_seed1234_%j.out
#SBATCH --error=/pscratch/sd/k/krish_m/depthvit/runs/logs/jets_150p_vit_small_90epoch_seed1234_%j.err
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

CFG=configs/jets_150p_vit_small_90epoch_seed1234.json

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Branch:    $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
echo "Commit:    $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "Config:    $CFG"
echo "Entry:     vit_small_trainer.py (NOT imagenet_trainer.py)"
echo "Model:     ViT-Small, seed 1234"
echo

python3 -c "
import json
c = json.load(open('$CFG'))
assert 'k_factor' not in c.get('model', {}), \
    'FATAL: this config has k_factor but is being run as ViT-Small -- wrong config file?'
assert 'embed_dim' in c.get('model', {}), \
    'FATAL: no embed_dim in model config -- not a ViT-Small config?'
assert c['seed'] == 1234, f\"FATAL: config seed is {c['seed']}, expected 1234\"
print('Pre-flight OK: embed_dim present, k_factor absent, seed=1234, as expected.')
"

python3 -m torch.distributed.run \
  --standalone \
  --nproc_per_node=4 \
  vit_small_trainer.py \
  --config "$CFG"
