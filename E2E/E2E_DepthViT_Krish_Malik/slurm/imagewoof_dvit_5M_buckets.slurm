#!/bin/bash
# DepthViT ~5M on Imagewoof, BUCKETED variable resolution, 90 epochs, 1 seed.
# Resolution-independence arm: no crop, each image resized to nearest-aspect
# bucket. Pairs with imagewoof_vit384_5M.slurm.
#SBATCH --account=m4392
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=iwoof_dvit_5M
#SBATCH --output=/pscratch/sd/k/krish_m/depthvit/runs/logs/imagewoof_dvit_5M_buckets_%j.out
#SBATCH --error=/pscratch/sd/k/krish_m/depthvit/runs/logs/imagewoof_dvit_5M_buckets_%j.err
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

CFG=configs/imagewoof_dvit_5M_buckets.json

echo "Job ID:  $SLURM_JOB_ID   Node: $(hostname)"
echo "Branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "Config:  $CFG"
echo

# ---- pre-flight: this config MUST be a DepthViT config, and MUST route to the
# DepthViT trainer. Guards against the recurring k_factor KeyError from configs
# being sent through the wrong trainer.
python3 - "$CFG" << 'PY'
import json, sys
c = json.load(open(sys.argv[1]))
m = c["model"]
assert "k_factor" in m, "not a DepthViT config (no k_factor)"
assert m.get("pos_embed_2d") is True, "pos_embed_2d must be True for bucketed runs"
assert c["data"]["format"] == "imagewoof_bucket", "wrong data format for DepthViT bucket arm"
print(f"pre-flight OK: DepthViT k={m['k_factor']} pos_embed_2d=True buckets={c['data']['suite']}")
PY

python3 -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  imagenet_trainer.py --config "$CFG"
