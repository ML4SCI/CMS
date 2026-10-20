#!/bin/bash
# ViT-384 ~5M on Imagewoof, FIXED square (short-side resize -> center crop),
# 90 epochs, 1 seed. Baseline arm that pays the crop/upscale tax.
# Pairs with imagewoof_dvit_5M_buckets.slurm.
#SBATCH --account=m4392
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=iwoof_vit384
#SBATCH --output=/pscratch/sd/k/krish_m/depthvit/runs/logs/imagewoof_vit384_5M_%j.out
#SBATCH --error=/pscratch/sd/k/krish_m/depthvit/runs/logs/imagewoof_vit384_5M_%j.err
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

CFG=configs/imagewoof_vit384_5M.json

echo "Job ID:  $SLURM_JOB_ID   Node: $(hostname)"
echo "Branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)  Commit: $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "Config:  $CFG"
echo

# ---- pre-flight: this MUST be a ViT config (no k_factor) routed to the ViT
# trainer. A DepthViT config here, or vice-versa, is the recurring routing bug.
python3 - "$CFG" << 'PY'
import json, sys
c = json.load(open(sys.argv[1]))
m = c["model"]
assert "k_factor" not in m, "DepthViT config sent to ViT trainer (k_factor present)"
assert "embed_dim" in m, "not a ViT config (no embed_dim)"
assert c["data"]["format"] == "imagewoof_fixed", "wrong data format for ViT fixed arm"
assert m.get("img_size") == 384, "ViT arm must be img_size=384"
print(f"pre-flight OK: ViT D={m['embed_dim']} L={m['num_layers']} h={m['num_heads']} img=384")
PY

python3 -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  vit_small_trainer.py --config "$CFG"
