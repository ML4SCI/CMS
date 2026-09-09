#!/bin/bash
#SBATCH --account=m4392
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=iwoof_vit_s1234
#SBATCH --output=/pscratch/sd/k/krish_m/depthvit/runs/logs/imagewoof_vit384_5M_seed1234_%j.out
#SBATCH --error=/pscratch/sd/k/krish_m/depthvit/runs/logs/imagewoof_vit384_5M_seed1234_%j.err
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
CFG=configs/imagewoof_vit384_5M_seed1234.json

echo "Job ID:  $SLURM_JOB_ID   Node: $(hostname)"
echo "Config:  $CFG"

python3 - "$CFG" << 'PY'
import json, sys
c = json.load(open(sys.argv[1]))
m = c["model"]
assert "k_factor" not in m, "DepthViT config sent to ViT trainer"
assert "embed_dim" in m, "not a ViT config"
assert c["data"]["format"] == "imagewoof_fixed"
assert c["seed"] == 1234, f"seed mismatch: {c['seed']}"
print(f"pre-flight OK: ViT D={m['embed_dim']} seed={c['seed']}")
PY

python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
  vit_small_trainer.py --config "$CFG"
