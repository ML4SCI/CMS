#!/bin/bash
#SBATCH --account=m4392
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-gpu=16
#SBATCH --job-name=iwoof_dvit_s123
#SBATCH --output=/pscratch/sd/k/krish_m/depthvit/runs/logs/imagewoof_dvit_5M_seed123_%j.out
#SBATCH --error=/pscratch/sd/k/krish_m/depthvit/runs/logs/imagewoof_dvit_5M_seed123_%j.err
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
CFG=configs/imagewoof_dvit_5M_seed123.json

echo "Job ID:  $SLURM_JOB_ID   Node: $(hostname)"
echo "Config:  $CFG"

python3 - "$CFG" << 'PY'
import json, sys
c = json.load(open(sys.argv[1]))
m = c["model"]
assert "k_factor" in m, "not a DepthViT config"
assert m.get("pos_embed_2d") is True, "pos_embed_2d must be True"
assert c["data"]["format"] == "imagewoof_bucket"
assert c["seed"] == 123, f"seed mismatch: {c['seed']}"
print(f"pre-flight OK: DepthViT k={m['k_factor']} seed={c['seed']}")
PY

python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
  imagenet_trainer.py --config "$CFG"
