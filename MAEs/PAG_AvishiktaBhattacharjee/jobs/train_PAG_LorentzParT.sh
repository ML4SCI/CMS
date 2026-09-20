#!/bin/bash
#SBATCH -A m4392_g
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -q shared
#SBATCH -t 36:00:00
#SBATCH -J train_PAG_LorentzParT
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 32
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
##SBATCH --mail-user=<your email>
##SBATCH --mail-type=ALL

# Fine-tune the gated LorentzParT for classification. Submit from the project root with the pretrained model,
# or with the pretraining job id (jobs/run_all_PAG_LorentzParT.sh does this for you):
#     sbatch jobs/train_PAG_LorentzParT.sh logs/LorentzParT/best/<pretrain run>.pt
#     sbatch jobs/train_PAG_LorentzParT.sh <pretraining job id>

PRETRAINED=${1:?"usage: sbatch jobs/train_PAG_LorentzParT.sh <pretrained model .pt | pretraining job id>"}
NPZ_PATH=/global/homes/t/tapur/jetclass_balanced_1M.npz

cd "$SLURM_SUBMIT_DIR"
source jobs/setup_env.sh

echo "Node list: $SLURM_NODELIST"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source jobs/lib.sh
PRETRAINED=$(best_model "$PRETRAINED") || exit 1
echo "Pretrained model: $PRETRAINED"

srun --unbuffered --export=ALL python -m scripts.train_LorentzParT \
    --seed 42 \
    --config-path ./configs/train_PAG_LorentzParT.yaml \
    --weights "$PRETRAINED" \
    --npz-path "$NPZ_PATH"
