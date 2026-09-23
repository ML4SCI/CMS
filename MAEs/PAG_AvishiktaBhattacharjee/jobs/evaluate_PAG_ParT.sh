#!/bin/bash
#SBATCH -A m4392_g
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -q shared
#SBATCH -t 01:00:00
#SBATCH -J evaluate_PAG_ParT
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 32
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
##SBATCH --mail-user=<your email>
##SBATCH --mail-type=ALL

# Evaluate a gated ParT on the test split. Submit from the project root with the model, or with the
# id of the job that trained it:
#     sbatch jobs/evaluate_PAG_ParT.sh logs/ParticleTransformer/best/<classifier run>.pt
# For a pretraining (masked) model, also pass its config:
#     sbatch jobs/evaluate_PAG_ParT.sh logs/ParticleTransformer/best/<pretrain run>.pt ./configs/pretrain_PAG_ParT.yaml
# Plots (PNG) are written to plots/ParticleTransformer/.

BEST_MODEL=${1:?"usage: sbatch jobs/evaluate_PAG_ParT.sh <best model .pt | training job id> [config .yaml]"}
CONFIG=${2:-./configs/train_PAG_ParT.yaml}
NPZ_PATH=/global/homes/t/tapur/jetclass_balanced_1M.npz

cd "$SLURM_SUBMIT_DIR"
source jobs/setup_env.sh

echo "Node list: $SLURM_NODELIST"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source jobs/lib.sh
BEST_MODEL=$(best_model "$BEST_MODEL") || exit 1
echo "Model to evaluate: $BEST_MODEL"

srun --unbuffered --export=ALL python -m scripts.evaluate_ParT \
    --config-path "$CONFIG" \
    --best-model-path "$BEST_MODEL" \
    --npz-path "$NPZ_PATH"
