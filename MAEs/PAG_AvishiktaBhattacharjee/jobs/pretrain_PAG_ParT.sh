#!/bin/bash
#SBATCH -A m4392_g
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -q shared
#SBATCH -t 36:00:00
#SBATCH -J pretrain_PAG_ParT
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 32
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
##SBATCH --mail-user=<your email>
##SBATCH --mail-type=ALL

# Self-supervised pretraining of the gated ParT. Submit from the project root:
#     sbatch jobs/pretrain_PAG_ParT.sh
# The log ends with "Best model saved to: logs/ParticleTransformer/best/<run>.pt", the input of train_PAG_ParT.sh.

NPZ_PATH=/global/homes/t/tapur/jetclass_balanced_1M.npz

cd "$SLURM_SUBMIT_DIR"
source jobs/setup_env.sh

echo "Node list: $SLURM_NODELIST"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun --unbuffered --export=ALL python -m scripts.train_ParT \
    --seed 42 \
    --config-path ./configs/pretrain_PAG_ParT.yaml \
    --npz-path "$NPZ_PATH"
