#!/bin/bash
#SBATCH -A m4392_g
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -t 36:00:00
#SBATCH -J train_ParT
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 4
#SBATCH --cpus-per-task 128
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.out
##SBATCH --mail-user=<your email>
##SBATCH --mail-type=ALL

cd "$SLURM_SUBMIT_DIR"
source jobs/setup_env.sh

echo "Node list: $SLURM_NODELIST"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO

srun --unbuffered --export=ALL python -m scripts.train_ParT \
    --seed 49 \
    --config-path ./configs/train_ParT.yaml \
    --train-data-dir ./data/train_100M \
    --val-data-dir ./data/val_5M
