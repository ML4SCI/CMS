#!/bin/bash
#SBATCH -A m4392
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -q regular
#SBATCH -t 36:00:00
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 4
#SBATCH --cpus-per-task 128
#SBATCH --image=docker:thanh12273203/gsoc25_cms:latest
#SBATCH --output=/pscratch/sd/t/thanh/logs/slurm-%j.out
#SBATCH --error=/pscratch/sd/t/thanh/logs/slurm-%j.out
#SBATCH --mail-user=tpnguyen8@crimson.ua.edu
#SBATCH --mail-type=ALL

echo "Node list: $SLURM_NODELIST"
nvidia-smi || true

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=INFO

srun --unbuffered --export=ALL shifter python -m scripts.train_ParT \
    --seed 43 \
    --config-path ./configs/pretrain_ParT.yaml \
    --train-data-dir ./data/train_100M \
    --val-data-dir ./data/val_5M