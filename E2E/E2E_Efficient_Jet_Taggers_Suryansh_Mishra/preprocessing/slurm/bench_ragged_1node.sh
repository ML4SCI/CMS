#!/bin/bash
# Ragged DataLoader benchmark — 1 node, 4×A100 (Perlmutter).
#
# Runs flat-shuffle (A) and shard-coherent (B) modes, prints jets/s comparison.
#
# DDP topology is read from Slurm by ablation/distributed.py (same pattern
# as ablation/slurm/train_arm.sh).
#
#   sbatch -A m4392_g preprocessing/slurm/bench_ragged_1node.sh
#
# Perlmutter GPU node: 1× AMD EPYC 7763, 4× A100-80GB.

#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --job-name=bench-ragged-1n
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail

PT_DIR="${PT_DIR:-/pscratch/sd/o/omasho/jetclass/pt_ragged/train_100M}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_BATCHES="${NUM_BATCHES:-200}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PACKAGE_DIR="${PACKAGE_DIR:-$SLURM_SUBMIT_DIR}"

cd "$PACKAGE_DIR"
mkdir -p logs

# --- environment ----------------------------------------------------------
module load pytorch

export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 20000))

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export SLURM_CPU_BIND=cores
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "=========================================================="
echo "job          : ${SLURM_JOB_ID}  (${SLURM_JOB_NAME})"
echo "nodes/tasks  : ${SLURM_NNODES} / ${SLURM_NTASKS}"
echo "pt_dir       : ${PT_DIR}"
echo "batch_size   : ${BATCH_SIZE}"
echo "num_batches  : ${NUM_BATCHES}"
echo "master       : ${MASTER_ADDR}:${MASTER_PORT}"
echo "started      : $(date -Is)"
echo "=========================================================="

# Step 1: flat-shuffle baseline (single process, no DDP contention).
# Running flat under srun would penalise it with N-way Lustre contention
# that coherent avoids by design — mixing sampler quality with I/O noise.
# The bench script also enforces this (rank-0 only) but running flat
# outside srun is cleaner.
python preprocessing/bench_ragged_loader.py \
    --pt-dir "$PT_DIR" \
    --modes flat \
    --batch-size "$BATCH_SIZE" \
    --num-batches "$NUM_BATCHES" \
    --num-workers "$NUM_WORKERS"

echo "--- flat baseline done, starting DDP coherent + cuda ---"

# Step 2: coherent + cuda (4 tasks, DDP partitioned shards).
srun --cpu-bind=cores python preprocessing/bench_ragged_loader.py \
    --pt-dir "$PT_DIR" \
    --modes coherent cuda \
    --batch-size "$BATCH_SIZE" \
    --num-batches "$NUM_BATCHES" \
    --num-workers "$NUM_WORKERS" \
    --distributed

echo "finished     : $(date -Is)"
