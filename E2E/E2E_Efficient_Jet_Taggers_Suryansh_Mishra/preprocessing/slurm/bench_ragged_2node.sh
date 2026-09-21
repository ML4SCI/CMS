#!/bin/bash
# Ragged DataLoader smoke + bench — 2 nodes, 8×A100 (Perlmutter).
#
# Validates multi-node DDP: shard-coherent loading on 8 ranks (2 nodes × 4
# GPUs), NCCL allreduce of a dummy tensor to prove internode connectivity,
# and a shard partition overlap check.
#
# DDP topology is read from Slurm by ablation/distributed.py (same pattern
# as ablation/slurm/train_arm.sh):
#   RANK=SLURM_PROCID, LOCAL_RANK=SLURM_LOCALID, WORLD_SIZE=SLURM_NTASKS,
#   MASTER_ADDR from scontrol, NCCL backend.
#
#   sbatch -A m4392_g preprocessing/slurm/bench_ragged_2node.sh
#
# Success bars (locked):
#   1. All 8 ranks complete ≥1 coherent epoch-slice without hang.
#   2. NCCL allreduce succeeds.
#   3. No two ranks share the same shard ID in a given epoch partition.

#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --job-name=bench-ragged-2n
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail

PT_DIR="${PT_DIR:-/pscratch/sd/o/omasho/jetclass/pt_ragged/train_100M}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_BATCHES="${NUM_BATCHES:-100}"
NUM_WORKERS="${NUM_WORKERS:-2}"
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
echo "num_workers  : ${NUM_WORKERS}"
echo "master       : ${MASTER_ADDR}:${MASTER_PORT}"
echo "started      : $(date -Is)"
echo "=========================================================="

# --- coherent bench + shard partition check + cuda smoke ---
# The bench script auto-detects DDP from SLURM_NTASKS and initialises
# via ablation.distributed.setup_distributed (NCCL backend).
# The built-in shard overlap check uses all_gather_object to verify
# no two ranks share the same shard ID.
srun --cpu-bind=cores python preprocessing/bench_ragged_loader.py \
    --pt-dir "$PT_DIR" \
    --modes coherent cuda \
    --batch-size "$BATCH_SIZE" \
    --num-batches "$NUM_BATCHES" \
    --num-workers "$NUM_WORKERS" \
    --distributed

echo "finished     : $(date -Is)"
