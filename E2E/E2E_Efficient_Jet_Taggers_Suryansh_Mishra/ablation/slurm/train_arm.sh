#!/bin/bash
# Train one ParT ablation arm on NERSC Perlmutter.
#
# One task per GPU; DDP topology is read from Slurm by ablation/distributed.py.
# Do not run this directly -- use ablation/slurm/submit_all.sh, which supplies the
# account, the arm, and the job chaining.
#
#   ARM=lloca TRAIN_PT_DIR=$PSCRATCH/jetclass/pt_ragged/train_100M \
#     VAL_PT_DIR=$PSCRATCH/jetclass/pt_ragged/val_5M \
#     sbatch -A m1234_g ablation/slurm/train_arm.sh
#
# Perlmutter GPU node: 1x AMD EPYC 7763 (64 cores / 128 threads), 4x A100.
# 4 tasks x 32 logical CPUs saturates the node without oversubscribing.

#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --requeue
#SBATCH --job-name=part-ablation
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail

# --- required inputs ------------------------------------------------------
: "${ARM:?set ARM to one of: baseline lloca moe sparsemax diff_v1 diff_v2 n8 urot}"
: "${TRAIN_PT_DIR:?set TRAIN_PT_DIR to the directory holding ragged .pt training shards}"
: "${VAL_PT_DIR:?set VAL_PT_DIR to the directory holding ragged .pt validation shards}"

PACKAGE_DIR="${PACKAGE_DIR:-$SLURM_SUBMIT_DIR}"
CONFIG="${CONFIG:-ablation/configs/${ARM}.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/part_ablation/runs}"
RUN_NAME="${RUN_NAME:-$ARM}"
PYTORCH_MODULE="${PYTORCH_MODULE:-pytorch}"
NORM_STATS="${NORM_STATS:-${NORM_STATS_PATH:-}}"
# Space-free env for Slurm --export; train_arm turns it into --set.
# Override with USE_PART_KERNELS=false (submit_all forwards this).
USE_PART_KERNELS="${USE_PART_KERNELS:-true}"
case "${USE_PART_KERNELS}" in
  1|true|TRUE|yes|YES) KERNEL_SET="--set use_part_kernels=true" ;;
  *) KERNEL_SET="--set use_part_kernels=false" ;;
esac
# torch.compile after part_kernels; off by default (dynamic P_max can recompile).
USE_COMPILE="${USE_COMPILE:-false}"
case "${USE_COMPILE}" in
  1|true|TRUE|yes|YES) COMPILE_SET="--set compile_model=true" ;;
  *) COMPILE_SET="--set compile_model=false" ;;
esac
# Optional extra --set flags for interactive/debug runs only (not via --export).
EXTRA_SET="${EXTRA_SET:-}"

cd "$PACKAGE_DIR"
mkdir -p logs "$OUTPUT_DIR"

# --- environment ----------------------------------------------------------
module load "$PYTORCH_MODULE"

VENV="${VENV:-$(dirname "$PACKAGE_DIR")/.venv_train}"
if [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
  export PYTHONPATH="$(dirname "$PACKAGE_DIR"):$PACKAGE_DIR"
fi

# One rendezvous host, and a port keyed to the job so the six arms can run
# concurrently on the same node without colliding.
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((29500 + SLURM_JOB_ID % 20000))

# Each task gets cpus-per-task threads; leaving OMP unset makes every rank
# spawn threads for the whole node and thrash.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export SLURM_CPU_BIND=cores
# Bounded, predictable allocator behaviour across the long run.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# weaver and the loader are import-heavy; keep tokenizer/BLAS threads in check.
export TOKENIZERS_PARALLELISM=false

echo "=========================================================="
echo "job          : ${SLURM_JOB_ID}  (${SLURM_JOB_NAME})"
echo "arm          : ${ARM}"
echo "config       : ${CONFIG}"
echo "nodes/tasks  : ${SLURM_NNODES} / ${SLURM_NTASKS}"
echo "train_pt_dir : ${TRAIN_PT_DIR}"
echo "val_pt_dir   : ${VAL_PT_DIR}"
echo "norm_stats   : ${NORM_STATS}"
echo "use_part_kernels : ${USE_PART_KERNELS}"
echo "use_compile  : ${USE_COMPILE}"
echo "extra_set    : ${EXTRA_SET:-<none>}"
echo "output_dir   : ${OUTPUT_DIR}"
echo "master       : ${MASTER_ADDR}:${MASTER_PORT}"
echo "started      : $(date -Is)"
echo "=========================================================="

# Build the norm_stats override if provided
NORM_SET=""
[ -n "$NORM_STATS" ] && NORM_SET="--set norm_stats_path=${NORM_STATS}"

# The trainer resumes from <output_dir>/<run_name>/last.pt automatically, so a
# requeued or chained job continues rather than restarting from scratch.
srun --cpu-bind=cores python -m ablation.train \
    --config "$CONFIG" \
    --set "arm=${ARM}" \
    --set "run_name=${RUN_NAME}" \
    --set "train_pt_dir=${TRAIN_PT_DIR}" \
    --set "val_pt_dir=${VAL_PT_DIR}" \
    --set "output_dir=${OUTPUT_DIR}" \
    ${NORM_SET} \
    ${KERNEL_SET} \
    ${COMPILE_SET} \
    ${EXTRA_SET}

echo "finished     : $(date -Is)"
