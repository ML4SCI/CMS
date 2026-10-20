#!/bin/bash
# Verify the part_kernels-optimized model trains as close to the pristine
# baseline as floating-point noise allows.
#
# Runs arm=baseline twice on one 4-GPU debug node -- once with the stock weaver
# math (use_part_kernels=false) and once with the fused Triton PairEmbed
# (use_part_kernels=true) -- then diffs the per-step loss/accuracy trajectories.
# Both runs share the same seed, config, data order and dropout masks, so the
# only source of divergence is the kernels' numerics.  The final
# ablation.parity_check PASSes/FAILs and the job exits accordingly.
#
#   export NERSC_ACCOUNT=m1234
#   export JETCLASS_PT_DIR=$PSCRATCH/jetclass/pt_ragged
#   sbatch -A ${NERSC_ACCOUNT}_g ablation/slurm/smoke_parity.sh
#
# Or interactively:
#   salloc -A ${NERSC_ACCOUNT}_g -C gpu -q interactive -t 00:30:00 -N 1 --gpus 4
#   ./ablation/slurm/smoke_parity.sh

#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --job-name=part-parity
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -uo pipefail

: "${JETCLASS_PT_DIR:?set JETCLASS_PT_DIR to the ragged .pt base directory (containing train_100M/ and val_5M/)}"

TRAIN_PT_DIR="${TRAIN_PT_DIR:-${JETCLASS_PT_DIR}/train_100M}"
VAL_PT_DIR="${VAL_PT_DIR:-${JETCLASS_PT_DIR}/val_5M}"

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p logs

module load "${PYTORCH_MODULE:-pytorch}"

VENV="${VENV:-$(dirname "$PWD")/.venv_train}"
if [ -f "$VENV/bin/activate" ]; then
  source "$VENV/bin/activate"
  export PYTHONPATH="$(dirname "$PWD"):$PWD"
fi

# Both runs must see the same normalized features; fail fast if absent (same
# resolution as smoke.sh / submit_all.sh).
NORM_STATS="${NORM_STATS:-${NORM_STATS_PATH:-${JETCLASS_PT_DIR}/norm_stats.json}}"
if [ ! -f "$NORM_STATS" ]; then
  echo "error: norm stats not found at $NORM_STATS" >&2
  exit 1
fi
NORM_SET="--set norm_stats_path=${NORM_STATS}"

export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n 1)
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 20000))
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export SLURM_CPU_BIND=cores

OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/part_ablation/smoke_parity}"
rm -rf "$OUTPUT_DIR"

echo "norm_stats   : ${NORM_STATS}"
echo "output       : ${OUTPUT_DIR}"
echo ""

run_arm() {
  local run_name="$1"
  shift
  srun --cpu-bind=cores python -m ablation.train \
      --config ablation/configs/smoke.yaml \
      --set "arm=baseline" \
      --set "run_name=${run_name}" \
      --set "train_pt_dir=${TRAIN_PT_DIR}" \
      --set "val_pt_dir=${VAL_PT_DIR}" \
      --set "output_dir=${OUTPUT_DIR}" \
      ${NORM_SET} "$@"
}

echo "################ parity: pristine baseline ################"
run_arm baseline --set use_part_kernels=false || exit 1

echo ""
echo "################ parity: part_kernels baseline ################"
run_arm baseline_part --set use_part_kernels=true || exit 1

echo ""
echo "################ parity: comparing trajectories ################"
python -m ablation.parity_check \
    --base "${OUTPUT_DIR}/baseline" \
    --optimized "${OUTPUT_DIR}/baseline_part"
