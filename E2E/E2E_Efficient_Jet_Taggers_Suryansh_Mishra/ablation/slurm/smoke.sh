#!/bin/bash
# Validate the pipeline on Perlmutter before spending allocation on real runs.
#
# Runs every arm for ~60 steps on one GPU node in the debug QOS. This exercises
# the same code paths a multi-day job uses -- data loading, DDP over 4 GPUs, AMP,
# evaluation, checkpointing and resume -- so a configuration mistake costs a few
# minutes instead of a day. After each arm's 60-step run, a second invocation
# with max_steps=2 reloads last.pt so resume (Lookahead, sampler skip, scaler)
# is actually executed.
#
#   export NERSC_ACCOUNT=m1234
#   export JETCLASS_PT_DIR=$PSCRATCH/jetclass/pt_ragged
#   sbatch -A ${NERSC_ACCOUNT}_g ablation/slurm/smoke.sh
#
# Or interactively, which is usually faster to iterate on:
#   salloc -A ${NERSC_ACCOUNT}_g -C gpu -q interactive -t 00:30:00 -N 1 --gpus 4
#   ./ablation/slurm/smoke.sh

#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --job-name=part-smoke
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

# Resolve normalization stats once: NORM_STATS, else NORM_STATS_PATH, else the
# conventional location next to the split dirs.  Fail fast if the file is
# missing -- smoke must never pass on raw features while production requires
# norms, or a green smoke hides a NormStats.load crash a day into the run.
NORM_STATS="${NORM_STATS:-${NORM_STATS_PATH:-${JETCLASS_PT_DIR}/norm_stats.json}}"
if [ ! -f "$NORM_STATS" ]; then
  echo "error: norm stats not found at $NORM_STATS" >&2
  echo "  Export NORM_STATS (or NORM_STATS_PATH) or generate it with:" >&2
  echo "    python -m preprocessing.compute_norm_stats \\" >&2
  echo "        --pt-dir ${TRAIN_PT_DIR} --output ${JETCLASS_PT_DIR}/norm_stats.json" >&2
  exit 1
fi
NORM_SET="--set norm_stats_path=${NORM_STATS}"

# Exercise the production kernel path (fused Triton PairEmbed) unless told
# otherwise; a smoke that only passes on stock weaver math would be a false
# green light for the real runs. Space-free USE_PART_KERNELS matches submit_all.
USE_PART_KERNELS="${USE_PART_KERNELS:-true}"
case "${USE_PART_KERNELS}" in
  1|true|TRUE|yes|YES) KERNEL_SET="--set use_part_kernels=true" ;;
  *) KERNEL_SET="--set use_part_kernels=false" ;;
esac
USE_COMPILE="${USE_COMPILE:-false}"
case "${USE_COMPILE}" in
  1|true|TRUE|yes|YES) COMPILE_SET="--set compile_model=true" ;;
  *) COMPILE_SET="--set compile_model=false" ;;
esac

export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n 1)
export MASTER_PORT=$((29500 + ${SLURM_JOB_ID:-0} % 20000))
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export SLURM_CPU_BIND=cores

OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/part_ablation/smoke}"
rm -rf "$OUTPUT_DIR"

echo "norm_stats   : ${NORM_STATS}"
echo "use_part_kernels : ${USE_PART_KERNELS}"
echo "use_compile  : ${USE_COMPILE}"

failed=""
for arm in baseline lloca moe sparsemax diff_v1 diff_v2; do
  echo ""
  echo "################ smoke: $arm ################"
  # LLoCa must stay in fp32; the config layer enforces it, this makes it explicit.
  extra=""
  [ "$arm" = "lloca" ] && extra="--set precision=fp32"

  srun --cpu-bind=cores python -m ablation.train \
      --config ablation/configs/smoke.yaml \
      --set "arm=${arm}" \
      --set "run_name=${arm}" \
      --set "train_pt_dir=${TRAIN_PT_DIR}" \
      --set "val_pt_dir=${VAL_PT_DIR}" \
      --set "output_dir=${OUTPUT_DIR}" \
      ${NORM_SET} ${extra} ${KERNEL_SET} ${COMPILE_SET}

  if [ $? -ne 0 ]; then
    failed="$failed $arm"
    echo "!!!! $arm FAILED"
  elif [ ! -f "${OUTPUT_DIR}/${arm}/last.pt" ]; then
    failed="$failed ${arm}(ckpt)"
    echo "!!!! $arm produced no last.pt"
  else
    echo "################ smoke resume: $arm ################"
    srun --cpu-bind=cores python -m ablation.train \
        --config ablation/configs/smoke.yaml \
        --set "arm=${arm}" \
        --set "run_name=${arm}" \
        --set "train_pt_dir=${TRAIN_PT_DIR}" \
        --set "val_pt_dir=${VAL_PT_DIR}" \
        --set "output_dir=${OUTPUT_DIR}" \
        --set "max_steps=2" \
        ${NORM_SET} ${extra} ${KERNEL_SET} ${COMPILE_SET}
    if [ $? -ne 0 ]; then
      failed="$failed ${arm}(resume)"
      echo "!!!! $arm resume FAILED"
    fi
  fi
done

echo ""
echo "=========================================================="
if [ -n "$failed" ]; then
  echo "FAILED arms:$failed"
  echo "Fix these before running ablation/slurm/submit_all.sh."
  exit 1
fi
echo "all six arms completed; checkpoints under $OUTPUT_DIR"
echo "safe to submit the real runs:  ./ablation/slurm/submit_all.sh"
