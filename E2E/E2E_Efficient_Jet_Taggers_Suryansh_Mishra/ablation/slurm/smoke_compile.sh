#!/bin/bash
# Smoke test with torch.compile enabled (plus part_kernels).
#
# Exercises compile + fused PairEmbed on all six arms at smoke scale (~60 steps).
# First step per arm pays a compile warmup; expect several extra minutes vs smoke.sh.
#
# Run from ml4sci_26/ (same as smoke.sh):
#   export JETCLASS_PT_DIR=$PSCRATCH/jetclass/pt_ragged
#   sbatch -A m4392_g ablation/slurm/smoke_compile.sh
#
# Or: USE_COMPILE=true ./ablation/slurm/smoke.sh

#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:25:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --gpu-bind=none
#SBATCH --job-name=part-smoke-compile
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

export USE_PART_KERNELS="${USE_PART_KERNELS:-true}"
export USE_COMPILE=true
export OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/part_ablation/smoke_compile}"

# Slurm copies only this script to the spool dir — smoke.sh is not copied with
# it, so resolve via SLURM_SUBMIT_DIR (cwd when sbatch was invoked).
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
SMOKE_SH="${SUBMIT_DIR}/ablation/slurm/smoke.sh"
if [ ! -f "$SMOKE_SH" ]; then
  echo "error: smoke.sh not found at ${SMOKE_SH}" >&2
  echo "  Run sbatch from ml4sci_26/:  sbatch ablation/slurm/smoke_compile.sh" >&2
  exit 1
fi
exec bash "$SMOKE_SH"
