#!/bin/bash
# Submit the N8 K6 screen (RoPE + rank-4 Minkowski, PairEmbed off) — one job, no chain.
#
#   export NERSC_ACCOUNT=m4392
#   export JETCLASS_PT_DIR=$PSCRATCH/jetclass/pt_ragged
#   ./ablation/slurm/submit_n8.sh
#
# One 12h 4xA100 job, capped at 200000 steps by n8_k6.yaml (the LR schedule
# still spans the 1M-step recipe, so steps are directly comparable to the
# existing `baseline` and `baseline_nopair` evals at 150k/175k/200k).
#
# Leave the running `part-baseline-ca` and `part-moe-e8` chains untouched.
#
# 60-step smoke (debug QOS, tiny model, kernels off):
#   SMOKE=1 ./ablation/slurm/submit_n8.sh -q debug -t 00:25:00
#
# Options
#   -t HH:MM:SS      walltime per job (default 12:00:00)
#   -q QOS           Slurm QOS (default regular)
#   -n               dry run: print the sbatch commands and exit

set -euo pipefail

cd "$(dirname "$0")/../.."
PACKAGE_DIR="$PWD"

WALLTIME=""
QOS="regular"
DRY_RUN=0

while getopts "t:q:nh" opt; do
  case "$opt" in
    t) WALLTIME="$OPTARG" ;;
    q) QOS="$OPTARG" ;;
    n) DRY_RUN=1 ;;
    h) sed -n '2,26p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done

# --- validate inputs before spending anything -----------------------------
if [ -z "${NERSC_ACCOUNT:-}" ]; then
  echo "error: NERSC_ACCOUNT is not set." >&2
  echo "  Then:  export NERSC_ACCOUNT=m1234   # no _g suffix" >&2
  exit 1
fi

if [ -z "${JETCLASS_PT_DIR:-}" ]; then
  echo "error: JETCLASS_PT_DIR is not set." >&2
  echo "  export JETCLASS_PT_DIR=\$PSCRATCH/jetclass/pt_ragged" >&2
  exit 1
fi

TRAIN_PT_DIR="${TRAIN_PT_DIR:-${JETCLASS_PT_DIR}/train_100M}"
VAL_PT_DIR="${VAL_PT_DIR:-${JETCLASS_PT_DIR}/val_5M}"
NORM_STATS="${NORM_STATS:-${NORM_STATS_PATH:-${JETCLASS_PT_DIR}/norm_stats.json}}"
if [ ! -f "$NORM_STATS" ]; then
  echo "error: norm stats not found at $NORM_STATS" >&2
  exit 1
fi
for split_dir in "$TRAIN_PT_DIR" "$VAL_PT_DIR"; do
  if [ ! -d "$split_dir" ]; then
    echo "error: missing directory $split_dir" >&2
    exit 1
  fi
done

OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/part_ablation/runs}"
mkdir -p logs "$OUTPUT_DIR"

if [ "${SMOKE:-0}" = "1" ]; then
  RUN_NAME="${RUN_NAME:-n8_k6_smoke}"
  CONFIG="ablation/configs/smoke.yaml"
  WALLTIME="${WALLTIME:-00:25:00}"
else
  RUN_NAME="${RUN_NAME:-n8_k6_v2}"
  CONFIG="${CONFIG:-ablation/configs/n8_k6.yaml}"
  WALLTIME="${WALLTIME:-12:00:00}"
fi

if [ ! -f "$PACKAGE_DIR/$CONFIG" ]; then
  echo "error: config $CONFIG not found" >&2
  exit 1
fi

echo "account      : ${NERSC_ACCOUNT}_g"
echo "arm          : n8"
echo "run_name     : $RUN_NAME"
echo "config       : $CONFIG"
echo "train_pt_dir : $TRAIN_PT_DIR"
echo "val_pt_dir   : $VAL_PT_DIR"
echo "norm_stats   : $NORM_STATS"
echo "output       : $OUTPUT_DIR"
echo "walltime/qos : ${WALLTIME} / ${QOS}"
echo "kernels      : false"
echo ""

args=(
  --account "${NERSC_ACCOUNT}_g"
  --qos "$QOS"
  --time "$WALLTIME"
  --job-name "part-n8-k6"
  --export "ALL,ARM=n8,RUN_NAME=${RUN_NAME},CONFIG=${CONFIG},TRAIN_PT_DIR=${TRAIN_PT_DIR},VAL_PT_DIR=${VAL_PT_DIR},OUTPUT_DIR=${OUTPUT_DIR},PACKAGE_DIR=${PACKAGE_DIR},NORM_STATS=${NORM_STATS},USE_PART_KERNELS=false,USE_COMPILE=false"
)

echo "submit: ${RUN_NAME}  (N8 K6 rotary + Minkowski)"
echo "  kernels=false  config=${CONFIG}  walltime=${WALLTIME} qos=${QOS}"
if [ "$DRY_RUN" -eq 1 ]; then
  echo "  [dry run] sbatch ${args[*]} ablation/slurm/train_arm.sh"
else
  job_id=$(sbatch --parsable "${args[@]}" ablation/slurm/train_arm.sh)
  echo "  -> job $job_id"
  echo ""
  echo "monitor with:  squeue --me"
  echo "results in  :  $OUTPUT_DIR/${RUN_NAME}/metrics.jsonl"
  echo "compare with:  python -m ablation.plot_metrics --runs $OUTPUT_DIR --arm n8_k6_v2 --arm n8_k6 --arm baseline --arm baseline_nopair"
fi
