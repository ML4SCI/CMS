#!/bin/bash
# Submit the two T0 cheap-kill screen jobs (HERON §0.1) — independent, no chaining.
#
#   export NERSC_ACCOUNT=m1234
#   export JETCLASS_PT_DIR=$PSCRATCH/jetclass/pt_ragged
#   ./ablation/slurm/submit_t0.sh
#
# Two 12h 4xA100 jobs, one per arm, both capped at 200000 steps by their
# configs (the LR schedule still spans the 1M-step recipe, so steps are
# directly comparable to the existing `baseline` run's evals at 150k/175k/200k):
#
#   T0.4  baseline_nopair  pair bias off (pair_embed_dims=null), kernels off
#   C/A   baseline_ca_t0   train-only C/A coarsening, fused kernels on
#
# The C/A screen runs under baseline_ca_t0, NOT baseline_ca: a pre-existing
# chained baseline_ca job series (submitted before this plan) owns the
# baseline_ca run dir, and two writers would corrupt each other's resumes.
#
# No --dependency chain: each screen is capped at 200k steps and must fit
# one 12h allocation. Lookahead resume was fixed in 5068b99, but a single
# job still avoids mixing this screen with the older CA chain's last.pt.
#
# Options
#   -t HH:MM:SS      walltime per job (default 12:00:00)
#   -q QOS           Slurm QOS (default regular)
#   -n               dry run: print the sbatch commands and exit

set -euo pipefail

cd "$(dirname "$0")/../.."
PACKAGE_DIR="$PWD"

WALLTIME="12:00:00"
QOS="regular"
DRY_RUN=0

while getopts "t:q:nh" opt; do
  case "$opt" in
    t) WALLTIME="$OPTARG" ;;
    q) QOS="$OPTARG" ;;
    n) DRY_RUN=1 ;;
    h) sed -n '2,34p' "$0"; exit 0 ;;
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

echo "account      : ${NERSC_ACCOUNT}_g"
echo "train_pt_dir : $TRAIN_PT_DIR"
echo "val_pt_dir   : $VAL_PT_DIR"
echo "norm_stats   : $NORM_STATS"
echo "output       : $OUTPUT_DIR"
echo "walltime/qos : ${WALLTIME} / ${QOS}"
echo ""

# arm | config | kernels | description
# baseline_nopair config sets use_part_kernels=false itself; exporting it here
# stops train_arm.sh's default `--set use_part_kernels=true` from overriding.
JOBS="
baseline_nopair|ablation/configs/baseline_nopair.yaml|false|T0.4 pair bias off
baseline_ca_t0|ablation/configs/baseline_ca.yaml|true|C/A train-only coarsening
"

while IFS='|' read -r run_name config kernels label; do
  [ -z "$run_name" ] && continue
  if [ ! -f "$PACKAGE_DIR/$config" ]; then
    echo "error: config $config not found" >&2
    exit 1
  fi

  args=(
    --account "${NERSC_ACCOUNT}_g"
    --qos "$QOS"
    --time "$WALLTIME"
    --job-name "part-t0-${run_name#baseline_}"
    --export "ALL,ARM=baseline,RUN_NAME=${run_name},CONFIG=${config},TRAIN_PT_DIR=${TRAIN_PT_DIR},VAL_PT_DIR=${VAL_PT_DIR},OUTPUT_DIR=${OUTPUT_DIR},PACKAGE_DIR=${PACKAGE_DIR},NORM_STATS=${NORM_STATS},USE_PART_KERNELS=${kernels},USE_COMPILE=false"
  )

  echo "submit: ${run_name}  (${label})"
  echo "  kernels=${kernels}  config=${config}  walltime=${WALLTIME} qos=${QOS}"
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  [dry run] sbatch ${args[*]} ablation/slurm/train_arm.sh"
    continue
  fi

  job_id=$(sbatch --parsable "${args[@]}" ablation/slurm/train_arm.sh)
  echo "  -> job $job_id"
done <<< "$JOBS"

if [ "$DRY_RUN" -eq 0 ]; then
  echo ""
  echo "monitor with:  squeue --me"
  echo "results in  :  $OUTPUT_DIR/{baseline_nopair,baseline_ca_t0}/metrics.jsonl"
  echo "compare with:  python -m ablation.report --runs $OUTPUT_DIR"
fi
