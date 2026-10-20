#!/bin/bash
# Submit the full six-arm ParT ablation on NERSC Perlmutter.
#
#   export NERSC_ACCOUNT=m1234              # without the _g suffix
#   export JETCLASS_PT_DIR=$PSCRATCH/jetclass/pt_ragged
#   ./ablation/slurm/submit_all.sh
#
# Options
#   -a "arm1 arm2"   arms to submit (default: all six)
#   -c N             chain N dependent jobs per arm (default 6)
#   -t HH:MM:SS      walltime per job (default 12:00:00)
#   -q QOS           Slurm QOS (default regular)
#   -n               dry run: print the sbatch commands and exit
#
# Chaining exists because the run does not fit in one allocation: ParT's recipe
# is 1e6 steps, roughly 2-3 days on 4x A100. Each job in a chain resumes from
# last.pt, and they are submitted with --dependency=afterany so a requeue or a
# timeout does not break the chain.

set -euo pipefail

cd "$(dirname "$0")/../.."
PACKAGE_DIR="$PWD"

ALL_ARMS="baseline lloca moe sparsemax diff_v1 diff_v2"
ARMS="$ALL_ARMS"
CHAIN=6
WALLTIME="12:00:00"
QOS="regular"
DRY_RUN=0

while getopts "a:c:t:q:nh" opt; do
  case "$opt" in
    a) ARMS="$OPTARG" ;;
    c) CHAIN="$OPTARG" ;;
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
  echo "  Find yours with:  iris  (or see https://iris.nersc.gov)" >&2
  echo "  Then:  export NERSC_ACCOUNT=m1234   # no _g suffix" >&2
  exit 1
fi

if [ -z "${JETCLASS_PT_DIR:-}" ]; then
  echo "error: JETCLASS_PT_DIR is not set." >&2
  echo "  Convert ROOT to ragged .pt shards, then point at them:" >&2
  echo "    python -m preprocessing.convert_jetclass_ragged_pt \\" >&2
  echo "        --root-dir \$SCRATCH/jetclass --out-dir \$PSCRATCH/jetclass/pt_ragged" >&2
  echo "    export JETCLASS_PT_DIR=\$PSCRATCH/jetclass/pt_ragged" >&2
  exit 1
fi

TRAIN_PT_DIR="${TRAIN_PT_DIR:-${JETCLASS_PT_DIR}/train_100M}"
VAL_PT_DIR="${VAL_PT_DIR:-${JETCLASS_PT_DIR}/val_5M}"
NORM_STATS="${NORM_STATS:-${NORM_STATS_PATH:-${JETCLASS_PT_DIR}/norm_stats.json}}"
if [ ! -f "$NORM_STATS" ]; then
  echo "error: norm stats not found at $NORM_STATS" >&2
  echo "  Export NORM_STATS or NORM_STATS_PATH, or generate it with:" >&2
  echo "    python -m preprocessing.compute_norm_stats \\" >&2
  echo "        --pt-dir $TRAIN_PT_DIR --output $JETCLASS_PT_DIR/norm_stats.json" >&2
  exit 1
fi
# Space-free flags for Slurm --export (values with spaces get truncated).
# Override with: USE_PART_KERNELS=false USE_COMPILE=true ./ablation/slurm/submit_all.sh
USE_PART_KERNELS="${USE_PART_KERNELS:-true}"
USE_COMPILE="${USE_COMPILE:-false}"

for split_dir in "$TRAIN_PT_DIR" "$VAL_PT_DIR"; do
  if [ ! -d "$split_dir" ]; then
    echo "error: missing directory $split_dir" >&2
    echo "  Run preprocessing/convert_jetclass_ragged_pt.py first (see -h)." >&2
    exit 1
  fi
done

for arm in $ARMS; do
  case " $ALL_ARMS " in
    *" $arm "*) ;;
    *) echo "error: unknown arm '$arm'; expected from: $ALL_ARMS" >&2; exit 1 ;;
  esac
done

OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH/part_ablation/runs}"
mkdir -p logs "$OUTPUT_DIR"

echo "account      : ${NERSC_ACCOUNT}_g"
echo "train_pt_dir : $TRAIN_PT_DIR"
echo "val_pt_dir   : $VAL_PT_DIR"
echo "norm_stats   : $NORM_STATS"
echo "use_part_kernels : $USE_PART_KERNELS"
echo "use_compile  : $USE_COMPILE"
echo "output       : $OUTPUT_DIR"
echo "arms         : $ARMS"
echo "chain        : $CHAIN job(s) per arm, ${WALLTIME} each, qos=${QOS}"
echo ""

for arm in $ARMS; do
  previous=""
  for link in $(seq 1 "$CHAIN"); do
    args=(
      --account "${NERSC_ACCOUNT}_g"
      --qos "$QOS"
      --time "$WALLTIME"
      --job-name "part-${arm}"
      --export "ALL,ARM=${arm},TRAIN_PT_DIR=${TRAIN_PT_DIR},VAL_PT_DIR=${VAL_PT_DIR},OUTPUT_DIR=${OUTPUT_DIR},PACKAGE_DIR=${PACKAGE_DIR},NORM_STATS=${NORM_STATS},USE_PART_KERNELS=${USE_PART_KERNELS},USE_COMPILE=${USE_COMPILE}"
    )
    # afterany, not afterok: a job that hits the walltime exits non-zero but has
    # still written last.pt, and the chain must continue from it.
    [ -n "$previous" ] && args+=(--dependency "afterany:${previous}")

    if [ "$DRY_RUN" -eq 1 ]; then
      echo "sbatch ${args[*]} ablation/slurm/train_arm.sh"
      previous="<job${link}>"
      continue
    fi

    job_id=$(sbatch --parsable "${args[@]}" ablation/slurm/train_arm.sh)
    printf "  %-10s link %d/%s -> job %s%s\n" \
      "$arm" "$link" "$CHAIN" "$job_id" \
      "$([ -n "$previous" ] && echo " (after $previous)")"
    previous="$job_id"
  done
done

if [ "$DRY_RUN" -eq 0 ]; then
  echo ""
  echo "monitor with:  squeue --me"
  echo "results in  :  $OUTPUT_DIR/<arm>/metrics.jsonl"
  echo "compare with:  python -m ablation.report --runs $OUTPUT_DIR"
fi
