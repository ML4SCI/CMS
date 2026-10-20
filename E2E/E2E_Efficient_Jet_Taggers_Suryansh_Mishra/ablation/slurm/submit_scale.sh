#!/bin/bash
# Submit the Wave 0 dense-capacity trio (FFN 2× / PairEmbed 2× / attention width).
#
#   export NERSC_ACCOUNT=m4392
#   export JETCLASS_PT_DIR=$PSCRATCH/jetclass/pt_ragged
#   ./ablation/slurm/submit_scale.sh
#
# Three independent 12h 4xA100 jobs, each capped at 200000 steps. The LR
# schedule still spans the 1M-step recipe, so evals line up with baseline /
# baseline_nopair at 150k / 175k / 200k.
#
# Leave n8_k6, n8_k6_v2, baseline_ca, and moe_e8_top2 run dirs untouched.
# Do not submit K2. Do not port Kimi K3.
#
# GATE.txt is written next to each empty run dir *before* sbatch.
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
    h) sed -n '2,28p' "$0"; exit 0 ;;
    *) exit 2 ;;
  esac
done

if [ -z "${NERSC_ACCOUNT:-}" ]; then
  echo "error: NERSC_ACCOUNT is not set." >&2
  echo "  Then:  export NERSC_ACCOUNT=m4392   # no _g suffix" >&2
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

# run_name | config | short label
# ONLY=baseline_ffn2x ./ablation/slurm/submit_scale.sh  submits one arm.
JOBS="
baseline_ffn2x|ablation/configs/baseline_ffn2x.yaml|FFN expansion_factor 8
baseline_pair2x|ablation/configs/baseline_pair2x.yaml|PairEmbed [128,128,128]
baseline_wide|ablation/configs/baseline_wide.yaml|embed_dims [160,640,160]
"
ONLY="${ONLY:-}"

write_gate() {
  local run_name="$1"
  local label="$2"
  local dest="$OUTPUT_DIR/$run_name"
  mkdir -p "$dest"
  if [ -f "$dest/GATE.txt" ]; then
    echo "  GATE.txt already exists at $dest (leaving it)"
    return
  fi
  cat > "$dest/GATE.txt" << EOF
Wave 0 dense-capacity screen — registered before any eval
run_name: ${run_name}
what:     ${label}
arm:      baseline (stock ParT; one capacity knob)

Comparators at 200k (matched 1M LR schedule, val_5M):
  baseline        0.85453
  baseline_nopair 0.83696   (T0.4 gap 0.01757)

Three-way gate at 200k val acc vs baseline 0.85453:
  1. noise / kill     |Δacc| < 0.002          this site does not scale
  2. interesting      0.002 ≤ Δacc < 0.009    keep watching, not a U-scale effect
  3. this site scales Δacc ≥ 0.009            ≥ half the nopair gap; Wave 1 may follow

Also report params, active FLOPs, jets/s. Do not resume baseline/, moe/, n8_k6/.
Do not submit K2. Do not port Kimi K3 (KDA / MLA / AttnRes / LatentMoE).
EOF
  echo "  wrote $dest/GATE.txt"
}

submitted=0
while IFS='|' read -r run_name config label; do
  [ -z "$run_name" ] && continue
  if [ -n "$ONLY" ] && [ "$run_name" != "$ONLY" ]; then
    continue
  fi
  if [ ! -f "$PACKAGE_DIR/$config" ]; then
    echo "error: config $config not found" >&2
    exit 1
  fi
  if [ -f "$OUTPUT_DIR/$run_name/metrics.jsonl" ]; then
    echo "error: $OUTPUT_DIR/$run_name already has metrics.jsonl — pick a new run_name" >&2
    exit 1
  fi

  write_gate "$run_name" "$label"

  args=(
    --account "${NERSC_ACCOUNT}_g"
    --qos "$QOS"
    --time "$WALLTIME"
    --job-name "part-scale-${run_name#baseline_}"
    --export "ALL,ARM=baseline,RUN_NAME=${run_name},CONFIG=${config},TRAIN_PT_DIR=${TRAIN_PT_DIR},VAL_PT_DIR=${VAL_PT_DIR},OUTPUT_DIR=${OUTPUT_DIR},PACKAGE_DIR=${PACKAGE_DIR},NORM_STATS=${NORM_STATS},USE_PART_KERNELS=true,USE_COMPILE=false"
  )

  echo "submit: ${run_name}  (${label})"
  echo "  kernels=true  config=${config}  walltime=${WALLTIME} qos=${QOS}"
  submitted=$((submitted + 1))
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "  [dry run] sbatch ${args[*]} ablation/slurm/train_arm.sh"
    continue
  fi

  job_id=$(sbatch --parsable "${args[@]}" ablation/slurm/train_arm.sh)
  echo "  -> job $job_id"
done <<< "$JOBS"

if [ -n "$ONLY" ] && [ "$submitted" -eq 0 ]; then
  echo "error: ONLY=$ONLY matched no job" >&2
  exit 1
fi

if [ "$DRY_RUN" -eq 0 ]; then
  echo ""
  echo "monitor with:  squeue --me"
  if [ -n "$ONLY" ]; then
    echo "results in  :  $OUTPUT_DIR/${ONLY}/metrics.jsonl"
  else
    echo "results in  :  $OUTPUT_DIR/{baseline_ffn2x,baseline_pair2x,baseline_wide}/metrics.jsonl"
  fi
  echo "compare with:  python -m ablation.plot_metrics --runs $OUTPUT_DIR --arm baseline --arm baseline_nopair --arm baseline_ffn2x --arm baseline_pair2x --arm baseline_wide"
fi
