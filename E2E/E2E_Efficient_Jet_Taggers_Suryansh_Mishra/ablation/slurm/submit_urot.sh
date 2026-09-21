#!/bin/bash
# Submit the U-as-rotation screen — PairEmbed on, U consumed as a pairwise
# rotation instead of an additive logit. Inverse of K6; not K2.
#
#   export NERSC_ACCOUNT=m4392
#   export JETCLASS_PT_DIR=$PSCRATCH/jetclass/pt_ragged
#   ./ablation/slurm/submit_urot.sh
#
# One 12h 4xA100 job, capped at 200000 steps by urot.yaml (the LR schedule
# still spans the 1M-step recipe, so steps line up with baseline /
# baseline_nopair / n8_k6_v2 at 150k/175k/200k).
#
# Leave Wave 0 (ffn2x / pair2x / wide), n8_k6, n8_k6_v2, baseline_ca,
# moe_e8_top2, and the in-flight 1-θ urot (57410779) run dirs untouched.
# Default submits urot_rope (RoPE apply from pooled U). Do not submit K2.
#
# GATE.txt is written next to the empty run dir *before* sbatch.
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

# Default is RoPE apply (pool U → per-token phase, rotate Q/K, QK^T).
# The pairwise fused 1-θ d=128 job is already 57410779 (runs/urot).
# Do not resubmit that dir.
#   ONLY=urot      ./ablation/slurm/submit_urot.sh  # fused θ_ij (in flight)
#   ONLY=urot_rope ./ablation/slurm/submit_urot.sh  # default
#   ONLY=urot_mf   ./ablation/slurm/submit_urot.sh  # per-plane + width (failed)
ONLY="${ONLY:-urot_rope}"
case "$ONLY" in
  urot)
    RUN_NAME="${RUN_NAME:-urot}"
    CONFIG="${CONFIG:-ablation/configs/urot.yaml}"
    GATE_WHAT="PairEmbed on; pairwise fused θ_ij; embed 128 (d_head=16). Not apply-then-matmul."
    GATE_HOW="urot            logits = [(Q·K) cos θ + (Q⋆K) sin θ]/√d
                  θ = π tanh(α_h U_ij)   (pairwise; cannot pre-rotate Q)"
    ;;
  urot_rope)
    RUN_NAME="${RUN_NAME:-urot_rope}"
    CONFIG="${CONFIG:-ablation/configs/urot_rope.yaml}"
    GATE_WHAT="PairEmbed on; RoPE apply. ψ_i = α · mean_j U_ij; Q_i←R(ψ_i)Q_i, K_j←R(ψ_j)K_j, then QK^T. LLaMA ω_p = 10000^{-2p/d}. Same width as urot. Rank-1 factorization of U (T0.1 p90=20)."
    GATE_HOW="urot_rope       ψ_i = α_h mean_j U_ij
                  Q,K = apply_rope(Q, K, ψ)   (LLaMA rotate_half + geometric freqs)
                  logits = QK^T/√d
                  relative structure is ψ_j − ψ_i, not U_ij"
    ;;
  urot_mf)
    RUN_NAME="${RUN_NAME:-urot_mf}"
    CONFIG="${CONFIG:-ablation/configs/urot_mf.yaml}"
    GATE_WHAT="PairEmbed on; per-plane RoPE freqs of U; embed 256 (d_head=32, 16 planes). Two knobs vs 57410779 (freqs AND width). Wave 0 baseline_wide is the additive-U width control."
    GATE_HOW="urot_mf         logits = Σ_p [(q^p·k^p) cos θ_p + (q^p⋆k^p) sin θ_p]/√d
                  θ_p = π tanh(α_{h,p} U_ij)"
    ;;
  *)
    echo "error: ONLY=$ONLY — expected urot, urot_rope, or urot_mf" >&2
    exit 2
    ;;
esac
mkdir -p logs "$OUTPUT_DIR"

if [ ! -f "$PACKAGE_DIR/$CONFIG" ]; then
  echo "error: config $CONFIG not found" >&2
  exit 1
fi

if [ -f "$OUTPUT_DIR/$RUN_NAME/metrics.jsonl" ]; then
  echo "error: $OUTPUT_DIR/$RUN_NAME already has metrics.jsonl — pick a new run_name" >&2
  exit 1
fi

DEST="$OUTPUT_DIR/$RUN_NAME"
mkdir -p "$DEST"
if [ ! -f "$DEST/GATE.txt" ]; then
  cat > "$DEST/GATE.txt" << EOF
U-as-rotation screen — registered before any eval
run_name: ${RUN_NAME}
arm:      urot
what:     ${GATE_WHAT}
          U is NOT added as a scalar.
not:      K2, N8 extras, resume of n8_k6 / n8_k6_v2 / runs/urot, Wave 0 dirs

Single delta vs baseline: how U is consumed.
  baseline        logits = QK^T/√d + U          (additive scalar)
  ${GATE_HOW}
  n8_k6_v2        PairEmbed off; extras from momenta as a scalar
  baseline_nopair no U

Comparators at 200k (matched 1M LR schedule, val_5M):
  baseline        0.85453
  baseline_nopair 0.83696   (T0.4 gap 0.01757)
  n8_k6_v2        0.83823   (not-U)
  half-gap floor  0.84574

Four-way gate at 200k val acc:
  1. rotation ≥ additive   |Δ| ≤ 0.001 vs 0.85453 and Rej99 within 1% relative
     → U should be a rotation generator, not a logit bias
  2. uses U, additive better   acc ≥ 0.84574 but miss (1)
     → keep additive U; optional later hybrid (residual λU), not K2
  3. rotation is not-U     |Δacc| < 0.002 vs nopair 0.83696
     → RoPE/fused consumption discarded PairEmbed information
  4. bug                   worse than nopair by > 0.01 or collapse
     → wiring / angle scale; not physics

Do not resume baseline/, moe/, n8_k6/, n8_k6_v2/, runs/urot, Wave 0 dirs.
Do not submit K2.
EOF
  echo "  wrote $DEST/GATE.txt"
else
  echo "  GATE.txt already exists at $DEST (leaving it)"
fi

echo "account      : ${NERSC_ACCOUNT}_g"
echo "arm          : urot"
echo "run_name     : $RUN_NAME"
echo "config       : $CONFIG"
echo "train_pt_dir : $TRAIN_PT_DIR"
echo "val_pt_dir   : $VAL_PT_DIR"
echo "norm_stats   : $NORM_STATS"
echo "output       : $OUTPUT_DIR"
echo "walltime/qos : ${WALLTIME} / ${QOS}"
echo "kernels      : true (PairEmbed); attention fusion off"
echo ""

args=(
  --account "${NERSC_ACCOUNT}_g"
  --qos "$QOS"
  --time "$WALLTIME"
  --job-name "part-${RUN_NAME}"
  --export "ALL,ARM=urot,RUN_NAME=${RUN_NAME},CONFIG=${CONFIG},TRAIN_PT_DIR=${TRAIN_PT_DIR},VAL_PT_DIR=${VAL_PT_DIR},OUTPUT_DIR=${OUTPUT_DIR},PACKAGE_DIR=${PACKAGE_DIR},NORM_STATS=${NORM_STATS},USE_PART_KERNELS=true,USE_COMPILE=false"
)

echo "submit: ${RUN_NAME}  (U as rotation, PairEmbed on)"
echo "  kernels=true  config=${CONFIG}  walltime=${WALLTIME} qos=${QOS}"
if [ "$DRY_RUN" -eq 1 ]; then
  echo "  [dry run] sbatch ${args[*]} ablation/slurm/train_arm.sh"
else
  job_id=$(sbatch --parsable "${args[@]}" ablation/slurm/train_arm.sh)
  echo "  -> job $job_id"
  echo ""
  echo "monitor with:  squeue --me"
  echo "results in  :  $OUTPUT_DIR/${RUN_NAME}/metrics.jsonl"
  echo "compare with:  python -m ablation.plot_metrics --runs $OUTPUT_DIR --arm urot --arm urot_rope --arm baseline --arm baseline_nopair --arm n8_k6_v2"
fi
