#!/bin/bash
# Submit the whole PAG / PAunG LorentzParT pipeline with one command. Every stage waits in the queue
# until the stage before it has finished successfully, so nothing has to be started by hand.
# Run it on a login node (not with sbatch), from the project root:
#     bash jobs/run_all_PAG_LorentzParT.sh            # also submits the pretraining
#     bash jobs/run_all_PAG_LorentzParT.sh <jobid>    # chains after a pretraining job already queued
#
#   pretrain_PAG ──┬── evaluate_PAG (pretraining: reconstruction plot)
#                  ├── train_PAG   ── evaluate_PAG
#                  └── train_PAunG ── evaluate_PAunG
#
# If a stage fails, the stages after it are cancelled automatically.
set -euo pipefail
cd "$(dirname "$0")/.."

submit() { sbatch --parsable "$@" | cut -d';' -f1; }
after() { submit --kill-on-invalid-dep=yes --dependency=afterok:"$1" "${@:2}"; }

if [ $# -ge 1 ]; then
    PRE=$1
else
    queued=$(squeue --me --noheader --name=pretrain_PAG_LorentzParT --format=%i) \
        || { echo "Could not read the queue with squeue; please try again." >&2; exit 1; }
    queued=$(echo $queued)
    if [ -n "$queued" ]; then
        echo "A pretrain_PAG_LorentzParT job is already in the queue: $queued"
        echo "Chain after it:      bash jobs/run_all_PAG_LorentzParT.sh <jobid>"
        echo "or cancel it first:  scancel <jobid>"
        exit 1
    fi
    PRE=$(submit jobs/pretrain_PAG_LorentzParT.sh)
fi

EVAL_PRE=$(after "$PRE" jobs/evaluate_PAG_LorentzParT.sh "$PRE" ./configs/pretrain_PAG_LorentzParT.yaml)
TRAIN_PAG=$(after "$PRE" jobs/train_PAG_LorentzParT.sh "$PRE")
TRAIN_PAUNG=$(after "$PRE" jobs/train_PAunG_LorentzParT.sh "$PRE")
EVAL_PAG=$(after "$TRAIN_PAG" jobs/evaluate_PAG_LorentzParT.sh "$TRAIN_PAG")
EVAL_PAUNG=$(after "$TRAIN_PAUNG" jobs/evaluate_PAunG_LorentzParT.sh "$TRAIN_PAUNG")

echo "PAG / PAunG LorentzParT pipeline submitted:"
printf '  %-36s %s\n' \
    "pretrain_PAG_LorentzParT" "$PRE" \
    "evaluate pretraining (after $PRE)" "$EVAL_PRE" \
    "train_PAG_LorentzParT (after $PRE)" "$TRAIN_PAG" \
    "train_PAunG_LorentzParT (after $PRE)" "$TRAIN_PAUNG" \
    "evaluate_PAG (after $TRAIN_PAG)" "$EVAL_PAG" \
    "evaluate_PAunG (after $TRAIN_PAUNG)" "$EVAL_PAUNG"
echo "Check with: squeue --me   (waiting stages show (Dependency))"
