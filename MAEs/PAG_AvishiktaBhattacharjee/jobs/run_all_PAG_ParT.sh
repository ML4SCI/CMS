#!/bin/bash
# Submit the PAG ParticleTransformer pipeline with one command. Every stage waits in the queue
# until the stage before it has finished successfully, so nothing has to be started by hand.
# Run it on a login node (not with sbatch), from the project root:
#     bash jobs/run_all_PAG_ParT.sh            # also submits the pretraining
#     bash jobs/run_all_PAG_ParT.sh <jobid>    # chains after a pretraining job already queued
#
#   pretrain_PAG_ParT ──┬── evaluate_PAG_ParT (pretraining: reconstruction plot)
#                       └── train_PAG_ParT ── evaluate_PAG_ParT (ROC, confusion matrix)
#
# If a stage fails, the stages after it are cancelled automatically.
set -euo pipefail
cd "$(dirname "$0")/.."

submit() { sbatch --parsable "$@" | cut -d';' -f1; }
after() { submit --kill-on-invalid-dep=yes --dependency=afterok:"$1" "${@:2}"; }

if [ $# -ge 1 ]; then
    PRE=$1
else
    queued=$(squeue --me --noheader --name=pretrain_PAG_ParT --format=%i) \
        || { echo "Could not read the queue with squeue; please try again." >&2; exit 1; }
    queued=$(echo $queued)
    if [ -n "$queued" ]; then
        echo "A pretrain_PAG_ParT job is already in the queue: $queued"
        echo "Chain after it:      bash jobs/run_all_PAG_ParT.sh <jobid>"
        echo "or cancel it first:  scancel <jobid>"
        exit 1
    fi
    PRE=$(submit jobs/pretrain_PAG_ParT.sh)
fi

EVAL_PRE=$(after "$PRE" jobs/evaluate_PAG_ParT.sh "$PRE" ./configs/pretrain_PAG_ParT.yaml)
TRAIN_PAG=$(after "$PRE" jobs/train_PAG_ParT.sh "$PRE")
EVAL_PAG=$(after "$TRAIN_PAG" jobs/evaluate_PAG_ParT.sh "$TRAIN_PAG")

echo "PAG ParticleTransformer pipeline submitted:"
printf '  %-36s %s\n' \
    "pretrain_PAG_ParT" "$PRE" \
    "evaluate pretraining (after $PRE)" "$EVAL_PRE" \
    "train_PAG_ParT (after $PRE)" "$TRAIN_PAG" \
    "evaluate_PAG_ParT (after $TRAIN_PAG)" "$EVAL_PAG"
echo "Check with: squeue --me   (waiting stages show (Dependency))"
