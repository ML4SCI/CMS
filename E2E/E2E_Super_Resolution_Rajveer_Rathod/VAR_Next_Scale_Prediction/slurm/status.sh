#!/bin/bash
# Progress of one experiment: queued / running jobs, what is finished, and the end of the latest logs.
#
#   bash slurm/status.sh                 the most recently submitted experiment
#   bash slurm/status.sh <experiment>    a given one, e.g. 2026-10-06_v2_multiseed
#   bash slurm/status.sh --smoke         the latest smoke test
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings

load_experiment "${1:-}"
exp="$(experiment_dir)"
pfx="$(prefix)"
cache="$ROOT/cache/$DATASET"
logdir="$ROOT/logs/$EXPERIMENT"

echo "experiment $EXPERIMENT  ($exp)"
echo
echo "---- jobs in the queue"
ids="$(cut -f3 "$logdir/jobs.tsv" 2> /dev/null | paste -sd, -)"
if [[ -z "$ids" ]]; then
    echo "(no jobs submitted yet)"
elif command -v squeue > /dev/null; then
    squeue -j "$ids" -o '%.14i %.36j %.9T %.10M %.10l %.24R' 2> /dev/null || echo "(none of this experiment's jobs are queued)"
else
    echo "(squeue not available)"
fi

mark() { if compgen -G "$1" > /dev/null; then echo "done   "; else echo "-      "; fi; }
echo
echo "---- finished work"
echo "$(mark "$cache/meta.json") prepare (HR / LR cache)"
echo "$(mark "$exp/models/vqvae_tokenizer/metrics.json") tokenizer  models/vqvae_tokenizer"
for l in "${LEVEL_LIST[@]}"; do
    echo "$(mark "$exp/models/var_transformer_${l}/metrics.json") transformer models/var_transformer_${l}"
    echo "$(mark "$cache/test_${pfx}_sr-*_${l}.npy") super-resolved test split, $l"
    echo "$(mark "$cache/train_${pfx}_sr-*_${l}.npy") super-resolved train split, $l"
    echo "$(mark "$exp/evaluation/sr_${l}/observables.csv") evaluation evaluation/sr_${l}"
done
arch=resnet18                 # the qg default (superres/config.py); calo is always cnn
[[ "$EXTRA_SET" == *tagger.arch=*cnn* ]] && arch=cnn
tagger_model=${arch}_tagger
[[ "$DATASET" == calo ]] && tagger_model=cnn_regressor
for group in early late; do
    while read -r kind seed; do
        echo "$(mark "$exp/models/${tagger_model}_${kind/:/-}_seed${seed}/metrics.json") tagger     models/${tagger_model}_${kind/:/-}_seed${seed}"
    done < <(tagger_tasks "$group")
done
echo "$(mark "$exp/summary/${pfx}_${tagger_model/_/-}-2x3_*.csv") 2x3 tagger table"
echo "$(mark "$exp/summary/${pfx}_report.md") summary    summary/${pfx}_report.md"
compgen -G "$ROOT/results_${pfx}*.tar.gz" > /dev/null && echo "packed: $(ls "$ROOT"/results_"${pfx}"*.tar.gz)"

echo
echo "---- end of the latest log of each stage ($logdir)"
for stage in prepare tune-vqvae vqvae tune-var var generate eval tune-tagger taggers-early taggers-late summary smoke; do
    # shellcheck disable=SC2012
    f="$(ls -t "$logdir/$stage"-*.out 2> /dev/null | head -1)"
    [[ -n "$f" ]] || continue
    echo "== $(basename "$f")"
    tail -n 3 "$f"
done
grep -l -i -E "traceback|error:|out of memory|DUE TO TIME LIMIT" "$logdir"/*.out 2> /dev/null \
    | sed 's/^/!! problem reported in /'
exit 0
