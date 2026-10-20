#!/bin/bash
# Submitted by "bash slurm/submit_all.sh --smoke" (debug QOS, 1 GPU, 30 min).
# Every stage on a few hundred synthetic events with tiny settings, under $ROOT (= <ROOT>/smoke),
# then checks that the report and the new tables were written. Prints "SMOKE OK" on success.
#SBATCH -J superres-smoke
set -euo pipefail
: "${RUN_ENV:?submit this job with slurm/submit_all.sh --smoke}"
# shellcheck disable=SC1090
source "$RUN_ENV"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python
job_header "smoke test"

raw="$ROOT/raw/$DATASET"
if [[ -z "$(ls -A "$raw" 2> /dev/null)" ]]; then
    echo "writing synthetic $DATASET data to $raw"
    python -c "import sys; from tests.smoke_test import fake_qg, fake_calo; \
(fake_qg if sys.argv[1] == 'qg' else fake_calo)(sys.argv[2])" "$DATASET" "$raw"
fi

run_pipeline prepare,train_vqvae,train_var,generate,eval_sr,train_taggers,tagger_xeval,summarize --smoke

exp="$(experiment_dir)"
pfx="$(prefix)"
missing=0
check() {
    if compgen -G "$1" > /dev/null; then
        echo "  ok      $1"
    else
        echo "  MISSING $1"
        missing=1
    fi
}
echo "checking outputs in $exp"
check "$exp/summary/${pfx}_report.md"
check "$exp/summary/${pfx}_*-tagger-2x3_*.csv"
check "$exp/summary/figures/${pfx}_*-tagger-curves_*.png"
check "$exp/evaluation/sr_*/paired.csv"
check "$exp/models/vqvae_tokenizer/best.pt"
check "$exp/models/var_transformer_*/best.pt"
check "$exp/models/cnn_*_sr-*_seed*/best.pt"
check "$exp/run_info.json"
((missing == 0)) || die "smoke test finished but outputs are missing (see above)"
echo "SMOKE OK"
