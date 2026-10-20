#!/bin/bash
# Pack one experiment's results (tables, figures, logs, configs, metrics) into a tarball to send back.
# Model checkpoints (*.pt) are left out unless --with-checkpoints is given. The SR caches are never
# included (they are large and can be regenerated). Copies the tarball to CFS_COPY if that is set.
#
#   bash slurm/pack_results.sh [<experiment> | --smoke] [--with-checkpoints]
# Runs automatically at the end of the summary job.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "${REPO}/slurm/common.sh"

which_run=""
with_ckpt=0
for a in "$@"; do
    case "$a" in
        --with-checkpoints) with_ckpt=1 ;;
        *) which_run="$a" ;;
    esac
done
if [[ -n "${RUN_ENV:-}" ]]; then  # inside the summary job
    load_settings
else
    load_settings
    load_experiment "$which_run"
fi

exp="$(experiment_dir)"
pfx="$(prefix)"
[[ -d "$exp" ]] || die "no results at $exp"
suffix=""
((with_ckpt)) && suffix="_with_checkpoints"
out="$ROOT/results_${pfx}${suffix}.tar.gz"
exclude=()
((with_ckpt)) || exclude=(--exclude='*.pt')

# experiment folder + the job logs, under one top-level directory named after the experiment
stage_dir="$(mktemp -d)"
trap 'rm -rf "$stage_dir"' EXIT
ln -s "$exp" "$stage_dir/$pfx"
[[ -d "$ROOT/logs/$EXPERIMENT" ]] && ln -s "$ROOT/logs/$EXPERIMENT" "$stage_dir/${pfx}_slurm-logs"
tar -czhf "$out" ${exclude[@]+"${exclude[@]}"} --exclude='*.tmp' -C "$stage_dir" .
echo "packed $(du -h "$out" | cut -f1)  $out"

if [[ -n "${CFS_COPY:-}" ]]; then
    mkdir -p "$CFS_COPY"
    cp "$out" "$CFS_COPY/"
    echo "copied to $CFS_COPY/$(basename "$out")"
fi
