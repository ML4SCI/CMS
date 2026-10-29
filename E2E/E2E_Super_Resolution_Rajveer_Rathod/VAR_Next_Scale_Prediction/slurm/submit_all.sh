#!/bin/bash
# Submit the whole pipeline as a chain of Slurm jobs (run on a login node).
#
#   bash slurm/submit_all.sh                  everything, from download + prepare
#   bash slurm/submit_all.sh --from var       restart from a stage (earlier stages are not submitted)
#   bash slurm/submit_all.sh --smoke          the 30-minute check on synthetic data (debug QOS)
#   bash slurm/submit_all.sh --dry-run        print the sbatch commands, submit nothing
#
# Stages: prepare -> vqvae -> var[level] -> generate[level] -> eval[level]
#                                                  \-> taggers[kind x seed] -> summary
# Finished work is skipped by the code, so re-submitting after a failure or timeout is safe.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"

STAGES=(prepare vqvae var generate eval taggers summary)
FROM=prepare
DRY=0
SMOKE=0
while (($#)); do
    case "$1" in
        --from) FROM="${2:?--from needs a stage}"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        -h | --help) sed -n '2,12p' "$0"; exit 0 ;;
        *) die "unknown option $1" ;;
    esac
done

stage_index() {
    local i
    for i in "${!STAGES[@]}"; do [[ "${STAGES[$i]}" == "$1" ]] && { echo "$i"; return; }; done
    die "unknown stage '$1' (choose from: ${STAGES[*]})"
}
FROM_I=$(stage_index "$FROM")
wanted() { (($(stage_index "$1") >= FROM_I)); }

load_settings
unset RUN_ENV
if ((!SMOKE)) && [[ "$FROM" != prepare && -z "$RUN_DATE" ]]; then
    # a restart continues the latest experiment with this tag, even on a later day
    RUN_DATE="$(latest_experiment_date)"
    [[ -n "$RUN_DATE" ]] && echo "continuing the experiment from $RUN_DATE (set RUN_DATE in settings.sh to pick another)"
fi
if ((SMOKE)); then
    ROOT="$ROOT/smoke"
    resolve_experiment -smoke
    EXTRA_SET=""
else
    resolve_experiment
fi
if [[ "$ACCOUNT" == mXXXX* ]]; then
    ((DRY)) || die "set ACCOUNT in slurm/settings.sh"
    echo "(dry run: ACCOUNT is still the placeholder '$ACCOUNT')"
fi

LOGDIR="$ROOT/logs/$EXPERIMENT"
ENV_FILE="$LOGDIR/run.env"
echo "experiment : $(experiment_dir)"
echo "logs       : $LOGDIR"

# ---- freeze the settings for the jobs ------------------------------------------------------------
write_env() {
    local v
    for v in REPO ROOT EXPERIMENT ACCOUNT CPU_ACCOUNT QOS GPU_MEM DATASET VERSION RUN_TAG RUN_DATE LEVELS SEEDS \
             TUNE EXTRA_SET CFS_COPY; do
        printf 'export %s=%q\n' "$v" "${!v}"
    done
}
if ((DRY)); then
    echo "--- run.env would be:"
    write_env
else
    mkdir -p "$LOGDIR"
    [[ -f "$ENV_FILE" ]] && cp "$ENV_FILE" "$ENV_FILE.$(date +%Y%m%d-%H%M%S)"
    write_env > "$ENV_FILE"
fi

# ---- sbatch wrapper ------------------------------------------------------------------------------
# sub <stage-name> <time> <script> <dependency or ""> [array range or ""] [extra exports]
sub() {
    local name="$1" time="$2" script="$3" dep="$4" array="${5:-}" exports="${6:-}"
    local -a args=(--parsable -J "sr_${RUN_TAG}_${name}" -t "$time" --kill-on-invalid-dep=yes
                   --export="ALL,RUN_ENV=$ENV_FILE${exports:+,$exports}")
    if [[ "$name" == prepare && -n "$CPU_ACCOUNT" ]]; then
        args+=(-A "$CPU_ACCOUNT" -C cpu -q shared -c 32)
    elif ((SMOKE)); then
        args+=(-A "$ACCOUNT" -C gpu -q debug --gpus=1 -c 32)
    else
        local constraint=gpu
        [[ "$GPU_MEM" == 80 ]] && constraint="gpu&hbm80g"
        args+=(-A "$ACCOUNT" -C "$constraint" -q "$QOS" --gpus=1 -c 32)
        [[ "$QOS" == regular ]] && args+=(-N 1)
        [[ "$QOS" == preempt ]] && args+=(--requeue)  # training resumes from last.pt
    fi
    if [[ -n "$array" ]]; then
        args+=(--array="$array" -o "$LOGDIR/${name}-%A_%a.out")
    else
        args+=(-o "$LOGDIR/${name}-%j.out")
    fi
    [[ -n "$dep" ]] && args+=(--dependency="$dep")
    args+=("$REPO/slurm/$script")
    if ((DRY)); then
        echo "sbatch ${args[*]}" >&2
        echo "<$name>"
    else
        local id
        id=$(sbatch "${args[@]}")
        id="${id%%;*}"
        printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "$name" "$id" "${array:--}" >> "$LOGDIR/jobs.tsv"
        echo "  submitted $name: $id${array:+ (array $array)}" >&2
        echo "$id"
    fi
}

# dep afterok:<id>,aftercorr:<id> ... from "type:id" pairs whose id is not empty
dep() {
    local out="" p
    for p in "$@"; do
        [[ -n "${p#*:}" ]] && out+="${out:+,}$p"
    done
    echo "$out"
}

if ((SMOKE)); then
    sub smoke 00:30:00 00_smoke.slurm "" > /dev/null
    echo "smoke job submitted; wait for 'SMOKE OK' at the end of $LOGDIR/smoke-<jobid>.out"
    exit 0
fi

n_levels=${#LEVEL_LIST[@]}
P="" TV="" V="" TR="" VAR="" G="" E="" TT="" TE="" TL=""
if wanted prepare; then
    P=$(sub prepare "$TIME_PREPARE" 01_prepare.slurm "")
fi
if wanted vqvae; then
    if ((TUNE)); then
        TV=$(sub tune-vqvae "$TIME_TUNE" tune.slurm "$(dep "afterok:$P")" "" TUNE_STAGE=tune_vqvae)
    fi
    V=$(sub vqvae "$TIME_VQVAE" 02_vqvae.slurm "$(dep "afterok:$P" "afterok:$TV")")
fi
if wanted var; then
    if ((TUNE)); then
        TR=$(sub tune-var "$TIME_TUNE" tune.slurm "$(dep "afterok:$V")" "" TUNE_STAGE=tune_var)
    fi
    VAR=$(sub var "$TIME_VAR" 03_var.slurm "$(dep "afterok:$V" "afterok:$TR")" "0-$((n_levels - 1))")
fi
if wanted generate; then
    G=$(sub generate "$TIME_GENERATE" 04_generate.slurm "$(dep "aftercorr:$VAR")" "0-$((n_levels - 1))")
fi
if wanted eval; then
    E=$(sub eval "$TIME_EVAL" 05_eval_sr.slurm "$(dep "aftercorr:$G")" "0-$((n_levels - 1))")
fi
if wanted taggers; then
    if ((TUNE)); then
        TT=$(sub tune-tagger "$TIME_TUNE" tune.slurm "$(dep "afterok:$P")" "" TUNE_STAGE=tune_tagger)
    fi
    # HR and LR taggers do not need SR, so they start after prepare - unless SR is capped to fewer
    # events (var.gen_max_per_split): every input must then wait for SR to use the same events.
    early_dep="$(dep "afterok:$P" "afterok:$TT")"
    [[ "$EXTRA_SET" == *gen_max_per_split* ]] && early_dep="$(dep "afterok:$G" "afterok:$TT")"
    TE=$(sub taggers-early "$TIME_TAGGER" 06_taggers.slurm "$early_dep" \
        "0-$(($(n_tagger_tasks early) - 1))" TAGGER_GROUP=early)
    TL=$(sub taggers-late "$TIME_TAGGER" 06_taggers.slurm "$(dep "afterok:$G" "afterok:$TT")" \
        "0-$(($(n_tagger_tasks late) - 1))" TAGGER_GROUP=late)
fi
if wanted summary; then
    sub summary "$TIME_SUMMARY" 07_xeval_summary.slurm "$(dep "afterok:$E" "afterok:$TE" "afterok:$TL")" > /dev/null
fi

echo
if ((DRY)); then
    echo "dry run: nothing submitted (the commands above would be)"
else
    echo "submitted. Follow with:  bash slurm/status.sh $EXPERIMENT   (job list: $LOGDIR/jobs.tsv)"
    echo "squeue -u \$USER -o '%.18i %.40j %.10T %.10M %.20R'"
fi
