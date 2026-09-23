#!/bin/bash
# Helpers shared by the job scripts (sourced from the project root).

# best_model <.pt path | job id>
# A job id is turned into the model that job saved, read from its log line
# "Best model saved to: ..." in logs/slurm-<job name>-<job id>.out. This lets a job that is
# queued now use a model that only exists once an earlier job has finished.
best_model() {
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        local log path
        log=$(ls logs/slurm-*-"$1".out 2>/dev/null | head -n 1)
        path=$(grep -m1 "Best model saved to:" "$log" 2>/dev/null | sed 's/.*Best model saved to: //')
        if [ -z "$path" ] || [ ! -f "$path" ]; then
            echo "ERROR: no saved model found in the log of job $1 (${log:-logs/slurm-*-$1.out})" >&2
            return 1
        fi
        echo "$path"
    else
        echo "$1"
    fi
}
