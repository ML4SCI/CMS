#!/bin/bash
# Compute per-feature mean/std over ALL train shards (1000 × train_100M).
#
# Writes: $PSCRATCH/jetclass/pt_ragged/norm_stats.json
# Stats cover all 16 particle channels (including charge + PID one-hots);
# NormStats.apply later leaves channels 10–15 raw — that is apply-time, not
# compute-time skipping.
#
#   sbatch -A m4392 preprocessing/slurm/compute_norm_stats.sh
#
# Perlmutter CPU node (shared partition is fine for this I/O-bound pass).

#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --job-name=norm-stats
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail

PT_DIR="${PT_DIR:-/pscratch/sd/o/omasho/jetclass/pt_ragged/train_100M}"
OUT="${OUT:-/pscratch/sd/o/omasho/jetclass/pt_ragged/norm_stats.json}"
PACKAGE_DIR="${PACKAGE_DIR:-$HOME/code/ml4sci_26}"

cd "$PACKAGE_DIR"
mkdir -p logs "$HOME/code/jetclass/logs"

module load python
source "$HOME/code/jetclass/.venv/bin/activate"

echo "=========================================================="
echo "job          : ${SLURM_JOB_ID}  (${SLURM_JOB_NAME})"
echo "node         : $(hostname)"
echo "pt_dir       : ${PT_DIR}"
echo "output       : ${OUT}"
echo "started      : $(date -Is)"
echo "=========================================================="

# All shards (omit --num-shards). Includes PID / charge channels in mean/std.
python3 -m preprocessing.compute_norm_stats \
    --pt-dir "$PT_DIR" \
    --output "$OUT"

echo "finished     : $(date -Is)"
ls -la "$OUT"
python3 - <<PY
import json
from pathlib import Path
p = Path("$OUT")
stats = json.loads(p.read_text())
print(f"shards={stats['num_shards']}  particles={stats['num_particles']:,}")
print("mean[0:4]=", [round(x, 4) for x in stats["mean"][:4]])
print("std [0:4]=", [round(x, 4) for x in stats["std"][:4]])
PY
