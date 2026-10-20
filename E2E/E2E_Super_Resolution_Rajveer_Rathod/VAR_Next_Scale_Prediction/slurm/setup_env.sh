#!/bin/bash
# One-time setup, on a Perlmutter login node:  bash slurm/setup_env.sh
# Loads the NERSC PyTorch module, installs the few missing packages into the module's user site
# ($PYTHONUSERBASE, kept between sessions) and checks that the code imports.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=common.sh
source "$REPO/slurm/common.sh"
load_settings
setup_python

echo "pytorch module: $(module list 2>&1 | grep -o 'pytorch/[^ ]*' || echo '?')"
python - << 'EOF'
import torch
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
print("torch", torch.__version__, "(ok)" if v >= (2, 6) else "- older than 2.6: run 'module avail pytorch' and load a newer one")
EOF

missing=()
for spec in numpy scipy sklearn:scikit-learn pandas matplotlib h5py pyarrow requests optuna tabulate; do
    mod="${spec%%:*}"
    pkg="${spec#*:}"
    python -c "import $mod" 2> /dev/null || missing+=("$pkg")
done
if ((${#missing[@]})); then
    echo "installing into the user site of this module: ${missing[*]}"
    pip install --user "${missing[@]}"
fi

python -c "import superres.pipeline, superres.evaluate, superres.train; print('superres imports ok')"
mkdir -p "$ROOT"
echo "data / results root: $ROOT  ($(df -h "$ROOT" | awk 'NR == 2 {print $4 " free"}'))"
[[ "$ACCOUNT" == mXXXX* ]] && echo "NEXT: set ACCOUNT in slurm/settings.sh (your projects: run 'iris' or check https://iris.nersc.gov)"
echo "setup done. Next: bash slurm/submit_all.sh --smoke"
