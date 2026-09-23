#!/bin/bash
# Python environment for the jobs on NERSC Perlmutter, matching the PAG notebook
# (NERSC Python module + user-installed packages). Source it from the project root:
#     source jobs/setup_env.sh

module load python

# LGATr version the PAG models are built with (the scripts refuse to run with another version)
python -m pip install --user "lgatr==1.4.4" uproot awkward tqdm vector

# No display on compute nodes: matplotlib saves the plots as PNG files
export MPLBACKEND=Agg

python -c 'import torch; from importlib.metadata import version; print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "| lgatr", version("lgatr"))'
