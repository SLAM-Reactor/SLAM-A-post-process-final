#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/home/rgaur/anaconda3}"
DESC_SOURCE="${DESC_SOURCE:-/home/rgaur/DESC3/DESC}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate desc-env

# /home/rgaur/DESC currently contains unresolved conflict markers. Prepend the
# clean, compatible checkout without modifying the user's active DESC tree.
export PYTHONPATH="${DESC_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "${HERE}"
python plot_boozer.py "$@"
python plot_equilibrium_coils.py
