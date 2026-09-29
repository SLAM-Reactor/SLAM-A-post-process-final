#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

module load conda
conda activate desc-env2

export DESC_SOURCE="${DESC_SOURCE:-/pscratch/sd/r/rgaur/DESC3/DESC}"
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/pscratch/sd/r/rgaur/.cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/pscratch/sd/r/rgaur/.cache/matplotlib}"

mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

cd "$HERE"
python3 -u plot_boozer_rhop95.py
python3 -u plot_equilibrium_coils_rhop95.py
python3 -u plot_poincare_comparison.py
