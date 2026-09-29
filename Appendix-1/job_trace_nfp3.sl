#!/bin/bash
#SBATCH -A m5194_g
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH -c 32
#SBATCH --mem=55G
#SBATCH -t 00:45:00
#SBATCH -J trace_nfp3
#SBATCH -o logs/trace_nfp3-%j.out

set -euo pipefail
unset LD_LIBRARY_PATH
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_MEM_FRACTION=.93
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME=/pscratch/sd/r/rgaur/.cache
export MPLCONFIGDIR=/pscratch/sd/r/rgaur/.cache/matplotlib
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
module load conda
conda activate desc-env2
cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/Appendix-1
python -u trace_desc_poincare_nfp3.py
