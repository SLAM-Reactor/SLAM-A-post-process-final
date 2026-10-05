#!/bin/bash
#SBATCH -J desc_poincare
#SBATCH --account=m5194_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=55G
#SBATCH --time=01:00:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_f67/post-process/logs/trace_desc_poincare_%j.out

set -euo pipefail
unset LD_LIBRARY_PATH

export DESC_SOURCE=/pscratch/sd/r/rgaur/DESC3/DESC
export DESC_DEVICE=gpu
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_MEM_FRACTION=.93
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME=/pscratch/sd/r/rgaur/.cache
export MPLCONFIGDIR=/pscratch/sd/r/rgaur/.cache/matplotlib
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_f67/post-process/logs

module load conda
conda activate desc-env2

cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_f67/post-process
python -u trace_desc_poincare.py \
  --source-grid-n 32 \
  --bounds-r=0.3,2.6 \
  --bounds-z=-1.0,1.0
