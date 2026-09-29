#!/bin/bash

#SBATCH --job-name=slnam_pc
#SBATCH --qos=premium
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=55G
#SBATCH --account=m5194_g
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=trace_poincare_%A_%a.out

unset LD_LIBRARY_PATH

export DESC_SOURCE=/pscratch/sd/r/rgaur/DESC3/DESC
export MPLBACKEND=Agg
export XLA_PYTHON_CLIENT_MEM_FRACTION=.93
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XDG_CACHE_HOME=/pscratch/sd/r/rgaur/.cache
export MPLCONFIGDIR=/pscratch/sd/r/rgaur/.cache/matplotlib

mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

module load conda
conda activate desc-env2

cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/QI_to_SLNAM/Poincare_plot/Poincare_plot
python3 -u trace_poincare_gpu.py
