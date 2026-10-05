#!/bin/bash
#SBATCH -J desc_pc_case
#SBATCH --account=m5194_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=55G
#SBATCH --time=01:00:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_f67/post-process/logs/trace_desc_poincare_case_%j.out

set -euo pipefail
unset LD_LIBRARY_PATH

: "${CASE_LABEL:?set CASE_LABEL, e.g. SLAM_N128_RHO065}"

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
  --labels "${CASE_LABEL}" \
  --sections "data/gvec_sections_for_poincare_${CASE_LABEL}.npz" \
  --output "data/desc_coil_poincare_traces_${CASE_LABEL}.npz" \
  --source-grid-n 32 \
  --bounds-r=0.3,2.6 \
  --bounds-z=-1.0,1.0
