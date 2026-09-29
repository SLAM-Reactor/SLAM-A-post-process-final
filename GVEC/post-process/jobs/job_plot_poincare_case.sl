#!/bin/bash
#SBATCH -J plot_pc_case
#SBATCH --account=m5194
#SBATCH --constraint=cpu
#SBATCH --qos=premium
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/post-process/logs/plot_poincare_case_%j.out

set -euo pipefail
unset LD_LIBRARY_PATH

: "${CASE_LABEL:?set CASE_LABEL, e.g. SLAM_N128_RHO065}"

module load conda
conda activate desc-env

cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/post-process
python -u plot_poincare_vs_gvec.py \
  --sections "data/gvec_sections_for_poincare_${CASE_LABEL}.npz" \
  --traces "data/desc_coil_poincare_traces_${CASE_LABEL}.npz" \
  --output-stem figures/poincare_vs_gvec
