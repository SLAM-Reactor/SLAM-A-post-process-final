#!/bin/bash
#SBATCH -J gvec_sec_case
#SBATCH --account=m5194
#SBATCH --constraint=cpu
#SBATCH --qos=premium
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/post-process/logs/export_gvec_sections_case_%j.out

set -euo pipefail
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=4

: "${CASE_LABEL:?set CASE_LABEL, e.g. SLAM_N128_RHO065}"

module load conda
conda activate gvec-env

cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/post-process
python -u export_gvec_sections.py \
  --labels "${CASE_LABEL}" \
  --output "data/gvec_sections_for_poincare_${CASE_LABEL}.npz"
