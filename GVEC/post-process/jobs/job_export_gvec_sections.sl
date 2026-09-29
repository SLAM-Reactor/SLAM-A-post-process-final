#!/bin/bash
#SBATCH -J gvec_sections
#SBATCH --account=m5194
#SBATCH --constraint=cpu
#SBATCH --qos=premium
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/post-process/logs/export_gvec_sections_%j.out

set -euo pipefail
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=12

module load conda
conda activate gvec-env

cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/post-process
python -u export_gvec_sections.py
