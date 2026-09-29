#!/bin/bash
#SBATCH -A m5194
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J export_nfp3
#SBATCH -o logs/export_nfp3-%j.out

set -euo pipefail
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=12
module load conda
conda activate gvec-env
cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/Appendix-1
python -u export_gvec_sections_nfp3.py
