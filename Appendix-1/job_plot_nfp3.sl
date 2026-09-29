#!/bin/bash
#SBATCH -A m5194
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -J plot_nfp3
#SBATCH -o logs/plot_nfp3-%j.out

set -euo pipefail
unset LD_LIBRARY_PATH
module load conda
conda activate desc-env2
cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/Appendix-1
python -u plot_poincare_vs_gvec_nfp3.py
