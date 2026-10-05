#!/bin/bash
#SBATCH -A m5194
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -J pyvista_nfp3
#SBATCH -o logs/pyvista_nfp3-%j.out

set -euo pipefail
unset LD_LIBRARY_PATH
module load conda
conda activate gvec-env
cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/Appendix-A_fA1/fA1b
python -u make_coils_modB_html_svg_nfp3.py
