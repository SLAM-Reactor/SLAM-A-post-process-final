#!/bin/bash
#SBATCH -J rip_SLNAM_BEST
#SBATCH --account=m5194
#SBATCH --constraint=cpu
#SBATCH --qos=premium
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_f67/post-process/logs/ripple_SLNAM_BEST_rho_%j.out

# Effective ripple, one surface per job (RHO from sbatch --export), 10 turns, 751 points, GVEC default Boozer resolution
set -euo pipefail
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=32
module load conda
conda activate gvec-env
cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_f67
python -u post-process/compute_gvec_effective_ripple.py SLNAM_BEST "${RHO}"
