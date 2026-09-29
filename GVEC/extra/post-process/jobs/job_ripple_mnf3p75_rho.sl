#!/bin/bash
#SBATCH -J rip_mnf3p75
#SBATCH --account=m5194
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/post-process/logs/ripple_%x_%j.out

# SLAM_BEST ripple resolution check: Boozer M, N = int(3.75 x (24, 32)) = (90, 120), vs (72, 96) in the MNfactor 3 jobs.
# One surface per job (CASE and RHO from sbatch --export), 10 turns, 751 points. Estimated ~364 GB, ~80 min per surface.
set -euo pipefail
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=32
module load conda
conda activate gvec-env
cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC
python -u post-process/compute_gvec_effective_ripple.py "${CASE}" "${RHO}" 3.75
