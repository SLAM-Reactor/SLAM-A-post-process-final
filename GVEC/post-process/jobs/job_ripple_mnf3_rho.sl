#!/bin/bash
#SBATCH -J rip_mnf3
#SBATCH --account=m5194
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/post-process/logs/ripple_%x_%j.out

# Effective ripple at Boozer MNfactor 3 (GVEC default is 5), one surface per job, 10 turns, 751 points.
# CASE and RHO from sbatch --export. SLAM_BEST runs out of memory at MNfactor 5 (job 58800587);
# SLNAM_BEST rho=0.2 at MNfactor 3 checks the change against its MNfactor 5 value (2.939320e-03).
set -euo pipefail
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=32
module load conda
conda activate gvec-env
cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC
python -u post-process/compute_gvec_effective_ripple.py "${CASE}" "${RHO}" 3
