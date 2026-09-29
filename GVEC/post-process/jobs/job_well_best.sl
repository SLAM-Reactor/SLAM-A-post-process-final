#!/bin/bash
#SBATCH -J well_best
#SBATCH --account=m5194
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/post-process/logs/well_best_%j.out

set -euo pipefail
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=32
module load conda
conda activate gvec-env
cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC
python -u post-process/compute_gvec_magnetic_well.py
