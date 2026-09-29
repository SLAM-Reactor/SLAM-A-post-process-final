#!/bin/bash
#SBATCH -J desc_well
#SBATCH --account=m5194
#SBATCH --constraint=cpu
#SBATCH --qos=premium
#SBATCH -N 1
#SBATCH -c 12
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/post-process/logs/desc_magnetic_well_%j.out

set -euo pipefail
unset LD_LIBRARY_PATH
export DESC_REPO=/pscratch/sd/r/rgaur/DESC
export JAX_PLATFORMS=cpu

module load conda
conda activate desc-env

cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/post-process
python -u compute_desc_magnetic_well.py
