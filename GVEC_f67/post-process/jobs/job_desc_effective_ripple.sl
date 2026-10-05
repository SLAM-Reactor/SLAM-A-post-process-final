#!/bin/bash
#SBATCH -J desc_ripple
#SBATCH --account=m5194_g
#SBATCH --constraint=gpu&hbm80g
#SBATCH --qos=premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_f67/post-process/logs/desc_effective_ripple_%j.out

set -euo pipefail
unset LD_LIBRARY_PATH

export DESC_REPO=/pscratch/sd/r/rgaur/DESC
export DESC_DEVICE=gpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=.93
export XLA_PYTHON_CLIENT_PREALLOCATE=false

module load conda
conda activate desc-env

cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_f67/post-process
python -u compute_desc_effective_ripple.py
