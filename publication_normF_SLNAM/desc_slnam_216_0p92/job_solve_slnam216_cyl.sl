#!/bin/bash
#SBATCH -A m5194_g
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 00:45:00
#SBATCH -J desc_slnam216_cyl
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/publication_normF_SLNAM/desc_slnam_216_0p92/logs/desc_slnam216_cyl-%j.out

# 3 cylindrical fits on the SLNAM_BEST boundary, one per GPU. SLAM runs at the same basis took 24-26 min
# (jobs 58788651, 58841853, 58853789) -> 45 min cap.
unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
module load conda
conda activate desc-env
cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/publication_normF_SLNAM/desc_slnam_216_0p92

CUDA_VISIBLE_DEVICES=0 python -u solve_slnam216_cyl.py 7 14 > logs/solve_f7n14.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -u solve_slnam216_cyl.py 8 16 > logs/solve_f8n16.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -u solve_slnam216_cyl.py 9 18 > logs/solve_f9n18.out 2>&1 &
wait
