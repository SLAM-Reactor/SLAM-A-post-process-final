#!/bin/bash

#SBATCH --qos=premium
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus 4
#SBATCH -c 96
#SBATCH --mem=200G
#SBATCH --account=m5194_g
#SBATCH --time=01:00:00

unset LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

module load conda

conda activate desc-env


python3 boozer_spline.py 1p0 100 0.9 0 &
python3 boozer_spline.py 1p0 128 0.9 1 &
python3 boozer_spline.py 1p0 152 0.9 2 &
python3 boozer_spline.py 1p0 196 0.9 3 &
wait

python3 boozer_spline.py 1p0 216 0.9 0 &
python3 boozer_spline.py 1p0 256 0.9 1 &
python3 boozer_spline.py 1p0 100 0.95 2 &
python3 boozer_spline.py 1p0 128 0.95 3 &
wait

python3 boozer_spline.py 1p0 152 0.95 0 &
python3 boozer_spline.py 1p0 196 0.95 1 &
python3 boozer_spline.py 1p0 216 0.95 2 &
python3 boozer_spline.py 1p0 256 0.95 3 &
wait


