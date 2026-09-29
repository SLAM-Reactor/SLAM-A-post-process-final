#!/bin/bash

#SBATCH --qos=premium
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus 4
#SBATCH -c 96
#SBATCH --mem=200G
#SBATCH --account=m5194_g
#SBATCH --time=02:00:00

unset LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

module load conda

conda activate desc-env


python3 boozer_spline.py 1p0 100 0.7 0 &
python3 boozer_spline.py 1p0 128 0.7 1 &
python3 boozer_spline.py 1p0 152 0.7 2 &
python3 boozer_spline.py 1p0 196 0.7 3 &
wait

python3 boozer_spline.py 1p0 216 0.7 0 &
python3 boozer_spline.py 1p0 256 0.7 1 &
python3 boozer_spline.py 1p0 100 0.85 2 &
python3 boozer_spline.py 1p0 128 0.85 3 &
wait

python3 boozer_spline.py 1p0 152 0.85 0 &
python3 boozer_spline.py 1p0 196 0.85 1 &
python3 boozer_spline.py 1p0 216 0.85 2 &
python3 boozer_spline.py 1p0 256 0.85 3 &
wait

python3 boozer_spline.py 1p0 100 0.88 0 &
python3 boozer_spline.py 1p0 128 0.88 1 &
python3 boozer_spline.py 1p0 152 0.88 2 &
python3 boozer_spline.py 1p0 196 0.88 3 &
wait


python3 boozer_spline.py 1p0 216 0.88 0 &
python3 boozer_spline.py 1p0 256 0.88 1 &
python3 boozer_spline.py 1p0 100 0.92 2 &
python3 boozer_spline.py 1p0 128 0.92 3 &
wait
