#!/bin/bash

#SBATCH --qos=premium
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH --account=m5194_g
#SBATCH --time=00:20:00

unset LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

module load conda

conda activate desc-env


python3 boozer_spline.py 1p5 100 0.7 
python3 boozer_spline.py 1p5 128 0.7
python3 boozer_spline.py 1p5 152 0.7
python3 boozer_spline.py 1p5 196 0.7
python3 boozer_spline.py 1p5 216 0.7
python3 boozer_spline.py 1p5 256 0.7

##python3 boozer_spline.py 1p5 100 0.90
##python3 boozer_spline.py 1p5 128 0.90
##python3 boozer_spline.py 1p5 152 0.90
##python3 boozer_spline.py 1p5 196 0.90
##python3 boozer_spline.py 1p5 216 0.90
##python3 boozer_spline.py 1p5 256 0.90
##
##python3 boozer_spline.py 1p5 100 0.91
##python3 boozer_spline.py 1p5 128 0.91
##python3 boozer_spline.py 1p5 152 0.91
##python3 boozer_spline.py 1p5 196 0.91
##python3 boozer_spline.py 1p5 216 0.91
##python3 boozer_spline.py 1p5 256 0.91
##
##
##
##
###python3 boozer_spline.py 1p0 100 0.87
###python3 boozer_spline.py 1p0 128 0.87
###python3 boozer_spline.py 1p0 152 0.87
###python3 boozer_spline.py 1p0 196 0.87
###python3 boozer_spline.py 1p0 216 0.87
###python3 boozer_spline.py 1p0 256 0.87
###
###python3 boozer_spline.py 1p0 100 0.90
###python3 boozer_spline.py 1p0 128 0.90
###python3 boozer_spline.py 1p0 152 0.90
###python3 boozer_spline.py 1p0 196 0.90
###python3 boozer_spline.py 1p0 216 0.90
###python3 boozer_spline.py 1p0 256 0.90
###
###python3 boozer_spline.py 1p0 100 0.92
###python3 boozer_spline.py 1p0 128 0.92
###python3 boozer_spline.py 1p0 152 0.92
##python3 boozer_spline.py 1p0 196 0.92
##python3 boozer_spline.py 1p0 216 0.92
##python3 boozer_spline.py 1p0 256 0.92
#
#

python3 boozer_spline.py 2p0 100 0.84
python3 boozer_spline.py 2p0 128 0.84
python3 boozer_spline.py 2p0 152 0.84
python3 boozer_spline.py 2p0 196 0.84
python3 boozer_spline.py 2p0 216 0.84
python3 boozer_spline.py 2p0 256 0.84

python3 boozer_spline.py 2p0 100 0.83
python3 boozer_spline.py 2p0 128 0.83
python3 boozer_spline.py 2p0 152 0.83
python3 boozer_spline.py 2p0 196 0.83
python3 boozer_spline.py 2p0 216 0.83
python3 boozer_spline.py 2p0 256 0.83

python3 boozer_spline.py 2p0 100 0.85
python3 boozer_spline.py 2p0 128 0.85
python3 boozer_spline.py 2p0 152 0.85
python3 boozer_spline.py 2p0 196 0.85
python3 boozer_spline.py 2p0 216 0.85
python3 boozer_spline.py 2p0 256 0.85


#

#python3 boozer_spline.py 0p5 100 0.87
#python3 boozer_spline.py 0p5 128 0.87
#python3 boozer_spline.py 0p5 152 0.87
#python3 boozer_spline.py 0p5 196 0.87
#python3 boozer_spline.py 0p5 216 0.87
#python3 boozer_spline.py 0p5 256 0.87
#
#python3 boozer_spline.py 0p5 100 0.90
#python3 boozer_spline.py 0p5 128 0.90
#python3 boozer_spline.py 0p5 152 0.90
#python3 boozer_spline.py 0p5 196 0.90
#python3 boozer_spline.py 0p5 216 0.90
#python3 boozer_spline.py 0p5 256 0.90
#
#python3 boozer_spline.py 0p5 100 0.91
#python3 boozer_spline.py 0p5 128 0.91
#python3 boozer_spline.py 0p5 152 0.91
#python3 boozer_spline.py 0p5 196 0.91
#python3 boozer_spline.py 0p5 216 0.91
#python3 boozer_spline.py 0p5 256 0.91



#python3 boozer_spline.py 0p0 100 0.82
#python3 boozer_spline.py 0p0 128 0.82
#python3 boozer_spline.py 0p0 152 0.82
#python3 boozer_spline.py 0p0 196 0.82
#python3 boozer_spline.py 0p0 216 0.82
#python3 boozer_spline.py 0p0 256 0.82
#
#python3 boozer_spline.py 0p0 100 0.85
#python3 boozer_spline.py 0p0 128 0.85
#python3 boozer_spline.py 0p0 152 0.85
#python3 boozer_spline.py 0p0 196 0.85
#python3 boozer_spline.py 0p0 216 0.85
#python3 boozer_spline.py 0p0 256 0.85
#
#python3 boozer_spline.py 0p0 100 0.86
#python3 boozer_spline.py 0p0 128 0.86
#python3 boozer_spline.py 0p0 152 0.86
#python3 boozer_spline.py 0p0 196 0.86
#python3 boozer_spline.py 0p0 216 0.86
#python3 boozer_spline.py 0p0 256 0.86


