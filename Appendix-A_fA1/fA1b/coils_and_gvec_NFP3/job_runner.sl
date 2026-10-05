#!/bin/bash

#SBATCH --qos=premium
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH --mem=55G
#SBATCH --account=m5194_g
#SBATCH --time=00:25:00

unset LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

module load conda

conda activate desc-env




#python3 boozer_spline.py 2p0 100 0.95
#python3 boozer_spline.py 2p0 100 0.98
#python3 boozer_spline.py 2p0 100 1.00
#
#python3 boozer_spline.py 2p0 116 0.95
#python3 boozer_spline.py 2p0 116 0.98
#python3 boozer_spline.py 2p0 116 1.00
#
#python3 boozer_spline.py 2p0 128 0.95
#python3 boozer_spline.py 2p0 128 0.98
#python3 boozer_spline.py 2p0 128 1.00
#
#python3 boozer_spline.py 2p0 152 0.95
python3 boozer_spline.py 2p0 152 0.98
#python3 boozer_spline.py 2p0 152 1.00

#python3 boozer_spline.py 2p0 196 0.95
#python3 boozer_spline.py 2p0 196 0.98
#python3 boozer_spline.py 2p0 196 1.00
#
#python3 boozer_spline.py 2p0 216 0.95
#python3 boozer_spline.py 2p0 216 0.98
#python3 boozer_spline.py 2p0 216 1.00

#python3 boozer_spline.py 2p0 256 0.95
#python3 boozer_spline.py 2p0 256 0.98
#python3 boozer_spline.py 2p0 256 1.00



#python3 boozer_spline.py 0p0 100 0.95
#python3 boozer_spline.py 0p0 100 0.98
#python3 boozer_spline.py 0p0 100 1.00
#
#python3 boozer_spline.py 0p0 116 0.95
#python3 boozer_spline.py 0p0 116 0.98
#python3 boozer_spline.py 0p0 116 1.00
#
#python3 boozer_spline.py 0p0 128 0.95
#python3 boozer_spline.py 0p0 128 0.98
#python3 boozer_spline.py 0p0 128 1.00
#
#python3 boozer_spline.py 0p0 152 0.95
#python3 boozer_spline.py 0p0 152 0.98
#python3 boozer_spline.py 0p0 152 1.00
#
#python3 boozer_spline.py 0p0 196 0.95
#python3 boozer_spline.py 0p0 196 0.98
#python3 boozer_spline.py 0p0 196 1.00
#
#python3 boozer_spline.py 0p0 216 0.95
#python3 boozer_spline.py 0p0 216 0.98
#python3 boozer_spline.py 0p0 216 1.00
#
#python3 boozer_spline.py 0p0 256 0.95
#python3 boozer_spline.py 0p0 256 0.98
#python3 boozer_spline.py 0p0 256 1.00
#


#python3 boozer_spline.py 0p5 100 0.95
#python3 boozer_spline.py 0p5 100 0.98
#python3 boozer_spline.py 0p5 100 1.00
#
#python3 boozer_spline.py 0p5 116 0.95
#python3 boozer_spline.py 0p5 116 0.98
#python3 boozer_spline.py 0p5 116 1.00
#
#python3 boozer_spline.py 0p5 128 0.95
#python3 boozer_spline.py 0p5 128 0.98
#python3 boozer_spline.py 0p5 128 1.00
#
#python3 boozer_spline.py 0p5 152 0.95
#python3 boozer_spline.py 0p5 152 0.98
#python3 boozer_spline.py 0p5 152 1.00
#
#python3 boozer_spline.py 0p5 196 0.95
#python3 boozer_spline.py 0p5 196 0.98
#python3 boozer_spline.py 0p5 196 1.00
#
#python3 boozer_spline.py 0p5 216 0.95
#python3 boozer_spline.py 0p5 216 0.98
#python3 boozer_spline.py 0p5 216 1.00
#
#python3 boozer_spline.py 0p5 256 0.95
#python3 boozer_spline.py 0p5 256 0.98
#python3 boozer_spline.py 0p5 256 1.00

#
#python3 boozer_spline.py 0p1 100 0.95
#python3 boozer_spline.py 0p1 100 0.98
#python3 boozer_spline.py 0p1 100 1.00
#
#python3 boozer_spline.py 0p1 116 0.95
#python3 boozer_spline.py 0p1 116 0.98
#python3 boozer_spline.py 0p1 116 1.00
#
#python3 boozer_spline.py 0p1 128 0.95
#python3 boozer_spline.py 0p1 128 0.98
#python3 boozer_spline.py 0p1 128 1.00
#
#python3 boozer_spline.py 0p1 152 0.95
#python3 boozer_spline.py 0p1 152 0.98
#python3 boozer_spline.py 0p1 152 1.00
#
#python3 boozer_spline.py 0p1 196 0.95
#python3 boozer_spline.py 0p1 196 0.98
#python3 boozer_spline.py 0p1 196 1.00
#
#python3 boozer_spline.py 0p1 216 0.95
#python3 boozer_spline.py 0p1 216 0.98
#python3 boozer_spline.py 0p1 216 1.00
#
#python3 boozer_spline.py 0p1 256 0.95
#python3 boozer_spline.py 0p1 256 0.98
#python3 boozer_spline.py 0p1 256 1.00



