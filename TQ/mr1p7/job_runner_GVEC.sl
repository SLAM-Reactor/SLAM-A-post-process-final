#!/bin/bash

#SBATCH --qos=test
#SBATCH --constraint=intel
#SBATCH --time=00:59:00

module load anaconda3/2023.3

export OMP_NUM_THREADS=12

conda activate gvec-env

pygvec run surface_2_1p0_100_0p85-parameters.toml 

