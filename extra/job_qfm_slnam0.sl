#!/bin/bash
#SBATCH -J slnam0_qfm
#SBATCH -A m5194_g
#SBATCH -C gpu&hbm80g
#SBATCH -q premium
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=55G
#SBATCH -o /pscratch/sd/r/rgaur/generalized_toroidal/SLNAM0_equilibrium/QFM_optimization/slurm-qfm-%j.out

set -euo pipefail

module load conda
conda activate desc-env

unset LD_LIBRARY_PATH
unset JAX_PLATFORMS

export DESC_REPO=/pscratch/sd/r/rgaur/DESC
export PYTHONPATH=/pscratch/sd/r/rgaur/DESC
export DESC_DEVICE=gpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=.93
export JAX_CACHE_DIR=/pscratch/sd/r/rgaur/generalized_toroidal/SLNAM0_equilibrium/QFM_optimization/jax-caches

export M_SURF=${M_SURF:-8}
export N_SURF=${N_SURF:-16}
export EVAL_M=${EVAL_M:-24}
export EVAL_N=${EVAL_N:-32}
export BS_CHUNK_SIZE=${BS_CHUNK_SIZE:-10}
export QFLUX_WEIGHT=${QFLUX_WEIGHT:-1e-1}
export TFLUX_WEIGHT=${TFLUX_WEIGHT:-1e3}
export MAXITER=${MAXITER:-100}
export FTOL=${FTOL:-5e-4}

cd /pscratch/sd/r/rgaur/generalized_toroidal/SLNAM0_equilibrium/QFM_optimization
python3 -u qfm_slnam0.py
