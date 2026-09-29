#!/bin/bash

#SBATCH --qos=premium
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus 1
#SBATCH -c 32
#SBATCH --mem=55G
#SBATCH --account=m5194_g
#SBATCH --time=03:30:00

unset LD_LIBRARY_PATH

export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

export OMP_NUM_THREADS=12

module load conda

conda activate gvec-env

# NFP3 │ 0p0_216_0p98 │ 0p5_256_1p00 │ 1p0_128_1p00 │ 1p5_216_0p95 │ 2p0_152_0p98 │

python3 GVEC_Boozer_spline_symmetrize.py 0p0 216 0p98 3 1
pygvec load-quasr -f surface_3_0p0_216_0p98.nc -v --stellsym --tol 1e-4 --clean 2e-3
python add_flag.py surface_3_0p0_216_0p98-parameters.toml
#python3 GVEC_Boozer_spline_symmetrize2.py 0p0 216 0p82 2
pygvec run surface_3_0p0_216_0p98-parameters.toml

python3 GVEC_Boozer_spline_symmetrize.py 0p5 256 1p00 3 1
pygvec load-quasr -f surface_3_0p5_256_1p00.nc -v --stellsym --tol 1e-4 --clean 3e-3
python add_flag.py surface_3_0p5_256_1p00-parameters.toml
#python3 GVEC_Boozer_spline_symmetrize2.py 0p5 100 0p87 2
pygvec run surface_3_0p5_256_1p00-parameters.toml

python3 GVEC_Boozer_spline_symmetrize.py 1p0 128 1p00 3 1
pygvec load-quasr -f surface_3_1p0_128_1p00.nc -v --stellsym --tol 1e-4 --clean 3e-3
python add_flag.py surface_3_1p0_128_1p00-parameters.toml
#python3 GVEC_Boozer_spline_symmetrize2.py 1p0 216 0p90 2
pygvec run surface_3_1p0_128_1p00-parameters.toml

python3 GVEC_Boozer_spline_symmetrize.py 1p5 216 0p95 3 1
pygvec load-quasr -f surface_3_1p5_216_0p95.nc -v --stellsym --tol 1e-4 --clean 3e-3
python add_flag.py surface_3_1p5_216_0p95-parameters.toml
#python3 GVEC_Boozer_spline_symmetrize2.py 1p5 128 0p87 2
pygvec run surface_3_1p5_216_0p95-parameters.toml

python3 GVEC_Boozer_spline_symmetrize.py 2p0 152 0p98 3 1
pygvec load-quasr -f surface_3_2p0_152_0p98.nc -v --stellsym --tol 1e-4 --clean 3e-3
python add_flag.py surface_3_2p0_152_0p98-parameters.toml
#python3 GVEC_Boozer_spline_symmetrize2.py 2p0 128 0p83 2
pygvec run surface_3_2p0_152_0p98-parameters.toml



