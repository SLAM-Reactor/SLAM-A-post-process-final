#!/bin/bash
#SBATCH -J slnam_res
#SBATCH --account=m5194
#SBATCH --constraint=cpu
#SBATCH --qos=premium
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/res_scan/logs/slnam_res_%x_%j.out

# SLNAM 216/0.92 resolution scan. Boundary file is byte-identical to the
# production SLNAM run. Set per job with
#   sbatch -J <tag> -t <time> --export=ALL,NELEMS=..,MM=..,NN=.. job_gvec_slnam_res.sl

set -euo pipefail
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=12

module load conda
conda activate gvec-env

tag="nelems${NELEMS}_mn${MM}_${NN}"
cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/res_scan/slnam_res_scan
mkdir -p "${tag}"
cp surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01-Gframe.nc "${tag}/"
sed -e "s/TAG/${tag}/" -e "s/MM/${MM}/g" -e "s/NN/${NN}/g" -e "s/NELEMS/${NELEMS}/" \
  template-parameters.toml > "${tag}/${tag}-parameters.toml"
cd "${tag}"
grep -n "mn_max\|nElems\|totalIter\|minimize_tol" "${tag}-parameters.toml"

pygvec run "${tag}-parameters.toml"
