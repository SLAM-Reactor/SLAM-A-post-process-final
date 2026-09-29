#!/bin/bash
#SBATCH -J slam_res_bundle
#SBATCH --account=m5194
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 10:00:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/res_scan/logs/slam_res_bundle_%j.out

# SLAM 128/0.65/clean1e-4 (Poincare comparison) resolution scan: all five
# cases side by side on ONE node (billing is per node). Boundary file is
# byte-identical to the SLAM Poincare-comparison run. Past runs peaked at
# ~7.5 GB each. Walltime = estimated longest case (nE16 (32,40), ~6 h at
# 12 threads, UNVERIFIED) with margin.

set -uo pipefail
unset LD_LIBRARY_PATH

module load conda
conda activate gvec-env

cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC/res_scan/slam_res_scan

for c in "16 24 32" "24 24 32" "8 16 24" "8 32 40" "16 32 40"; do
  set -- $c
  tag="nelems${1}_mn${2}_${3}"
  mkdir -p "${tag}"
  cp surface_2_SLAM_1p0_128_0p65_clean1em4_axis_x1m0p01-Gframe.nc "${tag}/"
  sed -e "s/TAG/${tag}/" -e "s/MM/${2}/g" -e "s/NN/${3}/g" -e "s/NELEMS/${1}/" \
    template-parameters.toml > "${tag}/${tag}-parameters.toml"
  (cd "${tag}" && OMP_NUM_THREADS=48 pygvec run "${tag}-parameters.toml" > "${tag}.log" 2>&1; \
   echo "${tag} exit=$? $(date)") &
done
wait
grep -a -o "GVEC finished.*minimize_tol[^)]*)" */nelems*.log
