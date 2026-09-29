#!/bin/bash
#SBATCH -J gx_geo_best
#SBATCH --account=m5194
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_GX/logs/export_gx_geometry_best_%j.out

# GX geometry for SLAM_BEST / SLNAM_BEST at 5 field lines, then the GX inputs.
set -euo pipefail
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=32
module load conda
conda activate gvec-env
cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_GX
mkdir -p logs data

for a in "0.00 alpha0p00" "0.25 alpha0p25pi" "0.50 alpha0p50pi" "0.75 alpha0p75pi" "1.00 alpha1p00pi"; do
  set -- $a
  GX_ALPHA=$(python -c "import math; print($1*math.pi)") GX_ALPHA_LABEL=$2 python -u export_gvec_gx_geometry.py
done
python -u make_gx_best_alpha5_inputs.py
