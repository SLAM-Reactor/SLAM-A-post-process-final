#!/bin/bash
#SBATCH -J gvec_slam_scan
#SBATCH --account=m5194
#SBATCH --constraint=cpu
#SBATCH --qos=premium
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -c 12
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH -o gvec_slam_scan-%j.out

set -euo pipefail

unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=12
export OM_NUM_THREADS=12

module load conda 2>/dev/null || module load anaconda3/2023.3
conda activate gvec-env

cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC

: "${NTURNS:?set NTURNS, e.g. 128}"
: "${RHO:?set RHO, e.g. 0.65}"
CLEAN=${CLEAN:-3e-3}
CLEAN_TAG=${CLEAN_TAG:-3e-3}
LOAD_TOL=${LOAD_TOL:-1e-4}
MINIMIZE_TOL=${MINIMIZE_TOL:-1e-7}
FIT_M_MAX=${FIT_M_MAX:-14}
FIT_N_MAX=${FIT_N_MAX:-20}
FIT_TAG=${FIT_TAG:-}

rho_suffix=$(python3 - <<PY
rho = float("$RHO")
print(f"{rho:.2f}".replace(".", "p"))
PY
)

label=SLAM
base="surface_2_SLAM_1p0_${NTURNS}_${rho_suffix}_clean${CLEAN_TAG}${FIT_TAG}_axis_x1m0p01"

echo "Running ${label} GVEC setup from ${base}"

python3 -u GVEC_Boozer_spline_symmetrize.py \
  --label "${label}" \
  --fname1 1p0 \
  --nturns "${NTURNS}" \
  --fname2 "${rho_suffix}" \
  --nfp 2 \
  --reverse-theta \
  --m-max "${FIT_M_MAX}" \
  --n-max "${FIT_N_MAX}" \
  --output-name "${base}"

pygvec load-quasr \
  -f "${base}.nc" \
  -v \
  --stellsym \
  --tol "${LOAD_TOL}" \
  --clean "${CLEAN}" \
  --name "${base}"

python3 -u patch_gvec_params.py "${base}-parameters.toml" \
  --n-elems 8 \
  --phi-edge 0.01 \
  --total-iter 100000 \
  --minimize-tol "${MINIMIZE_TOL}" \
  --init-average-axis true \
  --axis-move-x1 -0.01 \
  --axis-move-x2 0.0

pygvec run "${base}-parameters.toml"
