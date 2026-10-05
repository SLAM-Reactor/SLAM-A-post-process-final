#!/bin/bash
#SBATCH -J gvec_0p92
#SBATCH --account=m5194
#SBATCH --constraint=cpu
#SBATCH --qos=premium
#SBATCH -N 1
#SBATCH --ntasks=2
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH -o gvec_0p92-%j.out

# Run run_gvec_smoke_check.py for both SLAM and SLNAM before submitting this
# production solve; the smoke check catches negative-Jacobian initialization
# failures without spending compute-node time.

set -euo pipefail

unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=12
export OM_NUM_THREADS=12

module load conda 2>/dev/null || module load anaconda3/2023.3
conda activate gvec-env

cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_f67

run_case() {
  local label=$1
  local clean=$2
  local base="surface_2_${label}_1p0_216_0p92_clean${clean}_axis_x1m0p01"

  echo "Running ${label} GVEC setup from ${base}"

  python3 -u GVEC_Boozer_spline_symmetrize.py \
    --label "${label}" \
    --fname1 1p0 \
    --nturns 216 \
    --fname2 0p92 \
    --nfp 2 \
    --reverse-theta \
    --output-name "${base}"

  pygvec load-quasr \
    -f "${base}.nc" \
    -v \
    --stellsym \
    --tol 1e-4 \
    --clean "${clean}" \
    --name "${base}"

  python3 -u patch_gvec_params.py "${base}-parameters.toml" \
    --n-elems 8 \
    --phi-edge 0.01 \
    --init-average-axis true \
    --axis-move-x1 -0.01 \
    --axis-move-x2 0.0

  pygvec run "${base}-parameters.toml"
}

run_case SLAM 3e-3 > "gvec_0p92-SLAM-${SLURM_JOB_ID}.log" 2>&1 &
pid_slam=$!

run_case SLNAM 1e-3 > "gvec_0p92-SLNAM-${SLURM_JOB_ID}.log" 2>&1 &
pid_slnam=$!

status=0
wait "${pid_slam}" || status=$?
wait "${pid_slnam}" || status=$?
exit "${status}"
