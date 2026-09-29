#!/bin/bash
#SBATCH -J ftgen_prod
#SBATCH --qos=premium
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus 4
#SBATCH -c 96
#SBATCH --mem=200G
#SBATCH --account=m5194_g
#SBATCH --time=04:00:00
#SBATCH -o data/ftgen-prod-%j.out

# Fresh FTGen production.  Submit FROM the FTGen directory:
#
#     cd .../FTGen && sbatch run_ftgen.sl
#
# Every path here is relative.  The -o line above is resolved by Slurm against
# the submission directory, and the cd below uses $SLURM_SUBMIT_DIR, so this
# job script moves with the tree and needs no editing.
#
#   coils/         inputs, read only
#   data/traces    boundary_RZ_data_*.npz   (the expensive field-line traces)
#   data/alpha     alpha_data_*.npz         (native (alpha, phi) arrays)
#   data/pest      PEST_data_*.npz
#   data/boozer    Boozer_data_*.npz + Boozer_splines_*.pkl
#   data/figures   test*.png
#
# boozer_spline.py reuses data/traces/*.npz if present, so re-running after a
# code change is cheap.  Delete data/traces to force a from-scratch trace.

unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

module load conda
conda activate desc-env

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
echo "running in: $PWD"

FNAME=${FNAME:-1p0}
NTURNS=${NTURNS:-"102 128 150 196 216 256"}
RHOS=${RHOS:-"0.92 0.94"}
NGPU=${NGPU:-4}

# Build the case list, then run NGPU at a time.
cases=()
for rho in $RHOS; do
  for n in $NTURNS; do
    cases+=("$n $rho")
  done
done
echo "${#cases[@]} cases, ${NGPU} at a time"

i=0
for c in "${cases[@]}"; do
  gpu=$(( i % NGPU ))
  echo "launch: boozer_spline.py $FNAME $c (gpu $gpu)"
  python3 -u boozer_spline.py $FNAME $c $gpu &
  i=$(( i + 1 ))
  if (( i % NGPU == 0 )); then wait; fi
done
wait

echo "=== done ==="
ls -la data/traces data/alpha data/pest data/boozer
