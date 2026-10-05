#!/bin/bash
#SBATCH -J qfm_ftgen
#SBATCH --qos=premium
#SBATCH --constraint=gpu&hbm80g
#SBATCH --gpus 4
#SBATCH -c 96
#SBATCH --mem=200G
#SBATCH --account=m5194_g
#SBATCH --time=04:00:00
#SBATCH -o ftgen-qfm-%j.out

# Submit from this directory:
#
#     cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_f67
#     sbatch run_ftgen_qfm.sl
#
# Default scan:
#   labels: SLNAM and SLAM
#   rho:    0.78, 0.82, 0.85, 0.87, 0.90, 0.92
#   nturns: 128, 196, 216, 256
#
# Override at submission, for example:
#   sbatch --export=ALL,RHOS="0.85 0.87",NTURNS="128 196 216" run_ftgen_qfm.sl

unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95

module load conda
conda activate desc-env

cd "${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "$0")")}"
echo "running in: $PWD"

FNAME=${FNAME:-1p0}
LABELS=${LABELS:-"SLNAM SLAM"}
NTURNS=${NTURNS:-"128 196 216 256"}
RHOS=${RHOS:-"0.78 0.82 0.85 0.87 0.90 0.92"}
NGPU=${NGPU:-4}
EQ_FILE=${EQ_FILE:-"eq_NFP2_A+9.0_tor+0.0_mr+0.10_shift+0.00_wd+0.00_elong+3.00_wellwt+2.0_vacuum_final2.h5"}

coil_for_label() {
  case "$1" in
    SLNAM) echo "SLNAM-field-09-09-26.h5" ;;
    SLAM)  echo "SLAM-field-09-09-26_2.h5" ;;
    *)
      echo "unknown label: $1" >&2
      return 1
      ;;
  esac
}

cases=()
for label in $LABELS; do
  coil_file=$(coil_for_label "$label") || exit 2
  if [[ ! -f "coils/$coil_file" ]]; then
    echo "missing coils/$coil_file" >&2
    exit 3
  fi
  for rho in $RHOS; do
    for n in $NTURNS; do
      rho_suffix=$(python3 - <<PY
rho = float("$rho")
print(f"{rho:.2f}".replace(".", "p"))
PY
)
      if [[ -f "data/$label/Boozer_splines_${FNAME}_${n}_${rho_suffix}.pkl" ]]; then
        echo "skip existing: $label nturns=$n rho=$rho"
        continue
      fi
      cases+=("$label $coil_file $n $rho")
    done
  done
done

echo "${#cases[@]} FTGen cases, ${NGPU} at a time"
printf '  %s\n' "${cases[@]}"

i=0
for c in "${cases[@]}"; do
  read -r label coil_file n rho <<< "$c"
  gpu=$(( i % NGPU ))
  export FTGEN_EQ="$EQ_FILE"
  export FTGEN_COIL_TEMPLATE="$coil_file"
  export FTGEN_DATA_DIR="$PWD/data/$label"
  mkdir -p "$FTGEN_DATA_DIR"
  echo "launch: $label coil=$coil_file nturns=$n rho=$rho gpu=$gpu"
  python3 -u boozer_spline.py "$FNAME" "$n" "$rho" "$gpu" &
  i=$(( i + 1 ))
  if (( i % NGPU == 0 )); then wait; fi
done
wait

echo "=== done ==="
find data -maxdepth 2 -type f \( -name '*.npz' -o -name '*.pkl' -o -name '*.png' \) | sort
