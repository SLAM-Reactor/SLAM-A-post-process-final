#!/bin/bash
#SBATCH -J gx_qi_rerun
#SBATCH --account=m5194_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_GX/logs/gx_qi_rerun_dt0p02_%j.out

# Rerun of QI_precursor alpha=0.5pi a/LT=1.5 (pyvmec ns512), NaN at t~10.6 in job 58828140_15 with dt CFL-cut to 0.042.
# Same input except dt = 0.02 (30000 steps, ~38 min). GX is stopped as soon as the log shows NaN.
set -euo pipefail

module purge
module load PrgEnv-nvidia
module load nvidia/25.5
module load cudatoolkit/12.9
module load cray-mpich
module load nccl/2.24.3
module load cray-hdf5-parallel
module load cray-netcdf-hdf5parallel
module load python

export MPICH_GPU_SUPPORT_ENABLED=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216

RUN_DIR=/pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_GX/gx_runs/qi_precursor_pyvmec_ns512_t600_alpha5/QI_precursor/alpha0p50pi/aLT1p50
LOG=/pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_GX/logs/gx_qi_rerun_dt0p02_${SLURM_JOB_ID}.out
GX=/global/homes/r/rgaur/gx/gx

cd "${RUN_DIR}"
rm -f gx.out.nc gx.big.nc gx.restart.nc

srun -n 4 --gpus-per-task=1 --gpu-bind=none "${GX}" gx.in &
GX_PID=$!
while kill -0 "${GX_PID}" 2>/dev/null; do
  sleep 60
  if grep -q "Q_i = nan" "${LOG}"; then
    echo "NaN in log, stopping GX"
    kill "${GX_PID}"
    exit 1
  fi
done
wait "${GX_PID}"

# only gx.out.nc is used; drop the 1.8 GB big/restart files
rm -f gx.big.nc gx.restart.nc
