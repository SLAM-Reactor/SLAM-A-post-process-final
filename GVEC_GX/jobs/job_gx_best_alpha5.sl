#!/bin/bash
#SBATCH -J gx_best
#SBATCH --account=m5194_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --array=0-69
#SBATCH --time=00:45:00
#SBATCH -o /pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_GX/logs/gx_best_%A_%a.out

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

RUN_ROOT=/pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_GX/gx_runs/best_gvec_t600_alpha5
MANIFEST=${RUN_ROOT}/manifest.txt
GX=/global/homes/r/rgaur/gx/gx

mapfile -t RUN_DIRS < "${MANIFEST}"
RUN_DIR=${RUN_DIRS[$SLURM_ARRAY_TASK_ID]}

echo "RUN_DIR=${RUN_DIR}"
cd "${RUN_DIR}"
rm -f gx.out.nc gx.big.nc gx.restart.nc

srun -n 4 --gpus-per-task=1 --gpu-bind=none "${GX}" gx.in
