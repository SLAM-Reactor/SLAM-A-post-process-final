#!/bin/bash
#SBATCH -A m5194
#SBATCH -C cpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH -t 01:30:00
#SBATCH -J gvec_boozer
#SBATCH -o logs/gvec_boozer-%j.out
# Boozer |B| planes of the best GVEC SLAM / SLNAM equilibria (gvec_boozer_plane.py; user 09-30 00:05). Six identical jobs
# share the six (state, rho) tasks through lock files; one surface per child process (150 GB SLAM_BEST, 320 GB SLNAM_BEST
# per surface - jobs 58849241 / 58849240). -t: the ripple jobs incl. the Boozer transform took 21 min (SLAM_BEST) / 65 min
# (SLNAM_BEST) per surface; this computes only the transform (no radial derivative, no bounce integrals) -> 65 x 1.3 ~ 1:30.
unset LD_LIBRARY_PATH
export OMP_NUM_THREADS=32
module load conda
conda activate gvec-env
cd "$SLURM_SUBMIT_DIR"
echo "$(date +%H:%M:%S) start on $(hostname)"
python -u gvec_boozer_plane.py
echo "$(date +%H:%M:%S) end"
