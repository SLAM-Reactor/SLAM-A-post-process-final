# GVEC FTGen and Equilibrium Staging

This directory stages the QFM-surface coilsets for FTGen, symmetrizes the
selected FTGen surface, then runs GVEC. The important reproducibility note for
the completed `rho=0.92`, `nturns=216` equilibria is:

```text
GVEC_0p92_REPRODUCIBILITY.md
```

Read that runbook before changing or rerunning GVEC. In particular, do the
login-node smoke checks before submitting any full Slurm solve.

## FTGen Source

Canonical FTGen generator:

```text
boozer_spline.py
```

This file is copied byte-for-byte from:

```text
/pscratch/sd/r/rgaur/mirrors/SLAM/SLAM-A-post-process/FTGen/boozer_spline.py
```

Inputs:

```text
coils/SLNAM-field-09-09-26.h5
coils/SLAM-field-09-09-26_2.h5
coils/eq_NFP2_A+9.0_tor+0.0_mr+0.10_shift+0.00_wd+0.00_elong+3.00_wellwt+2.0_vacuum_final2.h5
reference_surfaces/SLNAM_QFM_surface_09-09-26_3.h5
reference_surfaces/SLAM_QFM_surface_09-09-26_2.h5
```

Run or rerun the FTGen scan with:

```bash
cd /pscratch/sd/r/rgaur/mirrors/SLAM/SLAM-A-post-process/GVEC
sbatch run_ftgen_qfm.sl
```

Current scan:

```text
LABELS = SLNAM SLAM
FNAME  = 1p0
NTURNS = 128 196 216 256
RHOS   = 0.78 0.82 0.85 0.87 0.90 0.92
```

Outputs are separated by field label:

```text
data/SLNAM/
data/SLAM/
```

The FTGen stopping products are the `Boozer_data_*.npz` and
`Boozer_splines_*.pkl` files under each `data/<label>/` directory.

The high-resolution GVEC run used:

```text
data/SLAM/Boozer_splines_1p0_216_0p92.pkl
data/SLNAM/Boozer_splines_1p0_216_0p92.pkl
```

## Completed High-Resolution GVEC Products

Final equilibria:

```text
surface_2_SLAM_1p0_216_0p92_clean3e-3_axis_x1m0p01_State_final.dat
parameter_surface_2_SLAM_1p0_216_0p92_clean3e-3_axis_x1m0p01_final.ini

surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01_State_final.dat
parameter_surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01_final.ini
```

Both final parameter files use `sgrid_nElems = 8`.

## Reproducible GVEC Workflow

Activate the GVEC environment:

```bash
module load conda 2>/dev/null || module load anaconda3/2023.3
conda activate gvec-env
cd /pscratch/sd/r/rgaur/mirrors/SLAM/SLAM-A-post-process/GVEC
```

Run cheap local smoke checks first:

```bash
python3 run_gvec_smoke_check.py --label SLAM --clean 3e-3 --axis-move-x1 -0.01
python3 run_gvec_smoke_check.py --label SLNAM --clean 1e-3 --axis-move-x1 -0.01
```

Submit the full high-resolution CPU solve only after both smoke checks pass:

```bash
sbatch run_gvec_0p92_cpu.sl
```

Critical production settings are:

```text
reverse theta: yes
axis move: average_axis_move_X1=-0.01, average_axis_move_X2=0.0
SLAM clean: 3e-3
SLNAM clean: 1e-3
nElems: 8
OMP_NUM_THREADS: 12
```

See `GVEC_0p92_REPRODUCIBILITY.md` for the full provenance, known bad branches,
and verification commands.

## Post-Processing

The post-GVEC comparison workflow lives in:

```text
post-process/
```

Read this first:

```text
post-process/POST_PROCESS_OBJECTIVES_PROCEDURE.md
```

It documents the Poincare boundary comparison, effective-ripple comparison,
magnetic-well comparison, cached `.npz` files, environment split, and known
pitfalls from the first run.
