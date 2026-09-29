# GVEC 0p92 Reproducibility Runbook

This is the human-readable handoff for the completed SLAM and SLNAM GVEC
equilibria built from the QFM FTGen surfaces at:

```text
rho = 0.92
nturns = 216
fname1 = 1p0
fname2 = 0p92
NFP = 2
```

The purpose of this file is simple: reproduce the successful path without
rediscovering the theta orientation, smoothing, axis shift, or GVEC settings.

All paths below are relative to:

```bash
cd /pscratch/sd/r/rgaur/mirrors/SLAM/SLAM-A-post-process/GVEC
```

## One-Screen Summary

Use this branch:

```text
reverse theta: yes
SLAM clean: 3e-3
SLNAM clean: 1e-3
axis move: average_axis_move_X1 = -0.01, average_axis_move_X2 = 0.0
GVEC radial resolution: sgrid_nElems = 8
PHIEDGE = 0.01
CPU threads: OMP_NUM_THREADS = 12
Slurm QOS: premium
```

Do this before any full solve:

```text
1. Activate gvec-env.
2. Run the login-node smoke check for SLAM.
3. Run the login-node smoke check for SLNAM.
4. Submit run_gvec_0p92_cpu.sl only after both smoke checks pass.
```

The completed high-resolution Slurm job was:

```text
job id: 58430713
state: COMPLETED
elapsed: 00:26:48
```

## Final High-Resolution Equilibria

These are the production equilibrium outputs:

```text
surface_2_SLAM_1p0_216_0p92_clean3e-3_axis_x1m0p01_State_final.dat
parameter_surface_2_SLAM_1p0_216_0p92_clean3e-3_axis_x1m0p01_final.ini

surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01_State_final.dat
parameter_surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01_final.ini
```

Interpretation:

```text
State_final.dat: final GVEC equilibrium state
parameter_..._final.ini: final GVEC parameter file written by GVEC
clean3e-3 / clean1e-3: pygvec load-quasr smoothing used before GVEC
axis_x1m0p01: average axis shifted by X1 = -0.01
```

The final parameter files both contain:

```text
average_axis_move_X1 = -0.01
average_axis_move_X2 = 0
PHIEDGE = 0.01
sgrid_nElems = 8
```

The final logs reported:

```text
SLAM:  |force| = 1.00e-07, rms delta iota = 5.67e-14, 26529 iterations
SLNAM: |force| = 1.00e-07, rms delta iota = 1.28e-14, 38194 iterations
```

## Directory Organization

Keep the top level for reproducible production inputs, scripts, and final
products:

```text
README.md
GVEC_0p92_REPRODUCIBILITY.md
boozer_spline.py
GVEC_Boozer_spline_symmetrize.py
patch_gvec_params.py
run_gvec_smoke_check.py
run_ftgen_qfm.sl
run_gvec_0p92_cpu.sl
surface_2_*_axis_x1m0p01_State_final.dat
parameter_surface_2_*_axis_x1m0p01_final.ini
gvec_0p92-*58430713*
```

Keep FTGen outputs flat. Do not make extra subdirectories inside these:

```text
data/SLAM/
data/SLNAM/
```

Keep failed trials, smoke-test products, and plots here:

```text
diagnostics_0p92/
```

Important diagnostic names in `diagnostics_0p92/`:

```text
surface_diagnostics_1p0_216_0p92.png
smoke_*
parameter_smoke_*
*noflip*
*noavgaxis*
*nelems5*
*initLAfalse*
```

Those are not production equilibria.

## Data Flow

The successful workflow is:

```text
QFM/coil inputs
  -> FTGen using boozer_spline.py
  -> data/<label>/Boozer_splines_1p0_216_0p92.pkl
  -> GVEC_Boozer_spline_symmetrize.py with --reverse-theta
  -> symmetrized surface_2_<label>...nc
  -> pygvec load-quasr with case-specific --clean
  -> <base>-parameters.toml
  -> patch_gvec_params.py
  -> pygvec run
  -> State_final.dat and parameter_*_final.ini
```

The high-resolution GVEC run used these FTGen spline files:

```text
data/SLAM/Boozer_splines_1p0_216_0p92.pkl
data/SLNAM/Boozer_splines_1p0_216_0p92.pkl
```

The `boozer_spline.py` in this directory is the FTGen copy used for this scan.
Do not substitute an older TQ-side copy.

## Source Inputs

The FTGen staging in this directory references:

```text
coils/SLNAM-field-09-09-26.h5
coils/SLAM-field-09-09-26_2.h5
coils/eq_NFP2_A+9.0_tor+0.0_mr+0.10_shift+0.00_wd+0.00_elong+3.00_wellwt+2.0_vacuum_final2.h5
reference_surfaces/SLNAM_QFM_surface_09-09-26_3.h5
reference_surfaces/SLAM_QFM_surface_09-09-26_2.h5
```

The FTGen scan script is:

```text
run_ftgen_qfm.sl
```

The scan covered:

```text
LABELS = SLNAM SLAM
FNAME  = 1p0
NTURNS = 128 196 216 256
RHOS   = 0.78 0.82 0.85 0.87 0.90 0.92
```

For the high-resolution GVEC equilibria documented here, use only the
`nturns=216`, `rho=0.92` FTGen spline products listed above.

## Reproduce From Existing FTGen Splines

Activate the GVEC environment:

```bash
module load conda 2>/dev/null || module load anaconda3/2023.3
conda activate gvec-env
cd /pscratch/sd/r/rgaur/mirrors/SLAM/SLAM-A-post-process/GVEC
```

Run cheap initialization checks on the login node first:

```bash
python3 run_gvec_smoke_check.py \
  --label SLAM \
  --clean 3e-3 \
  --axis-move-x1 -0.01 \
  --axis-move-x2 0.0

python3 run_gvec_smoke_check.py \
  --label SLNAM \
  --clean 1e-3 \
  --axis-move-x1 -0.01 \
  --axis-move-x2 0.0
```

A passing check ends with:

```text
SMOKE PASS
```

Only then submit the full CPU solve:

```bash
sbatch run_gvec_0p92_cpu.sl
```

The Slurm script runs both labels concurrently with:

```text
SBATCH --account=m5194
SBATCH --constraint=cpu
SBATCH --qos=premium
SBATCH --ntasks=2
SBATCH -c 12
SBATCH --mem=128G
SBATCH --time=04:00:00
OMP_NUM_THREADS=12
OM_NUM_THREADS=12
```

The script writes:

```text
gvec_0p92-SLAM-<jobid>.log
gvec_0p92-SLNAM-<jobid>.log
gvec_0p92-<jobid>.out
```

## Exact Per-Case Production Settings

SLAM:

```text
base name: surface_2_SLAM_1p0_216_0p92_clean3e-3_axis_x1m0p01
FTGen spline: data/SLAM/Boozer_splines_1p0_216_0p92.pkl
reverse theta: yes
pygvec load-quasr --clean: 3e-3
axis move X1: -0.01
axis move X2: 0.0
sgrid_nElems: 8
PHIEDGE: 0.01
```

SLNAM:

```text
base name: surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01
FTGen spline: data/SLNAM/Boozer_splines_1p0_216_0p92.pkl
reverse theta: yes
pygvec load-quasr --clean: 1e-3
axis move X1: -0.01
axis move X2: 0.0
sgrid_nElems: 8
PHIEDGE: 0.01
```

## What The Production Script Does

For each case, `run_gvec_0p92_cpu.sl` does the following.

First it symmetrizes the FTGen surface and reverses theta:

```bash
python3 -u GVEC_Boozer_spline_symmetrize.py \
  --label <SLAM-or-SLNAM> \
  --fname1 1p0 \
  --nturns 216 \
  --fname2 0p92 \
  --nfp 2 \
  --reverse-theta \
  --output-name <base>
```

Then it converts the surface for GVEC:

```bash
pygvec load-quasr \
  -f <base>.nc \
  -v \
  --stellsym \
  --tol 1e-4 \
  --clean <case-specific-clean> \
  --name <base>
```

Then it patches the generated GVEC parameter file:

```bash
python3 -u patch_gvec_params.py <base>-parameters.toml \
  --n-elems 8 \
  --phi-edge 0.01 \
  --init-average-axis true \
  --axis-move-x1 -0.01 \
  --axis-move-x2 0.0
```

Then it runs GVEC:

```bash
pygvec run <base>-parameters.toml
```

## Why These Choices Were Made

Theta:

```text
No theta flip gave negative Jacobian essentially from the axis outward.
Reverse theta moved the problem to the boundary before axis/smoothing fixes.
Therefore reverse theta is the correct branch for these surfaces.
```

Axis:

```text
The useful manual axis shift was average_axis_move_X1 = -0.01.
X2 was kept at 0.0.
This fixed the initialization sign issue together with the correct smoothing.
```

Smoothing:

```text
SLNAM was smooth enough with clean = 1e-3.
SLAM required clean = 3e-3.
SLAM with only clean = 1e-3 and the axis shift still failed.
```

Radial resolution:

```text
The production high-resolution equilibria use sgrid_nElems = 8.
Lower nElems was only a diagnostic branch, not the final result.
```

## Known Bad Or Diagnostic Branches

These were tested and should not be treated as final equilibria:

```text
no theta flip
init_average_axis = false
nElems = 5
init_LA = false
SLAM clean = 1e-3 with only the axis shift
```

Related files were moved under:

```text
diagnostics_0p92/
```

## Negative Jacobian Triage

Common failure:

```text
InitializationError: NEGATIVE JACOBIAN FOUND AFTER INITIALIZATION!
```

How to read it:

```text
Negative Jacobian throughout the radius usually means wrong theta orientation.
Negative Jacobian mostly near the boundary usually means surface smoothness or
axis placement is still bad.
```

What worked here:

```text
1. Use --reverse-theta.
2. Use average_axis_move_X1 = -0.01.
3. Use clean = 3e-3 for SLAM.
4. Use clean = 1e-3 for SLNAM.
5. Keep the full solve on CPU with OMP_NUM_THREADS = 12.
```

## Verification Commands

Check a Slurm job:

```bash
sacct -j <jobid> --format=JobID,JobName%24,State,Elapsed,AllocCPUS,MaxRSS,ExitCode -P
```

Check final logs and parameter files:

```bash
rg -n "GVEC finished|sgrid_nElems|ERROR|rms|force" \
  gvec_0p92-SLAM-<jobid>.log \
  gvec_0p92-SLNAM-<jobid>.log \
  parameter_surface_2_*_axis_x1m0p01_final.ini
```

For the documented successful job, the important result is:

```text
job 58430713 completed
both final INIs have sgrid_nElems = 8
both final INIs have average_axis_move_X1 = -0.01
both logs reached |force| = 1.00e-07
```

## Reproducibility Rules

Use these rules for future reruns:

```text
1. Do not create nested directories under data/SLAM or data/SLNAM.
2. Do not use an old TQ-side boozer_spline.py.
3. Do not submit the full GVEC solve before the login-node smoke checks pass.
4. Do not treat files in diagnostics_0p92/ as production equilibria.
5. Keep the case-specific clean values unless a new surface scan proves otherwise.
6. Keep the manual axis shift unless a new smoke check proves another axis is better.
7. Keep the final high-resolution run at sgrid_nElems = 8.
```
