# GVEC Post-Processing Runbook

This directory contains the reproducible post-processing workflow for the two
high-resolution GVEC equilibria:

```text
SLAM
SLNAM
```

The detailed objectives, procedure, current status, and known pitfalls are in:

```text
POST_PROCESS_OBJECTIVES_PROCEDURE.md
```

The workflow has three goals:

```text
1. Compare DESC coil-field Poincare traces with GVEC flux surfaces.
2. Compare effective ripple for SLAM GVEC, SLNAM GVEC, and the original DESC equilibrium.
3. Compare magnetic-well profiles for SLAM GVEC, SLNAM GVEC, and the original DESC equilibrium.
```

All expensive outputs are cached as `.npz` files in `data/`. Plotting scripts
read those caches and do not recompute field-line traces, ripple, or well data.

Run from:

```bash
cd /pscratch/sd/r/rgaur/mirrors/SLAM/SLAM-A-post-process/GVEC/post-process
```

## Directory Layout

```text
post-process/
  README.md
  paths.py
  export_gvec_sections.py
  trace_desc_poincare.py
  plot_poincare_vs_gvec.py
  compute_gvec_effective_ripple.py
  combine_gvec_effective_ripple.py
  compute_desc_effective_ripple.py
  plot_effective_ripple.py
  compute_gvec_magnetic_well.py
  compute_desc_magnetic_well.py
  plot_magnetic_well.py
  data/
  figures/
  logs/
  jobs/
```

`paths.py` is the single place where the final GVEC equilibria, original DESC
equilibrium, and DESC coilsets are named.

## Inputs

GVEC final equilibria:

```text
../parameter_surface_2_SLAM_1p0_216_0p92_clean3e-3_axis_x1m0p01_final.ini
../surface_2_SLAM_1p0_216_0p92_clean3e-3_axis_x1m0p01_State_final.dat

../parameter_surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01_final.ini
../surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01_State_final.dat
```

Original DESC equilibrium:

```text
../coils/eq_NFP2_A+9.0_tor+0.0_mr+0.10_shift+0.00_wd+0.00_elong+3.00_wellwt+2.0_vacuum_final2.h5
```

DESC coilsets:

```text
../coils/SLAM-field-09-09-26_2.h5
../coils/SLNAM-field-09-09-26.h5
```

## Environment Split

The local environments do not all contain both DESC and GVEC.

Use `gvec-env` for:

```text
export_gvec_sections.py
compute_gvec_effective_ripple.py
compute_gvec_magnetic_well.py
```

Use `desc-env2` for:

```text
trace_desc_poincare.py
```

Use `desc-env` for:

```text
compute_desc_effective_ripple.py
compute_desc_magnetic_well.py
plot_*.py
```

## Poincare Comparison

The Poincare comparison is intentionally split into two cache steps.

First, in `gvec-env`, export GVEC surfaces, practical R/Z trace bounds, and
seed points:

```bash
sbatch jobs/job_export_gvec_sections.sl
```

This writes:

```text
data/gvec_sections_for_poincare.npz
```

Important detail: this file contains two GVEC cross-section conventions.

```text
R_phi, Z_phi:
  GVEC surfaces interpolated onto constant physical cylindrical phi.
  This is the default comparison against the DESC coil Poincare trace.

R_logical, Z_logical:
  GVEC surfaces sampled at constant GVEC logical zeta.
  This is saved only to diagnose coordinate-convention or mirror-image issues.
```

The DESC trace starts from the outboard point of each GVEC surface in the
physical `phi=0` plane. This avoids guessing R0/Z0 by hand.

Then, in `desc-env2`, trace the DESC coil field:

```bash
sbatch jobs/job_trace_desc_poincare.sl
```

This writes:

```text
data/desc_coil_poincare_traces.npz
```

The trace is only sampled at:

```text
phi = 0
phi = pi/NFP
```

The trace script can use R/Z bounds from the GVEC surface cache, but the
production job intentionally overrides them with broader DESC trace bounds:

```text
R = 0.3, 2.6
Z = -1.0, 1.0
```

The tighter GVEC-derived bounds produced a misleading cache with only the seed
points finite. The script now prints finite-fraction diagnostics and fails if
any seed has poor finite coverage.

Plot after both caches exist:

```bash
python plot_poincare_vs_gvec.py
```

If the GVEC and coil traces look like mirror images, do not retrace first.
Try plotting variants:

```bash
python plot_poincare_vs_gvec.py --mirror-gvec-z
python plot_poincare_vs_gvec.py --mirror-poincare-z
python plot_poincare_vs_gvec.py --use-logical-zeta
```

Figures are written to:

```text
figures/poincare_vs_gvec_SLAM.png
figures/poincare_vs_gvec_SLNAM.png
```

## Effective Ripple

GVEC-side ripple uses the independent GVEC bounce calculation from:

```text
/pscratch/sd/r/rgaur/GVEC_benchmark/GVEC_eps_eff
```

Run:

```bash
sbatch jobs/job_gvec_effective_ripple.sl
```

This writes:

```text
data/effective_ripple_gvec.npz
```

If the all-in-one GVEC ripple job runs out of memory, use the split jobs
instead:

```bash
sbatch jobs/job_gvec_effective_ripple_SLAM.sl
sbatch jobs/job_gvec_effective_ripple_SLNAM.sl

# After both complete:
sbatch jobs/job_combine_gvec_effective_ripple.sl
```

The split jobs write:

```text
data/effective_ripple_gvec_SLAM.npz
data/effective_ripple_gvec_SLNAM.npz
```

The combiner restores the plotting input:

```text
data/effective_ripple_gvec.npz
```

DESC-side ripple uses DESC's `old effective ripple` / Bounce1D-style path on
the original DESC equilibrium:

```bash
sbatch jobs/job_desc_effective_ripple.sl
```

This writes:

```text
data/effective_ripple_desc_original.npz
```

Plot after both caches exist:

```bash
python plot_effective_ripple.py
```

Figure:

```text
figures/effective_ripple_comparison.png
```

Default rho grid:

```text
0.2, 0.4, 0.6, 0.8, 0.92, 0.95
```

Override with, for example:

```bash
python compute_gvec_effective_ripple.py --rho 0.2,0.4,0.6,0.8,0.92
python compute_desc_effective_ripple.py --rho 0.2,0.4,0.6,0.8,0.92
```

## Magnetic Well

DESC's native quantity is:

```text
magnetic well
```

GVEC's native well quantity is not the same object. GVEC registers a scalar:

```text
vacuum_magnetic_well_depth
```

defined as:

```text
(dV/dPhi_n(axis) - dV/dPhi_n(edge)) / dV/dPhi_n(axis)
```

For the radial comparison, the GVEC script therefore saves two GVEC-side
profiles:

```text
desc_style_well:
  V(r) * d<|B|^2>/dr / (V_r(r) * <|B|^2>) for this vacuum case.
  This matches DESC's magnetic-well proxy formula with p_r = 0.

cumulative_vacuum_well:
  (dV/dPhi_n(axis) - dV/dPhi_n(rho)) / dV/dPhi_n(axis).
  Its edge value matches GVEC's native scalar well depth.
```

Run:

```bash
sbatch jobs/job_gvec_magnetic_well.sl
sbatch jobs/job_desc_magnetic_well.sl
```

These write:

```text
data/magnetic_well_gvec.npz
data/magnetic_well_desc_original.npz
```

Plot:

```bash
python plot_magnetic_well.py
```

Figure:

```text
figures/magnetic_well_comparison.png
```

## Full Run Order

Recommended order:

```bash
sbatch jobs/job_export_gvec_sections.sl

# After data/gvec_sections_for_poincare.npz exists:
sbatch jobs/job_trace_desc_poincare.sl

sbatch jobs/job_gvec_effective_ripple.sl
sbatch jobs/job_desc_effective_ripple.sl
sbatch jobs/job_gvec_magnetic_well.sl
sbatch jobs/job_desc_magnetic_well.sl

# After all data files exist:
sbatch jobs/job_plot_all.sl
```

Check job status:

```bash
sacct -j <jobid> --format=JobID,JobName%24,State,Elapsed,AllocCPUS,MaxRSS,ExitCode -P
```

## Expected Cache Files

```text
data/gvec_sections_for_poincare.npz
data/desc_coil_poincare_traces.npz
data/effective_ripple_gvec.npz
data/effective_ripple_desc_original.npz
data/magnetic_well_gvec.npz
data/magnetic_well_desc_original.npz
```

Once these exist, all plotting should be local and cheap.

## Cheap Smoke Tests

Before running a full Poincare trace, make a tiny trace:

```bash
module load conda
conda activate desc-env2
python trace_desc_poincare.py --ntransit 3 --output data/smoke_desc_coil_poincare_traces.npz
```

Before running full ripple, test one rho:

```bash
module load conda
conda activate gvec-env
python compute_gvec_effective_ripple.py --rho 0.8 --nz 301 --npitch 9 --num-well 40 --output data/smoke_effective_ripple_gvec.npz
```

These smoke files are diagnostics only. Do not use them for final plots.
