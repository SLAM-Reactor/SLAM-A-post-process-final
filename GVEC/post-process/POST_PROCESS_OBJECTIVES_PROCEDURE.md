# Post-Process Objectives And Procedure

This document is the detailed handoff for the GVEC post-processing workflow.
It records what we are trying to prove, how the scripts are organized, what
data products are cached, and the traps already encountered.

All paths are relative to:

```bash
cd /pscratch/sd/r/rgaur/mirrors/SLAM/SLAM-A-post-process/GVEC/post-process
```

## Objective

The post-processing has three physics objectives.

1. Boundary verification by Poincare comparison.

   Trace the magnetic field from the DESC coilsets using DESC, then compare the
   resulting Poincare points against the flux-surface cross-sections from the
   final GVEC equilibria.

   The comparison only needs the two toroidal cuts:

   ```text
   phi = 0
   phi = pi / NFP
   ```

   This demonstrates that the GVEC runs used the intended coil-produced
   boundary, rather than a visually similar but wrong toroidal or mirrored
   surface.

2. Effective-ripple comparison.

   Compute effective ripple for:

   ```text
   SLAM GVEC equilibrium
   SLNAM GVEC equilibrium
   original DESC equilibrium
   ```

   The goal is to compare the neoclassical quality of the two GVEC equilibria
   against the original DESC equilibrium using cached data, not by recomputing
   the bounce integrals every time a plot is made.

3. Magnetic-well comparison.

   Compute magnetic-well information for:

   ```text
   SLAM GVEC equilibrium
   SLNAM GVEC equilibrium
   original DESC equilibrium
   ```

   The goal is to compare the radial stability proxy from DESC with the closest
   GVEC-side equivalents, while being explicit about convention differences.

## Final Equilibrium Inputs

SLAM GVEC:

```text
../parameter_surface_2_SLAM_1p0_216_0p92_clean3e-3_axis_x1m0p01_final.ini
../surface_2_SLAM_1p0_216_0p92_clean3e-3_axis_x1m0p01_State_final.dat
```

SLNAM GVEC:

```text
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

The script `paths.py` is the single source of truth for these paths.

## Cache Contract

Every expensive calculation writes an `.npz` cache under:

```text
data/
```

Plotting scripts read only these caches. They should not retrace field lines or
rerun ripple/well calculations.

Expected final caches:

```text
data/gvec_sections_for_poincare.npz
data/desc_coil_poincare_traces.npz
data/effective_ripple_gvec.npz
data/effective_ripple_desc_original.npz
data/magnetic_well_gvec.npz
data/magnetic_well_desc_original.npz
```

Expected figures:

```text
figures/poincare_vs_gvec_SLAM.png
figures/poincare_vs_gvec_SLNAM.png
figures/effective_ripple_comparison.png
figures/magnetic_well_comparison.png
```

## Environment Matrix

Do not assume one environment can import both DESC and GVEC.

Use `gvec-env` for:

```text
export_gvec_sections.py
compute_gvec_effective_ripple.py
combine_gvec_effective_ripple.py
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
plot_poincare_vs_gvec.py
plot_effective_ripple.py
plot_magnetic_well.py
```

Why this split exists:

```text
gvec-env imports GVEC but not DESC.
desc-env and desc-env2 import DESC but not GVEC.
desc-env2 was the environment already used for earlier GPU Poincare traces.
```

The scripts communicate through `.npz` files, not through mixed imports.

## Poincare Procedure

### Step 1: Export GVEC Sections And Trace Seeds

Run:

```bash
sbatch jobs/job_export_gvec_sections.sl
```

Script:

```text
export_gvec_sections.py
```

Environment:

```text
gvec-env
```

Output:

```text
data/gvec_sections_for_poincare.npz
```

This file contains:

```text
labels
rho
theta
phi_planes
R_phi
Z_phi
zeta_at_phi
R_logical
Z_logical
phi_at_logical
seed_R0
seed_Z0
seed_theta
bounds_R
bounds_Z
metadata
```

Important convention:

```text
R_phi, Z_phi
  GVEC surfaces interpolated to constant physical cylindrical phi.
  This is the default comparison against DESC coil Poincare traces.

R_logical, Z_logical
  GVEC surfaces sampled at constant GVEC logical zeta.
  Use this only to diagnose coordinate-convention mistakes.
```

Seed selection:

```text
seed_R0, seed_Z0 are taken from the outboard point of the GVEC surface at phi=0.
```

Reason:

```text
The field-line trace must start on the same surface that GVEC is being asked to
represent. Guessing R0/Z0 by hand is unreliable. Pulling the seed from GVEC
keeps the comparison tied to the actual final equilibrium.
```

Bounds:

```text
bounds_R and bounds_Z are practical integration bounds derived from the GVEC
sections plus padding.
```

Reason:

```text
DESC field_line_integrate is faster and safer when it is not allowed to wander
into irrelevant R/Z space.
```

Current production setting:

```text
The submitted trace job overrides the GVEC-derived bounds with broader bounds:
R = 0.3, 2.6 and Z = -1.0, 1.0.
```

Reason:

```text
The tighter GVEC-derived bounds caused field_line_integrate to return a cache
where only the initial phi=0 seed points were finite. The corrected trace uses
the broader bounds, source_grid=LinearGrid(N=32), options={"throw": False}, and
finite-fraction validation.
```

### Step 2: Trace DESC Coil Fields

Run:

```bash
sbatch jobs/job_trace_desc_poincare.sl
```

Script:

```text
trace_desc_poincare.py
```

Environment:

```text
desc-env2
```

Output:

```text
data/desc_coil_poincare_traces.npz
```

This uses:

```python
from desc.io import load
from desc.magnetic_fields import field_line_integrate
```

Trace cuts:

```text
phi = 0
phi = pi / NFP
```

Trace seeds:

```text
seed_R0 and seed_Z0 from data/gvec_sections_for_poincare.npz
```

Trace bounds:

```text
bounds_R and bounds_Z from data/gvec_sections_for_poincare.npz
```

### Step 3: Plot Poincare Comparison

Run after both Poincare caches exist:

```bash
python plot_poincare_vs_gvec.py
```

Outputs:

```text
figures/poincare_vs_gvec_SLAM.png
figures/poincare_vs_gvec_SLNAM.png
```

If surfaces and Poincare points are mirrored, do not retrace first. Try plotting
diagnostics:

```bash
python plot_poincare_vs_gvec.py --mirror-gvec-z
python plot_poincare_vs_gvec.py --mirror-poincare-z
python plot_poincare_vs_gvec.py --use-logical-zeta
```

Reason:

```text
GVEC uses its own logical coordinate convention. A mirrored or shifted section
can be a coordinate-convention issue, not a wrong coil trace.
```

## Effective-Ripple Procedure

### DESC Original Equilibrium

Run:

```bash
sbatch jobs/job_desc_effective_ripple.sl
```

Script:

```text
compute_desc_effective_ripple.py
```

Environment:

```text
desc-env
```

Output:

```text
data/effective_ripple_desc_original.npz
```

Method:

```text
DESC `old effective ripple` / Bounce1D-style path.
```

Default rho grid:

```text
0.2, 0.4, 0.6, 0.8, 0.92, 0.95
```

### GVEC Equilibria

The GVEC ripple calculation uses the independent implementation in:

```text
/pscratch/sd/r/rgaur/GVEC_benchmark/GVEC_eps_eff
```

The all-in-one job is:

```bash
sbatch jobs/job_gvec_effective_ripple.sl
```

However, the all-in-one job is not recommended after this run. It OOM-killed
after completing SLAM and part of SLNAM. Use split jobs instead:

```bash
sbatch jobs/job_gvec_effective_ripple_SLAM.sl
sbatch jobs/job_gvec_effective_ripple_SLNAM.sl
```

Outputs:

```text
data/effective_ripple_gvec_SLAM.npz
data/effective_ripple_gvec_SLNAM.npz
```

Then combine:

```bash
sbatch jobs/job_combine_gvec_effective_ripple.sl
```

Combined output:

```text
data/effective_ripple_gvec.npz
```

### Effective-Ripple Plot

Run after both DESC and combined GVEC caches exist:

```bash
python plot_effective_ripple.py
```

Output:

```text
figures/effective_ripple_comparison.png
```

## Magnetic-Well Procedure

### DESC Original Equilibrium

Run:

```bash
sbatch jobs/job_desc_magnetic_well.sl
```

Script:

```text
compute_desc_magnetic_well.py
```

Output:

```text
data/magnetic_well_desc_original.npz
```

This saves DESC's native quantity:

```text
magnetic well
```

### GVEC Equilibria

Run:

```bash
sbatch jobs/job_gvec_magnetic_well.sl
```

Script:

```text
compute_gvec_magnetic_well.py
```

Output:

```text
data/magnetic_well_gvec.npz
```

Important convention warning:

```text
DESC `magnetic well` and GVEC `vacuum_magnetic_well_depth` are not the same
registered quantity.
```

GVEC's native registered scalar is:

```text
vacuum_magnetic_well_depth =
(dV/dPhi_n(axis) - dV/dPhi_n(edge)) / dV/dPhi_n(axis)
```

The GVEC script therefore saves two profiles:

```text
desc_style_well
  V(r) * d<|B|^2>/dr / (V_r(r) * <|B|^2>) for this vacuum case.
  This is the closest match to DESC's native magnetic-well proxy with p_r=0.

cumulative_vacuum_well
  (dV/dPhi_n(axis) - dV/dPhi_n(rho)) / dV/dPhi_n(axis).
  This is the radial analogue of GVEC's native edge scalar.
```

### Magnetic-Well Plot

Run:

```bash
python plot_magnetic_well.py
```

Output:

```text
figures/magnetic_well_comparison.png
```

The plot has two panels:

```text
DESC-style proxy comparison
GVEC native cumulative vacuum-well convention
```

## Recommended Full Run Order

Use this order from a clean state:

```bash
cd /pscratch/sd/r/rgaur/mirrors/SLAM/SLAM-A-post-process/GVEC/post-process

sbatch jobs/job_export_gvec_sections.sl

# After data/gvec_sections_for_poincare.npz exists:
sbatch jobs/job_trace_desc_poincare.sl

sbatch jobs/job_desc_effective_ripple.sl
sbatch jobs/job_desc_magnetic_well.sl
sbatch jobs/job_gvec_magnetic_well.sl

# GVEC ripple should be split by case to avoid memory growth:
sbatch jobs/job_gvec_effective_ripple_SLAM.sl
sbatch jobs/job_gvec_effective_ripple_SLNAM.sl

# After both per-case GVEC ripple jobs complete:
sbatch jobs/job_combine_gvec_effective_ripple.sl

# After all final caches exist:
sbatch jobs/job_plot_all.sl
```

Check status:

```bash
sacct -j <jobid> --format=JobID,JobName%24,State,Elapsed,AllocCPUS,MaxRSS,ExitCode -P
```

## Current Run Status

Completed successfully:

```text
58450608  gvec_sections   -> data/gvec_sections_for_poincare.npz
58450609  desc_poincare   -> data/desc_coil_poincare_traces.npz
58450611  desc_ripple     -> data/effective_ripple_desc_original.npz
58450612  gvec_well       -> data/magnetic_well_gvec.npz
58450613  desc_well       -> data/magnetic_well_desc_original.npz
```

Failed or obsolete:

```text
58450610  gvec_ripple     FAILED quickly because of a wrong import in the first adapter.
58450682  gvec_ripple     OUT_OF_MEMORY after completing SLAM and SLNAM through rho=0.8.
58450614  gvec_pp_plot    CANCELLED because its dependency chain included the failed ripple job.
58450684  gvec_pp_plot    CANCELLED after being left with DependencyNeverSatisfied.
```

The missing final cache is:

```text
data/effective_ripple_gvec.npz
```

Use the split GVEC ripple jobs and combiner to produce it.

## Pitfalls Already Encountered

### 1. DESC And GVEC Are Not In One Environment

Do not write a single script that imports both DESC and GVEC.

What happened:

```text
gvec-env could import GVEC but not DESC.
desc-env and desc-env2 could import DESC but not GVEC.
```

Resolution:

```text
Use `.npz` cache boundaries between GVEC-only and DESC-only scripts.
```

### 2. GVEC Logical Zeta Is Not Necessarily Physical Cylindrical Phi

The Poincare trace is a physical cylindrical-phi diagnostic.

What can go wrong:

```text
GVEC cross-sections sampled at logical zeta can look shifted or mirrored
relative to DESC coil Poincare points.
```

Resolution:

```text
export_gvec_sections.py saves constant-physical-phi sections as R_phi/Z_phi.
plot_poincare_vs_gvec.py uses those by default.
R_logical/Z_logical are only for diagnosing convention mistakes.
```

### 3. Mirror Images Are A Known Diagnostic Possibility

What can go wrong:

```text
The Poincare points and GVEC surface curves can appear as mirror images.
```

Resolution:

```bash
python plot_poincare_vs_gvec.py --mirror-gvec-z
python plot_poincare_vs_gvec.py --mirror-poincare-z
python plot_poincare_vs_gvec.py --use-logical-zeta
```

Do this before retracing field lines.

### 4. R0/Z0 Seeds Should Come From GVEC

What can go wrong:

```text
Starting field lines from hand-guessed R0/Z0 can land on the wrong surface.
```

Resolution:

```text
export_gvec_sections.py picks the outboard point on each GVEC surface at phi=0.
trace_desc_poincare.py uses those seeds.
```

### 5. DESC Field-Line Tracing Should Use Practical Bounds

What can go wrong:

```text
Unbounded field-line tracing is slower and can wander into irrelevant R/Z space.
```

Resolution:

```text
export_gvec_sections.py derives bounds_R and bounds_Z from the GVEC surfaces.
trace_desc_poincare.py passes those to field_line_integrate.
```

### 6. The First GVEC Ripple Adapter Had A Bad Import

What happened:

```text
build_spline_and_bounce_points_rho was imported from _bounce_points.py.
In this checkout it actually lives in _find_bounce_pairs.py.
```

Correct imports:

```python
from _bounce_points import bounce_quad_to_zeta
from _find_bounce_pairs import build_spline_and_bounce_points_rho
```

### 7. The All-In-One GVEC Ripple Job OOM-Killed

What happened:

```text
The all-in-one job completed all SLAM rho values, then SLNAM through rho=0.8,
then was OOM-killed at 128G.
```

Resolution:

```text
Run SLAM and SLNAM as separate jobs.
Combine the per-case NPZ files afterward.
```

### 8. A Failed Dependency Can Leave A Pending Plot Job

What happened:

```text
plot job 58450684 was left pending with DependencyNeverSatisfied because it
depended on the failed/OOM GVEC ripple job.
```

Resolution:

```bash
scancel <stale-plot-jobid>
```

Then submit a new plot job after the split GVEC ripple cache has been combined.

### 9. GVEC And DESC Magnetic-Well Names Do Not Mean The Same Thing

What can go wrong:

```text
Comparing DESC `magnetic well` directly to GVEC `vacuum_magnetic_well_depth`
as if they were identical is misleading.
```

Resolution:

```text
Save both:
  desc_style_well
  cumulative_vacuum_well

Plot and label them separately.
```

### 10. Smoke Files Are Diagnostics Only

Smoke-test commands may write small `.npz` files. Do not use those in final
plots.

Examples:

```text
data/smoke_desc_coil_poincare_traces.npz
data/smoke_effective_ripple_gvec.npz
```

Delete or clearly ignore them.

## Recovery Procedure From The Current State

The completed caches already present are:

```text
data/gvec_sections_for_poincare.npz
data/desc_coil_poincare_traces.npz
data/effective_ripple_desc_original.npz
data/magnetic_well_gvec.npz
data/magnetic_well_desc_original.npz
```

To finish from here:

```bash
cd /pscratch/sd/r/rgaur/mirrors/SLAM/SLAM-A-post-process/GVEC/post-process

sbatch jobs/job_gvec_effective_ripple_SLAM.sl
sbatch jobs/job_gvec_effective_ripple_SLNAM.sl
```

After both per-case GVEC ripple jobs complete:

```bash
sbatch jobs/job_combine_gvec_effective_ripple.sl
```

After `data/effective_ripple_gvec.npz` exists:

```bash
sbatch jobs/job_plot_all.sl
```

## Useful Verification Commands

List cache files:

```bash
find data -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

List figures:

```bash
find figures -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
```

Inspect cache keys:

```bash
python - <<'PY'
import numpy as np
for f in [
    "data/gvec_sections_for_poincare.npz",
    "data/desc_coil_poincare_traces.npz",
    "data/effective_ripple_gvec.npz",
    "data/effective_ripple_desc_original.npz",
    "data/magnetic_well_gvec.npz",
    "data/magnetic_well_desc_original.npz",
]:
    d = np.load(f, allow_pickle=True)
    print(f)
    for k in d.files:
        print("  ", k, getattr(d[k], "shape", "scalar"))
PY
```

Check relevant jobs:

```bash
sacct -j <jobid> --format=JobID,JobName%24,State,Elapsed,AllocCPUS,MaxRSS,ExitCode -P
```

## Do Not Repeat These

Do not:

```text
Use hand-picked R0/Z0 seeds when GVEC surface seeds are available.
Compare logical GVEC zeta sections to physical coil Poincare traces without
checking the physical-phi section.
Treat a mirrored plot as a failed trace before trying the mirror diagnostics.
Run the combined SLAM+SLNAM GVEC ripple job again unless memory usage has been
fixed.
Compare DESC `magnetic well` directly to GVEC `vacuum_magnetic_well_depth`
without naming the convention.
Leave DependencyNeverSatisfied plot jobs in the queue and assume they will run.
```
