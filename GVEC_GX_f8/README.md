# GVEC_GX: nonlinear ITG heat-flux data

GX heat-flux data for the GVEC best equilibria and the OP precursor. This
directory contains the retained outputs needed to reproduce the two separate
paper-style GX panels:

- `figures/gx_Q_vs_alpha_aLT3p0.{pdf,png}` - `Q_i/Q_GB` versus field-line label
  at `a/LT_i = 3`.
- `figures/gx_Q_vs_aLT_max_alpha.{pdf,png}` - `Q_i/Q_GB` versus `a/LT_i`,
  using the field line with the largest heat flux at each gradient.

The full pre-cleanup backup is:

```text
/home/rgaur/GVEC_GX_backup_20261005_032031
```

## Files

- `plot_gx_heat_flux_scan_best.py` - the only retained Python script. It reads
  the retained `gx.out.nc` files, rewrites the CSV summaries, and redraws the two
  paper-style figures.
- `gx_runs/best_gvec_t600_alpha5/heat_flux_per_fieldline.csv` - one row per
  case, field line, and retained gradient.
- `gx_runs/best_gvec_t600_alpha5/heat_flux_max_alpha.csv` - one row per case and
  retained gradient after applying the max-alpha rule.
- `data/*.eik.nc` - GX geometry files for SLAM_BEST, SLNAM_BEST, and
  QI_precursor. They are kept for provenance; the retained plotting script reads
  the GX run outputs directly.
- `data/qi_precursor_vmec_ns512_*.toml` and `data/gx_geometry_vmec_*.log` -
  provenance files for the QI precursor DESC-to-VMEC-to-pyvmec geometry.
- `gx_runs/*/manifest.txt` and `gx_runs/manifest_aLT1p25.txt` - retained run
  manifests matching the kept run directories.

## Retained Data

Cases:

- `SLNAM_BEST` - pre-SLAM, from the best GVEC equilibrium in the 09-23
  resolution scans.
- `SLAM_BEST` - SLAM, from the best GVEC equilibrium in the 09-23 resolution
  scans.
- `QI_precursor` - original OP / DESC OP precursor, exported through VMEC ns512
  and pyvmec geometry.

Field lines:

```text
alpha0p00, alpha0p25pi, alpha0p50pi, alpha0p75pi, alpha1p00pi
```

Retained gradients:

```text
a/LT_i = 1.0, 1.25, 1.5, 2.0, 2.5, 3.0
```

Run trees:

- `gx_runs/best_gvec_t600_alpha5` - SLAM_BEST and SLNAM_BEST t=600 runs for
  `a/LT_i = 1.0, 1.5, 2.0, 2.5, 3.0`. The superseded t=600 `a/LT_i = 1.25`
  SLAM/SLNAM outputs were removed.
- `gx_runs/best_gvec_t1200_alpha5` - SLAM_BEST and SLNAM_BEST t=1200 runs for
  `a/LT_i = 1.25` only. These replace the removed t=600 `a/LT_i = 1.25`
  SLAM/SLNAM outputs.
- `gx_runs/qi_precursor_pyvmec_ns512_t600_alpha5` - QI_precursor t=600 runs for
  all retained gradients, including `a/LT_i = 1.25`.

Removed data:

- `a/LT_i = 3.5` and `a/LT_i = 4.0` runs.
- `gx.big.nc`, `gx.restart.nc`, logs, job scripts, old run-generation scripts,
  old geometry-export scripts, old time-history plots, and debug/window plots.

## Redraw

Use the DESC environment with `netCDF4`:

```bash
cd /home/rgaur/SLAM-A-post-process-final/GVEC_GX_f8
/home/rgaur/anaconda3/envs/desc-env3/bin/python plot_gx_heat_flux_scan_best.py
```

This rewrites:

```text
figures/gx_Q_vs_alpha_aLT3p0.pdf
figures/gx_Q_vs_alpha_aLT3p0.png
figures/gx_Q_vs_aLT_max_alpha.pdf
figures/gx_Q_vs_aLT_max_alpha.png
gx_runs/best_gvec_t600_alpha5/heat_flux_per_fieldline.csv
gx_runs/best_gvec_t600_alpha5/heat_flux_max_alpha.csv
```

## Conventions

- `Q_i/Q_GB` is read directly from GX `Diagnostics/HeatFlux_st[:, 0]`.
- The value and error bar are the mean and standard deviation of the last 100
  diagnostics.
- The `a/LT_i` panel uses the alpha with the largest mean heat flux at each
  gradient; alpha spread is not used as an error bar.
- No `<|grad rho|>` re-normalization is applied.
- Plot styling follows Figure 8 of the paper: separate panels,
  no titles, inward ticks, simple in-axis legends, and labels
  `Q_i/Q_GB`, `alpha`, and `a/LT_i`.

## Current Results

Max-alpha values after the cleanup:

| case | alpha used | a/LT_i | Q_i/Q_GB |
|---|---|---:|---:|
| pre-SLAM | alpha0p75pi | 1.25 | 0.221 |
| SLAM | alpha0p50pi | 1.25 | 0.266 |
| DESC OP | alpha0p50pi | 1.25 | 0.952 |
| pre-SLAM | alpha0p75pi | 3.0 | 30.026 |
| SLAM | alpha0p50pi | 3.0 | 51.046 |
| DESC OP | alpha0p50pi | 3.0 | 41.889 |
