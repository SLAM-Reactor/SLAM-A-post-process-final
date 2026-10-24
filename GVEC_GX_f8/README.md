# Nonlinear ITG heat-flux scan

GX runs for pre-SLAM (`SLNAM_BEST`), SLAM (`SLAM_BEST`), and the precursor OP
(`QI_precursor`), at rho=0.65 with five-poloidal-turn flux tubes. The hybrids
use BEST GVEC equilibria; OP uses DESC -> VMEC ns512 -> pyvmec geometry.

## Scan

Nine field lines, alpha/pi = 0, .125, .25, .375, .5, .625, .75, .875, 1,
at a/LTi = 1, 1.25, 1.5, 2, 2.5, 3: 162 total outputs. The original
five-alpha scan is retained; 72 added runs fill the four intervening alphas.
All run to t=600 except pre-SLAM/SLAM at a/LTi=1.25, which run to t=1200.

- `gx_runs/best_gvec_t600_alpha5`: original hybrid runs, excluding 1.25.
- `gx_runs/best_gvec_t1200_alpha5`: original hybrid runs at 1.25.
- `gx_runs/qi_precursor_pyvmec_ns512_t600_alpha5`: original OP runs.
- `gx_runs/refined_alpha9`: the 72 added inputs and diagnostic outputs;
  `manifest.txt` lists paths relative to this directory.
- `data`: GX geometry, including the four additional alphas for all cases.

Added inputs point to geometry via relative paths from each run directory.
Large original outputs previously omitted from Git are now included after
lossless NetCDF compression. No numerical values or GX normalizations are
changed by compression. Restart files, large field dumps, and run logs are
not included in this figure-data repository.

## Reproduce plots

With Python, NumPy, Matplotlib, and netCDF4 installed:

```bash
python plot_gx_heat_flux_scan_best.py
python plot_gx_peak_time_traces.py
```

The scan plotter reads repository-relative outputs and rewrites:

- `figures/gx_Q_vs_alpha_aLT3p0.{pdf,png}`: nine-alpha scan at a/LTi=3.
- `figures/gx_Q_vs_aLT_max_alpha.{pdf,png}`: maximum over all nine alphas
  separately at each gradient.
- `figures/gx_Q_vs_alpha_all_gradients.{pdf,png}`: six gradient panels.
- `gx_runs/best_gvec_t600_alpha5/heat_flux_per_fieldline.csv`: 162 rows.
- `gx_runs/best_gvec_t600_alpha5/heat_flux_max_alpha.csv`: 18 selected rows.

The time-trace script writes
`figures/gx_Q_vs_t_alpha3pi8_peaks.{pdf,png,npz}` and `_stats.json`, showing
full histories and late-time detail for pre-SLAM/SLAM at alpha=3pi/8,
a/LTi=3. The NPZ contains both cases' time and Q_i arrays.

## Conventions

Q_i/Q_GB is GX `Diagnostics/HeatFlux_st[:, 0]`, used as reported. Scan
points are arithmetic means of the final 100 diagnostics; error bars are
the temporal standard deviations of those samples. Diagnostic time spacing
may differ between runs. The gradient plot selects the largest mean flux
across alpha, with that run's temporal standard deviation as the error bar.
No grad-rho or other additional normalization is applied.

Alpha=theta-iota*zeta labels the field line; consistently transformed
Boozer and PEST coordinates describe the same label. Spatial plotting-origin
choices do not alter alpha or the heat-flux data.

## Execution provenance

The added a/LTi=3 runs were completed using regular job 60150410 and
Jupyter allocation 60185129. The other 60 added runs were completed using
Jupyter allocations 60185129 and 60202337. Physics settings and gradient-
matched numerical inputs were retained; only new field lines were simulated.
