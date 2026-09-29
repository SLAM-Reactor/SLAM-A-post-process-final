# GVEC_GX Clean Working Tree

Backup:

`/pscratch/sd/r/rgaur/mirrors/SLAM/SLAM-A-post-process/GVEC_GX_bkp`

The backup is the full pre-cleanup copy. The cleaned `GVEC_GX` tree keeps only:

- GX input files under `gx_runs`
- restored `gx.out.nc` diagnostics needed to recompute heat fluxes
- geometry files and geometry-generation TOML/log files under `data`
- scripts that generate geometry, generate GX inputs, or make retained Q-vs-t / geometry plots
- retained plots under `figures`
- SLURM job scripts that correspond to retained input trees

`gx_runs` (updated 2026-09-24):

- `best_gvec_t600_alpha5` - SLAM_BEST and SLNAM_BEST (best GVEC equilibria from the 09-23 resolution scans), 70 runs, `gx.in` + `gx.out.nc`
- `qi_precursor_pyvmec_ns512_t600_alpha5` - QI precursor OP with DESC->VMEC ns512->pyvmec geometry (`data/gx_geometry_vmec_QI_precursor_ns512_*.eik.nc`, geo_option nc), 35 runs, job 58828140 (09-24); inputs from `make_qi_pyvmec_ns512_inputs.py`, job `jobs/job_gx_qi_pyvmec_ns512.sl`. This is the QI case used in the plots. 34/35 clean to t=600; alpha0p50pi/aLT1p50 went NaN at t~10 (not rerun), so the QI a/LT=1.5 point is left out. The DESC-direct QI runs (wrong geometry, Q=1239 vs 42 on alpha=0.5pi) were moved to `../GVEC_GX_bkp/moved_old_equilibria_09-24/gx_runs/`.

Old-equilibrium runs (old SLAM_N128 / old SLNAM: t600, t1200, t2400), their geometry,
figures, jobs and scripts, and the 09-24 kxfac/pyvmec test, were moved to
`../GVEC_GX_bkp/moved_old_equilibria_09-24/`.

Removed from the cleaned tree:

- `gx.big.nc` and `gx.restart.nc`
- failed/debug/smoke/fixaspect/npol2 benchmark run trees
- old heat-flux-vs-a/LT scan CSVs and scan figures
- Python cache, run logs, local dependency scratch, and stale W7X comparison material
- lower-time duplicates of the exact SLAM `alpha0p50pi`, `a/L_T=1.5` run superseded by the `t=2400` output

Conventions:

- Scripts are hardcoded and have editable variables near the top.
- No retained Python script uses argparse.
- Retained Python scripts use at most three top-level helper functions.
- Q vs a/LT uses the GX-reported Q_i of the highest-flux field line; error bar = last-100 time std. No `<|grad rho|>` re-normalization.
