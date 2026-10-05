# GVEC post-processing: Figures 6 and 7

All equilibrium, coil and DESC paths are in `paths.py`. Expensive results are
cached in `data/`; the `plot_*.py` scripts only read those caches, so replotting is
cheap. Slurm jobs are in `jobs/` (they write logs to `logs/`, created on demand).

## Figure 6: GVEC surfaces vs. coil Poincaré traces

Cases `SLNAM` (pre-SLAM) and `SLAM_N128_RHO065_CLEAN1E4` (SLAM).

1. `jobs/job_export_gvec_sections*.sl` → `export_gvec_sections.py` (gvec-env) →
   `data/gvec_sections_for_poincare*.npz`
2. `jobs/job_trace_desc_poincare*.sl` → `trace_desc_poincare.py` (desc-env2, GPU) →
   `data/desc_coil_poincare_traces*.npz`
3. `plot_poincare_vs_gvec.py` → `figures/poincare_vs_gvec_SLNAM.pdf`,
   `figures/poincare_vs_gvec_SLAM_N128_RHO065_CLEAN1E4.pdf`

## Figure 7: effective ripple and magnetic well

Cases `SLNAM_BEST`, `SLAM_BEST` and the DESC OP equilibrium.

- GVEC ripple, one ρ per job: `jobs/job_ripple_SLNAM_BEST_rho.sl`,
  `jobs/job_ripple_mnf3_rho.sl` → `compute_gvec_effective_ripple.py` →
  `data/effective_ripple_gvec_<case>_rho*.npz`. This uses the external code
  `GVEC_eps_eff` (https://github.com/rahulgaur104/GVEC_eps_eff).
- GVEC well: `jobs/job_well_best.sl` → `compute_gvec_magnetic_well.py` →
  `data/magnetic_well_gvec_best.npz`
- DESC OP: `jobs/job_desc_effective_ripple.sl`, `jobs/job_desc_magnetic_well.sl` →
  `data/effective_ripple_desc_original.npz`, `data/magnetic_well_desc_original.npz`
- Plots: `plot_effective_ripple_best.py`, `plot_magnetic_well_best.py` →
  `figures/effective_ripple_comparison_best.pdf`,
  `figures/magnetic_well_comparison_best.pdf`

## Environments

- gvec-env: `export_gvec_sections.py`, `compute_gvec_*.py`
- desc-env2: `trace_desc_poincare.py`
- desc-env: `compute_desc_*.py`, `plot_*.py`
