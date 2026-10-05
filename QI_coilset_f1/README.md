# Figure 1: OP precursor equilibrium and coil set

Inputs (from the DESC omnigenity database):

- `eq_NFP2_A+9.0_..._vacuum_final2.h5`: the NFP = 2 vacuum OP equilibrium.
- `NFP2_A8p7_coilset_40_coils_Bn_0p015_N9.h5`: the 40-coil OP coil set.

Everything for the figure is in `rhop9/`:

| Panel | Script | Cached data | Figure |
|---|---|---|---|
| 1a, Boozer \|B\| | `plot_boozer_rhop9.py` | `Boozer_DESC_rhop9.npz` | `Boozer_DESC_rhop9.{pdf,svg,png}` |
| 1b, coils + B·n/⟨\|B\|⟩ | `plot_equilibrium_coils_rhop9.py`, `compute_Bn_error_rhop9.py` | `QI_rhop9_Bn_error_data.npz` | `QI_rhop9_surface_and_40_coils_Bn.{pdf,svg}` (exported from the `_plotly.html`) |
| 1c, Poincaré vs. flux surfaces | `job_trace_poincare_gpu.sl` → `trace_poincare_gpu.py` (GPU), then `plot_poincare_comparison.py` | `QI_poincare_rhop9_data.npz` | `QI_poincare_rhop9_comparison.{pdf,svg,png}` |

`run_postprocess_rhop9.sh` runs the three plotting scripts (desc-env2). The
scripts are flat; settings are variables near the top of each file.

The coil design notebook (`coil_stage_two_optimization.ipynb`) is not included:
the only copy found is truncated at 512 KB and cannot be opened.
