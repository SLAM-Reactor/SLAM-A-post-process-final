# SLAMs: Stellarators Linking Axisymmetric Mirrors — figure data and scripts

This archive holds the inputs, outputs and scripts behind every figure in
*SLAMs: Stellarators Linking Axisymmetric Mirrors* (`SLAM_A_paper (3).pdf`).

Each top-level directory ends in `_f<figures>`, the paper figures it produces:
`_f1` is Figure 1, `_f23` is Figures 2 and 3, `_fA1` is Figure A1 (Appendix A), and so on.
Each directory holds the inputs (equilibria, coil sets, boundaries), the outputs
(cached `.npz`/`.nc` data and the figure files) and the scripts or Slurm jobs that
go from one to the other.

No step has to be rerun to replot a figure: the plotting scripts read the cached
data next to them. The expensive steps (field-line tracing, equilibrium solves, GX)
can be rerun from the files and jobs provided.

## Figure map

| Figure | What it shows | Directory | Figure file(s) |
|---|---|---|---|
| 1 | OP precursor: Boozer \|B\|, coils + B·n, coil Poincaré vs. flux surfaces | [`QI_coilset_f1`](QI_coilset_f1) | `rhop9/Boozer_DESC_rhop9.pdf`, `rhop9/QI_rhop9_surface_and_40_coils_Bn.pdf`, `rhop9/QI_poincare_rhop9_comparison.pdf` |
| 2 | Building the stellarator–mirror hybrid coil set | [`QI_to_SLNAM_f23`](QI_to_SLNAM_f23) | `final_html_figures/figure_{a,b,c}_*.pdf` |
| 3 | Hybrid coil sets at L/D = 1 and 2 and their Poincaré plots | [`QI_to_SLNAM_f23`](QI_to_SLNAM_f23) | `Poincare_plot/figure_d_hybrid_top_view_cross_sections.pdf`, `Poincare_plot/Poincare_plot/Poincare_plot_L_over_R_{1p0,2p0}_ntransit400.pdf` |
| 4 | Pre-SLAM and SLAM coil sets, QFM surfaces, \|B\| on axis | [`QFM_surf_f4`](QFM_surf_f4) | `SLNAM_QFM_surface_and_coils_Bn.pdf`, `SLAM_QFM_surface_and_coils_Bn.pdf`, `axis_B_comparison.pdf` |
| 5 | FTGen maps: (α, φ) → PEST → Boozer | [`FTGen_f5`](FTGen_f5) | `outfigures/FTGen_1p0_150_0p85_R_{alpha,pest,boozer}.pdf` |
| 6 | GVEC surfaces vs. coil Poincaré traces | [`GVEC_f67`](GVEC_f67) | `post-process/figures/poincare_vs_gvec_SLNAM.pdf`, `post-process/figures/poincare_vs_gvec_SLAM_N128_RHO065_CLEAN1E4.pdf` |
| 7 | Effective ripple and magnetic well | [`GVEC_f67`](GVEC_f67) | `post-process/figures/effective_ripple_comparison_best.pdf`, `post-process/figures/magnetic_well_comparison_best.pdf` |
| 8 | Nonlinear ITG heat flux from GX | [`GVEC_GX_f8`](GVEC_GX_f8) (+ [`DESC_GX_f8`](DESC_GX_f8)) | `figures/gx_Q_vs_alpha_aLT3p0.pdf`, `figures/gx_Q_vs_aLT_max_alpha.pdf` |
| A1a | NFP = 3 precursor OP (DESC) | [`Appendix-A_fA1/fA1a`](Appendix-A_fA1/fA1a) | `coils_op_modB_nfp3.pdf` |
| A1b | NFP = 3 hybrid (GVEC) | [`Appendix-A_fA1/fA1b`](Appendix-A_fA1/fA1b) | `SLNAM_coils_gvec_modB_nfp3.pdf` |
| B1 | Force-balance error, DESC (cylindrical angle) vs. GVEC (Boozer angle) | [`Appendix-B_fB1`](Appendix-B_fB1) | `publication_normF_SLNAM/normF_DESC_SLNAM_cyl_f9n18.pdf`, `publication_normF_SLNAM/normF_GVEC_SLNAM_BEST.pdf` |

Appendix C (SLAM shaping objectives) has no figures, so it has no directory.

Naming: **SLNAM** in file names is the *pre-SLAM* configuration in the paper,
**SLAM** is the SLAM configuration, and **QI / OP / QI_precursor** is the original
omnigenous precursor stellarator. File names say `L_over_R`; the paper calls
the same parameter L/D.

## How the pieces connect

```
OP equilibrium + OP coil set                        QI_coilset_f1/
        │  cut at the low-field point, add circular mirror coils
        ▼
hybrid (pre-SLAM) coil set, L/D = 0 … 2            QI_to_SLNAM_f23/   (SLAMify.ipynb)
        │  QFM surface + converter-coil optimization (Appendix C objectives)
        ▼
pre-SLAM and SLAM coil sets + QFM surfaces         QFM_surf_f4/       (QFM_…-slope2.ipynb)
        │  FTGen: trace one field line → (α, φ) → PEST → Boozer
        ▼
Boozer-angle boundary                              FTGen_f5/, GVEC_f67/  (boozer_spline.py)
        │  GVEC fixed-boundary solve
        ▼
GVEC equilibria                                    GVEC_f67/
        ├─ Poincaré check, ripple, well           GVEC_f67/post-process/
        ├─ GX flux-tube geometry + GX runs        GVEC_GX_f8/, DESC_GX_f8/
        └─ DESC vs. GVEC force balance            Appendix-B_fB1/
```

## Per-figure details

### Figure 1 — `QI_coilset_f1/`
- **Inputs:** `eq_NFP2_A+9.0_…_vacuum_final2.h5` (OP equilibrium) and
  `NFP2_A8p7_coilset_40_coils_Bn_0p015_N9.h5` (OP coil set, from the DESC
  omnigenity database).
- **Scripts:** `rhop9/run_postprocess_rhop9.sh` runs the Boozer plot, the coils + B·n
  plot and the Poincaré comparison. `rhop9/job_trace_poincare_gpu.sl` (GPU)
  traces the coil field and writes `QI_poincare_rhop9_data.npz`.
  `rhop9/compute_Bn_error_rhop9.py` writes `QI_rhop9_Bn_error_data.npz`.

### Figure 2 — `QI_to_SLNAM_f23/`
- **Inputs:** the OP coil set (`../QI_coilset_f1/`) and the hybrid coil sets
  `SLNAMified_NFP2_A8p7_coilset_L_over_R_{0p0,0p5,1p0,1p5,2p0}_SLAM-A.h5`, made in
  `SLAMify.ipynb`.
- **Script:** `final_html_figures/general_stellarator_mirror_figures_no_dash_a.py
  ORIGINAL_COILSET.h5 HYBRID_COILSET.h5` writes the interactive HTML figures; the
  PDF/SVG files were exported from them.

### Figure 3 — `QI_to_SLNAM_f23/Poincare_plot/`
- `Poincare_plot/job_trace_poincare_gpu.sl` → `trace_poincare_gpu.py` traces each
  L/D case (GPU) → `SLNAMified_L_over_R_*_poincare_data_ntransit400.npz` →
  `plot_four_poincare_pdfs.py` → the two PDFs (L/D = 1.0 and 2.0).

### Figure 4 — `QFM_surf_f4/`
- **Initial → final:** `QFM_SLNAM_to_SLAM_optimization_nonplanar-lower-finite-slope2.ipynb`
  starts from `SLNAMified_NFP2_A8p7_coilset_L_over_R_1p0_SLAM-A.h5` and the OP
  equilibrium. It produces the pre-SLAM QFM surface (`SLNAM_QFM_surface_09-09-26_3.h5`)
  and the SLAM coil set and surface (`SLAM-field-09-09-26_2.h5`,
  `SLAM_QFM_surface_09-09-26_2.h5`). `SLNAM-field-09-09-26.h5` is the pre-SLAM coil set.
  The notebook loads its starting coil set by an absolute path to the post-Simons
  directory; the identical file is here.
- **Scripts:** `plot_qfm_surfaces.py` (coils + QFM surface coloured by B·n/⟨|B|⟩,
  cached in `qfm_surface_Bn_data.npz`); `trace_axis_B.py` (axis tracing, expensive)
  → `axis_B_data.npz` → `post_process_axis_B.py`.

### Figure 5 — `FTGen_f5/`
- **Inputs:** `coils/` (OP equilibrium + `SLNAMified_NFP2_A8p7_coilset_L_over_R_1p0_SLAM-A-TQ.h5`).
- `run_ftgen.sl` → `boozer_spline.py FNAME NTURNS RHO GPU` → `data/{traces,alpha,pest,boozer}/`.
  Only the case in the figure, `1p0_150_0p85`, is kept.
- `plot_FTGen_coordinate_maps.py` (edit `CASE` at the top) → `outfigures/`.

### Figures 6 and 7 — `GVEC_f67/`
- **Equilibria** (`.ini` + `_State_final.dat` + `-Gframe.nc` boundary):
  - pre-SLAM for Fig. 6: `…SLNAM_1p0_216_0p92_clean1e-3…`
  - SLAM for Fig. 6: `…SLAM_1p0_128_0p65_clean1em4…`
  - Fig. 7 ("BEST", from the resolution scans): `SLNAM_216_0p92_nelems32_mn12_34_*`
    and `SLAM_128_0p65_nelems24_mn24_32_*`. These are also the equilibria behind
    Figures 8 and B1.
- **How the equilibria were made:** `run_ftgen_qfm.sl` (FTGen on `coils/` and
  `reference_surfaces/`) → `data/<label>/Boozer_splines_*.pkl` →
  `GVEC_Boozer_spline_symmetrize.py` + `patch_gvec_params.py` → `pygvec run`
  (`run_gvec_0p92_cpu.sl`, `run_gvec_slam_scan_case_cpu.sl`). The BEST runs come from
  `res_scan/{slnam,slam}_res_scan/` (template + job). See `GVEC_f67/README.md`.
- **Post-processing** (`post-process/`, all case paths in `paths.py`; see
  `GVEC_f67/post-process/README.md`):
  - Fig. 6: `jobs/job_export_gvec_sections*.sl` → `jobs/job_trace_desc_poincare*.sl` (GPU)
    → `plot_poincare_vs_gvec.py`
  - Fig. 7, GVEC: `jobs/job_ripple_SLNAM_BEST_rho.sl`, `jobs/job_ripple_mnf3_rho.sl`
    (one ρ per job) and `jobs/job_well_best.sl`
  - Fig. 7, DESC OP: `jobs/job_desc_effective_ripple.sl`, `jobs/job_desc_magnetic_well.sl`
  - Plots: `plot_effective_ripple_best.py`, `plot_magnetic_well_best.py`
  - The GVEC ripple uses the external code `GVEC_eps_eff`
    (https://github.com/rahulgaur104/GVEC_eps_eff).

### Figure 8 — `GVEC_GX_f8/` and `DESC_GX_f8/`
- **Geometry:** GX geometry files are in `GVEC_GX_f8/data/*.eik.nc`. SLAM and
  pre-SLAM come from the BEST GVEC equilibria; the OP goes DESC → VMEC
  (`DESC_GX_f8/export_qi_desc_to_vmec_ns512.py` →
  `DESC_GX_f8/data/wout_QI_precursor_OP_NFP2_A9_vacuum_final2_ns512.nc`) → pyvmec.
- **Runs:** `gx.in` and `gx.out.nc` files live under
  `gx_runs/*/<case>/<alpha>/<aLT>/`: 3 cases × 9 field lines × 6 gradients = 162 GX
  outputs, including 72 added field-line runs in `gx_runs/refined_alpha9`. SLAM/pre-SLAM `a/L_T = 1.25` come from t = 1200 runs; all other points
  come from t = 600 runs.
- **Plot:** `plot_gx_heat_flux_scan_best.py`. Q vs. a/L_T uses the GX-reported Q_i of
  the field line with the largest flux. Error bars are the standard deviation over
  the last 100 diagnostic times. See `GVEC_GX_f8/README.md`.

### Figure A1 — `Appendix-A_fA1/`
- **(a)** `fA1a/`: `coil_stage_one_optimization.ipynb` fits the NFP = 3 OP coil set
  `NFP3_coilset_42_coils_smoother_Bn_0p02_N5.h5` to
  `eq_NFP3_A+10.0_…_final2_resolve.h5`. `compute_op_surface_modB.py` (DESC) →
  `op_surface_modB_nfp3.npz` → `make_coils_modB_html_svg_op_nfp3.py` (gvec-env / pyvista).
- **(b)** `fA1b/coils_and_gvec_NFP3/`: `SLAMify5.ipynb` (hybrid coil set) →
  `boozer_spline.py` (FTGen, `Boozer_splines_1p0_128_1p00.pkl`) →
  `GVEC_Boozer_spline_symmetrize.py` + `add_flag.py` + `pygvec`
  (`job_runner_GVEC.sl`, block `1p0_128_1p00`) → `surface_3_1p0_128_1p00_*`.
  The figure is rendered by `fA1b/make_coils_modB_html_svg_nfp3.py`.
  The rest of `fA1b/` is the Poincaré check that the coil set reproduces this
  GVEC equilibrium (`job_export` → `job_trace` → `job_plot`).

### Figure B1 — `Appendix-B_fB1/`
- DESC equilibrium:
  `publication_normF_SLNAM/desc_slnam_216_0p92/eq_slnam216_0p92_cyl_f9n18_L20M10N20.h5`.
  GVEC equilibrium: the pre-SLAM BEST equilibrium in `../GVEC_f67/` (not duplicated).
- `export_desc_normF.py` (desc-env) and `export_gvec_normF.py` (gvec-env) → `*_phi_planes.npz`
  → `plot_normF_publication.py`. See `Appendix-B_fB1/README.md`.

## Environments

| Environment | Used for |
|---|---|
| `desc-env` / `desc-env2` (DESC) | DESC equilibria, coils, field-line tracing, QFM, FTGen |
| `gvec-env` (pyGVEC, pyvista, netCDF4) | GVEC solves, GVEC post-processing, 3D renders |
| GX (GPU) + pyvmec (`gx/geometry_modules/pyvmec`) | Figure 8 |

The Slurm jobs were run on NERSC Perlmutter (accounts `m5194` CPU / `m5194_g` GPU)
and `cd` into absolute `/pscratch/...` paths; change those before rerunning
elsewhere. DESC tracing jobs need a GPU node. Scripts keep their settings as
variables near the top of the file.

## Known issues

- The Figure 1 coil design notebook (`coil_stage_two_optimization.ipynb`) is not
  included: every copy found is truncated at 512 KB and cannot be opened.
- No script here produces `QI_to_SLNAM_f23/Poincare_plot/figure_d_hybrid_top_view_*`
  (Figure 3); only its outputs are present.
- The Figure 8 caption says ρ = 0.8, but all GX runs are at ρ = 0.65
  (torflux = ρ² = 0.4225).
- Figure A1 coil sets: `fA1a/` names the 42-coil set as the NFP = 3 OP coil set,
  but `fA1b/coils_and_gvec_NFP3/SLAMify5.ipynb` loads the 36-coil set
  (`NFP3_coilset_36_coils_smoother_Bn_0p03_N5.h5`) and saves `…_SLAM-A-low.h5`, while GVEC
  used `…_SLAM-A-lowcur.h5`.
- `QI_to_SLNAM_f23/plot_coilset.py` points to a local DESC path (`/home/rgaur/DESC3/DESC`).
