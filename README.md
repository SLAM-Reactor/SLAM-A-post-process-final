# SLAMs: Stellarators Linking Axisymmetric Mirrors — figure data and scripts

This repository holds the inputs, outputs and scripts behind every figure in
*SLAMs: Stellarators Linking Axisymmetric Mirrors* (`SLAM_A_paper (3).pdf`).
Each figure has one directory. In it you will find the input files (equilibria,
coil sets, boundaries), the output files (cached `.npz`/`.nc` data and the figure
itself), and the scripts or Slurm jobs that go from one to the other.

Nothing here has to be rerun to replot a figure: the plotting scripts read the
cached data that sits next to them. Rerunning the expensive steps (field-line
tracing, equilibrium solves, GX) is possible from the files and jobs provided.

## Figure map

| Figure | What it shows | Directory | Figure file(s) |
|---|---|---|---|
| 1 | OP precursor: Boozer \|B\|, coils + B·n, coil Poincaré vs. flux surfaces | [`QI_coilset/rhop9`](QI_coilset/rhop9) | `Boozer_DESC_rhop9.pdf`, `QI_rhop9_surface_and_40_coils_Bn.pdf`, `QI_poincare_rhop9_comparison.pdf` |
| 2 | Building the stellarator–mirror hybrid coil set | [`QI_to_SLNAM/final_html_figures`](QI_to_SLNAM/final_html_figures) | `figure_{a,b,c}_*.pdf` |
| 3 | Hybrid coil sets at two mirror lengths and their Poincaré plots | [`QI_to_SLNAM/Poincare_plot`](QI_to_SLNAM/Poincare_plot) | `figure_d_hybrid_top_view_cross_sections.pdf`, `Poincare_plot/Poincare_plot_L_over_R_*_ntransit400.pdf` |
| 4 | Pre-SLAM and SLAM coil sets, QFM surfaces, \|B\| on axis | [`QFM_surf`](QFM_surf) | `SLNAM_QFM_surface_and_coils_Bn.pdf`, `SLAM_QFM_surface_and_coils_Bn.pdf`, `axis_B_comparison.pdf` |
| 5 | FTGen maps: (α, φ) → PEST → Boozer | [`FTGen`](FTGen) | `outfigures/FTGen_1p0_150_0p85_*` |
| 6 | GVEC surfaces vs. coil Poincaré traces | [`GVEC/post-process`](GVEC/post-process) | `figures/poincare_vs_gvec_SLNAM.pdf`, `figures/poincare_vs_gvec_SLAM_N128_RHO065_CLEAN1E4.pdf` |
| 7 | Effective ripple and magnetic well | [`GVEC/post-process`](GVEC/post-process) | `figures/effective_ripple_comparison_best.pdf`, `figures/magnetic_well_comparison_best.pdf` |
| 8 | Nonlinear ITG heat flux from GX | [`GVEC_GX`](GVEC_GX) (+ [`DESC_GX`](DESC_GX)) | `figures/gx_Q_vs_alpha_aLT3p0.pdf`, `figures/gx_Q_vs_aLT_max_alpha.pdf` |
| A1a | NFP = 3 precursor OP (DESC) | [`Appendix-3`](Appendix-3) | `coils_op_modB_nfp3.pdf` |
| A1b | NFP = 3 hybrid (GVEC) | [`Appendix-1`](Appendix-1) (+ [`GVEC/coils_and_gvec/NFP3`](GVEC/coils_and_gvec/NFP3)) | `SLNAM_coils_gvec_modB_nfp3.pdf` |
| B1 | Force-balance error, DESC (cylindrical angle) vs. GVEC (Boozer angle) | [`publication_normF_SLNAM`](publication_normF_SLNAM) | `normF_DESC_SLNAM_cyl_f9n18.pdf`, `normF_GVEC_SLNAM_BEST.pdf` |

Naming: **SLNAM** in file names is the *pre-SLAM* configuration in the paper,
**SLAM** is the SLAM configuration, and **QI / OP / QI_precursor** is the original
omnigenous precursor stellarator.

## How the pieces connect

```
OP equilibrium + OP coil set                        QI_coilset/
        │  cut at the low-field point, add circular mirror coils
        ▼
hybrid (pre-SLAM) coil set, L/R = 0 … 2            QI_to_SLNAM/   (SLAMify.ipynb)
        │  QFM surface + converter-coil optimization
        ▼
pre-SLAM and SLAM coil sets + QFM surfaces         QFM_surf/      (QFM_…-slope2.ipynb)
        │  FTGen: trace one field line → (α, φ) → PEST → Boozer
        ▼
Boozer-angle boundary                              FTGen/, GVEC/  (boozer_spline.py)
        │  GVEC fixed-boundary solve
        ▼
GVEC equilibria                                    GVEC/, Appendix-2/
        ├─ Poincaré check, ripple, well           GVEC/post-process/
        ├─ GX flux-tube geometry + GX runs        GVEC_GX/
        └─ DESC vs. GVEC force balance            publication_normF_SLNAM/
```

## Per-figure details

### Figure 1 — `QI_coilset/`
- **Inputs:** `eq_NFP2_A+9.0_…_vacuum_final2.h5` (OP equilibrium) and
  `NFP2_A8p7_coilset_40_coils_Bn_0p015_N9.h5` (OP coil set, from the DESC
  omnigenity database).
- **Scripts:** `rhop9/run_postprocess_rhop9.sh` runs the Boozer plot, the coils + B·n
  plot and the Poincaré comparison. `rhop9/job_trace_poincare_gpu.sl` (GPU)
  traces the coil field and writes `QI_poincare_rhop9_data.npz`.
  `rhop9/compute_Bn_error_rhop9.py` writes `QI_rhop9_Bn_error_data.npz`.
- `coil_stage_two_optimization.ipynb` is the coil design notebook. **This copy is
  truncated at 512 KB and does not open**; see *Known issues*.

### Figure 2 — `QI_to_SLNAM/`
- **Inputs:** the OP coil set (`../QI_coilset/`) and the hybrid coil sets
  `SLNAMified_NFP2_A8p7_coilset_L_over_R_{0p0,0p5,1p0,1p5,2p0}_SLAM-A.h5`, made in
  `SLAMify.ipynb`.
- **Script:** `final_html_figures/general_stellarator_mirror_figures_no_dash_a.py
  ORIGINAL_COILSET.h5 HYBRID_COILSET.h5` writes the interactive HTML figures; the
  PDF/SVG files were exported from them.

### Figure 3 — `QI_to_SLNAM/Poincare_plot/`
- `Poincare_plot/job_trace_poincare_gpu.sl` → `trace_poincare_gpu.py` traces each
  L/R case (GPU) → `SLNAMified_L_over_R_*_poincare_data_ntransit400.npz` →
  `plot_four_poincare_pdfs.py` (or `plot_all_poincare.sh`) → the four PDFs.

### Figure 4 — `QFM_surf/`
- **Initial → final:** `QFM_SLNAM_to_SLAM_optimization_nonplanar-lower-finite-slope2.ipynb`
  starts from `SLNAMified_NFP2_A8p7_coilset_L_over_R_1p0_SLAM-A.h5` and the OP
  equilibrium. It produces the pre-SLAM QFM surface (`SLNAM_QFM_surface_09-09-26_3.h5`)
  and the SLAM coil set and surface (`SLAM-field-09-09-26_2.h5`,
  `SLAM_QFM_surface_09-09-26_2.h5`). `SLNAM-field-09-09-26.h5` is the pre-SLAM coil set.
  The notebook loads its starting coil set by an absolute path to the post-Simons
  directory; the identical file is here.
- **Scripts:** `plot_qfm_surfaces.py` (coils + QFM surface coloured by B·n/⟨|B|⟩,
  cached in `qfm_surface_Bn_data.npz`); `trace_axis_B.py` (axis tracing, expensive)
  → `axis_B_data.npz` → `post_process_axis_B.py`. See `SUMMARY.md`.
- `TQ/` holds earlier versions of these QFM notebooks and coil sets (provenance only).

### Figure 5 — `FTGen/`
- **Inputs:** `coils/` (OP equilibrium + `SLNAMified_NFP2_A8p7_coilset_L_over_R_1p0_SLAM-A-TQ.h5`).
- `run_ftgen.sl` → `boozer_spline.py FNAME NTURNS RHO GPU` → `data/{traces,alpha,pest,boozer}/`.
  The figure uses the case `1p0_150_0p85`; the rest of the (nturns, ρ) scan is kept
  because it is what the choice was based on.
- `plot_FTGen_coordinate_maps.py` (edit `CASE` at the top) → `outfigures/`.

### Figures 6 and 7 — `GVEC/`
- **Equilibria** (`.ini` + `_State_final.dat` + `-Gframe.nc` boundary):
  - pre-SLAM for Fig. 6: `Appendix-2/…SLNAM_1p0_216_0p92_clean1e-3…` (nElems 8)
  - SLAM for Fig. 6: `GVEC/…SLAM_1p0_128_0p65_clean1em4…`
  - Fig. 7 ("BEST", from the resolution scans): `GVEC/SLNAM_216_0p92_nelems32_mn12_34_*`
    and `GVEC/SLAM_128_0p65_nelems24_mn24_32_*`
- **How the equilibria were made:** `run_ftgen_qfm.sl` (FTGen on `coils/` and
  `reference_surfaces/`) → `data/<label>/Boozer_splines_*.pkl` →
  `GVEC_Boozer_spline_symmetrize.py` + `patch_gvec_params.py` → `pygvec run`
  (`run_gvec_0p92_cpu.sl`, `run_gvec_slam_scan_case_cpu.sl`). The BEST runs come from
  `res_scan/{slnam,slam}_res_scan/` (template + job). Read `GVEC_0p92_REPRODUCIBILITY.md`
  before rerunning.
- **Post-processing** (`post-process/`, all case paths in `paths.py`; read
  `POST_PROCESS_OBJECTIVES_PROCEDURE.md`):
  - Fig. 6: `jobs/job_export_gvec_sections*.sl` → `jobs/job_trace_desc_poincare*.sl` (GPU)
    → `plot_poincare_vs_gvec.py`
  - Fig. 7, GVEC: `jobs/job_ripple_SLNAM_BEST_rho.sl`, `jobs/job_ripple_mnf3_rho.sl`
    (one ρ per job) and `jobs/job_well_best.sl`
  - Fig. 7, DESC OP: `jobs/job_desc_effective_ripple.sl`, `jobs/job_desc_magnetic_well.sl`
  - Plots: `plot_effective_ripple_best.py`, `plot_magnetic_well_best.py`
  - The GVEC ripple uses the external code `/pscratch/sd/r/rgaur/GVEC_benchmark/GVEC_eps_eff`.

### Figure 8 — `GVEC_GX/` and `DESC_GX/`
- **Geometry:** SLAM/pre-SLAM from the BEST GVEC equilibria (`export_gvec_gx_geometry.py`,
  `jobs/job_export_gx_geometry_best.sl`). The OP goes DESC → VMEC
  (`../DESC_GX/export_qi_desc_to_vmec_ns512.py` →
  `DESC_GX/data/wout_QI_precursor_OP_NFP2_A9_vacuum_final2_ns512.nc`) → pyvmec
  (`export_qi_vmec_gx_geometry_ns512.py`). Results: `data/*.eik.nc`.
- **Runs:** `make_gx_best_alpha5_inputs.py` and `make_qi_pyvmec_ns512_inputs.py` write
  `gx_runs/*/<case>/<alpha>/<aLT>/gx.in`. `jobs/job_gx_best_alpha5.sl` and
  `jobs/job_gx_qi_pyvmec_ns512.sl` produce `gx.out.nc` next to each input.
  3 cases × 5 field lines × 7 a/L_T = 105 runs.
- **Plot:** `plot_gx_publication.py`. Q vs. a/L_T uses the GX-reported Q_i of the
  field line with the largest flux (error bar = time standard deviation), averaged
  over t ≥ 300 of t_max = 600. See `README_GVEC_GX_CLEAN.md`.

### Figure A1 — `Appendix-3/`, `Appendix-1/`, `GVEC/coils_and_gvec/NFP3/`
- **(a)** `Appendix-3/`: `coil_stage_one_optimization.ipynb` fits the NFP = 3 OP coil set
  `NFP3_coilset_42_coils_smoother_Bn_0p02_N5.h5` to
  `eq_NFP3_A+10.0_…_final2_resolve.h5`. `compute_op_surface_modB.py` (DESC) →
  `op_surface_modB_nfp3.npz` → `make_coils_modB_html_svg_op_nfp3.py` (gvec-env / pyvista).
- **(b)** `GVEC/coils_and_gvec/NFP3/`: `SLAMify5.ipynb` (hybrid coil set) →
  `boozer_spline.py` (FTGen, `Boozer_splines_1p0_128_1p00.pkl`) →
  `GVEC_Boozer_spline_symmetrize.py` + `add_flag.py` + `pygvec`
  (`job_runner_GVEC.sl`, block `1p0_128_1p00`) → `surface_3_1p0_128_1p00_*`.
  The figure is rendered by `Appendix-1/make_coils_modB_html_svg_nfp3.py`.
  The rest of `Appendix-1/` is the Poincaré check that the coil set reproduces this
  GVEC equilibrium (`job_export` → `job_trace` → `job_plot`).

### Figure B1 — `publication_normF_SLNAM/`
- `desc_slnam_216_0p92/`: `export_gvec_boundary.py` (boundary of GVEC SLNAM_BEST) →
  `solve_slnam216_cyl.py 9 18` (`job_solve_slnam216_cyl.sl`) →
  `eq_slnam216_0p92_cyl_f9n18_L20M10N20.h5`.
- `export_desc_normF.py` (desc-env) and `export_gvec_normF.py` (gvec-env) → `*_phi_planes.npz`
  → `plot_normF_publication.py`. See `README.md` in that directory.

## Environments

| Environment | Used for |
|---|---|
| `desc-env` / `desc-env2` (DESC from `/pscratch/sd/r/rgaur/DESC3/DESC`) | DESC equilibria, coils, field-line tracing, QFM, FTGen |
| `gvec-env` (pyGVEC, pyvista, netCDF4) | GVEC solves, GVEC post-processing, 3D renders |
| GX (GPU) + pyvmec (`gx/geometry_modules/pyvmec`) | Figure 8 |

Slurm jobs use the Perlmutter accounts `m5194` (CPU) / `m5194_g` (GPU). DESC tracing
jobs need a GPU node. Scripts keep their settings as variables near the top of the file.

## Large files

GitHub rejects files above 100 MB and warns above 50 MB, so the 14 files above 50 MB
are listed in `.gitignore`. All 14 are GX output files (`gx.out.nc`) in
`GVEC_GX/gx_runs/`. With them missing, `plot_gx_publication.py` cannot replot
Figure 8; restore them from the Perlmutter copy
(`/pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC_GX/gx_runs/`) or rerun those
cases with their `gx.in`. The regeneration command for the list is in `.gitignore`.
One of them, the OP α = 0.5π, a/L_T = 1.5 run, went NaN and is left out of the figure
anyway.

Even without those files the repository is about 4.2 GB, mostly the other `gx.out.nc`
files (72 files are between 25 and 50 MB). Git LFS or an external data archive
(e.g. Zenodo) is a better home for them than plain git.

## Known issues

- `QI_coilset/coil_stage_two_optimization.ipynb` (the Figure 1 coil design notebook) is
  truncated at 512 KB and cannot be opened. So is the notebook in the original
  Perlmutter tree; no complete copy has been found.
- No script here produces `QI_to_SLNAM/Poincare_plot/figure_d_hybrid_top_view_*`
  (Figure 3); only its outputs are present.
- The Figure 8 caption says ρ = 0.8, but all GX runs are at ρ = 0.65 (torflux = ρ² = 0.4225).
- Figure A1 coil sets: `Appendix-3/` names the 42-coil set as the NFP = 3 OP coil set,
  but `GVEC/coils_and_gvec/NFP3/SLAMify5.ipynb` loads the 36-coil set
  (`NFP3_coilset_36_coils_smoother_Bn_0p03_N5.h5`) and saves `…_SLAM-A-low.h5`, while GVEC
  used `…_SLAM-A-lowcur.h5`.
- `QI_to_SLNAM/plot_coilset.py` points to a laptop DESC path (`/home/rgaur/DESC3/DESC`).
- `TQ/mr1p7/GVEC_eps_eff/` is its own git repository; commit it as a submodule or remove
  its `.git/` before adding `TQ/`.

## Other folders

- `extra/` folders (top level and inside figure directories): files that no figure
  uses — old versions, other resolutions, diagnostics, SLAM0/SLNAM0 generalized-angle
  equilibria. Kept for review; not needed to make any figure.
- `TQ/`: earlier QFM notebooks and coil sets behind Figure 4 (provenance only).
