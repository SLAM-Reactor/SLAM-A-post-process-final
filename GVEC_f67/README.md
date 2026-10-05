# Figures 6 and 7: GVEC equilibria of pre-SLAM (SLNAM) and SLAM

`SLNAM` in file names is the pre-SLAM configuration of the paper.

## Equilibria

Each equilibrium is a `parameter_*.ini` + `*_State_final.dat` pair; the `.ini`
reads the boundary `*-Gframe.nc` from this directory.

| Use | Case | Files |
|---|---|---|
| Fig. 6a, pre-SLAM | 216 turns, ρ = 0.92, clean 1e-3 | `parameter_surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01_final.ini`, `surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01_State_final.dat` |
| Fig. 6b, SLAM | 128 turns, ρ = 0.65, clean 1e-4 | `parameter_surface_2_SLAM_1p0_128_0p65_clean1em4_axis_x1m0p01_final.ini`, `surface_2_SLAM_1p0_128_0p65_clean1em4_axis_x1m0p01_State_final.dat` |
| Fig. 7, Fig. 8, Fig. B1: pre-SLAM "BEST" | same boundary, nElems 32, mn_max (12, 34) | `parameter_SLNAM_216_0p92_nelems32_mn12_34_final.ini`, `SLNAM_216_0p92_nelems32_mn12_34_State_final.dat` |
| Fig. 7, Fig. 8: SLAM "BEST" | same boundary, nElems 24, mn_max (24, 32) | `parameter_SLAM_128_0p65_nelems24_mn24_32_final.ini`, `SLAM_128_0p65_nelems24_mn24_32_State_final.dat` |

The BEST equilibria have the lowest max |F|_normalized of the resolution scans in
`res_scan/{slnam,slam}_res_scan/` (template `.toml` + Slurm job for the chosen
resolution).

## How the equilibria were made

1. FTGen on the QFM-surface coil sets: `run_ftgen_qfm.sl` → `boozer_spline.py`,
   reading `coils/` (pre-SLAM and SLAM coil sets, OP equilibrium) and
   `reference_surfaces/` (QFM surfaces from `../QFM_surf_f4`). The two splines
   that were used are kept: `data/SLNAM/Boozer_splines_1p0_216_0p92.pkl`,
   `data/SLAM/Boozer_splines_1p0_128_0p65.pkl`.
2. `GVEC_Boozer_spline_symmetrize.py` → `surface_2_*.nc`, then
   `pygvec load-quasr` → `*-Gframe.nc` and `*-parameters.toml`, then
   `patch_gvec_params.py`, then `pygvec run`. Jobs: `run_gvec_0p92_cpu.sl`
   (pre-SLAM, ρ = 0.92) and `run_gvec_slam_scan_case_cpu.sl` (SLAM, with
   `NTURNS=128 RHO=0.65 CLEAN=1e-4 CLEAN_TAG=1em4`).
3. Run `run_gvec_smoke_check.py` on the login node before a full solve; it catches
   negative-Jacobian initialisations cheaply.

Production settings: reverse theta, `average_axis_move_X1 = -0.01`,
pre-SLAM clean 1e-3, SLAM clean 1e-4.

## Post-processing

See `post-process/README.md` (Poincaré comparison, effective ripple, magnetic well).
