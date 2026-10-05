# DESC cylindrical solves on the GVEC SLNAM_BEST boundary (09-24)

Same procedure as `../desc_slam_128_0p65` (SLAM_BEST), on the SLNAM_BEST boundary.

1. `export_gvec_boundary.py` (gvec-env, login node): rho=1 surface of GVEC SLNAM_BEST
   (`../../../GVEC_f67/parameter_SLNAM_216_0p92_nelems32_mn12_34_final.ini`, 216 turns, rho=0.92, clean1e-3)
   on 196 theta x 256 constant-phi planes -> `gvec_boundary_SLNAM_216_0p92_clean1em3.npz`.
2. `solve_slnam216_cyl.py M N` (desc-env): cylindrical fit (zeta=None, no omega) at M_fit, N_fit, padded to
   L20 M10 N20, vacuum, Psi = 0.038785 x 0.92^2, 500 iterations at tolerance 1e-8.
   Fits 7,14 / 8,16 / 9,18 in one 4-GPU premium job, `job_solve_slnam216_cyl.sl` (job 58854766).
   Login-node fit check: rms / max 0.89 / 4.89 mm, 0.68 / 3.92 mm, 0.37 / 2.67 mm; all initial guesses nested.
3. Section comparison with GVEC SLNAM_BEST at zeta = 0, pi/2:
   `../desc_slam_128_0p65/plot_normF_desc_gvec_best.py SLNAM_BEST` -> `../desc_slam_128_0p65/normF_desc_vs_gvec_SLNAM_BEST.png`.

## Result (job 58854766, 27 min)

| fit | fit rms / max [mm] | volume F max / mean | zeta=0 max / mean | zeta=pi/2 max / mean | iota rho 0.2 -> 1 | state |
|---|---|---|---|---|---|---|
| 7,14 | 0.89 / 4.89 | 3.7e+1 / 1.1e-2 | 4.5e-2 / 6.0e-3 | 7.0e-2 / 6.3e-3 | 0.555 -> 0.602 | 500-iteration cap, nested |
| 8,16 | 0.68 / 3.92 | 1.3e+11 / 2.4e+6 | 2.4e+3 / 2.3 | 1.3e+3 / 1.7 | 1.33, -0.11, 0.18, 0.60, 0.16 | FAILED: start force error ~6e13, stopped on xtol after 33 evaluations, not nested |
| 9,18 | 0.37 / 2.67 | 2.2 / 1.9e-2 | 1.9e-1 / 2.4e-2 | 2.2e-1 / 1.5e-2 | 0.601 -> 0.591 | 500-iteration cap, nested |
| GVEC SLNAM_BEST | - | 1.4e-1 / 4.1e-3 | 3.8e-2 / 4.4e-3 | 3.6e-2 / 2.6e-3 | 0.604 -> 0.608 | - |

Volume grid: rho 0.05-1 x 41, theta 91, zeta 32 over one field period. The 7,14 and 9,18 volume maxima (37, 2.2)
are far above their section maxima, so the worst point lies off the zeta = 0, pi/2 planes (not located yet).
