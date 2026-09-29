# Publication figures: force-balance error, DESC SLNAM cyl fit 9,18 vs GVEC SLNAM_BEST (09-25)

Quantity: |F| / <|grad(B^2/2mu0)|>_vol (DESC `|F|_normalized`, GVEC `mod_F / gradB2_volavg`; same definition).
Planes: cylindrical phi = 0 and pi/2, the same planes for both codes. Grid: rho 0.02-1 x 100, theta 0-2pi x 257.

1. `export_desc_normF.py` (desc-env): DESC `../desc_slnam_216_0p92/eq_slnam216_0p92_cyl_f9n18_L20M10N20.h5`
   -> `normF_DESC_SLNAM_cyl_f9n18_phi_planes.npz` (rho computed in chunks of 10; the volume average is taken on
   DESC's own grid, so it does not depend on the chunk).
2. `export_gvec_normF.py` (gvec-env): GVEC SLNAM_BEST (`../../GVEC/parameter_SLNAM_216_0p92_nelems32_mn12_34_final.ini`)
   exactly on the phi planes (Newton for GVEC zeta per (rho, theta), |phi - phi0| < 1e-12 rad, no interpolation of
   field values) -> `normF_GVEC_SLNAM_BEST_phi_planes.npz`.
3. `plot_normF_publication.py` (numpy + matplotlib only; all settings at the top) ->
   `normF_DESC_SLNAM_cyl_f9n18.{pdf,png}`, `normF_GVEC_SLNAM_BEST.{pdf,png}`.

npz contents: R, Z, F with shape (rho, theta, plane); rho, theta, phi; `info` string (source and method);
GVEC also `zeta_gvec` (its own zeta at each plotted point).

| | phi = 0 max / mean | phi = pi/2 max / mean |
|---|---|---|
| DESC cyl fit 9,18 | 1.86e-1 / 2.40e-2 | 2.38e-1 / 1.39e-2 |
| GVEC SLNAM_BEST | 5.29e-2 / 4.36e-3 | 4.18e-2 / 2.73e-3 |
