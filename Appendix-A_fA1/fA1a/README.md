# Appendix A, Figure A1a: NFP=3 precursor OP equilibrium + coilset, coils/|B| render

Moved here from `mirrors/SLAM/post-Simons/NFP3_A+10_.../`:

- `eq_NFP3_A+10.0_tor+0.0_mr+0.30_shift+0.00_wd+0.00_elong+1.50_wellwt+2.0_vacuum_final2_resolve.h5`
  - the precursor OP DESC equilibrium (aspect ratio A+10) fed into FTGen,
    *before* SLAM mirror insertion.
- `NFP3_coilset_42_coils_smoother_Bn_0p02_N5.h5`
  - the OP-era DESC coilset (42 coils: 7 base coils, NFP=3, stellarator
    symmetric) that this OP equilibrium's boundary was actually optimized
    against.

  Not `SLNAMified_NFP3_A10p1_L_over_R_1p0_SLAM-A-lowcur.h5` (used in
  fA1b): that coilset only exists post-mirror-insertion - it was fit to
  reproduce the OP boundary's rotational transform *through* the SLAM mirror
  sections (`boozer_spline.py`'s `field_line_integrate` loop over
  `L_over_R`), and its geometry does not correspond to this pre-mirror OP
  boundary. Plotting the OP equilibrium against the SLAM coilset was an
  earlier mistake in this Appendix - the OP equilibrium needs its own OP
  coilset.

Render (same recipe/style controls as fA1b's
`make_coils_modB_html_svg_nfp3.py`, but colored by the DESC equilibrium's own
`|B|` instead of a GVEC state's):

1. `compute_op_surface_modB.py` (**desc-env3**, not desc-env - the DESC
   checkout desc-env resolves is mid-edit and imports `adv_jax_math`, which
   is not installed) - loads the eq, evaluates `X,Y,Z,|B|` on the rho=1
   surface, caches to `op_surface_modB_nfp3.npz`. DESC and pyvista are not
   installed in the same conda env here, hence the split.

   The cache holds *flat* arrays in DESC's own node order (theta fastest,
   zeta slowest) and the script asserts that ordering before writing. Do not
   reshape them to `(THETA_COUNT, ZETA_COUNT)`: that silently scrambles the
   surface rather than failing, since both counts multiply to the same total.
2. `make_coils_modB_html_svg_op_nfp3.py` (gvec-env, has pyvista/netCDF4) -
   builds the coil tubes (with the same symmetry-expansion logic as
   fA1b, applied to the OP coilset) and the DESC surface mesh from that
   cache, writes `coils_op_modB_nfp3.html`, then opens the desktop window
   where you rotate to a view and press `s` to save
   `coils_op_modB_nfp3.svg`, same as fA1b. Set
   `START_DESKTOP_SVG_PICKER = False` to write only the HTML.

Sanity check on the rendered surface: its cross-sectional area gives an
aspect ratio of 10.17, matching the `A+10.0` in the equilibrium filename.
The |B| range (0.86-1.53 T) differs from fA1b's (0.16-0.43 T) because
this is a different equilibrium (precursor OP, before SLAM mirror insertion)
with a different coilset, not a shared coilset on two equilibria.

Both scripts ran directly on the login node (compute nodes here fail to get a
working GLX context even under Xvfb - see fA1b's notes).
