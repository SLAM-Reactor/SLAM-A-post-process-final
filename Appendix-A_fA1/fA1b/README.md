# Appendix A, Figure A1b: NFP=3 GVEC equilibrium vs. coilset Poincare check

Goal: before trusting the pyvista HTML render of the NFP=3 GVEC equilibrium
next to its DESC coilset (`coils_and_gvec_NFP3/SLNAMified_NFP3_A10p1_L_over_R_1p0_SLAM-A-lowcur.html`),
verify the two actually agree: trace the DESC coilset field and check the
Poincare points land on the GVEC flux surfaces. Same check as the NFP=2
SLAM/SLNAM comparison in `GVEC_f67/post-process/` (goal 1 there), redone
here for the NFP=3 case with hardcoded paths/flat layout.

Inputs (both already existed, not generated here):
- GVEC equilibrium: `coils_and_gvec_NFP3/surface_3_1p0_128_1p00_State_final.dat`
  + `parameter_surface_3_1p0_128_1p00_final.ini` (NFP=3, PHIEDGE=0.01)
- DESC coilset: `coils_and_gvec_NFP3/SLNAMified_NFP3_A10p1_L_over_R_1p0_SLAM-A-lowcur.h5`

Pipeline (3 scripts, chained by Slurm job dependency, run in this order):
1. `export_gvec_sections_nfp3.py` (gvec-env, CPU) - loads the GVEC state,
   extracts flux surfaces at rho=[0.5, 1.0] on constant *physical* phi planes
   phi=0 and phi=pi/NFP, and picks the outboard point at phi=0 as the
   Poincare seed. Writes `gvec_sections_nfp3.npz`.
2. `trace_desc_poincare_nfp3.py` (desc-env2, GPU) - loads the DESC coilset,
   integrates the coil field from those seeds for 300 toroidal transits at
   the same two phi planes. Writes `desc_coil_poincare_nfp3.npz`.
3. `plot_poincare_vs_gvec_nfp3.py` (desc-env2, CPU) - overlays the DESC
   Poincare points on the GVEC flux-surface curves. Writes
   `poincare_vs_gvec_nfp3.png`.

Submitted as a Slurm chain (`job_export_nfp3.sl` -> `job_trace_nfp3.sl` ->
`job_plot_nfp3.sl`, each `--dependency=afterok` on the previous), all
`qos=premium`. Check `poincare_vs_gvec_nfp3.png` and the finite-fraction
printouts in `logs/trace_nfp3-*.out` (the old logs are in the backup) once the chain finishes: if the DESC
points sit on top of the GVEC curves at both planes, the HTML render can be
trusted; if they diverge, the coilset does not reproduce this GVEC
equilibrium and the HTML comparison should not be used as-is.
