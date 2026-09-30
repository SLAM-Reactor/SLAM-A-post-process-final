# SLAM / SLNAM Boozer |B| data (GVEC best equilibria), 2026-09-30

Boozer |B| of the best GVEC equilibria of `../GVEC/post-process/paths.py` (lowest max |F|_normalized of the 09-23
resolution scans), on rho = 0.5 / 0.85 / 1.0, with the plots and the script that made them. The question behind it (09-29/30):
are the |B| contours closed in theta_B, as in the DESC SLAM_opt working equilibria? They are not.

| state | GVEC files (read only) | mn_max | Boozer M, N (MNfactor) | grid per field period |
|---|---|---|---|---|
| SLAM_BEST | `../GVEC/parameter_SLAM_128_0p65_nelems24_mn24_32_final.ini`, `SLAM_128_0p65_nelems24_mn24_32_State_final.dat` | (24, 32) | 72, 96 (3) | 160 x 224 |
| SLNAM_BEST | `../GVEC/parameter_SLNAM_216_0p92_nelems32_mn12_34_final.ini`, `SLNAM_216_0p92_nelems32_mn12_34_State_final.dat` | (12, 34) | 60, 170 (5) | 128 x 352 |

MNfactor as in the GVEC effective-ripple runs (SLAM_BEST 3: user decision 09-24, 5 needs ~490 GB; SLNAM_BEST 5).
Grid = at least 2 M + 2 by 2 N + 2 points, rounded up to a multiple of 32; theta_B over 2 pi, zeta_B over one field period
(NFP 2), endpoints excluded. No radial derivative of the Boozer transform.

## Files

* `data/gvec_boozer_<state>_rho<r>.nc` - the whole GVEC `EvaluationsBoozer` dataset of that surface (xarray / netCDF;
  dims rad 1, pol, tor, xyz 3): coordinates `theta_B`, `zeta_B`, `rho`; `mod_B`, `iota`, `pos` (x, y, z), the logical
  angles `theta`, `zeta` of each Boozer point, `LA`, `NU_B` and derivatives, `X1`, `X2` and derivatives, `B`, `B_contra_t`,
  `B_contra_z`, `Jac`, basis vectors. Open with `xarray.open_dataset(...)`.
* `data/gvec_boozer_<state>_rho<r>.npz` - `B` = |B|(theta_B, zeta_B) (pol, tor) of the same grid, `B_mn` = its 2D FFT
  (`np.fft.fft2(B) / B.size`; |B| = sum B_mn exp(i (m theta_B + n NFP zeta_B)), numpy frequency order), `theta`, `zeta`,
  `iota`, `rho`, `M`, `N`, `mnfactor`, `NFP`, `seconds` (wall time of that surface).
* `data/gvec_boozer_theta_closure.json` - per state and surface: `closed_frac` (share of the 49 trapped levels
  B* = Bmin + t (Bmax - Bmin), t = 0.02 ... 0.98, at which every region of {|B| < B*} and {|B| > B*} is a band winding once
  around theta_B and not around zeta_B), `t_open` (the levels that are not), `theta_var` (rms of |B| - <|B|>_theta over the
  surface / (Bmax - Bmin); 0 = |B| depends on zeta_B only), `theta_var_max`, Boozer Bmin / Bmax, iota, M_booz, N_booz.
* `figures/boozer_lowfield_gvec.png` - rows SLAM_BEST / SLNAM_BEST, columns rho 0.5 / 0.85 / 1.0: |B| contour lines (jet,
  20 even levels + 30 in [Bmin, Bmin + 0.08 (Bmax - Bmin)] so the low-field region shows), 4 field lines, as DESC
  `plot_boozer_surface(fill=False)`. `figures/boozer_lowfield_gvec_SLAM_BEST.png` - its first row, drawn while SLNAM was running.
* `gvec_boozer_plane.py` - the post-processing and plotting script (settings on top). Each (state, rho) is one task run in
  its own child process (memory builds up across surfaces); several identical jobs share the tasks through lock files in
  `data/gvec_boozer_locks/`. With all `data/*.npz` present it only redraws the figure and the json.
* `job_gvec_boozer_cpu.sl` - the premium CPU job (one exclusive node, gvec-env; five copies were submitted, 59115606-10).
  Measured per surface: SLAM_BEST ~650 s and ~113 GB, SLNAM_BEST ~2060 s and up to ~320 GB - compute nodes only, not
  the login node.
* `logs/` - the five job logs.

## Redraw / rerun

```bash
module load conda && conda activate gvec-env
cd /pscratch/sd/r/rgaur/SLAM-A-post-process-final/SLAM-SLNAM-Boozer
python -u gvec_boozer_plane.py          # redraw only (all npz present)
sbatch job_gvec_boozer_cpu.sl           # recompute: delete the npz / nc of the surfaces first
```

## Results (measured)

| | closed in theta_B, rho .5 / .85 / 1.0 | theta_var, rho .5 / .85 / 1.0 | Bmax/Bmin (rho .5) |
|---|---|---|---|
| GVEC SLAM_BEST | 78 / 57 / 53 % | 0.047 / 0.080 / 0.095 | 1.44 |
| GVEC SLNAM_BEST | 73 / 55 / 47 % | 0.065 / 0.113 / 0.135 | 1.45 |
| DESC SLAM_opt working finals (for comparison) | 96-98 / 90-92 / 84-86 % | 0.024-0.032 / 0.043-0.054 / 0.051-0.063 | 4.5-5.2 |

Both GVEC states have two wells (zeta_B ~ 1.3 and 1.8) with islands between and beside them - their |B| contours are not
closed in theta_B. The DESC comparison figures (SLAM_opt working equilibria, particle-traced low-loss states, LT3) are in
`/pscratch/sd/r/rgaur/generalized_toroidal/SLAM_LT3_09-27-26/figures/conclusion/boozer_lowfield_*.png`, made by
`boozer_theta_closure(_login).py` there; the originals of this folder's files are in that campaign's `data/conclusion/`.
Record: `generalized_toroidal/SLAM_LT3_RUNS_09-27-26.md` sec. 0S.0b.
