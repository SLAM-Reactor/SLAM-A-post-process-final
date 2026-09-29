# FTGen production

Fresh field-line-tracing / Boozer-transform production for the SLAM-A coilset.
Scripts live at this level; nothing writes next to the coils, and nothing
writes into the script directory.

```
FTGen/
├── boozer_spline.py                 the generator
├── plot_FTGen_coordinate_maps.py    the three-panel coordinate figure
├── run_ftgen.sl                     12-case production job
├── coils/                           INPUTS, read only
│   ├── SLNAMified_NFP2_A8p7_coilset_L_over_R_1p0_SLAM-A-TQ.h5
│   └── eq_NFP2_A+9.0_..._vacuum_final2.h5
└── data/
    ├── traces/    boundary_RZ_data_<case>.npz    field-line traces (expensive)
    ├── alpha/     alpha_data_<case>.npz          native (alpha, phi) arrays
    ├── pest/      PEST_data_<case>.npz
    ├── boozer/    Boozer_data_<case>.npz, Boozer_splines_<case>.pkl
    └── figures/   test*.png, FTGen_*_coordinate_maps.png
```

`<case>` is `<fname>_<nturns>_<rho>`, e.g. `1p0_216_0p85`.

## Running

```bash
sbatch run_ftgen.sl                        # all 12 cases, 4 GPUs
python3 boozer_spline.py 1p0 216 0.85 0    # one case, gpuid 0
python3 plot_FTGen_coordinate_maps.py      # figure, edit CASE/QUANTITY at top
```

Paths are overridable with `FTGEN_COILS_DIR`, `FTGEN_DATA_DIR`, `FTGEN_EQ`,
`FTGEN_COIL_TEMPLATE`.

`boozer_spline.py` reuses `data/traces/*.npz` when present, so re-running after
a code change costs about a minute instead of the full trace. **Delete
`data/traces/` to force a genuine from-scratch trace.**

## Cases

`fname = 1p0` (L/R = 1.0, SLAM-A-TQ), nturns ∈ {102, 128, 150, 196, 216, 256},
rho ∈ {0.85, 0.87}.

## Fixes carried into this production

Three defects were found in the previous copy of `boozer_spline.py` and are
fixed here. All three are verified, with the measurement that establishes each.

**1. `Boozer_data_*.npz` stored PEST-parameterised splines evaluated at Boozer
angles.** `Z_booz`, `modB_booz` and `nu_booz` were written as
`Z_spl1.ev(theta_booz_2D, phi_booz_2D)` — the right functions sampled at the
wrong arguments, an O(nu) error. `R_booz` and `phi_cyl` came from genuinely
Boozer-parameterised splines and were fine, so the stored (R, Z) pairs were
mismatched. Measured against the correct splines: `max|dZ|` from 4.2e-01 to
7.7e+00 m, up to 107% of the surface's own Z half-extent, while `max|dR|` was
2.2e-15 m. Now the arrays computed at the Newton-solved PEST angles are saved
directly.

**2. The alpha chart was never closed periodically.** The closure was present
but commented out. `alpha_sort` stops short of 2π and `RectBivariateSpline`
extrapolates rather than wrapping, so the ~0.6% of PEST queries landing in that
gap were cubic extrapolations — differing from the correct values by up to
0.121 m in X — and those values were then fitted by the next spline level, so
the error was not confined to the seam. Restoring the alpha closure:

| NFP=2 violation | R | Z | \|B\| |
|---|---|---|---|
| unclosed | 9.5% | 10.4% | 12.4% |
| alpha closed | **1.7%** | **1.6%** | **2.0%** |

`phi` is deliberately **not** closed. At fixed alpha the surface point is
`theta = alpha + iota*phi`, so `(alpha, phi + 2π)` sits at `theta + 2π*iota` —
a different poloidal position. `X(alpha, phi)` is genuinely discontinuous
across the phi seam, and wrapping it would erase the structure the alpha panel
exists to show. Closing alpha alone gives the same NFP improvement as closing
both, so nothing is lost.

**3. The native alpha arrays were never saved.** Plotting scripts had to
rebuild the alpha grid from `boundary_RZ_data_*.npz`, which stores only
`r, z, phi`, using a hardcoded iota. `alpha_data_*.npz` is now written straight
from the per-field-line traces, before any spline is fitted, and carries `iota`
and `G` so nothing downstream needs to hardcode them.

`Boozer_splines_*.pkl` now also stores the `B` and `phi_cyl` splines. Their
absence previously made `modB_booz` unrecoverable from a corrupted npz.

## Known residuals

* NFP=2 violation does not go to zero after the fix — about 1.6–2.0%. These
  traces are only approximately NFP=2 periodic, so some residual is expected.
  Not further characterised.
* The Boozer theta seam is still mildly non-periodic: median seam mismatch is
  4–12x a typical step for R, Z, |B| (max-vs-max is about 1x). The alpha
  closure barely moves this, so there is a second cause not yet identified.
* About 0.1% of Newton solves (28 of 25088 on `1p0_216_0p85`) miss
  `theta_B = theta_PEST + iota*nu` by more than 1e-4, clustered at the top of
  the theta_B range.
