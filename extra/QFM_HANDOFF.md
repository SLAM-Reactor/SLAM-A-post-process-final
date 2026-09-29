# QFM optimization for SLNAM0 in the generalized toroidal angle — HANDOFF

Written 2026-08-25.

**Status: grid fix applied and VERIFIED by dry run 57587821. The full solve has
NOT been run.** Job 57588612 was submitted and then cancelled at the user's
instruction before it started (elapsed 00:00:00, nothing consumed). Do not launch
the full solve without explicit instruction.

---

## 1. Goal

Repeat the QFM surface optimization from the reference notebook, but with the
seed surface carried in the **generalized toroidal angle** (phi = zeta + omega)
instead of a plain axisymmetric-angle tube.

Reference notebook:

    /pscratch/sd/r/rgaur/mirrors/SLAM/post-APS/OP_DB_coils/coil_design/
      NFP2_A+9_tor+0.00_mr+0.1_shift0.0_wd+0.00_elong+3.0_wellwt+2.0_iotashift+0.0_high_iota/
      QFM_SLNAM_to_SLAM_optimization.ipynb

---

## 2. What the reference notebook does

26 cells. The QFM-relevant path:

| cells | what |
|---|---|
| 0-7 | imports; `compute_average_normalized_field` = `<abs(B.n)>/<abs(B)>`, the QFM figure of merit |
| 9-12 | load coilset + eq; locate the magnetic axis by field-line integration; build seed `surf0` as a **circular tube around the axis**: `r = r_axis + 0.02*cos(theta)`, `z = z_axis + 0.04*sin(theta)`, then `FourierRZToroidalSurface.from_values(coords, theta, M=10, N=12, NFP=2, sym=True)` |
| 14-16 | split the coilset into shaping / non-shaping subsets -> `field1`, `field2` |
| **17** | **the QFM solve**: `surf0.rho=1.0; surf0.iota=-0.512; surf0.G=0.1`, then `ObjectiveFunction((SurfaceQuadraticFlux(...), ToroidalFlux(..., target=-0.050)))` with `Optimizer("lsq-exact")`, `ftol=5e-4`, `maxiter=100` -> `qfm_surf1` |
| 21 | elongation + volume of the QFM surface |
| 22 | **stage 2**: co-optimize surface *and* coils, adding `SurfaceMatch(surf0, qfm_surf1)`, `CoilLength`, `Elongation`, with `FixParameters` masks per coil subset |

Note `from_values` is called **without** `zeta=`, so the notebook's seed has no
`W_basis` — phi == zeta. That is the thing this work replaces.

---

## 3. What has been built here

    SLNAM0_equilibrium/QFM_optimization/
      qfm_slnam0.py           # the driver
      job_qfm_slnam0.sl       # full run, 4 h        <- NOT RUN
      job_qfm_dry.sl          # DRY_RUN=1 - builds + evaluates objectives, no optimize
      slurm-qfm-57582649.out  # dry run that hit the grid blocker (section 5)
      slurm-qfm-57587821.out  # dry run AFTER the fix - PASSED (section 4b)
      QFM_HANDOFF.md          # this file

`qfm_slnam0.py` covers notebook cells 9-21. **Cell 22 is NOT implemented** — no
coil co-optimization, no `SurfaceMatch`.

Substitutions vs the notebook:

- **Seed** = `eq.surface.copy()` from the SLNAM0 vacuum equilibrium, so it carries
  a `W_basis`. This is the whole point of the exercise.
- **No axis trace needed** — the equilibrium boundary already *is* the surface.
- **`iota`, `G`** taken from the equilibrium at rho=1 rather than hard-coded.
- **`ToroidalFlux` target** measured from the coil field through the seed surface,
  **not** `eq.Psi`. See section 6.

Inputs:

    coilset : ../SLNAM0_for_C5.h5
    eq      : ../eq_slnam0_1p5_196_0p82_m4n8_omega_m2n8_solved.h5

Env knobs: `M_SURF`, `N_SURF`, `EVAL_M`, `EVAL_N`, `BS_CHUNK_SIZE`,
`QFLUX_WEIGHT`, `TFLUX_WEIGHT`, `MAXITER`, `FTOL`, `TARGET_PSI`, `DRY_RUN`.

---

## 4a. Measured — dry run 57582649 (PRE-fix, failed)

Everything up to the `ToroidalFlux` build succeeded, then:

    File ".../qfm_slnam0.py", line 139, in main
        tflux_probe.build(verbose=0)
    File ".../desc/objectives/_coils.py", line 2502, in build
    ValueError: Evaluation grid should be at constant zeta

---

## 4b. Measured — dry run 57587821 (POST-fix, PASSED)

Full output:

    coilset: .../SLNAM0_for_C5.h5
    input eq: .../eq_slnam0_1p5_196_0p82_m4n8_omega_m2n8_solved.h5
    eq: L=20 M=8 N=16 Lz=4 Mz=4 Nz=16 NFP=2 sym=True Psi=2.421177e-03
    seed surface omega modes: 148
    max |W_lmn| = 4.136212e-01 rad
    seed surface resolution: M=8 N=16
    seed iota=-0.155884  G=0.120837
    measured toroidal flux through seed surface: -3.13041888e-03 Wb
    eq.Psi (inherited, known ~28% low): 2.42117653e-03 Wb
    ToroidalFlux target: -3.13041888e-03 Wb
    seed <|B.n|>/<|B|> = 1.850608e-02
    Building objective: Surface Quadratic Flux
    Building objective: toroidal-flux
    initial objective: dim_f=3186  |r|=3.057068e-02  max|r|=2.279427e-03
    DRY_RUN: objectives built and evaluated on the generalized-angle surface
      without error; exiting before optimization.

### What this establishes

1. **Both QFM objectives work on a generalized-angle surface.**
   `SurfaceQuadraticFlux` and `ToroidalFlux` both built and evaluated on a
   surface with **148 omega modes, max |omega| = 0.414 rad**. That is a large,
   real omega — not a rounding-level one. The generalized angle propagates
   through the Biot-Savart evaluation points as the source reading predicted
   (section 6). This was the central open question and it is now answered.

2. **Independent confirmation that `eq.Psi` is ~29 % low.** Two unrelated
   measurements agree to 1 %:

   | probe | quantity | ratio vs inherited |
   |---|---|---|
   | vacuum driver | `<abs(B)>` trace / eq on the boundary | **1.281** |
   | QFM dry run | measured toroidal flux / `eq.Psi` | **1.293** |

   `abs(-3.13041888e-03) / 2.42117653e-03 = 1.2929`. Different quantities,
   different code paths, same answer. The inherited-Psi error is confirmed, not
   merely suspected.

3. **The flux sign is negative** (-3.130e-03 Wb) while `eq.Psi` is positive. The
   measured-target design handles this automatically; a run that had inherited
   `eq.Psi` as the target would have driven the surface toward the wrong sign.

4. **Baseline to beat:** `seed <|B.n|>/<|B|> = 1.850608e-02`. A working QFM solve
   should drop this by roughly an order of magnitude. Objective has
   `dim_f = 3186`, initial `|r| = 3.057068e-02`, `max|r| = 2.279427e-03`.

### Still unmeasured

Whether the QFM optimization **converges** with an omega-carrying seed, and what
`<abs(B.n)>/<abs(B)>` it reaches. The full solve has not been run.

---

## 5. The blocker and the fix (APPLIED)

### Cause

`qfm_slnam0.py` built one grid and handed it to **both** objectives:

    eval_grid = LinearGrid(M=EVAL_M, N=EVAL_N, NFP=surf0.NFP)   # 24, 32 -> 65 zeta planes

The two objectives integrate over **different domains**:

- `SurfaceQuadraticFlux` — integral of |B.n|^2 over the *whole surface*.
  Domain = (theta, zeta) torus. Needs many zeta. **This grid is correct here.**
- `ToroidalFlux` — Psi over *one poloidal cross-section*. Domain = a single zeta
  plane. Needs exactly one zeta.

DESC enforces it literally (`_coils.py:2501`):

    errorif(
        not np.allclose(eval_grid.nodes[:, 2], eval_grid.nodes[0, 2]),
        ValueError, "Evaluation grid should be at constant zeta",
    )

65 distinct zeta values -> fails. **This had nothing to do with omega.** It was a
whole-surface grid used for a single-cut integral.

### Fix

Removed `eval_grid=eval_grid` from the three `ToroidalFlux` calls
(`tflux_probe` line 140, `tflux` line 160, `tflux_probe2` line 192).
`SurfaceQuadraticFlux` keeps it (line 155). DESC now builds its own grid:

    eval_grid = LinearGrid(
        L=eq.L_grid if not use_vector_potential else 0,
        M=eq.M_grid if hasattr(eq, "M_grid") else 3 * eq.M,
        zeta=jnp.array(0.0), NFP=eq.NFP,
    )

This is what the notebook does — it never passes `eval_grid` to `ToroidalFlux`.
Verified working by dry run 57587821.

---

## 6. Physics notes (verified against DESC source)

### Toroidal flux generalizes to a generalized angle for free

Two independent reasons:

1. **Cut-independence.** For a true flux surface (B.n = 0), two cross-sections at
   constant zeta and constant phi bound a volume closed by a band of the flux
   surface. B.n = 0 on the band, so div B = 0 forces equal flux through both cuts.
   The angle labelling the cut does not enter Psi.
2. **DESC's basis vectors already carry omega**
   (`desc/compute/_basis_vectors.py:1647`):

        data["e_rho"] = jnp.array([data["R_r"], data["R"] * data["omega_r"], data["Z_r"]]).T

   The `R*omega_r` term is explicit; `e_theta` carries `R*omega_t` likewise. So
   `|e_rho x e_theta|` and `n_zeta` describe the true constant-zeta **ribbon** in
   real space — twisted, not planar, when omega != 0. Biot-Savart is evaluated at
   `(R, phi, Z)` with phi = zeta + omega, not `(R, zeta, Z)`.

Dry run 57587821 confirms this empirically: the flux evaluated to a finite,
sensible value on a surface with max |omega| = 0.414 rad.

**Caveat:** reason 1 assumes B.n = 0, which a QFM surface satisfies only
approximately — that is exactly what the optimization minimizes. The seed here has
`<abs(B.n)>/<abs(B)> = 1.85e-02`, so the zeta-cut and phi-cut fluxes differ at
roughly that relative level. Psi on an unconverged surface inherits the QFM
residual as its error bar. This matters for interpreting `measured_psi_seed`.

### The vector-potential form is forced, not optional

For a `FourierRZToroidalSurface`, DESC *requires* it (`_coils.py:2483`):

    errorif(
        isinstance(eq, FourierRZToroidalSurface) and not self._use_vector_potential,
        ValueError, "Targeting a QFM surface requires the vector potential ...")

A bare surface has no `e_rho` (no radial extent), so the
integral-of-|e_rho x e_theta|*(B.n_zeta) form does not exist for it. Psi is then a
pure line integral around the poloidal curve:

    A_dot_e_theta = jnp.sum(A * data["e_theta"], axis=1)
    Psi = jnp.sum(grid.spacing[:, 1] * A_dot_e_theta)

which by Stokes depends only on the closed curve — the most robust form in a
generalized chart. Nothing to switch on; it is automatic.

### Why the flux target is measured, not inherited

`eq.Psi = 2.421177e-03` was inherited as `Psi_ref * rho**2` from the *reference*
rot_ellipse equilibrium, while the field here is SLNAM0's. Two independent probes
put it ~29 % low (section 4b, item 2), and with the **wrong sign**.
`qfm_slnam0.py` therefore measures the flux the coils actually put through the
seed surface and uses that as the `ToroidalFlux` target. Override with
`TARGET_PSI` if a different convention is wanted.

---

## 7. Open / unverified

- **Does the QFM solve converge with an omega-carrying seed?** Unknown — the full
  solve has never been run.
- **Is `SurfaceQuadraticFlux` fully omega-correct?** It built and evaluated
  without error, and its data keys are
  `["R", "Z", "n_rho", "phi", "|e_theta x e_zeta|"]` evaluated at `(R, phi, Z)`,
  which is the right evaluation point. `e_rho` was verified to carry omega
  explicitly; `n_rho` and `|e_theta x e_zeta|` were **not** separately verified.
  Building without error is not proof of correctness. Worth confirming before
  trusting a converged result.
- **Seed resolution.** `surf0.change_resolution(M=8, N=16)` keeps the
  equilibrium's boundary resolution. The notebook used M=10, N=12 on a much
  simpler seed. Not tuned.
- **Notebook cell 22 not implemented** — coil co-optimization with `SurfaceMatch`,
  `CoilLength`, `Elongation` and per-subset `FixParameters`. That also needs the
  coilset split into shaping/non-shaping subsets (cells 14-16), which is specific
  to that coilset's indexing and will not transfer to `SLNAM0_for_C5.h5`
  unchanged.
- **Weights** `QFLUX_WEIGHT=1e-1`, `TFLUX_WEIGHT=1e3` copied from the notebook;
  not tuned for this problem. The initial residual `|r| = 3.06e-02` is dominated
  by the quadratic-flux term (`dim_f = 3186`, of which 3185 are qflux and 1 is
  tflux), so the relative weighting is worth a look before a long run.

---

## 8. To resume (when explicitly instructed)

1. `sbatch job_qfm_slnam0.sl` — 4 h limit; the dry run's build+evaluate took well
   under a minute, so `MAXITER=100` at `ftol=5e-4` should fit comfortably.
2. Judge by `<abs(B.n)>/<abs(B)>` seed -> QFM in the printed summary and in
   `qfm_slnam0_196_0p82_metrics.json`. **Baseline is 1.850608e-02.** An order of
   magnitude drop is a good result; no drop means the solve is not working.
3. Cross-check `measured_psi_qfm` against `target_psi = -3.13041888e-03 Wb` in the
   same JSON.
4. Inspect `qfm_slnam0_196_0p82_boundaries.png` (seed vs QFM at 6 toroidal angles)
   for fold-over or spikes — the reference notebook family has a known
   "spikefix" variant (`QFM_SLNAM_to_SLAM_optimization_nonplanar-spikefix.ipynb`),
   so spiking is a known failure mode in this workflow.
5. Only then consider cell 22 (coil co-optimization), which needs the coilset
   subset split rebuilt for `SLNAM0_for_C5.h5`.
