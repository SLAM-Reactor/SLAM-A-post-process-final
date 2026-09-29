# Goal: Correct FTGen Coordinate Summary Plot

Build a reliable, publication-quality summary plot explaining the FTGen coordinate mapping. The desired figure has three `contourf` panels using the `viridis` colormap:

1. `(alpha, phi)`
2. `(theta_PEST, zeta)`
3. `(theta_B, zeta_B)`

The default plotted quantity should be Cartesian `X`. The plot should have black field-line overlays only on the `(alpha, phi)` panel, using about 10-15 field lines with linewidth `0.5`.

## Current Status

The current implementation in `plot_FTGen_coordinate_maps.py` should not be trusted.

The generated `FTGen_X_coordinate_maps.png` still looks wrong and is too noisy, especially in the `(alpha, phi)` panel. The attempted repair that avoided wrapping in `phi` did not produce an acceptable or convincing alpha-coordinate plot.

Treat the current plot as a failed draft, not as a validated diagnostic.

## Main Failure

The `(alpha, phi)` panel is wrong.

For the correct FTGen alpha-grid representation, `X(alpha, phi)` should be discontinuous in `phi` for a fixed `alpha`. The current plot does not represent this correctly and instead shows noisy vertical structure. This indicates that the alpha-grid construction, interpolation, or plotting data source is wrong.

## Do Not Repeat This Mistake

Do not build the alpha panel directly from only:

```text
boundary_RZ_data_*.npz
```

That file stores only the raw long field-line trace:

```text
r
z
phi
```

The current script tries to reconstruct the alpha panel from this trace and then regrid it. This is not robust enough for the intended figure and appears to introduce noisy artifacts.

## Correct Data Source

The alpha panel should be made from the native intermediate FTGen arrays created in `boozer_spline.py`, not from a later reconstruction in the plotting script.

Relevant code pattern in `boozer_spline.py`:

```python
alpha_list.append(np.mod(2 * np.pi * iota * i, 2 * np.pi))
R_list.append(r[i*npoints_per_turn:(i+1)*npoints_per_turn])
X_list.append(r[i*npoints_per_turn:(i+1)*npoints_per_turn] * np.cos(phi))
Y_list.append(r[i*npoints_per_turn:(i+1)*npoints_per_turn] * np.sin(phi))
Z_list.append(z[i*npoints_per_turn:(i+1)*npoints_per_turn])
nu_list.append(...)
B_list.append(...)
```

After sorting:

```python
alpha_sort = alpha[sort_idx]
R_array = np.array(R_list)[sort_idx]
X_array = np.array(X_list)[sort_idx]
Y_array = np.array(Y_list)[sort_idx]
Z_array = np.array(Z_list)[sort_idx]
nu_array = np.array(nu_list)[sort_idx]
B_array = np.array(B_list)[sort_idx]
```

These arrays are the correct starting point for the `(alpha, phi)` panel.

## Required Fix

Modify the FTGen generation script, especially the copy in the `NFP2_A+9*` directory, to save the alpha-grid arrays directly:

```python
np.savez(
    f"alpha_data_{fname}_{nturns_bdry}_{rho_suffix}.npz",
    alpha=alpha_sort,
    phi=phi,
    X_alpha=X_array,
    Y_alpha=Y_array,
    R_alpha=R_array,
    Z_alpha=Z_array,
    modB_alpha=B_array,
    nu_alpha=nu_array,
    iota=iota,
    G=G,
)
```

Then make the plotting script read `alpha_data_*.npz` for the first panel.

## Important Plotting Rule

For the alpha panel, avoid any interpolation that enforces continuity across the `phi = 0 / 2*pi` boundary.

Best first attempt:

```python
alpha_2D, phi_2D = np.meshgrid(alpha_sort, phi, indexing="ij")
plt.contourf(alpha_2D, phi_2D, X_array, levels=..., cmap="viridis")
```

If a smoother visual is needed, interpolate only after confirming the discontinuity is preserved. Do not wrap in `phi` for `X`.

## Validation Tests

Before accepting the figure, check:

```python
jump = np.abs(X_alpha[:, 0] - X_alpha[:, -1])
print(jump.max(), jump.mean())
```

This should show the expected `phi` discontinuity.

Also inspect:

```python
plt.figure()
plt.plot(phi, X_alpha[i, :])
```

for several fixed `alpha` rows. The line plot should make the discontinuity obvious and should not show unexplained high-frequency noise.

## PEST And Boozer Panels

The PEST and Boozer panels are less suspect than the alpha panel, but they should still be checked.

For PEST `.npz` files:

```python
X_PEST = R_PEST * np.cos(phi_cyl)
```

because `X_PEST` is usually not stored directly.

For Boozer `.npz` files:

```python
X_B = X_booz
```

when available.

If using a Boozer pickle, evaluate the saved `X` spline on `(theta_B, zeta_B)`.

## User Requirements

Keep the script as an editable research script:

- no argparse
- no `main()`
- no classes
- knobs set directly at the top of the script
- user can change files, quantity, resolution, ticks, colorbar ticks, font sizes, label pads, line thickness, contour levels, output name, and DPI
- field lines appear only on `(alpha, phi)`
- default field lines: 10-15
- default field-line linewidth: `0.5`

## Recommended Next Sequence

1. Add direct `alpha_data_*.npz` saving to `NFP2_A+9*/boozer_spline.py`.
2. Regenerate or create the alpha `.npz` for the same case used by the PEST and Boozer files.
3. Plot the native alpha array directly before doing any regridding.
4. Confirm `X_alpha` has the expected discontinuity and does not have unexplained noise.
5. Only then update `plot_FTGen_coordinate_maps.py`.
6. Re-render `FTGen_X_coordinate_maps.png`.
7. Visually compare fixed-alpha line cuts and the contour plot before trusting the figure.

