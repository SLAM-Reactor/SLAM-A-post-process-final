# QFM_surf summary

Run all scripts from this directory:

```bash
cd QFM_surf
```

Use the DESC environment, for example:

```bash
/home/rgaur/anaconda3/envs/desc-env3/bin/python script_name.py
```

## Current input files

The kept/current field and QFM surface pairs are:

```text
SLNAM:
  field   = SLNAM-field-09-09-26.h5
  surface = SLNAM_QFM_surface_09-09-26_3.h5

SLAM:
  field   = SLAM-field-09-09-26_2.h5
  surface = SLAM_QFM_surface_09-09-26_2.h5
```

The old non-`_2` SLAM files and old SLNAM `hres_08-09-26` surface were removed from this folder.

## Scripts

### `plot_qfm_surfaces.py`

Makes the two interactive 3D Plotly figures of coils plus the QFM surface colored by

```text
B.n / <|B|>
```

Run:

```bash
/home/rgaur/anaconda3/envs/desc-env3/bin/python plot_qfm_surfaces.py
```

Uses:

```text
SLNAM-field-09-09-26.h5
SLNAM_QFM_surface_09-09-26_3.h5
SLAM-field-09-09-26_2.h5
SLAM_QFM_surface_09-09-26_2.h5
qfm_surface_Bn_data.npz
qfm_surface_Bn_metrics.json
```

Writes:

```text
SLNAM_QFM_surface_and_coils_Bn.html
SLAM_QFM_surface_and_coils_Bn.html
qfm_surface_Bn_data.npz        only if recomputing
qfm_surface_Bn_metrics.json    only if recomputing
```

Normal mode is cached:

```python
USE_CACHED_DATA = True
RECOMPUTE_LABELS = []
```

To recompute one case, set for example:

```python
RECOMPUTE_LABELS = ["SLNAM"]
```

then set it back to `[]` after the run.

Current cached metrics:

```text
SLNAM: <|B.n|>/<|B|> = 2.798085e-04
SLAM:  <|B.n|>/<|B|> = 5.196327e-04
```

The Plotly camera export button is configured for SVG export from the HTML modebar.

### `compare_qfm_cross_sections.py`

Plots SLNAM and SLAM QFM cross-sections in one shared `R-Z` frame.

Run:

```bash
/home/rgaur/anaconda3/envs/desc-env3/bin/python compare_qfm_cross_sections.py
```

Uses:

```text
SLNAM_QFM_surface_09-09-26_3.h5
SLAM_QFM_surface_09-09-26_2.h5
```

Writes:

```text
SLAM_QFM_surface_cross_section_comparison.png
SLAM_QFM_surface_cross_section_comparison.pdf
```

This script directly evaluates high-resolution closed curves using `surface.compute(["R", "Z"])`.
It does not use `plot_boundaries`, because the lower-resolution boundary helper made misleading sharp connector artifacts.

### `trace_axis_B.py`

Traces the magnetic axis for the SLNAM and SLAM fields, evaluates `|B|` on the axis, and saves the data.

Run:

```bash
/home/rgaur/anaconda3/envs/desc-env3/bin/python trace_axis_B.py
```

Uses:

```text
SLNAM-field-09-09-26.h5
SLAM-field-09-09-26_2.h5
```

Writes:

```text
axis_B_data.npz
axis_B_metrics.json
```

This is the expensive axis-tracing step. Do not rerun it just to change plot styling.

### `post_process_axis_B.py`

Makes the simple Matplotlib comparison of cached `|B|` on axis.

Run after `trace_axis_B.py`:

```bash
/home/rgaur/anaconda3/envs/desc-env3/bin/python post_process_axis_B.py
```

Uses:

```text
axis_B_data.npz
axis_B_metrics.json
```

Writes:

```text
axis_B_comparison.png
axis_B_comparison.pdf
```

This script does not recompute field lines or magnetic fields. It only plots cached data.

## Common mistakes we hit

1. Wrong surface/field variants.

   The current intended pairings are SLNAM `_3` surface with the SLNAM field, and SLAM `_2` field/surface. Mixing in the old SLNAM `hres_08-09-26` surface or the non-`_2` SLAM files gives misleading comparisons and stale metrics.

2. Bad 3D surface geometry from hand-made symmetry replication.

   The 3D Plotly surface must not invent geometry by manually reflecting/rotating the cached low-resolution grid. The script now recomputes the displayed `X,Y,Z` directly from the DESC surface file on the display grid. The cached low-resolution data are used for the `B.n/<|B|>` color values/limits, not for making fake surface geometry.

3. SVG export confusion.

   Do not use Kaleido static export for these 3D Plotly figures. That caused Chrome/Kaleido failures and raster-looking output. The HTML files are written with the Plotly modebar export configured to SVG:

   ```python
   PLOTLY_EXPORT_FORMAT = "svg"
   ```

   Open the HTML and use the camera/export button. The old `.svg` files from failed attempts were removed.

