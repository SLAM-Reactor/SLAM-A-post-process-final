#!/usr/bin/env python3
"""
Generate the three interactive Plotly HTML construction figures for a
stellarator -> separated stellarator -> stellarator-mirror hybrid sequence.

Usage
-----
python generate_stellarator_mirror_figures.py \
    ORIGINAL_COILSET.h5 HYBRID_COILSET.h5 \
    --outdir figures

Outputs
-------
figure_a_original_stellarator.html
figure_b_separated_stellarator.html
figure_c_stellarator_mirror_hybrid.html

Requirements
------------
- DESC / desc-opt
- numpy
- plotly

Geometry assumptions
--------------------
1. The first N physical coils in the HYBRID coilset correspond, in the same
   order, to the N physical coils in the ORIGINAL coilset.
2. Any remaining physical coils in the HYBRID coilset are the added mirror
   coils.
3. The original stellarator is separated into two approximately rigidly
   translated groups. The script infers those two groups from the displacement
   of corresponding coil centroids.
4. The added mirror coils form two approximately straight rows. The script
   infers those rows from the mirror-coil centroids and uses the end coils of
   each row as terminal mirror coils.

The generated HTMLs are self-contained. Plotly controls remain outside the
exported SVG canvas.

Figure controls
---------------
(a), (b):
    - coil thickness
    - dashed-line thickness
    - X/Y axis thickness
    - X/Y label font size

(c):
    - coil thickness
    - dashed-line thickness
    - XYZ trident arrow thickness
    - XYZ label font thickness
    - XYZ label font size

The XYZ trident in figure (c) is a small second Plotly 3-D scene whose camera
is synchronized to the main scene, so it behaves like a ParaView orientation
widget.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from desc.grid import LinearGrid
from desc.io import load
from desc.plotting import plot_coils


# ============================================================================
# Publication defaults
# ============================================================================

COIL_POINTS = 500

RED = "red"
BLUE = "blue"
BLACK = "black"
GREEN = "green"
GUIDE_COLOR = "rgb(75,75,75)"
ENDPOINT_COLOR = "rgb(45,45,45)"

COIL_WIDTH = 7
GUIDE_WIDTH = 4
AXIS_WIDTH = 5
AXIS_FONT_SIZE = 28

# Manual dash geometry in physical units. These are actual short line segments,
# not Plotly's 3-D dash style, so they look like ----- instead of ||||||.
DASH_LENGTH = 0.065
DASH_GAP = 0.045

TOP_CAMERA = dict(
    eye=dict(x=0.0, y=0.0, z=2.6),
    up=dict(x=0.0, y=1.0, z=0.0),
    center=dict(x=0.0, y=0.0, z=0.0),
    projection=dict(type="orthographic"),
)

OBLIQUE_CAMERA = dict(
    eye=dict(x=1.55, y=-1.55, z=1.05),
    up=dict(x=0.0, y=0.0, z=1.0),
    center=dict(x=0.0, y=0.0, z=0.0),
    projection=dict(type="orthographic"),
)

# Figure-(c) orientation-widget position, in Plotly paper coordinates.
# Kept inside the main canvas so it is included in SVG exports without wasting
# a large amount of paper space.
TRIDENT_DOMAIN_X = [0.50, 0.68]
TRIDENT_DOMAIN_Y = [0.11, 0.31]


# ============================================================================
# DESC -> sampled physical coil curves
# ============================================================================

def sample_physical_coils(filename: Path, points: int = COIL_POINTS):
    """Load a DESC coilset and return one sampled xyz curve per physical coil."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"desc\.io\.hdf5_io",
        )
        coilset = load(str(filename))

    grid = LinearGrid(N=points, endpoint=True)

    # We use DESC itself to expand NFP / stellarator symmetry into every
    # physical coil. The temporary Plotly figure is just a convenient way to
    # obtain the sampled centerlines in exactly the same ordering DESC plots.
    fig = plot_coils(
        coilset,
        grid=grid,
        color="black",
        lw=1,
        showgrid=False,
        showticklabels=False,
        showaxislabels=False,
        zeroline=False,
    )

    curves = []
    for tr in fig.data:
        if not all(hasattr(tr, q) for q in ("x", "y", "z")):
            continue
        x = np.asarray(tr.x, dtype=float)
        y = np.asarray(tr.y, dtype=float)
        z = np.asarray(tr.z, dtype=float)
        if len(x) < 2:
            continue
        curves.append(dict(x=x, y=y, z=z))

    if not curves:
        raise RuntimeError(f"No physical coil traces were extracted from {filename}")

    return coilset, curves


def centroid(curve):
    return np.array(
        [
            np.mean(curve["x"]),
            np.mean(curve["y"]),
            np.mean(curve["z"]),
        ],
        dtype=float,
    )


def curve_points(curve):
    return np.column_stack([curve["x"], curve["y"], curve["z"]])


# ============================================================================
# Geometry inference
# ============================================================================

def split_by_largest_projected_gap(points: np.ndarray):
    """
    Split 3-D points into two groups.

    We inspect all PCA directions and choose the direction with the strongest
    normalized gap in the sorted projected coordinates. This works well for
    both:
      - the two rigid displacement vectors of the separated stellarator halves,
      - the two mirror-coil rows.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("Need at least two points to make two groups.")

    X = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(X, full_matrices=False)

    best = None
    for direction in vt:
        p = X @ direction
        order = np.argsort(p)
        ps = p[order]
        gaps = np.diff(ps)
        if len(gaps) == 0:
            continue
        j = int(np.argmax(gaps))
        scale = np.std(p)
        score = gaps[j] / max(scale, 1e-14)
        if best is None or score > best[0]:
            best = (score, order[: j + 1], order[j + 1 :], direction)

    if best is None or len(best[1]) == 0 or len(best[2]) == 0:
        raise RuntimeError("Could not infer two geometric groups.")

    return np.asarray(best[1], int), np.asarray(best[2], int)


def infer_stellarator_halves(original, hybrid_stellarator):
    """
    Infer the two moved stellarator halves from corresponding centroid shifts.

    The returned first group is labeled RED and the second BLUE. If x is the
    dominant separation direction, the more +x-shifted group is made RED to
    reproduce the convention used in these figures.
    """
    if len(original) != len(hybrid_stellarator):
        raise ValueError("Original and hybrid stellarator trace counts differ.")

    shifts = np.array(
        [centroid(h) - centroid(o) for o, h in zip(original, hybrid_stellarator)]
    )

    g0, g1 = split_by_largest_projected_gap(shifts)
    m0 = shifts[g0].mean(axis=0)
    m1 = shifts[g1].mean(axis=0)

    separation = m0 - m1
    dominant_axis = int(np.argmax(np.abs(separation)))

    # Prefer the group displaced toward positive coordinate on the dominant
    # separation axis as RED.
    if m0[dominant_axis] >= m1[dominant_axis]:
        red, blue = g0, g1
    else:
        red, blue = g1, g0

    return red.tolist(), blue.tolist(), shifts


def infer_mirror_rows(mirror_curves):
    """
    Split added mirror coils into two rows and identify the two end coils of
    each row.

    Returns
    -------
    rows : list[list[int]]
        Indices local to mirror_curves.
    connectors : list[(3,), (3,)]
        End-center coordinates for each row.
    terminal_local : set[int]
        Local mirror indices corresponding to the four terminal coils.
    """
    centers = np.array([centroid(c) for c in mirror_curves])
    r0, r1 = split_by_largest_projected_gap(centers)

    rows = [r0.tolist(), r1.tolist()]
    connectors = []
    terminal = set()

    for row in rows:
        c = centers[row]
        X = c - c.mean(axis=0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        direction = vt[0]
        proj = X @ direction

        i_min_local = int(np.argmin(proj))
        i_max_local = int(np.argmax(proj))

        idx_min = row[i_min_local]
        idx_max = row[i_max_local]

        terminal.update([idx_min, idx_max])
        connectors.append((centers[idx_min], centers[idx_max]))

    # Stable visual ordering: the row with larger mean y is returned first.
    mean_y = [np.mean(centers[row, 1]) for row in rows]
    if mean_y[0] < mean_y[1]:
        rows = [rows[1], rows[0]]
        connectors = [connectors[1], connectors[0]]

    return rows, connectors, terminal


def validate_correspondence(original, hybrid_stellarator):
    """
    Print a simple check that corresponding original/hybrid coils differ mostly
    by rigid translation rather than shape.
    """
    residuals = []
    shift_magnitudes = []

    for o, h in zip(original, hybrid_stellarator):
        po = curve_points(o)
        ph = curve_points(h)

        if po.shape != ph.shape:
            continue

        d = centroid(h) - centroid(o)
        residual = np.sqrt(np.mean(np.sum((ph - d - po) ** 2, axis=1)))
        residuals.append(residual)
        shift_magnitudes.append(np.linalg.norm(d))

    if residuals:
        print(
            "Median translated-shape RMS mismatch:",
            f"{np.median(residuals):.3e}",
        )
        print(
            "Median stellarator-half displacement:",
            f"{np.median(shift_magnitudes):.6g}",
        )


# ============================================================================
# Plotly traces
# ============================================================================

def coil_trace(curve, color, width=COIL_WIDTH, scene="scene", kind="coil"):
    return go.Scatter3d(
        x=curve["x"],
        y=curve["y"],
        z=curve["z"],
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip",
        showlegend=False,
        scene=scene,
        meta={"kind": kind},
    )


def line_trace(p0, p1, color, width, scene="scene", kind="guide"):
    return go.Scatter3d(
        x=[p0[0], p1[0]],
        y=[p0[1], p1[1]],
        z=[p0[2], p1[2]],
        mode="lines",
        line=dict(color=color, width=width),
        hoverinfo="skip",
        showlegend=False,
        scene=scene,
        meta={"kind": kind},
    )


def dash_segments(p0, p1, dash=DASH_LENGTH, gap=DASH_GAP):
    """Return physical short line segments with no overhang past the endpoints."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)

    v = p1 - p0
    L = np.linalg.norm(v)
    if L == 0:
        return []

    u = v / L
    out = []
    s = 0.0
    while s < L:
        e = min(s + dash, L)
        out.append((p0 + s * u, p0 + e * u))
        s += dash + gap
    return out


def add_dashed_connector(fig, p0, p1, scene="scene", endpoints=True):
    for a, b in dash_segments(p0, p1):
        fig.add_trace(
            line_trace(
                a,
                b,
                color=GUIDE_COLOR,
                width=GUIDE_WIDTH,
                scene=scene,
                kind="guide",
            )
        )

    if endpoints:
        fig.add_trace(
            go.Scatter3d(
                x=[p0[0], p1[0]],
                y=[p0[1], p1[1]],
                z=[p0[2], p1[2]],
                mode="markers",
                marker=dict(color=ENDPOINT_COLOR, size=5),
                hoverinfo="skip",
                showlegend=False,
                scene=scene,
                meta={"kind": "endpoint"},
            )
        )


def add_filled_arrow(
    fig,
    axis,
    L,
    head_len,
    *,
    z0=0.0,
    scene="scene",
    shaft_kind="axis",
    head_kind="axishead",
    head_sizeref=0.145,
):
    """
    Add a solid-tip arrow.

    Crucially, the line shaft stops at the BASE of the cone head; it does not
    continue underneath the arrowhead. Therefore increasing line thickness
    cannot make the shaft protrude beyond the arrow tip.
    """
    if axis == "x":
        shaft_start = (0.0, 0.0, z0)
        shaft_end = (L - head_len, 0.0, z0)
        tip = (L, 0.0, z0)
        vec = (head_len, 0.0, 0.0)
    elif axis == "y":
        shaft_start = (0.0, 0.0, z0)
        shaft_end = (0.0, L - head_len, z0)
        tip = (0.0, L, z0)
        vec = (0.0, head_len, 0.0)
    elif axis == "z":
        shaft_start = (0.0, 0.0, 0.0)
        shaft_end = (0.0, 0.0, L - head_len)
        tip = (0.0, 0.0, L)
        vec = (0.0, 0.0, head_len)
    else:
        raise ValueError(axis)

    fig.add_trace(
        line_trace(
            shaft_start,
            shaft_end,
            color=BLACK,
            width=AXIS_WIDTH,
            scene=scene,
            kind=shaft_kind,
        )
    )

    fig.add_trace(
        go.Cone(
            x=[tip[0]],
            y=[tip[1]],
            z=[tip[2]],
            u=[vec[0]],
            v=[vec[1]],
            w=[vec[2]],
            anchor="tip",
            sizemode="absolute",
            sizeref=head_sizeref,
            colorscale=[[0, BLACK], [1, BLACK]],
            showscale=False,
            hoverinfo="skip",
            scene=scene,
            meta={"kind": head_kind},
        )
    )


def add_xy_axes(fig, z0):
    """Joined X/Y Cartesian marker for top-view figures."""
    L = 0.48
    head_len = 0.105

    add_filled_arrow(fig, "x", L, head_len, z0=z0)
    add_filled_arrow(fig, "y", L, head_len, z0=z0)

    # Fixed 3-D anchor coordinates: changing font size does not create an
    # expanding text-position offset.
    fig.add_trace(
        go.Scatter3d(
            x=[L + 0.055, 0.085],
            y=[0.0, 0.33],
            z=[z0, z0],
            mode="text",
            text=["X", "Y"],
            textfont=dict(color=BLACK, size=AXIS_FONT_SIZE),
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False,
            meta={"kind": "axislabel"},
        )
    )


def add_xyz_trident(fig):
    """Small solid-tip XYZ orientation trident in Plotly scene2."""
    L = 0.78
    head_len = 0.18

    for axis in ("x", "y", "z"):
        add_filled_arrow(
            fig,
            axis,
            L,
            head_len,
            scene="scene2",
            shaft_kind="trident",
            head_kind="tridenthead",
            head_sizeref=0.22,
        )

    fig.add_trace(
        go.Scatter3d(
            x=[L + 0.11, 0.0, 0.0],
            y=[0.0, L + 0.11, 0.0],
            z=[0.0, 0.0, L + 0.11],
            mode="text",
            text=["X", "Y", "Z"],
            textfont=dict(size=22, color=BLACK),
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False,
            scene="scene2",
            meta={"kind": "tridentlabel"},
        )
    )


# ============================================================================
# Layout / HTML controls
# ============================================================================

def physical_ranges(curves, pad_fraction=0.08):
    xyz = np.vstack([curve_points(c) for c in curves])
    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    span = np.maximum(hi - lo, 1e-8)
    pad = pad_fraction * span
    return lo - pad, hi + pad


def scene_layout(camera, lo, hi):
    return dict(
        aspectmode="data",
        xaxis=dict(visible=False, range=[lo[0], hi[0]], autorange=False),
        yaxis=dict(visible=False, range=[lo[1], hi[1]], autorange=False),
        zaxis=dict(visible=False, range=[lo[2], hi[2]], autorange=False),
        bgcolor="white",
        camera=camera,
    )


def plotly_inner_html(fig, filename):
    return fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "filename": filename,
                "format": "svg",
                "scale": 2,
            },
        },
    )


def wrapped_html(fig_html, *, xy_controls=False, xyz_controls=False):
    """
    Wrap Plotly with sliders OUTSIDE the graph div.

    Because the controls are outside the Plotly canvas, they do not appear in
    SVG exports.
    """
    xy_html = """
      <label>X/Y axis thickness
        <input id="axisWidth" type="range" min="1" max="14" step="1" value="5">
      </label>
      <label>X/Y font size
        <input id="axisFont" type="range" min="12" max="60" step="1" value="28">
      </label>
    """ if xy_controls else ""

    xyz_html = """
      <label>XYZ arrow thickness
        <input id="tridentWidth" type="range" min="1" max="14" step="1" value="5">
      </label>
      <label>XYZ font thickness
        <input id="tridentFontWeight" type="range" min="100" max="900" step="100" value="400">
      </label>
      <label>XYZ font size
        <input id="tridentFontSize" type="range" min="10" max="60" step="1" value="22">
      </label>
    """ if xyz_controls else ""

    xy_js = """
const axisIndices = [];
const axisHeadIndices = [];
const axisLabelIndices = [];
gd.data.forEach((tr, i) => {
    if (tr.meta && tr.meta.kind === 'axis') axisIndices.push(i);
    if (tr.meta && tr.meta.kind === 'axishead') axisHeadIndices.push(i);
    if (tr.meta && tr.meta.kind === 'axislabel') axisLabelIndices.push(i);
});

document.getElementById('axisWidth').addEventListener('input', e => {
    const w = +e.target.value;
    Plotly.restyle(gd, {'line.width': w}, axisIndices);

    // Keep filled cone tips roughly proportional to the shaft.
    const sr = 0.11 + 0.007 * w;
    Plotly.restyle(gd, {'sizeref': sr}, axisHeadIndices);
});

document.getElementById('axisFont').addEventListener('input', e => {
    Plotly.restyle(gd, {'textfont.size': +e.target.value}, axisLabelIndices);
});
""" if xy_controls else ""

    xyz_js = """
const tridentIndices = [];
const tridentHeadIndices = [];
const tridentLabelIndices = [];
gd.data.forEach((tr, i) => {
    if (tr.meta && tr.meta.kind === 'trident') tridentIndices.push(i);
    if (tr.meta && tr.meta.kind === 'tridenthead') tridentHeadIndices.push(i);
    if (tr.meta && tr.meta.kind === 'tridentlabel') tridentLabelIndices.push(i);
});

document.getElementById('tridentWidth').addEventListener('input', e => {
    const w = +e.target.value;
    Plotly.restyle(gd, {'line.width': w}, tridentIndices);

    // Keep filled arrowheads proportional to the shaft thickness.
    const sr = 0.15 + 0.014 * w;
    Plotly.restyle(gd, {'sizeref': sr}, tridentHeadIndices);
});

document.getElementById('tridentFontWeight').addEventListener('input', e => {
    Plotly.restyle(gd, {'textfont.weight': +e.target.value}, tridentLabelIndices);
});

document.getElementById('tridentFontSize').addEventListener('input', e => {
    Plotly.restyle(gd, {'textfont.size': +e.target.value}, tridentLabelIndices);
});

// Synchronize the orientation widget to every main-camera change.
// Plotly may report either scene.camera or nested scene.camera.* relayout keys,
// so inspect the current layout camera whenever any such key changes.
let syncingCamera = false;
gd.on('plotly_relayout', ev => {
    if (syncingCamera) return;
    const changed = Object.keys(ev).some(k => k.startsWith('scene.camera'));
    if (!changed) return;

    const cam = gd.layout.scene.camera;
    if (!cam) return;

    syncingCamera = true;
    Plotly.relayout(gd, {
        'scene2.camera': {
            eye: cam.eye,
            up: cam.up,
            center: {x: 0, y: 0, z: 0},
            projection: {type: 'orthographic'}
        }
    }).then(() => {
        syncingCamera = false;
    });
});
""" if xyz_controls else ""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
html, body {{
    margin: 0;
    background: white;
    font-family: Arial, sans-serif;
}}
.controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    align-items: center;
    padding: 8px 14px;
    border-bottom: 1px solid #ddd;
    font-size: 14px;
}}
.controls label {{
    display: flex;
    gap: 7px;
    align-items: center;
    white-space: nowrap;
}}
.plotwrap {{
    width: 100%;
    height: calc(100vh - 58px);
}}
.plotwrap > div {{
    width: 100% !important;
    height: 100% !important;
}}
</style>
</head>
<body>
<div class="controls">
  <label>Coil thickness
    <input id="coilWidth" type="range" min="1" max="16" step="1" value="{COIL_WIDTH}">
  </label>
  <label>Dashed-line thickness
    <input id="guideWidth" type="range" min="1" max="12" step="1" value="{GUIDE_WIDTH}">
  </label>
  {xy_html}
  {xyz_html}
</div>

<div class="plotwrap">
{fig_html}
</div>

<script>
const gd = document.querySelector('.plotly-graph-div');

const coilIndices = [];
const guideIndices = [];

gd.data.forEach((tr, i) => {{
    if (tr.meta && tr.meta.kind === 'coil') coilIndices.push(i);
    if (tr.meta && tr.meta.kind === 'guide') guideIndices.push(i);
}});

document.getElementById('coilWidth').addEventListener('input', e => {{
    Plotly.restyle(gd, {{'line.width': +e.target.value}}, coilIndices);
}});

document.getElementById('guideWidth').addEventListener('input', e => {{
    Plotly.restyle(gd, {{'line.width': +e.target.value}}, guideIndices);
}});

{xy_js}
{xyz_js}
</script>
</body>
</html>
"""


# ============================================================================
# Three figures
# ============================================================================

def make_figure_a(
    original,
    red_indices,
    blue_indices,
    connectors,
    lo,
    hi,
):
    fig = go.Figure()

    for i in red_indices:
        fig.add_trace(coil_trace(original[i], RED))
    for i in blue_indices:
        fig.add_trace(coil_trace(original[i], BLUE))

    # In the original, the two separated connector endpoints collapse in the
    # TOP VIEW to one cut point per mirror row. The midpoint of each separated
    # connector provides that cut-point location without hardcoding x or y.
    cut0 = 0.5 * (np.asarray(connectors[0][0]) + np.asarray(connectors[0][1]))
    cut1 = 0.5 * (np.asarray(connectors[1][0]) + np.asarray(connectors[1][1]))

    # Keep the conceptual cut guide in the central z plane of the original.
    zmid = 0.5 * (lo[2] + hi[2])
    cut0 = cut0.copy()
    cut1 = cut1.copy()
    cut0[2] = zmid
    cut1[2] = zmid

    # Figure (a): show only the two cut-point dots, with NO dashed line.
    fig.add_trace(
        go.Scatter3d(
            x=[cut0[0], cut1[0]],
            y=[cut0[1], cut1[1]],
            z=[cut0[2], cut1[2]],
            mode="markers",
            marker=dict(color=ENDPOINT_COLOR, size=5),
            hoverinfo="skip",
            showlegend=False,
            meta={"kind": "endpoint"},
        )
    )

    # Lift the X/Y marker slightly above the geometry so it remains visible in
    # the top view while the coils themselves retain proper 3-D depth testing.
    z_axis = hi[2] - 0.02 * (hi[2] - lo[2])
    add_xy_axes(fig, z0=z_axis)

    fig.update_layout(
        width=1100,
        height=850,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        scene=scene_layout(TOP_CAMERA, lo, hi),
    )

    return fig


def make_figure_b(
    hybrid_stellarator,
    red_indices,
    blue_indices,
    connectors,
    lo,
    hi,
):
    fig = go.Figure()

    for i in red_indices:
        fig.add_trace(coil_trace(hybrid_stellarator[i], RED))
    for i in blue_indices:
        fig.add_trace(coil_trace(hybrid_stellarator[i], BLUE))

    for p0, p1 in connectors:
        add_dashed_connector(fig, p0, p1, endpoints=True)

    z_axis = hi[2] - 0.02 * (hi[2] - lo[2])
    add_xy_axes(fig, z0=z_axis)

    fig.update_layout(
        width=1100,
        height=850,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        scene=scene_layout(TOP_CAMERA, lo, hi),
    )

    return fig


def make_figure_c(
    hybrid_stellarator,
    mirror_curves,
    red_indices,
    blue_indices,
    connectors,
    terminal_local,
    lo,
    hi,
):
    fig = go.Figure()

    for i in red_indices:
        fig.add_trace(coil_trace(hybrid_stellarator[i], RED))
    for i in blue_indices:
        fig.add_trace(coil_trace(hybrid_stellarator[i], BLUE))

    for p0, p1 in connectors:
        add_dashed_connector(fig, p0, p1, endpoints=True)

    for j, c in enumerate(mirror_curves):
        color = GREEN if j in terminal_local else BLACK
        fig.add_trace(coil_trace(c, color))

    add_xyz_trident(fig)

    fig.update_layout(
        width=1100,
        height=850,
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,

        scene=dict(
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0]),
            **scene_layout(OBLIQUE_CAMERA, lo, hi),
        ),

        # Small detached camera-synchronized orientation widget, fully inside
        # the export canvas.
        scene2=dict(
            domain=dict(x=TRIDENT_DOMAIN_X, y=TRIDENT_DOMAIN_Y),
            aspectmode="cube",
            xaxis=dict(visible=False, range=[-0.25, 1.05]),
            yaxis=dict(visible=False, range=[-0.25, 1.05]),
            zaxis=dict(visible=False, range=[-0.25, 1.05]),
            bgcolor="rgba(0,0,0,0)",
            camera=OBLIQUE_CAMERA,
        ),
    )

    return fig


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate the three stellarator-mirror construction HTML figures."
    )
    parser.add_argument(
        "original_coilset",
        type=Path,
        help="DESC HDF5 file for the original stellarator coilset.",
    )
    parser.add_argument(
        "hybrid_coilset",
        type=Path,
        help="DESC HDF5 file for the final stellarator-mirror hybrid coilset.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory).",
    )
    parser.add_argument(
        "--coil-points",
        type=int,
        default=COIL_POINTS,
        help=f"Sample points per physical coil (default: {COIL_POINTS}).",
    )

    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Loading original coilset:", args.original_coilset)
    original_cs, original = sample_physical_coils(
        args.original_coilset,
        points=args.coil_points,
    )

    print("Loading hybrid coilset:", args.hybrid_coilset)
    hybrid_cs, hybrid = sample_physical_coils(
        args.hybrid_coilset,
        points=args.coil_points,
    )

    n_original = len(original)
    n_hybrid = len(hybrid)

    if n_hybrid <= n_original:
        raise RuntimeError(
            "The hybrid must contain more physical coils than the original.\n"
            f"Original physical traces: {n_original}\n"
            f"Hybrid physical traces:   {n_hybrid}"
        )

    hybrid_stellarator = hybrid[:n_original]
    mirror_curves = hybrid[n_original:]

    print("Original physical coils:", n_original)
    print("Hybrid physical coils:", n_hybrid)
    print("Added mirror coils:", len(mirror_curves))

    validate_correspondence(original, hybrid_stellarator)

    red_indices, blue_indices, shifts = infer_stellarator_halves(
        original,
        hybrid_stellarator,
    )

    rows, connectors, terminal_local = infer_mirror_rows(mirror_curves)

    print("Inferred red stellarator coils:", len(red_indices))
    print("Inferred blue stellarator coils:", len(blue_indices))
    print("Mirror row sizes:", [len(r) for r in rows])
    print(
        "Terminal mirror local indices:",
        sorted(terminal_local),
    )

    mean_red_shift = np.mean(shifts[red_indices], axis=0)
    mean_blue_shift = np.mean(shifts[blue_indices], axis=0)
    print("Mean red-half displacement:", mean_red_shift)
    print("Mean blue-half displacement:", mean_blue_shift)

    # Use a common physical frame for all three figures so apparent geometric
    # expansion is not caused by separate autoscaling.
    all_curves_for_frame = original + hybrid
    lo, hi = physical_ranges(all_curves_for_frame, pad_fraction=0.08)

    fig_a = make_figure_a(
        original,
        red_indices,
        blue_indices,
        connectors,
        lo,
        hi,
    )

    fig_b = make_figure_b(
        hybrid_stellarator,
        red_indices,
        blue_indices,
        connectors,
        lo,
        hi,
    )

    fig_c = make_figure_c(
        hybrid_stellarator,
        mirror_curves,
        red_indices,
        blue_indices,
        connectors,
        terminal_local,
        lo,
        hi,
    )

    outputs = [
        (
            args.outdir / "figure_a_original_stellarator.html",
            fig_a,
            "figure_a_original_stellarator",
            True,
            False,
        ),
        (
            args.outdir / "figure_b_separated_stellarator.html",
            fig_b,
            "figure_b_separated_stellarator",
            True,
            False,
        ),
        (
            args.outdir / "figure_c_stellarator_mirror_hybrid.html",
            fig_c,
            "figure_c_stellarator_mirror_hybrid",
            False,
            True,
        ),
    ]

    for path, fig, export_name, xy_controls, xyz_controls in outputs:
        inner = plotly_inner_html(fig, export_name)
        html = wrapped_html(
            inner,
            xy_controls=xy_controls,
            xyz_controls=xyz_controls,
        )
        path.write_text(html, encoding="utf-8")
        print("Saved:", path)


if __name__ == "__main__":
    main()
