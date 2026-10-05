#!/usr/bin/env python3
"""Post-process the NFP=3 GVEC |B| surface with its DESC coilset in PyVista,
same recipe/controls as GVEC_f67/post-process/plot_coils_gvec_modB_3d.py.

This script has two jobs:

  1. Write a static interactive HTML file (client-side vtk.js; can rotate in
     a browser but cannot save an SVG from the chosen view).
  2. Launch a desktop PyVista window where the view can be adjusted, then
     save an SVG from the current view by pressing ``s``. This needs a real
     display (X forwarding / NoMachine / VNC) - it will not work in a
     headless batch job or a plain SSH terminal.

Run in gvec-env.
"""

from __future__ import annotations

import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")

from pathlib import Path

import gvec
from netCDF4 import Dataset
import numpy as np
import pyvista as pv
from gvec.vtk import ev2vtk


# ----------------------------- run settings ----------------------------- #

CASE_DIR = Path("coils_and_gvec_NFP3")
INI = CASE_DIR / "parameter_surface_3_1p0_128_1p00_final.ini"
STATE = CASE_DIR / "surface_3_1p0_128_1p00_State_final.dat"
COILSET = CASE_DIR / "SLNAMified_NFP3_A10p1_L_over_R_1p0_SLAM-A-lowcur.h5"

WRITE_STATIC_HTML = True
START_DESKTOP_SVG_PICKER = True  # needs a real display; see module docstring

OUTPUT_STEM = Path("coils_gvec_modB_nfp3")
VTK_STEM = Path("pyGVEC_3d_visu_nfp3")


# ----------------------------- mesh settings ---------------------------- #

RHO_SURFACE = [1.0]
THETA_COUNT = 50
ZETA_COUNT = 128
COIL_POINTS = 400
COIL_TUBE_RADIUS = 0.01


# ----------------------------- view settings ---------------------------- #

WINDOW_SIZE = (1600, 1100)
BACKGROUND = "white"
CAMERA_POSITION = "iso"


# ----------------------------- visual style ----------------------------- #

COIL_COLOR = "black"
COIL_OPACITY = 1.0
COIL_AMBIENT = 0.9
COIL_DIFFUSE = 0.2

SURFACE_CMAP = "RdBu_r"
SURFACE_OPACITY = 1.0
SURFACE_SHOW_MESH = True
SURFACE_MESH_COLOR = "black"
SURFACE_MESH_LINE_WIDTH = 0.05
SURFACE_AMBIENT = 0.90
SURFACE_DIFFUSE = 0.45
SURFACE_SPECULAR = 0.0

COLORBAR_TITLE = "|B|"
COLORBAR_TITLE_SIZE = 28
COLORBAR_LABEL_SIZE = 24
COLORBAR_N_LABELS = 4
COLORBAR_FORMAT = "%.2f"
COLORBAR_WIDTH = 0.72
COLORBAR_HEIGHT = 0.10
COLORBAR_POSITION_X = 0.14
COLORBAR_POSITION_Y = 0.03

SHOW_ORIGIN_AXES = True
AXES_LINE_WIDTH = 2
AXES_LABEL_SIZE = 18


def read_variable(group, name: str) -> np.ndarray:
    return np.asarray(group.variables[name][:])


def decode_hdf5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_hdf5_string(value.item())
    return str(value)


def fourier_series(modes: np.ndarray, coefficients: np.ndarray, t: np.ndarray) -> np.ndarray:
    values = np.zeros_like(t)
    for mode, coefficient in zip(modes, coefficients):
        if mode == 0:
            values += coefficient
        elif mode > 0:
            values += coefficient * np.cos(mode * t)
        else:
            values += coefficient * np.sin(abs(mode) * t)
    return values


def make_tube(points: np.ndarray, radius: float) -> pv.PolyData:
    closed_points = np.vstack([points, points[0]])
    polyline = pv.PolyData(closed_points)
    polyline.lines = np.hstack([[len(closed_points)], np.arange(len(closed_points))])
    return polyline.tube(radius=radius)


def make_fourier_xyz_coil_points(group, t: np.ndarray) -> np.ndarray:
    x = fourier_series(read_variable(group.groups["_X_basis"], "_modes")[:, 2], read_variable(group, "_X_n"), t)
    y = fourier_series(read_variable(group.groups["_Y_basis"], "_modes")[:, 2], read_variable(group, "_Y_n"), t)
    z = fourier_series(read_variable(group.groups["_Z_basis"], "_modes")[:, 2], read_variable(group, "_Z_n"), t)

    points = np.column_stack([x, y, z])
    rotation = read_variable(group, "_rotmat").reshape(3, 3)
    shift = read_variable(group, "_shift")
    return points @ rotation.T + shift[None, :]


def make_fourier_planar_coil_points(group, t: np.ndarray) -> np.ndarray:
    center = read_variable(group, "_center")
    normal = read_variable(group, "_normal")
    normal = normal / np.linalg.norm(normal)

    radius = fourier_series(read_variable(group.groups["_r_basis"], "_modes")[:, 2], read_variable(group, "_r_n"), t)

    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, normal)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(normal, trial)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    points = center[None, :] + radius[:, None] * (
        np.cos(t)[:, None] * e1[None, :] + np.sin(t)[:, None] * e2[None, :]
    )

    rotation = read_variable(group, "_rotmat").reshape(3, 3)
    shift = read_variable(group, "_shift")
    return points @ rotation.T + shift[None, :]


def read_scalar(group, name: str, default):
    if name not in group.variables:
        return default
    value = read_variable(group, name)
    return value.item() if value.shape == () else value


def reflection_matrix(normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=float)
    return np.eye(3) - 2.0 * np.outer(normal, normal) / np.inner(normal, normal)


def rotate_z(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return points @ rotation.T


def expand_coilset_symmetry(points: list[np.ndarray], nfp: int, sym: bool) -> list[np.ndarray]:
    base_points = list(points)
    if sym:
        normal = np.array([-np.sin(np.pi / nfp), np.cos(np.pi / nfp), 0.0])
        flip = reflection_matrix(normal) @ reflection_matrix(np.array([0.0, 0.0, 1.0]))
        reflected_points = [coil_points @ flip.T for coil_points in reversed(base_points)]
        base_points = base_points + reflected_points

    expanded_points = []
    for period in range(nfp):
        angle = 2.0 * np.pi * period / nfp
        expanded_points.extend(rotate_z(coil_points, angle) for coil_points in base_points)
    return expanded_points


def coil_points_from_group(group, t: np.ndarray) -> list[np.ndarray]:
    if "__class__" not in group.variables:
        points = []
        for child in group.groups.values():
            points.extend(coil_points_from_group(child, t))
        return points

    class_name = decode_hdf5_string(read_variable(group, "__class__"))
    if "FourierXYZCoil" in class_name:
        return [make_fourier_xyz_coil_points(group, t)]
    if "FourierPlanarCoil" in class_name:
        return [make_fourier_planar_coil_points(group, t)]
    if "_coils" not in group.groups:
        return []

    coils = group.groups["_coils"]
    keys = sorted([key for key in coils.groups if key.isdigit()], key=int)
    child_points = []
    for key in keys:
        child_points.extend(coil_points_from_group(coils.groups[key], t))

    nfp = int(read_scalar(group, "_NFP", 1))
    sym = bool(read_scalar(group, "_sym", False))
    return expand_coilset_symmetry(child_points, nfp, sym)


def add_coils(plotter: pv.Plotter, coilset: Path) -> None:
    t = np.linspace(0.0, 2.0 * np.pi, COIL_POINTS, endpoint=False)
    with Dataset(coilset, "r") as root:
        coil_points = coil_points_from_group(root, t)
    print(f"plotting {len(coil_points)} symmetry-expanded coils from {coilset.name}", flush=True)

    for points in coil_points:
        tube = make_tube(points, COIL_TUBE_RADIUS)
        plotter.add_mesh(
            tube, color=COIL_COLOR, opacity=COIL_OPACITY,
            ambient=COIL_AMBIENT, diffuse=COIL_DIFFUSE, specular=0.0, smooth_shading=True,
        )


def scalar_bar_args() -> dict:
    return {
        "title": COLORBAR_TITLE, "label_font_size": COLORBAR_LABEL_SIZE,
        "title_font_size": COLORBAR_TITLE_SIZE, "n_labels": COLORBAR_N_LABELS,
        "fmt": COLORBAR_FORMAT, "width": COLORBAR_WIDTH, "height": COLORBAR_HEIGHT,
        "position_x": COLORBAR_POSITION_X, "position_y": COLORBAR_POSITION_Y, "color": "black",
    }


def write_gvec_surface_vtk() -> Path:
    state = gvec.State(str(INI), str(STATE))
    theta = np.linspace(0.0, 2.0 * np.pi, THETA_COUNT)
    zeta = np.linspace(0.0, 2.0 * np.pi, ZETA_COUNT)
    variables = ["X1", "X2", "LA", "pos", "mod_B", "B"]
    evaluations = state.evaluate(*variables, rho=RHO_SURFACE, theta=theta, zeta=zeta)
    ev2vtk(str(VTK_STEM), evaluations[variables])
    return VTK_STEM.with_suffix(".vts")


def add_gvec_surface(plotter: pv.Plotter) -> None:
    mesh = pv.read(write_gvec_surface_vtk())
    plotter.add_mesh(
        mesh, scalars="mod_B", cmap=SURFACE_CMAP, opacity=SURFACE_OPACITY,
        show_edges=SURFACE_SHOW_MESH, edge_color=SURFACE_MESH_COLOR, line_width=SURFACE_MESH_LINE_WIDTH,
        ambient=SURFACE_AMBIENT, diffuse=SURFACE_DIFFUSE, specular=SURFACE_SPECULAR,
        smooth_shading=True, scalar_bar_args=scalar_bar_args(),
    )


def style_axes(axes_actor, line_width: int, label_size: int) -> None:
    if axes_actor is None:
        return
    for axis in ("X", "Y", "Z"):
        shaft = getattr(axes_actor, f"Get{axis}AxisShaftProperty")()
        tip = getattr(axes_actor, f"Get{axis}AxisTipProperty")()
        caption = getattr(axes_actor, f"Get{axis}AxisCaptionActor2D")()
        shaft.SetLineWidth(line_width)
        tip.SetLineWidth(line_width)
        caption.GetCaptionTextProperty().SetFontSize(label_size)


def ensure_offscreen_display() -> None:
    """Make sure an X display exists for off-screen rendering.

    ``pv.start_xvfb()`` overwrites ``DISPLAY`` process-wide with ``:99``, which
    would send the later desktop window to the invisible virtual framebuffer.
    So only start Xvfb when there is no real display to use.
    """
    if os.environ.get("DISPLAY"):
        return
    pv.start_xvfb()


def build_plotter(off_screen: bool = True) -> pv.Plotter:
    if off_screen:
        ensure_offscreen_display()
    plotter = pv.Plotter(off_screen=off_screen, window_size=WINDOW_SIZE)
    plotter.set_background(BACKGROUND)
    add_coils(plotter, COILSET)
    add_gvec_surface(plotter)
    if SHOW_ORIGIN_AXES:
        axes_actor = plotter.add_axes(line_width=AXES_LINE_WIDTH)
        style_axes(axes_actor, AXES_LINE_WIDTH, AXES_LABEL_SIZE)
    plotter.camera_position = CAMERA_POSITION
    plotter.reset_camera()
    return plotter


def write_static_html() -> None:
    plotter = build_plotter(off_screen=True)
    html_path = OUTPUT_STEM.with_suffix(".html")
    plotter.export_html(html_path)
    plotter.close()
    print(f"wrote {html_path}", flush=True)


def save_svg(plotter: pv.Plotter, path: Path) -> None:
    try:
        plotter.save_graphic(path, title=path.stem, raster=False)
    except Exception as exc:
        print(f"vector SVG export failed ({exc}); writing raster SVG", flush=True)
        plotter.save_graphic(path, title=path.stem, raster=True)
    print(f"wrote {path}", flush=True)


def launch_desktop_svg_picker() -> None:
    if not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "no DISPLAY - the desktop window needs X forwarding, NoMachine or VNC"
        )
    plotter = build_plotter(off_screen=False)
    svg_path = OUTPUT_STEM.with_suffix(".svg")

    def save_current_view() -> None:
        plotter.render()
        save_svg(plotter, svg_path)

    plotter.add_key_event("s", save_current_view)
    print("rotate/zoom in the PyVista window, then press 's' to save SVG", flush=True)
    print(f"SVG output: {svg_path}", flush=True)
    print("close the PyVista window when done", flush=True)
    plotter.show(title="NFP3: coils + GVEC |B|")
    plotter.close()


def main() -> None:
    if WRITE_STATIC_HTML:
        write_static_html()
    if START_DESKTOP_SVG_PICKER:
        launch_desktop_svg_picker()


if __name__ == "__main__":
    main()
