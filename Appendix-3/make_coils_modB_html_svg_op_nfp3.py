#!/usr/bin/env python3
"""Same render as Appendix-1/make_coils_modB_html_svg_nfp3.py, but for the
NFP=3 precursor OP DESC equilibrium and its own OP-era coilset
(NFP3_coilset_42_coils_smoother_Bn_0p02_N5.h5) - not the SLNAMified/SLAM-A
coilset from Appendix-1, which was fit to the mirror-inserted equilibrium and
does not match this pre-mirror OP boundary. |B| comes from
compute_op_surface_modB.py's cached DESC output (op_surface_modB_nfp3.npz) -
run that first, in desc-env. This script itself runs in gvec-env (has
pyvista/netCDF4; DESC and pyvista are not installed in the same env here)."""

from __future__ import annotations

import os

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")

from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import pyvista as pv


# ----------------------------- run settings ----------------------------- #

SURFACE_CACHE = Path("op_surface_modB_nfp3.npz")
COILSET = Path("NFP3_coilset_42_coils_smoother_Bn_0p02_N5.h5")

WRITE_STATIC_HTML = True
START_DESKTOP_SVG_PICKER = True  # needs a real display; see Appendix-1 script

OUTPUT_STEM = Path("coils_op_modB_nfp3")


# ----------------------------- mesh settings ---------------------------- #

RHO_SURFACE = 1.0
THETA_COUNT = 50
ZETA_COUNT = 128
COIL_POINTS = 400
COIL_TUBE_RADIUS = 0.008


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


def build_desc_surface_mesh() -> pv.PolyData:
    """Build the rho=1 surface from the cached DESC output.

    The cache stores flat arrays in DESC's own node order (theta fastest,
    zeta slowest). VTK structured grids also take the first dimension as the
    fastest-varying one, so dimensions are [theta, zeta, 1] and both the
    points and the scalars are used flat, in that same single order. Raveling
    the two with different orders scrambles |B| across the surface.
    """
    cache = np.load(SURFACE_CACHE)
    x, y, z, modB = (np.asarray(cache[key]).ravel() for key in ("X", "Y", "Z", "mod_B"))
    theta_count, zeta_count = int(cache["theta_count"]), int(cache["zeta_count"])

    expected = theta_count * zeta_count
    if x.size != expected:
        raise RuntimeError(
            f"{SURFACE_CACHE} holds {x.size} points, expected "
            f"{expected} = {theta_count} theta x {zeta_count} zeta; "
            "re-run compute_op_surface_modB.py in desc-env"
        )
    if str(cache["layout"]) != "theta_fastest":
        raise RuntimeError(
            f"{SURFACE_CACHE} has an unexpected layout; "
            "re-run compute_op_surface_modB.py in desc-env"
        )

    mesh = pv.StructuredGrid()
    mesh.points = np.column_stack([x, y, z])
    mesh.dimensions = [theta_count, zeta_count, 1]
    mesh["mod_B"] = modB
    return mesh.extract_surface()


def add_desc_surface(plotter: pv.Plotter) -> None:
    mesh = build_desc_surface_mesh()
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
    add_desc_surface(plotter)
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
    plotter.show(title="NFP3 precursor OP: coils + DESC |B|")
    plotter.close()


def main() -> None:
    if WRITE_STATIC_HTML:
        write_static_html()
    if START_DESKTOP_SVG_PICKER:
        launch_desktop_svg_picker()


if __name__ == "__main__":
    main()
