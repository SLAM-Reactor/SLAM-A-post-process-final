"""
Render a combined 3D visualization of the mr1p7 SLAM coilset and GVEC plasma surface colored by |B|.

Usage:
    python plot_coilset_and_modB.py <coilset.h5> [tube_thickness]

Reads the GVEC state from the current directory and the DESC coilset HDF5 file given
as the first command-line argument. Renders both in a single PyVista scene with the
plasma surface colored by |B| (``mod_B``). Exports an interactive HTML file named
``<coilset_basename>.html``.

Args:
    argv[1]: Path to the DESC coilset HDF5 file.
    argv[2]: (optional) Tube thickness for coil rendering (default 0.01).
"""

import os
import sys
import h5py
import gvec
import numpy as np
import pyvista as pv
from gvec.vtk import ev2vtk
import matplotlib.pyplot as plt

# Set global font sizes
plt.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.labelsize': 18,      # Axis label font size
    'axes.titlesize': 18,      # Title font size
    'xtick.labelsize': 16,     # X-axis tick label size
    'ytick.labelsize': 16,     # Y-axis tick label size
    'legend.fontsize': 14,     # Legend font size
})



state = gvec.find_state(os.getcwd() + "/")

fname = sys.argv[1]
thickness = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
out = fname.rsplit(".", 1)[0] + ".html"


f = h5py.File(fname, "r")
t = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)

# ONE plotter for everything
p = pv.Plotter()
p.set_background("white")

stack = [f]

while stack:
    g = stack.pop()

    if "__class__" not in g:
        for k in g.keys():
            if isinstance(g[k], h5py.Group):
                stack.append(g[k])
        continue

    cls = g["__class__"][()]
    cls = cls.decode() if isinstance(cls, bytes) else str(cls)

    if "FourierXYZCoil" in cls:
        X_modes = g["_X_basis"]["_modes"][()][:, 2]
        Y_modes = g["_Y_basis"]["_modes"][()][:, 2]
        Z_modes = g["_Z_basis"]["_modes"][()][:, 2]

        X_n = g["_X_n"][()]
        Y_n = g["_Y_n"][()]
        Z_n = g["_Z_n"][()]

        rotmat = g["_rotmat"][()].reshape(3, 3)
        shift = g["_shift"][()]

        x = np.zeros_like(t)
        y = np.zeros_like(t)
        z = np.zeros_like(t)

        for n, c in zip(X_modes, X_n):
            x += c if n == 0 else c * (np.cos(n * t) if n > 0 else np.sin(abs(n) * t))
        for n, c in zip(Y_modes, Y_n):
            y += c if n == 0 else c * (np.cos(n * t) if n > 0 else np.sin(abs(n) * t))
        for n, c in zip(Z_modes, Z_n):
            z += c if n == 0 else c * (np.cos(n * t) if n > 0 else np.sin(abs(n) * t))

        pts = np.column_stack([x, y, z])
        pts = pts @ rotmat.T + shift[None, :]
        pts = np.vstack([pts, pts[0]])

        poly = pv.PolyData(pts)
        poly.lines = np.hstack([[len(pts)], np.arange(len(pts))])
        #p.add_mesh(poly.tube(radius=thickness), color="black")

        p.add_mesh(
            poly.tube(radius=thickness),
            color="black",
            ambient=0.9,
            diffuse=0.2,
            specular=0.0,
            smooth_shading=True,
        )

    elif "FourierPlanarCoil" in cls:
        center = g["_center"][()]
        normal = g["_normal"][()]
        r_modes = g["_r_basis"]["_modes"][()][:, 2]
        r_n = g["_r_n"][()]
        rotmat = g["_rotmat"][()].reshape(3, 3)
        shift = g["_shift"][()]

        r = np.zeros_like(t)
        for n, c in zip(r_modes, r_n):
            r += c if n == 0 else c * (np.cos(n * t) if n > 0 else np.sin(abs(n) * t))

        normal = normal / np.linalg.norm(normal)

        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, normal)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])

        e1 = np.cross(normal, trial)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(normal, e1)

        pts = center[None, :] + r[:, None] * (
            np.cos(t)[:, None] * e1[None, :] + np.sin(t)[:, None] * e2[None, :]
        )
        pts = pts @ rotmat.T + shift[None, :]
        pts = np.vstack([pts, pts[0]])

        poly = pv.PolyData(pts)
        poly.lines = np.hstack([[len(pts)], np.arange(len(pts))])
        #p.add_mesh(poly.tube(radius=thickness), color="black")

        p.add_mesh(
            poly.tube(radius=thickness),
            color="black",
            ambient=0.9,
            diffuse=0.2,
            specular=0.0,
            smooth_shading=True,
        )

    elif "_coils" in g:
        keys = sorted([k for k in g["_coils"].keys() if k.isdigit()], key=int)
        for key in keys:
            stack.append(g["_coils"][key])

f.close()

# ---------- add the 3D plasma surface to the SAME plotter ----------

variables_for_3D = ["X1", "X2", "LA", "pos", "mod_B", "B"]

rho = [1.0]
theta = np.linspace(0, 2 * np.pi, 50)
zeta = np.linspace(0, 2 * np.pi, 128)

ev = state.evaluate(*variables_for_3D, rho=rho, theta=theta, zeta=zeta)
visu_data_3D = ev[variables_for_3D]

ev2vtk("pyGVEC_3d_visu", visu_data_3D)

mesh = pv.read("pyGVEC_3d_visu.vts")

p.add_mesh(
    mesh,
    scalars="mod_B",
    cmap="RdBu_r",
    opacity=1.0,
    show_edges=True,
    edge_color="k",
    line_width=0.01,
    ambient=0.35,
    diffuse=0.92,
    specular=0.05,
    smooth_shading=True,
    scalar_bar_args={
        "title": "mod_B",
        "label_font_size": 24,
        "title_font_size": 28,
        "n_labels": 4,
        "fmt": "%.1f",
        "width": 0.72,
        "height": 0.10,
        "position_x": 0.14,
        "position_y": 0.03,
        "color": "black",
    },
)


p.add_axes()
p.export_html(out)
p.show()
