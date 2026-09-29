#!/usr/bin/env python
"""Trace the SLNAM and SLAM magnetic axes and save |B| on axis.

Run from QFM_surf:
    python trace_axis_B.py
"""

import json

from desc import set_device

set_device("cpu")

import numpy as np  # noqa: E402
from desc.io import load  # noqa: E402
from desc.magnetic_fields import field_line_integrate  # noqa: E402
from scipy.interpolate import InterpolatedUnivariateSpline  # noqa: E402


CASES = [
    ("SLNAM", "SLNAM-field-09-09-26.h5"),
    ("SLAM", "SLAM-field-09-09-26_2.h5"),
]

R0_GUESS = 2.08
DR_SCAN = 0.1
N_SCAN = 100
NPOINTS_PER_TURN = 513
NTURNS = 1
SOURCE_GRID = 50
BOUNDS_R = (0.5, 5.0)
BOUNDS_Z = (-1.2, 1.2)


def axis_and_B(field):
    phi_scan = np.linspace(0, 2 * np.pi, 2)
    x = np.linspace(R0_GUESS - DR_SCAN, R0_GUESS + DR_SCAN, N_SCAN)
    r, z = field_line_integrate(
        [x], [np.zeros_like(x)], phis=phi_scan, field=field,
        bounds_R=BOUNDS_R, bounds_Z=BOUNDS_Z,
    )
    roots = InterpolatedUnivariateSpline(x, z[0] - z[1], k=3).roots()
    if len(roots) == 0:
        raise RuntimeError("No magnetic-axis crossing found. Adjust R0_GUESS or DR_SCAN.")
    r_start = float(roots[np.argmin(np.abs(roots - R0_GUESS))])

    phi_axis = np.linspace(0, 2 * np.pi * NTURNS, NTURNS * NPOINTS_PER_TURN, endpoint=False)
    r_axis, z_axis = field_line_integrate(
        [r_start], [0.0], phis=phi_axis, field=field,
        bounds_R=BOUNDS_R, bounds_Z=BOUNDS_Z,
    )
    r_axis = np.asarray(r_axis).flatten()
    z_axis = np.asarray(z_axis).flatten()
    coords = np.vstack((r_axis * np.cos(phi_axis), r_axis * np.sin(phi_axis), z_axis)).T
    bvec = field.compute_magnetic_field(coords=coords, basis="xyz", source_grid=SOURCE_GRID)
    b_axis = np.linalg.norm(np.asarray(bvec), axis=-1)
    return phi_axis, r_axis, z_axis, b_axis, r_start


npz = {}
metrics = {}
for label, field_file in CASES:
    print(label, field_file)
    phi_axis, r_axis, z_axis, b_axis, r_start = axis_and_B(load(field_file))
    prefix = label.lower()
    npz[f"{prefix}_phi_axis"] = phi_axis
    npz[f"{prefix}_r_axis"] = r_axis
    npz[f"{prefix}_z_axis"] = z_axis
    npz[f"{prefix}_B_axis"] = b_axis
    metrics[label] = {
        "field": field_file,
        "R0_guess": R0_GUESS,
        "R_axis_start": r_start,
        "R_shift": r_start - R0_GUESS,
        "B_axis_avg": float(np.mean(b_axis)),
        "B_axis_min": float(np.min(b_axis)),
        "B_axis_max": float(np.max(b_axis)),
        "B_axis_ripple": float((np.max(b_axis) - np.min(b_axis)) / np.mean(b_axis)),
    }
    print("  R_start =", f"{r_start:.8f}", "  <|B|> =", f"{metrics[label]['B_axis_avg']:.6e}")

np.savez_compressed("axis_B_data.npz", **npz)
with open("axis_B_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print("wrote axis_B_data.npz")
print("wrote axis_B_metrics.json")
