#!/usr/bin/env python3
"""Export GVEC cross-sections and seed points for DESC coil Poincare traces.

Run in gvec-env. This script does not use DESC.

The Poincare trace is a geometric cylindrical-phi diagnostic, while GVEC's
logical zeta is not guaranteed to be identical to cylindrical phi. For that
reason this script exports two versions of each section:

  * constant physical phi, found by interpolating GVEC positions in zeta
  * raw constant GVEC logical zeta, saved only as a convention diagnostic

The DESC trace should normally be compared to the constant-physical-phi data.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import gvec

from paths import (
    CASES,
    DATA_DIR,
    DEFAULT_RHO_SURFACES,
    DEFAULT_ZETA_PLANES,
    NFP,
    TWOPI,
    ensure_dirs,
    object_array,
    parse_floats,
    selected_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", choices=sorted(CASES), default=None)
    parser.add_argument("--rho", default=None, help="comma-separated rho surfaces")
    parser.add_argument("--n-theta", type=int, default=360)
    parser.add_argument("--n-zeta-dense", type=int, default=721)
    parser.add_argument("--bound-padding", type=float, default=0.25)
    parser.add_argument("--output", default=str(DATA_DIR / "gvec_sections_for_poincare.npz"))
    return parser.parse_args()


def as_xyz(dataset) -> np.ndarray:
    arr = np.asarray(dataset["pos"])
    dims = list(dataset["pos"].dims)
    xyz_axis = dims.index("xyz")
    return np.moveaxis(arr, xyz_axis, 0)


def r_z_phi_from_pos(pos_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, z = pos_xyz
    return np.hypot(x, y), z, np.mod(np.arctan2(y, x), TWOPI)


def interp_periodic_phi(phi: np.ndarray, values: np.ndarray, target: float) -> float:
    """Interpolate one closed curve as a function of physical cylindrical phi."""
    phi_unwrapped = np.unwrap(phi)
    vals = np.asarray(values)

    if phi_unwrapped[-1] < phi_unwrapped[0]:
        phi_unwrapped = phi_unwrapped[::-1]
        vals = vals[::-1]

    order = np.argsort(phi_unwrapped)
    phi_sorted = phi_unwrapped[order]
    vals_sorted = vals[order]

    keep = np.concatenate([[True], np.diff(phi_sorted) > 1e-12])
    phi_sorted = phi_sorted[keep]
    vals_sorted = vals_sorted[keep]

    target_unwrapped = float(target)
    while target_unwrapped < phi_sorted[0]:
        target_unwrapped += TWOPI
    while target_unwrapped > phi_sorted[-1]:
        target_unwrapped -= TWOPI

    if target_unwrapped < phi_sorted[0] or target_unwrapped > phi_sorted[-1]:
        return float(vals_sorted[np.argmin(np.abs(phi_sorted - target_unwrapped))])
    return float(np.interp(target_unwrapped, phi_sorted, vals_sorted))


def constant_phi_sections(state, rho_values, theta, zeta_dense, phi_planes):
    r_out = np.empty((rho_values.size, theta.size, phi_planes.size))
    z_out = np.empty_like(r_out)
    logical_zeta_at_phi = np.empty_like(r_out)

    for ir, rho in enumerate(rho_values):
        ds = gvec.evaluate(
            state,
            "pos",
            rho=np.array([rho], dtype=float),
            theta=theta,
            zeta=zeta_dense,
        )
        pos = as_xyz(ds)[:, 0, :, :]
        r_grid, z_grid, phi_grid = r_z_phi_from_pos(pos)

        for it in range(theta.size):
            phi_line = phi_grid[it, :]
            for ip, phi0 in enumerate(phi_planes):
                r_out[ir, it, ip] = interp_periodic_phi(phi_line, r_grid[it, :], phi0)
                z_out[ir, it, ip] = interp_periodic_phi(phi_line, z_grid[it, :], phi0)
                logical_zeta_at_phi[ir, it, ip] = interp_periodic_phi(
                    phi_line, zeta_dense, phi0
                )

    return r_out, z_out, logical_zeta_at_phi


def logical_zeta_sections(state, rho_values, theta, zeta_planes):
    ds = gvec.evaluate(state, "pos", rho=rho_values, theta=theta, zeta=zeta_planes)
    pos = as_xyz(ds)
    r_logical, z_logical, phi_cyl = r_z_phi_from_pos(pos)
    return r_logical, z_logical, phi_cyl


def main() -> None:
    args = parse_args()
    ensure_dirs()

    labels = selected_cases(args.labels)
    rho_values = parse_floats(args.rho, DEFAULT_RHO_SURFACES)
    theta = np.linspace(0.0, TWOPI, args.n_theta, endpoint=False)
    zeta_dense = np.linspace(0.0, TWOPI, args.n_zeta_dense, endpoint=True)
    phi_planes = DEFAULT_ZETA_PLANES.copy()

    r_phi = []
    z_phi = []
    zeta_at_phi = []
    r_logical = []
    z_logical = []
    phi_at_logical = []
    seed_r0 = []
    seed_z0 = []
    seed_theta = []
    bounds_r = []
    bounds_z = []

    for label in labels:
        case = CASES[label]
        print(f"[{label}] loading GVEC state", flush=True)
        # boundary_filename in the .ini is a bare relative name resolved
        # against the current working directory, not the .ini's own
        # directory, so cd there first.
        os.chdir(case["ini"].parent)
        state = gvec.load_state(str(case["ini"]), str(case["state"]))

        print(f"[{label}] constant physical-phi sections", flush=True)
        rp, zp, za = constant_phi_sections(state, rho_values, theta, zeta_dense, phi_planes)
        print(f"[{label}] logical-zeta convention sections", flush=True)
        rl, zl, pl = logical_zeta_sections(state, rho_values, theta, phi_planes)

        # Use the outboard point in the phi=0 plane as the field-line seed for
        # each requested rho surface. This is the least ambiguous seed choice:
        # it is read from the GVEC surface that we later compare against.
        outboard_idx = np.argmax(rp[:, :, 0], axis=1)
        seed_r0.append(rp[np.arange(rho_values.size), outboard_idx, 0])
        seed_z0.append(zp[np.arange(rho_values.size), outboard_idx, 0])
        seed_theta.append(theta[outboard_idx])

        r_min = float(np.nanmin(rp))
        r_max = float(np.nanmax(rp))
        z_min = float(np.nanmin(zp))
        z_max = float(np.nanmax(zp))
        bounds_r.append([max(0.0, r_min - args.bound_padding), r_max + args.bound_padding])
        bounds_z.append([z_min - args.bound_padding, z_max + args.bound_padding])

        r_phi.append(rp)
        z_phi.append(zp)
        zeta_at_phi.append(za)
        r_logical.append(rl)
        z_logical.append(zl)
        phi_at_logical.append(pl)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "export_gvec_sections.py",
        "nfp": NFP,
        "labels": labels,
        "rho": rho_values.tolist(),
        "n_theta": args.n_theta,
        "n_zeta_dense": args.n_zeta_dense,
        "bound_padding_m": args.bound_padding,
        "section_note": "R_phi/Z_phi are constant cylindrical phi; R_logical/Z_logical are constant GVEC zeta.",
    }

    output = args.output
    np.savez(
        output,
        labels=object_array(labels),
        rho=rho_values,
        theta=theta,
        phi_planes=phi_planes,
        zeta_planes=phi_planes,
        R_phi=np.asarray(r_phi),
        Z_phi=np.asarray(z_phi),
        zeta_at_phi=np.asarray(zeta_at_phi),
        R_logical=np.asarray(r_logical),
        Z_logical=np.asarray(z_logical),
        phi_at_logical=np.asarray(phi_at_logical),
        seed_R0=np.asarray(seed_r0),
        seed_Z0=np.asarray(seed_z0),
        seed_theta=np.asarray(seed_theta),
        bounds_R=np.asarray(bounds_r),
        bounds_Z=np.asarray(bounds_z),
        metadata=json.dumps(metadata, indent=2),
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
