#!/usr/bin/env python3
"""Hardcoded GVEC Boozer geometry exports for GX.

Example:
    python export_gvec_gx_geometry.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
from scipy.constants import mu_0
from scipy.integrate import cumulative_trapezoid

import gvec
from netCDF4 import Dataset

from paths import CASES, DATA_DIR, ensure_dirs


# Edit this block, then run:
#   python export_gvec_gx_geometry.py
EXPORTS = (
    {"case": "SLAM_BEST", "rho": 0.65},
    {"case": "SLNAM_BEST", "rho": 0.65},
)
NTHETA = 641
NPOL = 5.0
MNFACTOR = 1
ALPHA = None if os.environ.get("GX_ALPHA") is None else float(os.environ["GX_ALPHA"])
ALPHA_LABEL = os.environ.get("GX_ALPHA_LABEL", "")
ZETA_CENTER = 0.0


def line(ds, name):
    return np.asarray(ds[name].isel(rad=0, pol=0), dtype=float).reshape(-1)


def reference_scales(case):
    params = {}
    for raw_line in case["ini"].read_text().splitlines():
        clean = raw_line.split("!", 1)[0].strip()
        if "=" in clean:
            lhs, rhs = clean.split("=", 1)
            params[lhs.strip().lower()] = rhs.strip().strip("\"'")

    nc_name = params.get("hmap_ncfile") or params.get("boundary_filename")
    if nc_name is None:
        raise ValueError(f"No hmap/boundary file listed in {case['ini']}")

    with Dataset(case["ini"].parent / nc_name) as nc:
        boundary = nc.groups["boundary"]
        x = np.asarray(boundary.variables["X(::)"][:], dtype=float)
        y = np.asarray(boundary.variables["Y(::)"][:], dtype=float)
        areas = 0.5 * np.abs(
            np.sum(x * np.roll(y, -1, axis=0) - y * np.roll(x, -1, axis=0), axis=0)
        )
        lref = float(np.sqrt(np.mean(areas) / np.pi))

        axis = nc.groups["axis"]
        xyz = np.asarray(axis.variables["xyz(::)"][:], dtype=float)
        dxyz = np.diff(np.column_stack([xyz, xyz[:, :1]]), axis=1)
        rmaj = float(np.sum(np.linalg.norm(dxyz, axis=0)) / (2.0 * np.pi))
    return lref, rmaj, "gframe_boundary"


def to_equal_arc(theta_native, gradpar, arrays):
    arc = cumulative_trapezoid(1.0 / np.abs(gradpar), theta_native, initial=0.0)
    gradpar_gx_value = 2.0 * np.pi / arc[-1]
    theta_equal_arc = gradpar_gx_value * arc - np.pi
    theta_gx = np.linspace(-np.pi, np.pi, theta_native.size)

    out = {}
    for name, values in arrays.items():
        out[name] = np.interp(theta_gx, theta_equal_arc, values)
    out["theta"] = theta_gx
    out["theta_Boozer"] = np.interp(theta_gx, theta_equal_arc, theta_native)
    out["gradpar"] = np.full(theta_native.size, gradpar_gx_value)
    out["scale"] = theta_native[-1] / theta_equal_arc[-1]
    return out, theta_gx, theta_equal_arc


ensure_dirs()

for item in EXPORTS:
    case_label = item["case"]
    rho = float(item["rho"])
    case = CASES[case_label]

    rho_label = f"{rho:.3f}".replace(".", "p")
    npol_label = f"{NPOL:g}".replace(".", "p")
    output = DATA_DIR / (
        f"gx_geometry_gvec_{case_label}_rho{rho_label}_npol{npol_label}_nt{NTHETA}"
        f"{('_' + ALPHA_LABEL) if ALPHA_LABEL else ''}.eik.nc"
    )

    print(f"\n[{case_label}] loading GVEC state")
    os.chdir(case["ini"].parent)  # boundary/hmap file names in the .ini are relative
    state = gvec.load_state(str(case["ini"]), str(case["state"]))
    lref, rmaj, scale_source = reference_scales(case)

    profiles = state.evaluate(
        "iota",
        "diota_dr",
        "shear",
        "Phi_edge",
        "dPhi_dr",
        "dp_dr",
        rho=np.array([rho], dtype=float),
        theta=None,
        zeta=None,
    )

    iota = float(np.asarray(profiles["iota"]).squeeze())
    diota_dr = float(np.asarray(profiles["diota_dr"]).squeeze())
    shat = float(np.asarray(profiles["shear"]).squeeze())
    phi_edge = float(np.asarray(profiles["Phi_edge"]).squeeze())
    dphi_dr = float(np.asarray(profiles["dPhi_dr"]).squeeze())
    dp_dr = float(np.asarray(profiles["dp_dr"]).squeeze())

    bref = 2.0 * abs(phi_edge) / lref**2
    toroidal_flux_sign = np.sign(phi_edge) if phi_edge != 0.0 else 1.0
    alpha = -iota * ZETA_CENTER if ALPHA is None else ALPHA

    theta_native = np.linspace(-NPOL * np.pi, NPOL * np.pi, NTHETA)
    zeta_b = (theta_native - alpha) / iota

    print(f"[{case_label}] evaluating Boozer field line")
    ev = state.evaluate_sfl(
        "mod_B",
        "B",
        "B_contra_t_B",
        "B_rho_B",
        "B_theta_B",
        "B_zeta_B",
        "dmod_B_dr_B",
        "dmod_B_dt_B",
        "dmod_B_dz_B",
        "grad_rho",
        "grad_theta_B",
        "grad_zeta_B",
        "Jac_B",
        "pos",
        rho=np.array([rho], dtype=float),
        theta=theta_native[None, :],
        zeta=zeta_b[None, :],
        sfl="boozer",
        MNfactor=MNFACTOR,
    )

    mod_b = line(ev, "mod_B")
    jac_b = line(ev, "Jac_B")
    b_r = line(ev, "B_rho_B")
    b_t = line(ev, "B_theta_B")
    b_z = line(ev, "B_zeta_B")
    db_r = line(ev, "dmod_B_dr_B")
    db_t = line(ev, "dmod_B_dt_B")
    db_z = line(ev, "dmod_B_dz_B")

    grad_rho = np.asarray(ev["grad_rho"].isel(rad=0, pol=0).transpose("xyz", "tor"), dtype=float)
    grad_theta = np.asarray(ev["grad_theta_B"].isel(rad=0, pol=0).transpose("xyz", "tor"), dtype=float)
    grad_zeta = np.asarray(ev["grad_zeta_B"].isel(rad=0, pol=0).transpose("xyz", "tor"), dtype=float)
    b_vec = np.asarray(ev["B"].isel(rad=0, pol=0).transpose("xyz", "tor"), dtype=float) / mod_b[None, :]
    pos = np.asarray(ev["pos"].isel(rad=0, pol=0).transpose("xyz", "tor"), dtype=float)

    alpha_r = -(zeta_b - ZETA_CENTER) * diota_dr
    alpha_t = np.ones_like(theta_native)
    alpha_z = -iota * np.ones_like(theta_native)
    grad_alpha = alpha_r[None, :] * grad_rho + grad_theta + alpha_z[None, :] * grad_zeta
    grad_psi = dphi_dr * grad_rho

    grad_alpha2 = np.sum(grad_alpha * grad_alpha, axis=0)
    grad_psi_alpha = np.sum(grad_psi * grad_alpha, axis=0)
    grad_psi2 = np.sum(grad_psi * grad_psi, axis=0)

    psi_norm = rho**2
    bmag = mod_b / bref
    grho = np.sqrt(grad_psi2 / (lref**2 * bref**2 * psi_norm))
    gds2 = grad_alpha2 * lref**2 * psi_norm
    gds21 = shat / bref * grad_psi_alpha
    gds22 = (shat / (lref * bref)) ** 2 / psi_norm * grad_psi2
    gradpar = np.abs(lref * line(ev, "B_contra_t_B") / mod_b)

    alpha_cross = (
        b_r * db_t * alpha_z
        + b_t * db_z * alpha_r
        + b_z * db_r * alpha_t
        - b_z * db_t * alpha_r
        - b_t * db_r * alpha_z
        - b_r * db_z * alpha_t
    )
    gbdrift = toroidal_flux_sign * (2.0 * bref * lref**2 * rho / mod_b**3) * alpha_cross / jac_b

    gbdrift0 = (
        toroidal_flux_sign
        * shat
        * 2.0
        / mod_b**3
        / rho
        * (b_t * db_z - b_z * db_t)
        * phi_edge
        / jac_b
        * 2.0
        * rho
    )
    cvdrift0 = gbdrift0.copy()

    b_cross_grad_alpha_radial = (b_z * alpha_t - b_t * alpha_z) / jac_b
    cvdrift = gbdrift + (
        2.0
        * bref
        * lref**2
        / mod_b**2
        * rho
        * mu_0
        / mod_b**2
        * dp_dr
        * b_cross_grad_alpha_radial
    )

    grad_x = shat * grad_psi / (lref * bref * rho)
    grad_y = lref * rho * grad_alpha
    rplot = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
    zplot = pos[2]

    gx, theta_gx, theta_equal_arc = to_equal_arc(
        theta_native,
        gradpar,
        {
            "bmag": bmag,
            "grho": grho,
            "gds2": gds2,
            "gds21": gds21,
            "gds22": gds22,
            "gbdrift": gbdrift,
            "gbdrift0": gbdrift0,
            "cvdrift": cvdrift,
            "cvdrift0": cvdrift0,
            "Rplot": rplot,
            "Zplot": zplot,
        },
    )

    gx["grad_x"] = np.vstack([np.interp(theta_gx, theta_equal_arc, grad_x[i]) for i in range(3)])
    gx["grad_y"] = np.vstack([np.interp(theta_gx, theta_equal_arc, grad_y[i]) for i in range(3)])
    gx["b_vec"] = np.vstack([np.interp(theta_gx, theta_equal_arc, b_vec[i]) for i in range(3)])

    gx["drhodpsi"] = abs(1.0 / dphi_dr)
    gx["kxfac"] = 1.0
    gx["Rmaj"] = rmaj
    gx["q"] = abs(1.0 / iota)
    gx["shat"] = shat
    gx["alpha"] = alpha
    gx["zeta_center"] = ZETA_CENTER
    gx["nfp"] = int(state.nfp)
    gx["jacob"] = 1.0 / np.abs(gx["drhodpsi"] * gx["gradpar"] * gx["bmag"])

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_label": case_label,
        "parameter_file": str(case["ini"]),
        "state_file": str(case["state"]),
        "rho": rho,
        "ntheta": NTHETA,
        "npol": NPOL,
        "mnfactor": MNFACTOR,
        "alpha": alpha,
        "zeta_center": ZETA_CENTER,
        "iota": iota,
        "Phi_edge_over_2pi": phi_edge,
        "Lref": lref,
        "Rmaj": rmaj,
        "Bref": bref,
        "reference_scale_source": scale_source,
    }

    print(f"[{case_label}] writing {output}")
    with Dataset(output, "w") as nc:
        nc.createDimension("z", NTHETA)
        nc.createDimension("3", 3)
        nc.setncattr("source", "GVEC")
        nc.setncattr("metadata", json.dumps(metadata, indent=2))

        for name in (
            "theta",
            "theta_Boozer",
            "theta_PEST",
            "bmag",
            "gradpar",
            "grho",
            "gds2",
            "gds21",
            "gds22",
            "gbdrift",
            "gbdrift0",
            "cvdrift",
            "cvdrift0",
            "jacob",
            "Rplot",
            "Zplot",
        ):
            var = nc.createVariable(name, "f8", ("z",))
            var[:] = gx["theta_Boozer"] if name == "theta_PEST" else gx[name]
            if name == "theta_PEST":
                var.setncattr("note", "Compatibility alias; GVEC exporter uses Boozer coordinates.")

        for name in ("grad_x", "grad_y", "b_vec"):
            var = nc.createVariable(name, "f8", ("3", "z"))
            var[:, :] = gx[name]

        for name in ("drhodpsi", "kxfac", "Rmaj", "q", "shat", "scale", "alpha", "zeta_center"):
            var = nc.createVariable(name, "f8")
            var[...] = gx[name]

        nfp = nc.createVariable("nfp", "i4")
        nfp[...] = gx["nfp"]

    print(
        f"[{case_label}] theta_Boozer=[{gx['theta_Boozer'][0]:+.6f}, "
        f"{gx['theta_Boozer'][-1]:+.6f}], scale={gx['scale']:.3f}, "
        f"iota={iota:+.8f}, shat={shat:+.8f}, "
        f"gradpar={gx['gradpar'][0]:.8e}, "
        f"bmag=[{np.min(gx['bmag']):.6f}, {np.max(gx['bmag']):.6f}]"
    )
