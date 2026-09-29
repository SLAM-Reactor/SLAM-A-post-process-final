#!/usr/bin/env python3
"""Trace the DESC NFP=3 SLAM-A coilset field from GVEC-derived seed points.
Run in desc-env2, after export_gvec_sections_nfp3.py has written
gvec_sections_nfp3.npz.
"""

import json
import sys
from datetime import datetime, timezone

import numpy as np

DESC_SOURCE = "/pscratch/sd/r/rgaur/DESC3/DESC"
sys.path.insert(0, DESC_SOURCE)

import desc

desc.set_device("gpu")
from desc.grid import LinearGrid
from desc.io import load
from desc.magnetic_fields import field_line_integrate

coilset_file = "../GVEC/coils_and_gvec/NFP3/SLNAMified_NFP3_A10p1_L_over_R_1p0_SLAM-A-lowcur.h5"
nfp = 3
ntransit = 300

sections = np.load("gvec_sections_nfp3.npz")
r0 = sections["seed_R0"]
z0 = sections["seed_Z0"]
bounds_R = tuple(float(x) for x in sections["bounds_R"])
bounds_Z = tuple(float(x) for x in sections["bounds_Z"])
phi_planes = sections["phi_planes"]

phis = (phi_planes[None, :] + np.arange(ntransit, dtype=float)[:, None] * 2.0 * np.pi / nfp).ravel()

print(f"loading coilset {coilset_file}", flush=True)
field = load(coilset_file)
if isinstance(field, (list, tuple)) and len(field) == 1:
    field = field[0]

print(f"tracing {r0.size} seeds for {ntransit} transits, bounds_R={bounds_R} bounds_Z={bounds_Z}", flush=True)
field_r, field_z = field_line_integrate(
    r0, z0, phis, field,
    source_grid=LinearGrid(N=32), rtol=1e-8, atol=1e-8,
    bounds_R=bounds_R, bounds_Z=bounds_Z,
    options={"throw": False},
)
r_trace = np.asarray(field_r).reshape((ntransit, phi_planes.size, r0.size))
z_trace = np.asarray(field_z).reshape((ntransit, phi_planes.size, z0.size))

finite = np.isfinite(r_trace) & np.isfinite(z_trace)
print("finite fraction by plane:", np.mean(finite, axis=(0, 2)), flush=True)
print("finite fraction by seed:", np.mean(finite, axis=(0, 1)), flush=True)

np.savez(
    "desc_coil_poincare_nfp3.npz",
    phi_planes=phi_planes, phis=phis, rho=sections["rho"],
    R_poincare=r_trace, Z_poincare=z_trace, seed_R0=r0, seed_Z0=z0,
    metadata=json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(),
                          "coilset": coilset_file, "ntransit": ntransit}),
)
print("wrote desc_coil_poincare_nfp3.npz", flush=True)
