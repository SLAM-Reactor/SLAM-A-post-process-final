#!/usr/bin/env python3
"""Export GVEC NFP=3 flux-surface sections at constant physical phi, plus
Poincare seed points, for comparison against the DESC coilset field. Run in
gvec-env.
"""

import json
from datetime import datetime, timezone

import numpy as np
import gvec

ini = "coils_and_gvec_NFP3/parameter_surface_3_1p0_128_1p00_final.ini"
state_file = "coils_and_gvec_NFP3/surface_3_1p0_128_1p00_State_final.dat"
nfp = 3
rho_values = np.array([0.5, 1.0])
n_theta = 360
n_zeta_dense = 721
TWOPI = 2.0 * np.pi
phi_planes = np.array([0.0, np.pi / nfp])


def as_xyz(dataset):
    arr = np.asarray(dataset["pos"])
    dims = list(dataset["pos"].dims)
    return np.moveaxis(arr, dims.index("xyz"), 0)


def r_z_phi_from_pos(pos_xyz):
    x, y, z = pos_xyz
    return np.hypot(x, y), z, np.mod(np.arctan2(y, x), TWOPI)


def interp_periodic_phi(phi, values, target):
    phi_u = np.unwrap(phi)
    vals = np.asarray(values)
    if phi_u[-1] < phi_u[0]:
        phi_u = phi_u[::-1]
        vals = vals[::-1]
    order = np.argsort(phi_u)
    phi_s, vals_s = phi_u[order], vals[order]
    keep = np.concatenate([[True], np.diff(phi_s) > 1e-12])
    phi_s, vals_s = phi_s[keep], vals_s[keep]
    t = float(target)
    while t < phi_s[0]:
        t += TWOPI
    while t > phi_s[-1]:
        t -= TWOPI
    if t < phi_s[0] or t > phi_s[-1]:
        return float(vals_s[np.argmin(np.abs(phi_s - t))])
    return float(np.interp(t, phi_s, vals_s))


print("loading GVEC state", flush=True)
state = gvec.load_state(ini, state_file)

theta = np.linspace(0.0, TWOPI, n_theta, endpoint=False)
zeta_dense = np.linspace(0.0, TWOPI, n_zeta_dense, endpoint=True)

R_out = np.empty((rho_values.size, theta.size, phi_planes.size))
Z_out = np.empty_like(R_out)

for ir, rho in enumerate(rho_values):
    print(f"rho={rho}: evaluating pos", flush=True)
    ds = gvec.evaluate(state, "pos", rho=np.array([rho]), theta=theta, zeta=zeta_dense)
    pos = as_xyz(ds)[:, 0, :, :]
    r_grid, z_grid, phi_grid = r_z_phi_from_pos(pos)
    for it in range(theta.size):
        phi_line = phi_grid[it, :]
        for ip, phi0 in enumerate(phi_planes):
            R_out[ir, it, ip] = interp_periodic_phi(phi_line, r_grid[it, :], phi0)
            Z_out[ir, it, ip] = interp_periodic_phi(phi_line, z_grid[it, :], phi0)

outboard_idx = np.argmax(R_out[:, :, 0], axis=1)
seed_R0 = R_out[np.arange(rho_values.size), outboard_idx, 0]
seed_Z0 = Z_out[np.arange(rho_values.size), outboard_idx, 0]

pad = 0.25
r_min, r_max = float(np.nanmin(R_out)), float(np.nanmax(R_out))
z_min, z_max = float(np.nanmin(Z_out)), float(np.nanmax(Z_out))
bounds_R = np.array([max(0.0, r_min - pad), r_max + pad])
bounds_Z = np.array([z_min - pad, z_max + pad])

np.savez(
    "gvec_sections_nfp3.npz",
    rho=rho_values, theta=theta, phi_planes=phi_planes,
    R_phi=R_out, Z_phi=Z_out,
    seed_R0=seed_R0, seed_Z0=seed_Z0,
    bounds_R=bounds_R, bounds_Z=bounds_Z,
    metadata=json.dumps({"created_utc": datetime.now(timezone.utc).isoformat(), "nfp": nfp,
                          "ini": ini, "state": state_file}),
)
print("wrote gvec_sections_nfp3.npz", flush=True)
