#!/usr/bin/env python3
"""Export DESC |F|_normalized of the SLNAM cylindrical solve (fit 9,18, L20 M10 N20) on the cylindrical planes
phi = 0 and pi/2 (DESC cylindrical: zeta = phi). Output: normF_DESC_SLNAM_cyl_f9n18_phi_planes.npz
Run in desc-env from this directory:  python export_desc_normF.py
"""

from desc import set_device

set_device("cpu")

import datetime

import numpy as np

from desc.grid import Grid
from desc.io import load

EQ = "desc_slnam_216_0p92/eq_slnam216_0p92_cyl_f9n18_L20M10N20.h5"
OUT = "normF_DESC_SLNAM_cyl_f9n18_phi_planes.npz"
NRHO, NTHETA = 100, 257  # same grid as export_gvec_normF.py
rho = np.linspace(0.02, 1.0, NRHO)
theta = np.linspace(0.0, 2 * np.pi, NTHETA, endpoint=True)  # closed curves
phi = np.array([0.0, np.pi / 2])

eq = load(EQ)
# <|grad(|B|^2)|/2mu0>_vol is a volume average that DESC takes on its own grid, so it is the same for every chunk
R = np.empty((NRHO, NTHETA, phi.size))
Z = np.empty_like(R)
F = np.empty_like(R)
CHUNK = 10  # rho values per compute call (all 100 at once runs out of login-node memory)
for i0 in range(0, NRHO, CHUNK):
    rr, tt, pp = np.meshgrid(rho[i0:i0 + CHUNK], theta, phi, indexing="ij")
    grid = Grid(np.stack([rr.ravel(), tt.ravel(), pp.ravel()], axis=1), sort=False)
    d = eq.compute(["R", "Z", "|F|_normalized"], grid=grid)
    shape = rr.shape
    R[i0:i0 + CHUNK] = np.asarray(d["R"]).reshape(shape)
    Z[i0:i0 + CHUNK] = np.asarray(d["Z"]).reshape(shape)
    F[i0:i0 + CHUNK] = np.asarray(d["|F|_normalized"]).reshape(shape)
    print(f"rho {rho[i0]:.3f}-{rho[min(i0 + CHUNK, NRHO) - 1]:.3f} done", flush=True)

info = (f"DESC {EQ}; |F|_normalized = |F| / <|grad(|B|^2)|/2mu0>_vol on cylindrical planes phi = {phi.tolist()} "
        f"(zeta = phi); rho {rho[0]}-{rho[-1]} x {NRHO}, theta 0-2pi x {NTHETA} (endpoint); "
        f"written {datetime.date.today()} by export_desc_normF.py")
np.savez(OUT, R=R, Z=Z, F=F, rho=rho, theta=theta, phi=phi, info=info)
for k in range(phi.size):
    print(f"phi={phi[k]:.4f}: max {F[:, :, k].max():.3e} mean {F[:, :, k].mean():.3e}")
print("wrote", OUT)
