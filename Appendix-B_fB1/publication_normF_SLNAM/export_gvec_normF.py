#!/usr/bin/env python3
"""Export GVEC SLNAM_BEST |F|_normalized (mod_F / gradB2_volavg, the same normalization as DESC |F|_normalized)
EXACTLY on the cylindrical planes phi = 0 and pi/2. GVEC's own zeta planes are tilted against constant phi, so
for every (rho, theta) the GVEC zeta with phi(rho, theta, zeta) = phi0 (mod pi, NFP=2) is found by Newton
iteration (finite-difference dphi/dzeta, to 1e-12 rad); position and mod_F are then evaluated at that point.
No field values are interpolated. Output: normF_GVEC_SLNAM_BEST_phi_planes.npz
Run in gvec-env from this directory:  python export_gvec_normF.py
"""

import datetime
import os
from pathlib import Path

import gvec
import numpy as np
import xarray as xr

GVEC_DIR = Path(__file__).resolve().parents[2] / "GVEC_f67"  # SLNAM_BEST (pre-SLAM) equilibrium
INI = "parameter_SLNAM_216_0p92_nelems32_mn12_34_final.ini"
STATE = "SLNAM_216_0p92_nelems32_mn12_34_State_final.dat"
OUT = os.path.abspath("normF_GVEC_SLNAM_BEST_phi_planes.npz")
NRHO, NTHETA = 100, 257  # same grid as export_desc_normF.py
rho = np.linspace(0.02, 1.0, NRHO)
theta = np.linspace(0.0, 2 * np.pi, NTHETA, endpoint=True)  # closed curves
phi = np.array([0.0, np.pi / 2])
period = np.pi  # 2 pi / NFP: phi0 and phi0 + pi are the same section

os.chdir(GVEC_DIR)  # boundary_filename in the .ini is relative to cwd
state = gvec.load_state(INI, STATE)
gradB2_volavg = float(state.evaluate("gradB2_volavg").gradB2_volavg.values)  # one number, computed once

R = np.empty((NRHO, NTHETA, phi.size))
Z = np.empty_like(R)
F = np.empty_like(R)
zeta_on_plane = np.empty_like(R)
worst = 0.0
for k, phi0 in enumerate(phi):
    for i, r in enumerate(rho):
        zeta = np.full(NTHETA, phi0)  # GVEC's own zeta plane is within ~1 deg of the phi plane
        for it in range(30):
            ev = gvec.Evaluations(rho=np.array([r]), theta=xr.DataArray(theta, dims=("pol",)),
                                  zeta=xr.DataArray(zeta, dims=("pol",)), state=state)
            ev_h = gvec.Evaluations(rho=np.array([r]), theta=xr.DataArray(theta, dims=("pol",)),
                                    zeta=xr.DataArray(zeta + 1e-7, dims=("pol",)), state=state)
            state.compute(ev, "pos")
            state.compute(ev_h, "pos")
            p = ev.pos.transpose("xyz", ...).values.reshape(3, -1)
            ph = ev_h.pos.transpose("xyz", ...).values.reshape(3, -1)
            res = (np.arctan2(p[1], p[0]) - phi0 + period / 2) % period - period / 2
            res_h = (np.arctan2(ph[1], ph[0]) - phi0 + period / 2) % period - period / 2
            if np.abs(res).max() < 1e-12:
                break
            zeta = zeta - res / ((res_h - res) / 1e-7)
        worst = max(worst, np.abs(res).max())
        ev = gvec.Evaluations(rho=np.array([r]), theta=xr.DataArray(theta, dims=("pol",)),
                              zeta=xr.DataArray(zeta, dims=("pol",)), state=state)
        state.compute(ev, "pos", "mod_F")
        p = ev.pos.transpose("xyz", ...).values.reshape(3, -1)
        R[i, :, k], Z[i, :, k] = np.hypot(p[0], p[1]), p[2]
        F[i, :, k] = np.abs(ev.mod_F.values.reshape(-1)) / gradB2_volavg
        zeta_on_plane[i, :, k] = zeta
    print(f"phi={phi0:.4f}: max {F[:, :, k].max():.3e} mean {F[:, :, k].mean():.3e}; GVEC zeta on plane "
          f"{zeta_on_plane[:, :, k].min():+.4f} .. {zeta_on_plane[:, :, k].max():+.4f}", flush=True)
print(f"max |phi - phi0| after Newton: {worst:.2e} rad", flush=True)

info = (f"GVEC {INI} / {STATE}; mod_F / gradB2_volavg (= DESC |F|_normalized) exactly on cylindrical planes "
        f"phi = {phi.tolist()} (Newton for GVEC zeta, |phi - phi0| <= {worst:.1e}); rho {rho[0]}-{rho[-1]} x {NRHO}, "
        f"theta 0-2pi x {NTHETA} (endpoint); written {datetime.date.today()} by export_gvec_normF.py")
np.savez(OUT, R=R, Z=Z, F=F, rho=rho, theta=theta, phi=phi, zeta_gvec=zeta_on_plane, info=info)
print("wrote", OUT, flush=True)
