#!/usr/bin/env python3
"""Export the boundary (rho=1) of the GVEC SLNAM_BEST equilibrium
(216 turns, rho=0.92, clean1e-3; nElems 32, mn (12,34)) on constant
cylindrical-phi planes, as a (theta, phi_cyl) tensor grid in the same layout
as PEST_data_*.npz, so DESC fits exactly the comparison boundary.
Run from Appendix-2/desc_slnam_216_0p92 in gvec-env."""

import os
import sys

import numpy as np
import gvec

sys.path.insert(0, os.path.abspath("../../GVEC/post-process"))
from export_gvec_sections import constant_phi_sections  # noqa: E402

gvec_dir = os.path.abspath("../../GVEC")
ini = "parameter_SLNAM_216_0p92_nelems32_mn12_34_final.ini"
state_file = "SLNAM_216_0p92_nelems32_mn12_34_State_final.dat"
out = os.path.abspath("gvec_boundary_SLNAM_216_0p92_clean1em3.npz")
ntheta, nphi, nzeta_dense = 196, 256, 1441

theta = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)
phi = np.linspace(0.0, 2 * np.pi, nphi, endpoint=False)
zeta_dense = np.linspace(0.0, 2 * np.pi, nzeta_dense, endpoint=True)

os.chdir(gvec_dir)  # boundary_filename in the .ini is relative to cwd
state = gvec.load_state(ini, state_file)
R, Z, _ = constant_phi_sections(state, np.array([1.0]), theta, zeta_dense, phi)
R, Z = R[0], Z[0]  # (theta, phi)

np.savez(out, theta_PEST=np.repeat(theta[:, None], nphi, axis=1),
         phi_cyl=np.repeat(phi[None, :], ntheta, axis=0), R_PEST=R, Z_PEST=Z)
print(f"wrote {out}  R {R.min():.4f}-{R.max():.4f}  Z {Z.min():.4f}-{Z.max():.4f}", flush=True)
