#!/usr/bin/env python3
"""SLNAM vacuum fixed-boundary DESC solve on the GVEC SLNAM_BEST boundary (216 turns, rho=0.92,
clean1e-3; rho=1 surface exported by export_gvec_boundary.py). Plain cylindrical fit (zeta=None, no omega),
padded to L20 M10 N20 - identical to the SLAM runs in ../desc_slam_128_0p65 (solve_slam128_cyl*.py)
except for the boundary file and Psi. Fit order from the command line:
    python solve_slnam216_cyl.py 7 14      (desc-env, from this directory)
Force error = max and mean of |F|_normalized over one field period."""

import json
import sys
import time

import matplotlib

matplotlib.use("Agg")

from desc import set_device

set_device("gpu")

import numpy as np
from matplotlib import pyplot as plt

from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.grid import Grid, LinearGrid
from desc.objectives import get_fixed_boundary_constraints
from desc.plotting import plot_section, plot_surfaces
from desc.profiles import PowerSeriesProfile

M_fit, N_fit = int(sys.argv[1]), int(sys.argv[2])
tag = f"slnam216_0p92_cyl_f{M_fit}n{N_fit}_L20M10N20"
datafile = "gvec_boundary_SLNAM_216_0p92_clean1em3.npz"
psi = 0.038785094488762877 * 0.92 ** 2  # reference eq Psi x rho^2
nfp = 2
L, M, N = 20, 10, 20
maxiter = 500

start = time.time()
m = dict(tag=tag, datafile=datafile, psi=psi, M_fit=M_fit, N_fit=N_fit, L=L, M=M, N=N,
         maxiter=maxiter)

d = np.load(datafile)
theta = d["theta_PEST"].ravel()
phi = d["phi_cyl"].ravel()
coords = np.stack([d["R_PEST"].ravel(), phi, d["Z_PEST"].ravel()], axis=1)

surf = FourierRZToroidalSurface.from_values(
    coords, theta, zeta=None, M=M_fit, N=N_fit, NFP=nfp, sym=True,
    check_orientation=True, w=np.ones(coords.shape[0]),
)

# fit error: per phi plane, distance from each data point to the fitted curve
phi_planes = d["phi_cyl"][0]
th_dense = np.linspace(0, 2 * np.pi, 1024, endpoint=False)
nodes = np.stack([np.ones(th_dense.size * phi_planes.size),
                  np.tile(th_dense, phi_planes.size),
                  np.repeat(phi_planes, th_dense.size)], axis=1)
fit = surf.compute(["R", "Z"], grid=Grid(nodes, sort=False))
R_fit = np.asarray(fit["R"]).reshape(phi_planes.size, th_dense.size)
Z_fit = np.asarray(fit["Z"]).reshape(phi_planes.size, th_dense.size)
err = np.concatenate([
    np.hypot(d["R_PEST"][:, j, None] - R_fit[j][None, :],
             d["Z_PEST"][:, j, None] - Z_fit[j][None, :]).min(axis=1)
    for j in range(phi_planes.size)])
m.update(boundary_fit_rms_m=float(np.sqrt(np.mean(err ** 2))), boundary_fit_max_m=float(err.max()))
print("boundary fit rms %.3e max %.3e m" % (m["boundary_fit_rms_m"], m["boundary_fit_max_m"]), flush=True)

surf.change_resolution(M=M, N=N)

eq = Equilibrium(
    Psi=psi, NFP=nfp, L=L, M=M, N=N,
    pressure=PowerSeriesProfile([0.0]), current=PowerSeriesProfile([0.0]),
    surface=surf, sym=True, check_orientation=True,
)
m.update(dim_x=int(eq.dim_x), nested_initial=bool(eq.is_nested()))
constraints = get_fixed_boundary_constraints(eq=eq)
eq, result = eq.solve(objective="force", ftol=1e-8, xtol=1e-8, gtol=1e-8, maxiter=maxiter,
                      verbose=3, constraints=constraints)
m.update(success=bool(result.success), message=str(result.message), nfev=int(result.nfev),
         nested_final=bool(eq.is_nested()))
eq.save("eq_%s.h5" % tag)

g_vol = LinearGrid(rho=np.linspace(0.05, 1.0, 41), theta=91, zeta=32, NFP=nfp, sym=False)
F = np.asarray(eq.compute("|F|_normalized", grid=g_vol)["|F|_normalized"])
m.update(F_max=float(np.nanmax(F)), F_mean=float(np.nanmean(F)))

fig, ax = plot_section(eq, "|F|_normalized", log=True)
fig.savefig("normF_%s.png" % tag, dpi=150, bbox_inches="tight")
plt.close(fig)
fig, ax = plot_surfaces(eq)
fig.savefig("surfaces_%s.png" % tag, dpi=150, bbox_inches="tight")
plt.close(fig)

rho = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
g = LinearGrid(rho=rho, M=2 * M, N=2 * N, NFP=nfp, sym=True)
iota = np.asarray(eq.compute("iota", grid=g)["iota"])
m["iota"] = [float(iota[g.inverse_rho_idx == k].mean()) for k in range(len(rho))]
m["wall_time_s"] = time.time() - start
json.dump(m, open("solve_%s.json" % tag, "w"), indent=2)
print(json.dumps(m, indent=2))
