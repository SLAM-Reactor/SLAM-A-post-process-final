#!/usr/bin/env python3
"""Trace coil-field Poincare data at zeta=0 and zeta=pi/NFP."""

import os
import sys
import warnings

DESC_SOURCE = os.environ.get("DESC_SOURCE", "/pscratch/sd/r/rgaur/DESC3/DESC")
sys.path.insert(0, DESC_SOURCE)

from desc import set_device

set_device("gpu")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from desc.coils import CoilSet
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.magnetic_fields import field_line_integrate
from desc.plotting import plot_surfaces


# ---------------------------------------------------------------------------
# Everything to change is hardcoded here.
# ---------------------------------------------------------------------------
equilibrium_filename = (
    "eq_NFP2_A+9.0_tor+0.0_mr+0.10_shift+0.00_wd+0.00_elong+3.00"
    "_wellwt+2.0_vacuum_final2.h5"
)
coilset_filename = "NFP2_A8p7_coilset_40_coils_Bn_0p015_N9.h5"
output_filename = "QI_poincare_zeta0_zetaPiOverNFP_data.npz"

ntransit = 500
rho_start = np.linspace(0, 1, 9)
rho_surfaces = np.linspace(0, 1, 9)
surface_theta_points = 360


# Load the equilibrium and the optimized coilset.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"desc\.io\.hdf5_io",
    )
    eq = Equilibrium.load(equilibrium_filename)

field = CoilSet.load(coilset_filename)


# Starting locations match the notebook: flux surfaces on the outboard midplane.
trace_grid = LinearGrid(rho=rho_start)
R0 = np.asarray(eq.compute("R", grid=trace_grid)["R"])
Z0 = np.asarray(eq.compute("Z", grid=trace_grid)["Z"])


# Direct field-line integration through the two geometric toroidal cuts.
zeta_planes = np.array([0.0, np.pi / eq.NFP])
zeta_plane_labels = np.array(["zeta=0", "zeta=pi/NFP"])
phis = (
    zeta_planes
    + np.arange(0, ntransit)[:, np.newaxis] * 2 * np.pi / eq.NFP
).ravel()

field_R, field_Z = field_line_integrate(R0, Z0, phis, field)
R_poincare = np.asarray(field_R).reshape((ntransit, zeta_planes.size, -1))
Z_poincare = np.asarray(field_Z).reshape((ntransit, zeta_planes.size, -1))


# Equilibrium flux-surface curves in the same two R-Z cross sections.
surface_fig, surface_ax, surface_data = plot_surfaces(
    eq,
    rho=rho_surfaces,
    theta=0,
    phi=zeta_planes,
    NT=surface_theta_points,
    return_data=True,
)
plt.close(surface_fig)

R_eq = np.asarray(surface_data["rho_R_coords"])
Z_eq = np.asarray(surface_data["rho_Z_coords"])


np.savez(
    output_filename,
    R_poincare=R_poincare,
    Z_poincare=Z_poincare,
    R_eq=R_eq,
    Z_eq=Z_eq,
    R0=R0,
    Z0=Z0,
    zeta_planes=zeta_planes,
    zeta_plane_labels=zeta_plane_labels,
    phis=phis,
    rho_start=rho_start,
    rho_surfaces=rho_surfaces,
    ntransit=ntransit,
    NFP=eq.NFP,
    surface_theta_points=surface_theta_points,
    equilibrium_filename=equilibrium_filename,
    coilset_filename=coilset_filename,
)

print("Equilibrium:", equilibrium_filename)
print("Coilset:", coilset_filename)
print("NFP:", eq.NFP)
print("Transit count:", ntransit)
print("Poincare R/Z shape:", R_poincare.shape)
print("Equilibrium surface R/Z shape:", R_eq.shape)
print("Saved:", output_filename)
