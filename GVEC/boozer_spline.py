#!/usr/bin/env python3
"""FTGen: trace field lines from a coilset, build the alpha / PEST / Boozer maps.

usage:  python3 boozer_spline.py <fname> <nturns> <rho> [gpuid]

Everything is resolved relative to THIS FILE, never to the working directory,
so the FTGen tree can be copied anywhere and run from anywhere.  Set
FTGEN_DEVICE=cpu to run without a GPU.
"""
import os
import sys

# Resolve the tree from the script's own location, not the cwd.
HERE = os.path.dirname(os.path.abspath(__file__))

_device = os.environ.get("FTGEN_DEVICE", "gpu").lower()
gpuid = int(eval(sys.argv[4])) if len(sys.argv) > 4 else 0
from desc import set_device

if _device == "cpu":
    set_device("cpu")
else:
    set_device("gpu", gpuid=gpuid)

import pdb
import pickle

sys.path.insert(0, HERE)

import jax
# Cache inside the tree, so nothing is written above FTGen/.
_cache = os.path.join(HERE, "jax-caches")
os.makedirs(_cache, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", _cache)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import numpy as np
from desc.coils import (
    CoilSet,
)
from desc.backend import jnp
from desc.grid import LinearGrid
from desc.equilibrium import Equilibrium
from desc.magnetic_fields import field_line_integrate

from scipy.optimize import root_scalar, root
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import RectBivariateSpline
from desc.optimize import Optimizer
from desc.integrals import compute_B_plasma
from scipy import integrate

import time
import plotly.express as px
import plotly.io as pio
from desc.io import load
from desc.plotting import *
from matplotlib import pyplot as plt



######################################################
####----Layout: scripts here, data in data/-------####
######################################################
# Inputs live in coils/.  In this GVEC staging copy, every product for a field
# label goes directly into data/SLNAM or data/SLAM; no nested output folders.
COILS_DIR  = os.environ.get("FTGEN_COILS_DIR",  os.path.join(HERE, "coils"))
DATA_DIR   = os.environ.get("FTGEN_DATA_DIR",   os.path.join(HERE, "data"))
TRACE_DIR  = DATA_DIR
ALPHA_DIR  = DATA_DIR
PEST_DIR   = DATA_DIR
BOOZER_DIR = DATA_DIR
FIG_DIR    = DATA_DIR
for _d in (TRACE_DIR, ALPHA_DIR, PEST_DIR, BOOZER_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

def _fig(name):
    return os.path.join(FIG_DIR, name)


fname = sys.argv[1]
val = float(f"{fname}".replace("p", "."))

EQ_FILE = os.environ.get("FTGEN_EQ",
    "eq_NFP2_A+9.0_tor+0.0_mr+0.10_shift+0.00_wd+0.00_elong+3.00_wellwt+2.0_vacuum_final2.h5")
COIL_TEMPLATE = os.environ.get("FTGEN_COIL_TEMPLATE",
    "SLNAMified_NFP2_A8p7_coilset_L_over_R_{fname}_SLAM-A-TQ.h5")
eq = Equilibrium.load(os.path.join(COILS_DIR, EQ_FILE))
coilset = CoilSet.load(os.path.join(COILS_DIR, COIL_TEMPLATE.format(fname=fname)))
print("coilset:", os.path.join(COILS_DIR, COIL_TEMPLATE.format(fname=fname)))
print("data   :", DATA_DIR)

nturns_bdry = eval(sys.argv[2])
rho_bdry = eval(sys.argv[3])


######################################
####----Find the magnetic axis----####
######################################
nturns = 1
npoints_per_turn = 1
phi_full = np.linspace(0, 2 * np.pi * nturns, nturns * npoints_per_turn+1)
grid_trace = LinearGrid(rho=np.linspace(0, 0.00, 1))

# Better than root finding because coupling a scipy root finding routine with the field line integrator
# is a bit unstable. Here we just find the minimum of r-r_axis, which should be zero at the magnetic axis.
# We can then use this as an initial guess for a root finding routine if we want to refine it further.
R0 = eq.compute("R", grid=grid_trace)["R"][0]
x = np.linspace(val - 0.1 + R0, val + 0.1 + R0, 100)
#pdb.set_trace()
r0 = [x]
z0 = [np.zeros_like(x)]
r, z = field_line_integrate(
    r0, z0, phis=phi_full, field=coilset, bounds_R=(0.5, 5.0), bounds_Z=(-1.0, 1.0)
)
spline = InterpolatedUnivariateSpline(x, z[0]-z[1], k=3)
R_shift = spline.roots() - R0
print(f"R_shift: {R_shift}")


nturns = 1
npoints_per_turn = 513
#phi_full = np.linspace(0, 2 * np.pi * nturns, nturns * npoints_per_turn)
phi_full = np.linspace(0, 2 * np.pi * nturns, nturns * npoints_per_turn, endpoint=False)
grid_trace = LinearGrid(rho=np.linspace(0, 0.00, 1))
r0 = [R_shift.item() + R0]
z0 = [0.]
r_axis, z_axis = field_line_integrate(
    r0, z0, phis=phi_full, field=coilset, bounds_R=(0.5, 5.0), bounds_Z=(-1.0, 1.0)
)

r_axis = r_axis.flatten()
z_axis = z_axis.flatten()

#####################################################
####-----Trace fieldline close to the boundary---####
#####################################################

nturns = nturns_bdry
phi_full = np.linspace(0, 2 * np.pi * nturns, nturns * npoints_per_turn+1)
rho0 = rho_bdry
rho_suffix = f"{rho0:.2f}".replace(".", "p")

grid_trace = LinearGrid(rho=rho0)
r0 = [(R_shift + eq.compute("R", grid=grid_trace)["R"][0]).item()]
z0 = [0.]

if os.path.exists(os.path.join(TRACE_DIR, f"boundary_RZ_data_{fname}_{nturns_bdry}_{rho_suffix}.npz")):
	data = np.load(os.path.join(TRACE_DIR, f"boundary_RZ_data_{fname}_{nturns_bdry}_{rho_suffix}.npz"))
	r = data["r"]
	z = data["z"]
	phi_full = data["phi"]
else:
    #pdb.set_trace()
    r, z = field_line_integrate(
        r0, z0, phis=phi_full, field=coilset, bounds_R=(0.5, R0.item() + R_shift.item() + 0.2), bounds_Z=(-0.7, 0.7)
    )
    r = r.flatten()
    z = z.flatten()
    phi_full = phi_full.flatten()

    np.savez(os.path.join(TRACE_DIR, f"boundary_RZ_data_{fname}_{nturns_bdry}_{rho_suffix}.npz"), r=r, z=z, phi=phi_full)


#############################################################
###-------Calculating various transformed coordinates-----###
#############################################################

z_axis_tiled = np.tile(z_axis.flatten(), nturns)
r_axis_tiled = np.tile(r_axis.flatten(), nturns)
arctan_theta = np.arctan2((z[:-1]-z_axis_tiled), (r[:-1]-r_axis_tiled))
iota = (np.unwrap(arctan_theta)[-1] - np.unwrap(arctan_theta)[0])/(phi_full[-1]-phi_full[0])
#iota = -0.6391740423353864 # When the phi grids mismatch
#iota = -0.6391913731458351
#iota = -0.6392013731458351
#iota = -0.6392113731458351
#iota = -0.6392513731458351
#iota = -0.6392813731458351
theta_PEST = iota * phi_full

#pdb.set_trace()
print(f"iota={iota}", arctan_theta)

coords = np.vstack((r*np.cos(phi_full), r*np.sin(phi_full), z)).T

Bvec = coilset.compute_magnetic_field(coords=coords, basis="xyz", source_grid=50)
grad_phi = np.vstack([-np.sin(phi_full)/r.flatten(), np.cos(phi_full)/r.flatten(), np.zeros_like(phi_full)]).T
B_dot_grad_phi = np.einsum('ij,ij->i', Bvec, grad_phi)
B2 = np.einsum('ij,ij->i', Bvec, Bvec)
B = np.sqrt(B2)


def func(x):
    nu_boozer = integrate.cumulative_simpson((B2/B_dot_grad_phi)*1/x-1, x=phi_full, initial=0)
    return np.max(nu_boozer) + np.min(nu_boozer)

G0 = eq.compute("G")["G"][-1]
G = root_scalar(func, bracket=[-0.2, -30], method='bisect', xtol=1e-10, rtol=1e-10).root

#G = -13.9586389013566

#G = G0 
nu_boozer = integrate.cumulative_simpson((B2/B_dot_grad_phi)*1/G-1, x=phi_full, initial=0)
#phi_boozer = integrate.simpson(B2[:npoints_per_turn]/B_dot_grad_phi[:npoints_per_turn]/G, phi_full[:npoints_per_turn])
phi_boozer = phi_full + nu_boozer
theta_boozer = iota * phi_boozer

phi = phi_full[:npoints_per_turn]

alpha_list = []
phi_list = []
R_list = []
X_list = []
Y_list = []
Z_list = []
R_list = []
nu_list = []
B_list = []
thetaPEST_list = []

for i in range(nturns-1):
    alpha_list.append(np.mod(2 * np.pi * iota * i, 2 * np.pi))
    R_list.append(r[i*npoints_per_turn:(i+1)*npoints_per_turn])
    X_list.append(r[i*npoints_per_turn:(i+1)*npoints_per_turn] * np.cos(phi)) 
    Y_list.append(r[i*npoints_per_turn:(i+1)*npoints_per_turn] * np.sin(phi))
    Z_list.append(z[i*npoints_per_turn:(i+1)*npoints_per_turn])
    nu_list.append(phi_boozer[i*npoints_per_turn:(i+1)*npoints_per_turn] - phi_full[i*npoints_per_turn:(i+1)*npoints_per_turn])
    #B2_list.append(B2[i*npoints_per_turn:(i+1)*npoints_per_turn])
    B_list.append(B[i*npoints_per_turn:(i+1)*npoints_per_turn])
    thetaPEST_list.append(theta_PEST[i*npoints_per_turn:(i+1)*npoints_per_turn])


alpha = np.array(alpha_list)
sort_idx = np.argsort(alpha)
alpha_sort = alpha[sort_idx]

R_array = np.array(R_list)[sort_idx]
X_array = np.array(X_list)[sort_idx]  
Y_array = np.array(Y_list)[sort_idx]
Z_array = np.array(Z_list)[sort_idx]
nu_array = np.array(nu_list)[sort_idx]
B_array = np.array(B_list)[sort_idx]
thetaPEST_array = np.array(thetaPEST_list)[sort_idx]

print(alpha_sort[0], alpha_sort[-1], R_array.shape)

# Save the NATIVE alpha-grid arrays, before any spline is fitted.  These are the
# raw per-field-line traces indexed (alpha, phi); X(alpha, phi) is genuinely
# discontinuous across phi = 0 / 2*pi at fixed alpha, and any regridding or
# wrapping applied later destroys that.  Plotting scripts should read this file
# rather than trying to rebuild the alpha grid from boundary_RZ_data_*.npz.
np.savez(os.path.join(ALPHA_DIR, f"alpha_data_{fname}_{nturns_bdry}_{rho_suffix}.npz"),
         alpha=alpha_sort,
         phi=phi,
         X_alpha=X_array,
         Y_alpha=Y_array,
         R_alpha=R_array,
         Z_alpha=Z_array,
         modB_alpha=B_array,
         nu_alpha=nu_array,
         theta_PEST_alpha=thetaPEST_array,
         iota=iota,
         G=G)

# Close the chart periodically in BOTH directions before fitting any spline.
#
# alpha_sort stops short of 2*pi (the traced alphas are mod(2*pi*iota*i, 2*pi),
# which never lands exactly on the seam) and phi stops one step short by
# construction.  RectBivariateSpline is not periodic and does not clamp: any
# query beyond the last knot is a cubic EXTRAPOLATION.  The PEST grid built
# below queries alpha over the full [0, 2*pi), so ~0.6% of its points land in
# that gap, and those extrapolated values are then fitted by the next spline
# level -- so the error is not confined to the seam.
#
# Measured effect of restoring this, alpha -> PEST step, case 1p0_216_0p85:
#     NFP=2 violation   R 9.5% -> 1.7%   Z 10.4% -> 1.6%   |B| 12.4% -> 2.0%
# and inside the alpha gap the closed and unclosed splines differ by up to
# 0.121 m in X.  alpha = 2*pi is the same field line as alpha = 0, and
# phi = 2*pi is the same point as phi = 0, so both closures are exact, not
# interpolated.
if alpha_sort[-1] != 2*np.pi:
    alpha_sort = np.append(alpha_sort, 2*np.pi)
    R_array = np.concatenate((R_array, R_array[0:1]))
    X_array = np.concatenate((X_array, X_array[0:1]))
    Y_array = np.concatenate((Y_array, Y_array[0:1]))
    Z_array = np.concatenate((Z_array, Z_array[0:1]))
    nu_array = np.concatenate((nu_array, nu_array[0:1]))
    B_array = np.concatenate((B_array, B_array[0:1]))
    thetaPEST_array = np.concatenate((thetaPEST_array, thetaPEST_array[0:1]))

# phi is deliberately NOT closed.  At fixed alpha the surface point is
# theta = alpha + iota*phi, so (alpha, phi + 2*pi) sits at theta + 2*pi*iota --
# a DIFFERENT poloidal position, not the same point.  X(alpha, phi) is genuinely
# discontinuous across the phi seam, and wrapping phi would erase exactly the
# structure the alpha panel exists to show.  Only alpha is periodic here.

print("closed chart in alpha: %d pts to %.6f (2pi = %.6f); phi left open, %d pts to %.6f"
      % (len(alpha_sort), alpha_sort[-1], 2*np.pi, len(phi), phi[-1]))

alpha_2D, phi_2D = np.meshgrid(alpha_sort, phi)

R_spl  = RectBivariateSpline(alpha_sort, phi, R_array,bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
X_spl  = RectBivariateSpline(alpha_sort, phi, X_array,bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
Y_spl  = RectBivariateSpline(alpha_sort, phi, Y_array,bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
Z_spl  = RectBivariateSpline(alpha_sort, phi, Z_array,bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
B_spl  = RectBivariateSpline(alpha_sort, phi, B_array,bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
nu_spl = RectBivariateSpline(alpha_sort, phi, nu_array,bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
phi_booz_spl = RectBivariateSpline(alpha_sort, phi, np.mod(phi_2D.T + nu_array, 2*np.pi) ,bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3) 

#plt.figure()
##plt.contourf(alpha_sort, phi, nu_array.T)
#plt.contourf(alpha_2D, phi_2D, nu_array.T)
#plt.title(r"$\nu$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig(_fig(f"test0_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)
#
#plt.figure()
#plt.contourf(alpha_sort, phi, X_array.T)
#plt.title(r"$X$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig(_fig(f"test1_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)
#
#plt.figure()
#plt.contourf(alpha_sort, phi, Y_array.T)
#plt.title(r"$Y$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig(_fig(f"test2_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)
#
#plt.figure()
#plt.contourf(alpha_sort, phi, Z_array.T)
#plt.title(r"$Z$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig(_fig(f"test3_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)
#
#plt.figure()
#plt.contour(alpha_sort, phi, B_array.T, levels=200, linewidths=0.05)
#plt.title(r"$|B|$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig(_fig(f"test4_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)
#
#plt.figure()
#plt.contourf(alpha_sort, phi, R_array.T)
#plt.title(r"$R$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig(_fig(f"test5_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)
#
#
#plt.figure()
#plt.contourf(alpha_sort, phi, np.mod((phi_2D.T+nu_array).T, 2*np.pi), levels=100)
#plt.title(r"$\phi_{boozer}$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig(_fig(f"test6_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)
#
#
#plt.figure()
#plt.contourf(alpha_sort, phi, np.mod(((alpha_2D + iota * phi_2D).T+ iota*nu_array).T, 2*np.pi), levels=100)
#plt.title(r"$\theta_{boozer}$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig(_fig(f"test7_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)

#########################################################################
###########--------Convert to theta_PEST coordinates--------------#######
#########################################################################

ntheta_PEST = 196
nphi_cyl = 256
theta_PEST = np.linspace(0, 2*np.pi, ntheta_PEST, endpoint=False)
phi_cyl = np.linspace(0, 2*np.pi, nphi_cyl, endpoint=False)
theta_PEST_2D, phi_cyl_2D = np.meshgrid(theta_PEST, phi_cyl, indexing="ij")
alpha1_2D = np.mod(theta_PEST_2D - iota * phi_cyl_2D, 2*np.pi)

nu_interp = nu_spl.ev(alpha1_2D, phi_cyl_2D)
phi_booz_interp = phi_booz_spl.ev(alpha1_2D, phi_cyl_2D)

X_spl1 = RectBivariateSpline(theta_PEST, phi_cyl, X_spl.ev(alpha1_2D, phi_cyl_2D) ,bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
Y_spl1 = RectBivariateSpline(theta_PEST, phi_cyl, Y_spl.ev(alpha1_2D, phi_cyl_2D) ,bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
R_spl1 = RectBivariateSpline(theta_PEST, phi_cyl, np.sqrt(X_spl.ev(alpha1_2D, phi_cyl_2D)**2 + Y_spl.ev(alpha1_2D, phi_cyl_2D)**2) ,bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
Z_spl1 = RectBivariateSpline(theta_PEST, phi_cyl, Z_spl.ev(alpha1_2D, phi_cyl_2D) ,bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
phi_spl1 = RectBivariateSpline(theta_PEST, phi_cyl, phi_cyl_2D ,bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
B_spl1 = RectBivariateSpline(theta_PEST, phi_cyl, B_spl.ev(alpha1_2D, phi_cyl_2D) ,bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
nu_spl1 = RectBivariateSpline(theta_PEST, phi_cyl, nu_spl.ev(alpha1_2D, phi_cyl_2D) ,bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)

plt.figure()
plt.contourf(theta_PEST_2D, phi_cyl_2D, X_spl1.ev(theta_PEST_2D, phi_cyl_2D))
plt.title(r"$X$", fontsize=18)
plt.xlabel(r"$\theta_{PEST}$", fontsize=18)
plt.ylabel(r"$\phi$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(_fig(f"test10_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)

plt.figure()
plt.contourf(theta_PEST_2D, phi_cyl_2D, Y_spl1.ev(theta_PEST_2D, phi_cyl_2D))
plt.title(r"$Y$", fontsize=18)
plt.xlabel(r"$\theta_{PEST}$", fontsize=18)
plt.ylabel(r"$\phi$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(_fig(f"test11_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)

plt.figure()
plt.contourf(theta_PEST_2D, phi_cyl_2D, Z_spl1.ev(theta_PEST_2D, phi_cyl_2D))
plt.title(r"$Z$", fontsize=18)
plt.xlabel(r"$\theta_{PEST}$", fontsize=18)
plt.ylabel(r"$\phi$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(_fig(f"test12_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)


plt.figure()
plt.contourf(theta_PEST_2D, phi_cyl_2D, B_spl1.ev(theta_PEST_2D, phi_cyl_2D))
plt.title(r"$B$", fontsize=18)
plt.xlabel(r"$\theta_{PEST}$", fontsize=18)
plt.ylabel(r"$\phi$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(_fig(f"test13_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)

plt.figure()
plt.contourf(theta_PEST_2D, phi_cyl_2D, nu_spl1.ev(theta_PEST_2D, phi_cyl_2D))
plt.title(r"$\nu$", fontsize=18)
plt.xlabel(r"$\theta_{PEST}$", fontsize=18)
plt.ylabel(r"$\phi$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(_fig(f"test14_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)

plt.figure()
print(theta_PEST_2D.shape, nu_interp.shape)
plt.contourf(theta_PEST_2D, phi_cyl_2D, np.mod((theta_PEST_2D+ iota*nu_interp), 2*np.pi), levels=100)
plt.title(r"$\theta_{boozer}$", fontsize=18)
plt.xlabel(r"$\theta_{PEST}$", fontsize=18)
plt.ylabel(r"$\phi$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(_fig(f"test70_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)

np.savez(os.path.join(PEST_DIR, f"PEST_data_{fname}_{nturns_bdry}_{rho_suffix}.npz"), theta_PEST=theta_PEST_2D, phi_cyl=phi_cyl_2D, Z_PEST=Z_spl1.ev(theta_PEST_2D, phi_cyl_2D), R_PEST=R_spl1.ev(theta_PEST_2D, phi_cyl_2D), modB_PEST=B_spl1.ev(theta_PEST_2D, phi_cyl_2D), nu_PEST=nu_spl1.ev(theta_PEST_2D, phi_cyl_2D))


ntheta_booz = 128
nphi_booz = 196

# Create a meshgrid in theta_B, phi_B
theta_booz = np.linspace(0, 2*np.pi, ntheta_booz, endpoint=False)
phi_booz   = np.linspace(0, 2*np.pi, nphi_booz,   endpoint=False)
theta_booz_2D, phi_booz_2D = np.meshgrid(theta_booz, phi_booz, indexing="ij")

theta_PEST_2D = np.zeros_like(theta_booz_2D)
phi_2D = np.zeros_like(phi_booz_2D)

def fun(inp, theta_b, phi_b):
	#return (phi_b - nu_spl1.ev(x, y))**2 + (theta_b -  iota * nu_spl.ev(x, y))**2
	x = inp[0]
	y = inp[1]
	return (phi_b - (y + nu_spl1.ev(x, y)),  theta_b -  (x + iota * nu_spl1.ev(x, y)))

def fprime(inp, theta_b, phi_b):
    x = inp[0]
    y = inp[1]
	#return -2*((phi_b - nu_spl1.ev(x, y))*nu_spl1.ev(x, y, dy=1) + 2*(theta_b -  iota * nu_spl.ev(x, y))* iota * nu_spl.ev(x, y, dx=1))
    jac = np.array([[-nu_spl1.ev(x, y, dx=1), -1-nu_spl1.ev(x, y, dy=1)],
                    [-1-iota * nu_spl1.ev(x, y, dx=1), -iota * nu_spl1.ev(x, y, dy=1)]])
    return jac


for i in range(ntheta_booz):
    for j in range(nphi_booz):
        #if i == 0 and j == 0:
        #    x0 = (theta_booz_2D[i, j].copy(), phi_booz_2D[i, j].copy())
        #else:
        #    x0 = (data[0], data[1])
        x0 = (theta_booz_2D[i, j].copy(), phi_booz_2D[i, j].copy())
        #res = root(fun, x0, jac=fprime, args=(theta_booz_2D[i, j], phi_booz_2D[i, j]), tol=1e-8, options={"maxiter":100})
        res = root(fun, x0,  args=(theta_booz_2D[i, j], phi_booz_2D[i, j]), tol=1e-8)
        data = res.x
        #print(res)
        theta_PEST_2D[i, j] = data[0]
        phi_2D[i, j] = data[1]


B_booz = B_spl1.ev(theta_PEST_2D, phi_2D)
X_booz = X_spl1.ev(theta_PEST_2D, phi_2D)
Y_booz = Y_spl1.ev(theta_PEST_2D, phi_2D)
Z_booz = Z_spl1.ev(theta_PEST_2D, phi_2D)


theta_booz_periodic = theta_booz
phi_booz_periodic = phi_booz

#theta_booz_periodic = np.append(theta_booz, 2*np.pi)
#phi_booz_periodic = np.append(phi_booz, 2*np.pi)

#X_booz = np.concatenate((X_booz, X_booz[0:1, :]), axis=0)
#X_booz = np.concatenate((X_booz, X_booz[:, 0:1]), axis=1)
#
#Y_booz = np.concatenate((Y_booz, Y_booz[0:1, :]), axis=0)
#Y_booz = np.concatenate((Y_booz, Y_booz[:, 0:1]), axis=1)
#
#Z_booz = np.concatenate((Z_booz, Z_booz[0:1, :]), axis=0)
#Z_booz = np.concatenate((Z_booz, Z_booz[:, 0:1]), axis=1)

#B_booz_spl1 = RectBivariateSpline(theta_booz, phi_booz, B_booz, bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
X_booz_spl1 = RectBivariateSpline(theta_booz_periodic, phi_booz_periodic, X_booz, bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
Y_booz_spl1 = RectBivariateSpline(theta_booz_periodic, phi_booz_periodic, Y_booz, bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
R_booz_spl1 = RectBivariateSpline(theta_booz_periodic, phi_booz_periodic, np.sqrt(X_booz**2 + Y_booz**2), bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
Z_booz_spl1 = RectBivariateSpline(theta_booz_periodic, phi_booz_periodic, Z_booz, bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
phi_booz_spl1 = RectBivariateSpline(theta_booz_periodic, phi_booz_periodic, phi_2D, bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)
#phi_booz_spl2 = RectBivariateSpline(theta_booz_periodic, phi_booz_periodic, np.arctan2(Y_booz, X_booz), bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3)

plt.figure()
plt.contourf(theta_booz_2D, phi_booz_2D, B_booz, levels=50, linewidths=0.1)
plt.title(r"$|B|$", fontsize=18)
plt.xlabel(r"$\theta_{B}$", fontsize=18)
plt.ylabel(r"$\phi_{B}$", fontsize=18)
plt.ylim([np.pi, 2*np.pi])
plt.colorbar()
plt.tight_layout()
plt.savefig(_fig(f"test90_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)

plt.figure()
plt.contourf(theta_booz_2D, phi_booz_2D, X_booz_spl1.ev(theta_booz_2D, phi_booz_2D))
plt.title(r"$X$", fontsize=18)
plt.xlabel(r"$\theta_{B}$", fontsize=18)
plt.ylabel(r"$\phi_{B}$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(_fig(f"test91_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)

plt.figure()
plt.contourf(theta_booz_2D, phi_booz_2D, Y_booz_spl1.ev(theta_booz_2D, phi_booz_2D))
plt.title(r"$Y$", fontsize=18)
plt.xlabel(r"$\theta_{B}$", fontsize=18)
plt.ylabel(r"$\phi_{B}$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(_fig(f"test92_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)

plt.figure()
plt.contourf(theta_booz_2D, phi_booz_2D, Z_booz_spl1.ev(theta_booz_2D, phi_booz_2D))
plt.title(r"$Z$", fontsize=18)
plt.xlabel(r"$\theta_{B}$", fontsize=18)
plt.ylabel(r"$\phi_{B}$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(_fig(f"test93_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)


plt.figure()
plt.contourf(theta_booz_2D, phi_booz_2D, R_booz_spl1.ev(theta_booz_2D, phi_booz_2D))
plt.title(r"$R$", fontsize=18)
plt.xlabel(r"$\theta_{B}$", fontsize=18)
plt.ylabel(r"$\phi_{B}$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(_fig(f"test94_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)


plt.figure()
plt.contourf(theta_booz_2D, phi_booz_2D, phi_booz_spl1.ev(theta_booz_2D, phi_booz_2D))
plt.title(r"$\phi$", fontsize=18)
plt.xlabel(r"$\theta_{B}$", fontsize=18)
plt.ylabel(r"$\phi_{B}$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(_fig(f"test95_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)


# Save the arrays computed at lines 418-421, NOT the PEST-parameterised splines
# re-evaluated at Boozer angles.  X_spl1/Y_spl1/Z_spl1/B_spl1/nu_spl1 are
# functions of (theta_PEST, phi_cyl); feeding them (theta_B, phi_B) samples the
# right function at the wrong arguments, an O(nu) error.  theta_PEST_2D and
# phi_2D are the PEST angles Newton-solved for at each Boozer grid point, so
# B_booz/X_booz/Y_booz/Z_booz above are already the correct values on this grid.
# nu follows the AXIS_FOLLOWING_ANGLE.md sec 8.5 convention, nu = zeta_B - phi_cyl.
np.savez(os.path.join(BOOZER_DIR, f"Boozer_data_{fname}_{nturns_bdry}_{rho_suffix}.npz"),
         theta_booz=theta_booz_2D, phi_booz=phi_booz_2D,
         phi_cyl=phi_2D, theta_PEST=theta_PEST_2D,
         X_booz=X_booz, Y_booz=Y_booz,
         R_booz=np.hypot(X_booz, Y_booz), Z_booz=Z_booz,
         modB_booz=B_booz,
         nu_booz=phi_booz_2D - phi_2D,
         iota=iota, G=G)

#plt.figure()
#plt.contourf(theta_booz_2D, phi_booz_2D, phi_booz_spl2.ev(theta_booz_2D, phi_booz_2D))
#plt.title(r"$Y$", fontsize=18)
#plt.xlabel(r"$\theta_{B}$", fontsize=18)
#plt.ylabel(r"$\phi_{B}$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig(_fig(f"test96_{fname}_{nturns_bdry}_{rho_suffix}.png"), dpi=600)

# Create a dictionary of splines
# |B| and phi_cyl are pickled as Boozer-parameterised splines as well: without
# them a corrupted npz cannot be rebuilt from the pickle, which is what made the
# earlier Boozer_data files unrecoverable for modB.
splines = {
    'X': X_booz_spl1,
    'Y': Y_booz_spl1,
    'Z': Z_booz_spl1,
    'B': RectBivariateSpline(theta_booz_periodic, phi_booz_periodic, B_booz, bbox=[0, 2*np.pi, -0*np.pi, 2*np.pi], kx=3, ky=3),
    'phi_cyl': phi_booz_spl1,
}

# Save all at once
with open(os.path.join(BOOZER_DIR, f"Boozer_splines_{fname}_{nturns_bdry}_{rho_suffix}.pkl"), 'wb') as f:
    pickle.dump(splines, f)

print(G, iota, np.max(nu_array), np.min(nu_array))





