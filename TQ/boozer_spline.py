#!/usr/bin/env python3
from desc import set_device
set_device("gpu")
#set_device("cpu")

import pdb
import sys
import os
import pickle

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

import jax
jax.config.update("jax_compilation_cache_dir", "../jax-caches")
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

eq = Equilibrium.load("eq_NFP2_A+9.0_tor+0.0_mr+0.10_shift+0.00_wd+0.00_elong+3.00_wellwt+2.0_vacuum_final2.h5")

fname = sys.argv[1]
val = float(f"{fname}".replace("p", "."))
coilset = CoilSet.load(f"QFM_coilset_tqB1p7.h5")

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
r0 = [x]
z0 = [np.zeros_like(x)]
r, z = field_line_integrate(
    r0, z0, phis=phi_full, field=coilset, bounds_R=(0.5, 5.0), bounds_Z=(-1.2, 1.2)
)
spline = InterpolatedUnivariateSpline(x, z[0]-z[1], k=3)
R_shift = spline.roots() - R0
print(f"R_shift: {R_shift}")
#pdb.set_trace()


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

if os.path.exists(f"boundary_RZ_data_{fname}_{nturns_bdry}_{rho_suffix}.npz"):
	data = np.load(f"boundary_RZ_data_{fname}_{nturns_bdry}_{rho_suffix}.npz")
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

    np.savez(f"boundary_RZ_data_{fname}_{nturns_bdry}_{rho_suffix}.npz", r=r, z=z, phi=phi_full)


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
G = root_scalar(func, bracket=[-0.2, -24], method='bisect', xtol=1e-10, rtol=1e-10).root

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

#if alpha_sort[-1] != 2*np.pi:
#    alpha_sort = np.append(alpha_sort, 2*np.pi)
#    R_array = np.concatenate((R_array, R_array[0:1]))
#    X_array = np.concatenate((X_array, X_array[0:1]))
#    Y_array = np.concatenate((Y_array, Y_array[0:1]))
#    Z_array = np.concatenate((Z_array, Z_array[0:1]))
#    nu_array = np.concatenate((nu_array, nu_array[0:1]))
#    B_array = np.concatenate((B_array, B_array[0:1]))

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
#plt.savefig("test0.png", dpi=600)
#
#plt.figure()
#plt.contourf(alpha_sort, phi, X_array.T)
#plt.title(r"$X$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig("test1.png", dpi=600)
#
#plt.figure()
#plt.contourf(alpha_sort, phi, Y_array.T)
#plt.title(r"$Y$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig("test2.png", dpi=600)
#
#plt.figure()
#plt.contourf(alpha_sort, phi, Z_array.T)
#plt.title(r"$Z$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig("test3.png", dpi=600)
#
#plt.figure()
#plt.contour(alpha_sort, phi, B_array.T, levels=200, linewidths=0.05)
#plt.title(r"$|B|$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig("test4.png", dpi=600)
#
#plt.figure()
#plt.contourf(alpha_sort, phi, R_array.T)
#plt.title(r"$R$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig("test5.png", dpi=600)
#
#
#plt.figure()
#plt.contourf(alpha_sort, phi, np.mod((phi_2D.T+nu_array).T, 2*np.pi), levels=100)
#plt.title(r"$\phi_{boozer}$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig("test6.png", dpi=600)
#
#
#plt.figure()
#plt.contourf(alpha_sort, phi, np.mod(((alpha_2D + iota * phi_2D).T+ iota*nu_array).T, 2*np.pi), levels=100)
#plt.title(r"$\theta_{boozer}$", fontsize=18)
#plt.xlabel(r"$\alpha$", fontsize=18)
#plt.ylabel(r"$\phi$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig("test7.png", dpi=600)

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
plt.savefig("test10.png", dpi=600)

plt.figure()
plt.contourf(theta_PEST_2D, phi_cyl_2D, Y_spl1.ev(theta_PEST_2D, phi_cyl_2D))
plt.title(r"$Y$", fontsize=18)
plt.xlabel(r"$\theta_{PEST}$", fontsize=18)
plt.ylabel(r"$\phi$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig("test11.png", dpi=600)

plt.figure()
plt.contourf(theta_PEST_2D, phi_cyl_2D, Z_spl1.ev(theta_PEST_2D, phi_cyl_2D))
plt.title(r"$Z$", fontsize=18)
plt.xlabel(r"$\theta_{PEST}$", fontsize=18)
plt.ylabel(r"$\phi$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig("test12.png", dpi=600)


plt.figure()
plt.contourf(theta_PEST_2D, phi_cyl_2D, B_spl1.ev(theta_PEST_2D, phi_cyl_2D))
plt.title(r"$B$", fontsize=18)
plt.xlabel(r"$\theta_{PEST}$", fontsize=18)
plt.ylabel(r"$\phi$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig("test13.png", dpi=600)

plt.figure()
plt.contourf(theta_PEST_2D, phi_cyl_2D, nu_spl1.ev(theta_PEST_2D, phi_cyl_2D))
plt.title(r"$\nu$", fontsize=18)
plt.xlabel(r"$\theta_{PEST}$", fontsize=18)
plt.ylabel(r"$\phi$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig("test14.png", dpi=600)

plt.figure()
print(theta_PEST_2D.shape, nu_interp.shape)
plt.contourf(theta_PEST_2D, phi_cyl_2D, np.mod((theta_PEST_2D+ iota*nu_interp), 2*np.pi), levels=100)
plt.title(r"$\theta_{boozer}$", fontsize=18)
plt.xlabel(r"$\theta_{PEST}$", fontsize=18)
plt.ylabel(r"$\phi$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig("test70.png", dpi=600)

np.savez(f"PEST_data_{fname}_{nturns_bdry}_{rho_suffix}.npz", theta_PEST=theta_PEST_2D, phi_cyl=phi_cyl_2D, Z_PEST=Z_spl1.ev(theta_PEST_2D, phi_cyl_2D), R_PEST=R_spl1.ev(theta_PEST_2D, phi_cyl_2D), modB_PEST=B_spl1.ev(theta_PEST_2D, phi_cyl_2D), nu_PEST=nu_spl1.ev(theta_PEST_2D, phi_cyl_2D))


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
plt.savefig(f"test90_{fname}_{nturns_bdry}_{rho_suffix}.png", dpi=600)

plt.figure()
plt.contourf(theta_booz_2D, phi_booz_2D, X_booz_spl1.ev(theta_booz_2D, phi_booz_2D))
plt.title(r"$X$", fontsize=18)
plt.xlabel(r"$\theta_{B}$", fontsize=18)
plt.ylabel(r"$\phi_{B}$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(f"test91_{fname}_{nturns_bdry}_{rho_suffix}.png", dpi=600)

plt.figure()
plt.contourf(theta_booz_2D, phi_booz_2D, Y_booz_spl1.ev(theta_booz_2D, phi_booz_2D))
plt.title(r"$Y$", fontsize=18)
plt.xlabel(r"$\theta_{B}$", fontsize=18)
plt.ylabel(r"$\phi_{B}$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(f"test92_{fname}_{nturns_bdry}_{rho_suffix}.png", dpi=600)

plt.figure()
plt.contourf(theta_booz_2D, phi_booz_2D, Z_booz_spl1.ev(theta_booz_2D, phi_booz_2D))
plt.title(r"$Z$", fontsize=18)
plt.xlabel(r"$\theta_{B}$", fontsize=18)
plt.ylabel(r"$\phi_{B}$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(f"test93_{fname}_{nturns_bdry}_{rho_suffix}.png", dpi=600)


plt.figure()
plt.contourf(theta_booz_2D, phi_booz_2D, R_booz_spl1.ev(theta_booz_2D, phi_booz_2D))
plt.title(r"$R$", fontsize=18)
plt.xlabel(r"$\theta_{B}$", fontsize=18)
plt.ylabel(r"$\phi_{B}$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(f"test94_{fname}_{nturns_bdry}_{rho_suffix}.png", dpi=600)


plt.figure()
plt.contourf(theta_booz_2D, phi_booz_2D, phi_booz_spl1.ev(theta_booz_2D, phi_booz_2D))
plt.title(r"$\phi$", fontsize=18)
plt.xlabel(r"$\theta_{B}$", fontsize=18)
plt.ylabel(r"$\phi_{B}$", fontsize=18)
plt.colorbar()
plt.tight_layout()
plt.savefig(f"test95_{fname}_{nturns_bdry}_{rho_suffix}.png", dpi=600)


np.savez(f"Boozer_data_{fname}_{nturns_bdry}_{rho_suffix}.npz", theta_booz=theta_booz_2D, phi_booz=phi_booz_2D, phi_cyl=phi_booz_spl1.ev(theta_booz_2D, phi_booz_2D), Z_booz=Z_spl1.ev(theta_booz_2D, phi_booz_2D), R_booz=R_booz_spl1.ev(theta_booz_2D, phi_booz_2D), modB_booz=B_spl1.ev(theta_booz_2D, phi_booz_2D), nu_booz=nu_spl1.ev(theta_booz_2D, phi_booz_2D))

#plt.figure()
#plt.contourf(theta_booz_2D, phi_booz_2D, phi_booz_spl2.ev(theta_booz_2D, phi_booz_2D))
#plt.title(r"$Y$", fontsize=18)
#plt.xlabel(r"$\theta_{B}$", fontsize=18)
#plt.ylabel(r"$\phi_{B}$", fontsize=18)
#plt.colorbar()
#plt.tight_layout()
#plt.savefig(f"test96_{fname}_{nturns_bdry}_{rho_suffix}.png", dpi=600)

# Create a dictionary of splines
splines = {
    'X': X_booz_spl1,
    'Y': Y_booz_spl1, 
    'Z': Z_booz_spl1, 
}

# Save all at once
with open(f'Boozer_splines_{fname}_{nturns_bdry}_{rho_suffix}.pkl', 'wb') as f:
    pickle.dump(splines, f)

print(G, iota, np.max(nu_array), np.min(nu_array))






