#!/usr/bin/env python3

import sys
import pdb
import numpy as np
from matplotlib import pyplot as plt
import pickle
import xarray as xr

import gvec 

fname1 = sys.argv[1]
nturns = sys.argv[2]
fname2 = sys.argv[3]
NFP = eval(sys.argv[4])
reverse_theta = eval(sys.argv[5])
#shift_zeta = eval(sys.argv[6])
shift_zeta = 0

# Load all back
with open(f'Boozer_splines_{fname1}_{nturns}_{fname2}.pkl', 'rb') as f:
    loaded_splines = pickle.load(f)

ntheta = 129
nzeta = 197
nzeta_by_two = int((nzeta - 1) / 2)

NFP = NFP

if reverse_theta:
    theta = np.linspace(2 * np.pi, 0, ntheta, endpoint=True)
else:
    theta = np.linspace(0., 2 * np.pi,ntheta, endpoint=True)

zeta = np.concatenate(
    (
        np.linspace(-np.pi / NFP, 0, nzeta_by_two, endpoint=False),
        np.linspace(0, np.pi / NFP, nzeta_by_two + 1, endpoint=True),
    )
)

theta_mesh, zeta_mesh = np.meshgrid(theta, zeta, indexing="ij")

spl_X = loaded_splines["X"]
spl_Y = loaded_splines["Y"]
spl_Z = loaded_splines["Z"]

X = spl_X.ev(theta_mesh, zeta_mesh)
Y = spl_Y.ev(theta_mesh, zeta_mesh)
Z = spl_Z.ev(theta_mesh, zeta_mesh)

# Ensure stellarator symmetry in half a field period
X[:, 0:nzeta_by_two] =  1 * X[:, nzeta_by_two + 1:][::-1, ::-1]
Y[:, 0:nzeta_by_two] = -1 * Y[:, nzeta_by_two + 1:][::-1, ::-1]
Z[:, 0:nzeta_by_two] = -1 * Z[:, nzeta_by_two + 1:][::-1, ::-1]

M = 14
N = 20

MZ = 14
NZ = 20

theta_mesh = theta_mesh[1:-2, 1:-1]
zeta_mesh = zeta_mesh[1:-2, 1:-1]

X = X[1:-2, 1:-1]
Y = Y[1:-2, 1:-1]
Z = Z[1:-2, 1:-1]


ntheta, nzeta = X.shape

assert theta_mesh.shape == (ntheta, nzeta)
assert zeta_mesh.shape == (ntheta, nzeta)

F = X + 1j * Y

theta_flat = theta_mesh.ravel()
zeta_flat = zeta_mesh.ravel()
F_flat = F.ravel()
Z_flat = Z.ravel()
npts = theta_flat.size

# ----- F = X + iY -----
m_vals_F = np.arange(-M, M + 1)
k_vals_F = np.arange(-N, N + 1)

n_F = NFP * k_vals_F + 1
m_arr_F = m_vals_F[None, :, None]
n_arr_F = n_F[None, None, :]

phaseF = theta_flat[:, None, None] * m_arr_F + zeta_flat[:, None, None] * n_arr_F
A_F = np.exp(1j * phaseF).reshape(npts, -1)

coef_F, residuals_F, rank_F, svals_F = np.linalg.lstsq(A_F, F_flat, rcond=None)
print("F: ncoef, rank:", coef_F.size, rank_F)

F_fit_flat = A_F @ coef_F
F_fit = F_fit_flat.reshape(ntheta, nzeta)
X_fit = F_fit.real
Y_fit = F_fit.imag

X_err = X_fit - X
Y_err = Y_fit - Y

rms_X = np.sqrt(np.mean(X_err**2))
rms_Y = np.sqrt(np.mean(Y_err**2))
rel_X = np.linalg.norm(X_err) / np.linalg.norm(X)
rel_Y = np.linalg.norm(Y_err) / np.linalg.norm(Y)

print("X  RMS abs error:", rms_X)
print("X  Rel  L2 error:", rel_X)
print("Y  RMS abs error:", rms_Y)
print("Y  Rel  L2 error:", rel_Y)

# ----- Z -----
m_vals_Z = np.arange(-MZ, MZ + 1)
k_vals_Z = np.arange(-NZ, NZ + 1)

n_Z = NFP * k_vals_Z
m_arr_Z = m_vals_Z[None, :, None]
n_arr_Z = n_Z[None, None, :]

phaseZ = theta_flat[:, None, None] * m_arr_Z + zeta_flat[:, None, None] * n_arr_Z
A_Z = np.exp(1j * phaseZ).reshape(npts, -1)

coef_Z, residuals_Z, rank_Z, svals_Z = np.linalg.lstsq(A_Z, Z_flat, rcond=None)
print("Z: ncoef, rank:", coef_Z.size, rank_Z)

Z_fit_flat = A_Z @ coef_Z
Z_fit = Z_fit_flat.reshape(ntheta, nzeta)
Z_err = Z_fit - Z

rms_Z = np.sqrt(np.mean(np.abs(Z_err)**2))
rel_Z = np.linalg.norm(Z_err) / np.linalg.norm(Z)

print("Z  RMS abs error:", rms_Z)
print("Z  Rel  L2 error:", rel_Z)

ntheta = 48
nzeta = 32

if reverse_theta:
    theta = np.linspace(2 * np.pi, 0, ntheta, endpoint=True)
else:
    theta = np.linspace(0., 2 * np.pi,ntheta, endpoint=True)

if shift_zeta:
    zeta = np.linspace(-np.pi + np.pi/(2*NFP), np.pi + np.pi/(2*NFP), nzeta * NFP, endpoint=False)
else:
    zeta = np.linspace(-np.pi, np.pi, nzeta * NFP, endpoint=False)

theta_mesh, zeta_mesh = np.meshgrid(theta, zeta, indexing="ij")

xyz = np.zeros((nzeta * NFP, ntheta, 3))

theta_flat = theta_mesh.ravel()
zeta_flat = zeta_mesh.ravel()
npts = theta_flat.size

phaseF = theta_flat[:, None, None] * m_arr_F + zeta_flat[:, None, None] * n_arr_F
A_F = np.exp(1j * phaseF).reshape(npts, -1)

F_fit_flat = A_F @ coef_F
F_fit = F_fit_flat.reshape(ntheta, nzeta * NFP)
X_fit = F_fit.real
Y_fit = F_fit.imag

phaseZ = theta_flat[:, None, None] * m_arr_Z + zeta_flat[:, None, None] * n_arr_Z
A_Z = np.exp(1j * phaseZ).reshape(npts, -1)

Z_fit_flat = A_Z @ coef_Z
Z_fit = Z_fit_flat.reshape(ntheta, nzeta * NFP)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X_fit, Y_fit, Z_fit.real, cmap='viridis', alpha=0.8)
ax.set_box_aspect([4, 2, 1])
plt.xlabel("X", fontsize=20)
plt.show()
plt.savefig(f"test_{NFP}_{fname1}_{nturns}_{fname2}.png")

for j in range(nzeta * NFP):
    for i in range(ntheta):
        #xyz[j, i, :] = np.array([X_fit[ntheta-1-i, j], Y_fit[ntheta-1-i, j], Z_fit[ntheta-1-i, j].real])
        xyz[j, i, :] = np.array([X_fit[i, j], Y_fit[i, j], Z_fit[i, j].real])

gvec.scripts.quasr.save_xyz(xyz, NFP, f"surface_{NFP}_{fname1}_{nturns}_{fname2}.nc")



