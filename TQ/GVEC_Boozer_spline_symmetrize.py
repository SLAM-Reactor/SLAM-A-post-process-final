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
#shift_zeta = eval(sys.argv[5])
shift_zeta = 0

# Load all back
with open(f'Boozer_splines_{fname1}_{nturns}_{fname2}.pkl', 'rb') as f:
    loaded_splines = pickle.load(f)

## Load all back
##with open('Boozer_splines_2p0_128_0p83.pkl', 'rb') as f:
#with open('Boozer_splines_1p5_196_0p91.pkl', 'rb') as f:
##with open('Boozer_splines_0p5_100_0p87.pkl', 'rb') as f:
##with open('Boozer_splines_0p0_256_0p82.pkl', 'rb') as f:
#    loaded_splines = pickle.load(f)

ntheta = 129
nzeta = 197
nzeta_by_two = int((nzeta-1)/2)

NFP = NFP

if reverse_theta:
    theta = np.linspace(2*np.pi, 0, ntheta, endpoint=True)
else:
    theta = np.linspace(0., 2*np.pi,ntheta, endpoint=True)

#zeta=np.linspace(0,2*np.pi/NFP,nzeta,endpoint=True)
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


## Ensure symmetry in a single field period
#X[:, 0:nzeta_by_two] = -1 * X[:, nzeta_by_two+1:][::-1, ::-1]
#Y[:, 0:nzeta_by_two] =  1 * Y[:, nzeta_by_two+1:][::-1, ::-1]
#Z[:, 0:nzeta_by_two] = -1 * Z[:, nzeta_by_two+1:][::-1, ::-1]

X[:, 0:nzeta_by_two] =  1 * X[:, nzeta_by_two+1:][::-1, ::-1]
Y[:, 0:nzeta_by_two] = -1 * Y[:, nzeta_by_two+1:][::-1, ::-1]
Z[:, 0:nzeta_by_two] = -1 * Z[:, nzeta_by_two+1:][::-1, ::-1]

#M = 12  
#N = 18

M = 14  
N = 20

# Exclude both theta= 0, 2pi and zeta = 0, 2pi
theta_mesh = theta_mesh[1:-1, 1:-1]
zeta_mesh = zeta_mesh[1:-1, 1:-1]

X = X[1:-1, 1:-1]
Y = Y[1:-1, 1:-1]
Z = Z[1:-1, 1:-1]

ntheta, nzeta = X.shape

assert theta_mesh.shape == (ntheta, nzeta)
assert zeta_mesh.shape  == (ntheta, nzeta)

# Fourier transform complex function onto which we can impose NFP symmetry
F = X + 1j * Y

theta_flat = theta_mesh.ravel()
zeta_flat  = zeta_mesh.ravel()
F_flat     = F.ravel()
Z_flat     = Z.ravel()
npts       = theta_flat.size

m_vals = np.arange(-M, M + 1)
k_vals = np.arange(-N, N + 1)

# F: only odd toroidal modes n = 2k+1  -> F(theta,zeta+pi) = -F(theta,zeta)
n_odd  = 2 * k_vals + 1
m_arr  = m_vals[None, :, None]
n_arrF = n_odd[None, None, :]

phaseF = theta_flat[:, None, None] * m_arr + zeta_flat[:, None, None] * n_arrF
A_F    = np.exp(1j * phaseF).reshape(npts, -1)

coef_F, residuals_F, rank_F, svals_F = np.linalg.lstsq(A_F, F_flat, rcond=None)
print("F: ncoef, rank:", coef_F.size, rank_F)

F_fit_flat = A_F @ coef_F
F_fit      = F_fit_flat.reshape(ntheta, nzeta)
X_fit      = F_fit.real
Y_fit      = F_fit.imag

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

#plt.contourf(theta_mesh, zeta_mesh, Y_fit)
#plt.show()

# Z: only even toroidal modes n = 2k  -> Z(theta,zeta+pi) = Z(theta,zeta)
n_even = 2 * k_vals
n_arrZ = n_even[None, None, :]

phaseZ = theta_flat[:, None, None] * m_arr + zeta_flat[:, None, None] * n_arrZ
A_Z    = np.exp(1j * phaseZ).reshape(npts, -1)

coef_Z, residuals_Z, rank_Z, svals_Z = np.linalg.lstsq(A_Z, Z_flat, rcond=None)
print("Z: ncoef, rank:", coef_Z.size, rank_Z)

Z_fit_flat = A_Z @ coef_Z
Z_fit      = Z_fit_flat.reshape(ntheta, nzeta)
Z_err      = Z_fit - Z

rms_Z = np.sqrt(np.mean(Z_err**2))
rel_Z = np.linalg.norm(Z_err) / np.linalg.norm(Z)

print("Z  RMS abs error:", rms_Z)
print("Z  Rel  L2 error:", rel_Z)


ntheta = 48
nzeta = 64

if reverse_theta:
    theta = np.linspace(2*np.pi, 0, ntheta, endpoint=False)
else:
    theta = np.linspace(0, 2*np.pi, ntheta, endpoint=False)

if shift_zeta:
    zeta=np.linspace(0 + np.pi/(2*NFP), 2*np.pi + np.pi/(2*NFP), nzeta*NFP,endpoint=False)
else:
    zeta=np.linspace(-np.pi, np.pi, nzeta*NFP,endpoint=False)

theta_mesh, zeta_mesh = np.meshgrid(theta, zeta, indexing="ij")

xyz=np.zeros((nzeta*NFP,ntheta,3))

theta_flat = theta_mesh.flatten()
zeta_flat = zeta_mesh.flatten()
npts       = theta_flat.size

phaseF = theta_flat[:, None, None] * m_arr + zeta_flat[:, None, None] * n_arrF
A_F    = np.exp(1j * phaseF).reshape(npts, -1)

F_fit_flat = A_F @ coef_F
F_fit      = F_fit_flat.reshape(ntheta, nzeta*NFP)
X_fit      = F_fit.real
Y_fit      = F_fit.imag


phaseZ = theta_flat[:, None, None] * m_arr + zeta_flat[:, None, None] * n_arrZ
A_Z    = np.exp(1j * phaseZ).reshape(npts, -1)

Z_fit_flat = A_Z @ coef_Z
Z_fit      = Z_fit_flat.reshape(ntheta, nzeta*NFP)


## 3D plot looks ok
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X_fit[:], Y_fit[:], Z_fit, cmap='viridis', alpha=0.8)
ax.set_box_aspect([4, 2, 1])  # Equal scaling for all axes
plt.xlabel("X", fontsize=20)
plt.show()
plt.savefig(f"test4_{NFP}_{fname1}_{nturns}_{fname2}.png")


for j in range(nzeta*NFP):
    for i in range(ntheta): 
        xyz[j,i,:] = np.array([X_fit[i, j], Y_fit[i, j], Z_fit[i, j].real])

gvec.scripts.quasr.save_xyz(xyz, NFP, f"surface_{NFP}_{fname1}_{nturns}_{fname2}.nc")


