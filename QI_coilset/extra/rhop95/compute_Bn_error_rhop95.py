#!/usr/bin/env python3
"""Compute coil-field normal error on the QI rho=0.95 surface."""

import os
import sys
import warnings

DESC_SOURCE = os.environ.get("DESC_SOURCE", "/pscratch/sd/r/rgaur/DESC3/DESC")
sys.path.insert(0, DESC_SOURCE)

from desc import set_device

set_device("gpu")

import numpy as np

from desc.coils import CoilSet
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid


here = os.path.dirname(os.path.abspath(__file__))
qi_dir = os.path.abspath(os.path.join(here, ".."))

equilibrium_filename = os.path.join(
    qi_dir,
    "eq_NFP2_A+9.0_tor+0.0_mr+0.10_shift+0.00_wd+0.00_elong+3.00"
    "_wellwt+2.0_vacuum_final2.h5",
)
coilset_filename = os.path.join(qi_dir, "NFP2_A8p7_coilset_40_coils_Bn_0p015_N9.h5")
output_filename = "QI_rhop95_Bn_error_data.npz"

rho = 0.95
surface_M = 80
surface_N = 80
coil_points = 400
biot_savart_chunk_size = 256

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"desc\.io\.hdf5_io",
    )
    eq = Equilibrium.load(equilibrium_filename)

field = CoilSet.load(coilset_filename)
surf = eq.get_surface_at(rho=rho)

eval_grid = LinearGrid(
    M=surface_M,
    N=surface_N,
    NFP=eq.NFP,
    endpoint=False,
)
field_grid = LinearGrid(N=coil_points, endpoint=False)

Bn, surf_coords = field.compute_Bnormal(
    surf,
    eval_grid=eval_grid,
    source_grid=field_grid,
    chunk_size=biot_savart_chunk_size,
)
B = field.compute_magnetic_field(
    surf_coords,
    source_grid=field_grid,
    chunk_size=biot_savart_chunk_size,
)

Bn = np.asarray(Bn, dtype=float)
B = np.asarray(B, dtype=float)
Bmag = np.linalg.norm(B, axis=1)
Bn_over_B = Bn / Bmag

abs_Bn = np.abs(Bn)
abs_Bn_over_B = np.abs(Bn_over_B)

mean_abs_Bn = np.mean(abs_Bn)
mean_Bmag = np.mean(Bmag)
notebook_error = mean_abs_Bn / mean_Bmag
mean_pointwise_error = np.mean(abs_Bn_over_B)
rms_pointwise_error = np.sqrt(np.mean(Bn_over_B**2))
max_pointwise_error = np.max(abs_Bn_over_B)

np.savez(
    output_filename,
    rho=rho,
    surface_M=surface_M,
    surface_N=surface_N,
    coil_points=coil_points,
    biot_savart_chunk_size=biot_savart_chunk_size,
    NFP=eq.NFP,
    eval_grid_nodes=np.asarray(eval_grid.nodes),
    surf_coords=np.asarray(surf_coords),
    Bn=Bn,
    B=B,
    Bmag=Bmag,
    Bn_over_B=Bn_over_B,
    mean_abs_Bn=mean_abs_Bn,
    mean_Bmag=mean_Bmag,
    notebook_error=notebook_error,
    mean_pointwise_error=mean_pointwise_error,
    rms_pointwise_error=rms_pointwise_error,
    max_pointwise_error=max_pointwise_error,
    min_Bn=np.min(Bn),
    max_Bn=np.max(Bn),
    equilibrium_filename=equilibrium_filename,
    coilset_filename=coilset_filename,
)

print("Equilibrium:", equilibrium_filename)
print("Coilset:", coilset_filename)
print("Surface rho:", rho)
print("Grid M, N, NFP:", surface_M, surface_N, eq.NFP)
print("Surface nodes:", eval_grid.num_nodes)
print("Mean |B.n| [T]:", mean_abs_Bn)
print("Mean |B| [T]:", mean_Bmag)
print("Notebook-style mean(|B.n|)/mean(|B|):", notebook_error)
print("Mean pointwise |B.n|/|B|:", mean_pointwise_error)
print("RMS pointwise B.n/|B|:", rms_pointwise_error)
print("Max pointwise |B.n|/|B|:", max_pointwise_error)
print("Min raw B.n [T]:", np.min(Bn))
print("Max raw B.n [T]:", np.max(Bn))
print("Saved:", output_filename)
