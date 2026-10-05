#!/usr/bin/env python3
"""Trace SLNAMified coilset Poincare data for one L_over_R case on a GPU."""

import os
import sys
import warnings

import numpy as np


DESC_SOURCE = os.environ.get("DESC_SOURCE", "/pscratch/sd/r/rgaur/DESC3/DESC")
sys.path.insert(0, DESC_SOURCE)

from desc import set_device

set_device("gpu")

import desc.io
from desc.grid import LinearGrid
from desc.magnetic_fields import field_line_integrate


HERE = os.path.dirname(os.path.abspath(__file__))
QI_TO_SLNAM_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
QI_COILSET_DIR = os.path.abspath(os.path.join(QI_TO_SLNAM_DIR, "..", "QI_coilset"))

EQUILIBRIUM_FILENAME = os.path.join(
    QI_COILSET_DIR,
    "eq_NFP2_A+9.0_tor+0.0_mr+0.10_shift+0.00_wd+0.00_elong+3.00_wellwt+2.0_vacuum_final2.h5",
)

CASE_ORDER = ("0p5", "1p0", "1p5", "2p0")
CASE_CONFIG = {
    "0p5": {
        "coilset_filename": os.path.join(
            QI_TO_SLNAM_DIR,
            "SLNAMified_NFP2_A8p7_coilset_L_over_R_0p5_SLAM-A.h5",
        ),
        "rho_start": np.linspace(0.0, 0.4, 4),
        "r_shift": 0.60,
        "axis_seed": (1.542915, -0.000234),
        "seed_R": np.array([1.568, 1.592, 1.617, 1.642, 1.667]),
        "seed_Z": np.zeros(5),
        "bounds_R": (0.30, 2.60),
        "bounds_Z": (-1.00, 1.00),
    },
    "1p0": {
        "coilset_filename": os.path.join(
            QI_TO_SLNAM_DIR,
            "SLNAMified_NFP2_A8p7_coilset_L_over_R_1p0_SLAM-A.h5",
        ),
        "local_truncated_filename": os.path.join(
            QI_TO_SLNAM_DIR,
            "SLNAMified_NFP2_A8p7_coilset_L_over_R_1p0_SLAM-A.h5",
        ),
        "rho_start": np.linspace(0.0, 0.9, 12),
        "r_shift": 1.04,
        "axis_seed": (2.035226, 0.000001),
        "seed_R": np.array([2.020, 1.995, 1.970, 1.945, 1.918]),
        "seed_Z": np.zeros(5),
        "bounds_R": (0.30, 2.60),
        "bounds_Z": (-1.00, 1.00),
    },
    "1p5": {
        "coilset_filename": os.path.join(
            QI_TO_SLNAM_DIR,
            "SLNAMified_NFP2_A8p7_coilset_L_over_R_1p5_SLAM-A.h5",
        ),
        "rho_start": np.linspace(0.0, 0.9, 4),
        "r_shift": 1.525,
        "axis_seed": (2.529017, 0.000069),
        "seed_R": np.array([2.510, 2.480, 2.450, 2.420, 2.412]),
        "seed_Z": np.zeros(5),
        "bounds_R": (0.30, 3.50),
        "bounds_Z": (-1.00, 1.00),
    },
    "2p0": {
        "coilset_filename": os.path.join(
            QI_TO_SLNAM_DIR,
            "SLNAMified_NFP2_A8p7_coilset_L_over_R_2p0_SLAM-A.h5",
        ),
        "rho_start": np.linspace(0.0, 0.93, 6),
        "r_shift": 2.01,
        "axis_seed": (3.026952, -0.000028),
        "seed_R": np.array([3.005, 2.980, 2.955, 2.930, 2.908]),
        "seed_Z": np.zeros(5),
        "bounds_R": (0.30, 4.00),
        "bounds_Z": (-1.00, 1.00),
    },
}

PHI = np.array([0.0, np.pi / 2])
NFP = 2
SOURCE_GRID_N = 32


def parse_pair(text):
    values = [float(value) for value in text.split(",")]
    if len(values) != 2:
        raise ValueError(f"Expected two comma-separated values, got {text!r}.")
    return tuple(values)


def parse_float_array(text):
    return np.array([float(value) for value in text.split(",")], dtype=float)


def unwrap_loaded(obj):
    if isinstance(obj, (list, tuple)):
        return obj[-1]
    if obj.__class__.__name__ == "EquilibriaFamily":
        return obj[-1]
    return obj


def selected_case():
    case = os.environ.get("POINCARE_CASE")
    if case:
        return case
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if array_task_id is not None:
        return CASE_ORDER[int(array_task_id)]
    return CASE_ORDER[0]


case = selected_case()
if case not in CASE_CONFIG:
    raise ValueError(f"Unknown POINCARE_CASE={case!r}. Valid cases: {CASE_ORDER}")

config = CASE_CONFIG[case]
ntransit = int(os.environ.get("POINCARE_NTRANSIT", "20"))
equilibrium_filename = os.environ.get("POINCARE_EQ_FILENAME", EQUILIBRIUM_FILENAME)
coilset_filename = os.environ.get(
    "POINCARE_COILSET_FILENAME", config["coilset_filename"]
)
if "POINCARE_BOUNDS_R" in os.environ:
    bounds_R = parse_pair(os.environ["POINCARE_BOUNDS_R"])
else:
    bounds_R = config["bounds_R"]
if "POINCARE_BOUNDS_Z" in os.environ:
    bounds_Z = parse_pair(os.environ["POINCARE_BOUNDS_Z"])
else:
    bounds_Z = config["bounds_Z"]
rho_start = parse_float_array(
    os.environ.get(
        "POINCARE_RHO_START",
        ",".join(str(value) for value in config["rho_start"]),
    )
)
r_shift = float(os.environ.get("POINCARE_R_SHIFT", str(config["r_shift"])))
include_axis_seed = os.environ.get("POINCARE_INCLUDE_AXIS_SEED", "0").lower() in (
    "1",
    "true",
    "yes",
)
max_steps_text = os.environ.get("POINCARE_MAX_STEPS", "")
max_steps = None if max_steps_text == "" else int(max_steps_text)
source_grid_N = int(os.environ.get("POINCARE_SOURCE_GRID_N", str(SOURCE_GRID_N)))
output_filename = os.environ.get(
    "POINCARE_OUTPUT_FILENAME",
    os.path.join(
        HERE,
        f"SLNAMified_L_over_R_{case}_poincare_data_ntransit{ntransit}.npz",
    ),
)

phis = (PHI + np.arange(ntransit)[:, np.newaxis] * 2 * np.pi / NFP).ravel()

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"desc\.io\.hdf5_io",
    )
    eq = unwrap_loaded(desc.io.load(equilibrium_filename))
    field = desc.io.load(coilset_filename)

trace_grid = LinearGrid(rho=rho_start)
if "seed_R" in config:
    r0 = np.asarray(config["seed_R"], dtype=float).copy()
    z0 = np.asarray(config.get("seed_Z", np.zeros_like(r0)), dtype=float).copy()
    seed_source = "case_config"
else:
    r0 = np.asarray(eq.compute("R", grid=trace_grid)["R"], dtype=float) + r_shift
    z0 = np.asarray(eq.compute("Z", grid=trace_grid)["Z"], dtype=float)
    seed_source = "equilibrium_rho"
if "POINCARE_SEED_R" in os.environ:
    r0 = parse_float_array(os.environ["POINCARE_SEED_R"])
    seed_source = "environment"
if "POINCARE_SEED_Z" in os.environ:
    z0 = parse_float_array(os.environ["POINCARE_SEED_Z"])
elif "POINCARE_SEED_R" in os.environ:
    z0 = np.zeros_like(r0)
if r0.shape != z0.shape:
    raise ValueError(f"Seed R/Z shapes differ: {r0.shape} vs {z0.shape}")
axis_seed_index = -1
if include_axis_seed:
    axis_seed_R = float(os.environ.get("POINCARE_AXIS_SEED_R", config["axis_seed"][0]))
    axis_seed_Z = float(os.environ.get("POINCARE_AXIS_SEED_Z", config["axis_seed"][1]))
    r0 = np.concatenate((np.array([axis_seed_R]), r0))
    z0 = np.concatenate((np.array([axis_seed_Z]), z0))
    axis_seed_index = 0

R, Z = field_line_integrate(
    r0,
    z0,
    phis=phis,
    field=field,
    source_grid=LinearGrid(N=source_grid_N),
    bounds_R=bounds_R,
    bounds_Z=bounds_Z,
    max_steps=max_steps,
    options={"throw": False},
)
R = np.asarray(R).reshape(ntransit, PHI.size, r0.size)
Z = np.asarray(Z).reshape(ntransit, PHI.size, r0.size)

finite = np.isfinite(R) & np.isfinite(Z)
finite_fraction_by_seed = np.mean(finite, axis=(0, 1))
finite_count_by_seed = np.sum(finite, axis=(0, 1))
last_finite_transit = np.full(r0.size, -1, dtype=int)
for seed_index in range(r0.size):
    finite_transits = np.where(np.any(finite[:, :, seed_index], axis=1))[0]
    if finite_transits.size:
        last_finite_transit[seed_index] = finite_transits[-1]

fully_finite = finite_fraction_by_seed == 1.0
outermost_full_seed_R = np.nan
if np.any(fully_finite):
    outermost_full_seed_R = np.max(r0[fully_finite])

np.savez_compressed(
    output_filename,
    R=R,
    Z=Z,
    r0=r0,
    z0=z0,
    phi=PHI,
    zeta_planes=PHI,
    zeta_plane_labels=np.array(["zeta=0", "zeta=pi/NFP"]),
    phis=phis,
    ntransit=ntransit,
    NFP=NFP,
    source_grid_N=source_grid_N,
    bounds_R=np.array(bounds_R),
    bounds_Z=np.array(bounds_Z),
    max_steps=-1 if max_steps is None else max_steps,
    equilibrium_filename=equilibrium_filename,
    coilset_filename=coilset_filename,
    local_truncated_filename=config.get("local_truncated_filename", ""),
    case=case,
    rho_start=rho_start,
    r_shift=r_shift,
    seed_source=seed_source,
    include_axis_seed=include_axis_seed,
    axis_seed_index=axis_seed_index,
    finite_fraction_by_seed=finite_fraction_by_seed,
    finite_count_by_seed=finite_count_by_seed,
    last_finite_transit=last_finite_transit,
    outermost_full_seed_R=outermost_full_seed_R,
)

print("Case:", case)
print("Equilibrium:", equilibrium_filename)
print("Coilset:", coilset_filename)
print("Transit count:", ntransit)
print("NFP:", NFP)
print("Rho start:", rho_start)
print("R shift:", r_shift)
print("Seed source:", seed_source)
print("Include axis seed:", include_axis_seed)
print("Axis seed index:", axis_seed_index)
print("Seed R:", r0)
print("Seed Z:", z0)
print("Phi planes:", PHI)
print("Bounds R:", bounds_R)
print("Bounds Z:", bounds_Z)
print("Source grid N:", source_grid_N)
print("Max steps:", "DESC default" if max_steps is None else max_steps)
print("Poincare R/Z shape:", R.shape)
print("Finite fraction by seed:", finite_fraction_by_seed)
print("Last finite transit by seed:", last_finite_transit)
print("Outermost full seed R:", outermost_full_seed_R)
print("Saved:", output_filename)
