"""Shared paths and case metadata for GVEC post-processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np


POST_DIR = Path(__file__).resolve().parent
GVEC_DIR = POST_DIR.parent
DATA_DIR = POST_DIR / "data"
FIGURE_DIR = POST_DIR / "figures"
LOG_DIR = POST_DIR / "logs"

NFP = 2
TWOPI = 2.0 * np.pi

DESC_EQ = (
    GVEC_DIR
    / "coils"
    / "eq_NFP2_A+9.0_tor+0.0_mr+0.10_shift+0.00_wd+0.00_elong+3.00_wellwt+2.0_vacuum_final2.h5"
)

CASES = {
    # Figure 6 equilibria (Poincare comparison)
    "SLNAM": {
        "coilset": GVEC_DIR / "coils" / "SLNAM-field-09-09-26.h5",
        "ini": GVEC_DIR
        / "parameter_surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01_final.ini",
        "state": GVEC_DIR
        / "surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01_State_final.dat",
    },
    "SLAM_N128_RHO065_CLEAN1E4": {
        "coilset": GVEC_DIR / "coils" / "SLAM-field-09-09-26_2.h5",
        "ini": GVEC_DIR
        / "parameter_surface_2_SLAM_1p0_128_0p65_clean1em4_axis_x1m0p01_final.ini",
        "state": GVEC_DIR
        / "surface_2_SLAM_1p0_128_0p65_clean1em4_axis_x1m0p01_State_final.dat",
    },
    # Best GVEC equilibria from the 09-23 resolution scans (lowest max
    # |F|_normalized over a field period); see ../res_scan/. Used for Figure 7.
    "SLNAM_BEST": {
        # SLNAM 216/0.92 boundary, nElems=32, mn_max=(12,34)
        "coilset": GVEC_DIR / "coils" / "SLNAM-field-09-09-26.h5",
        "ini": GVEC_DIR / "parameter_SLNAM_216_0p92_nelems32_mn12_34_final.ini",
        "state": GVEC_DIR / "SLNAM_216_0p92_nelems32_mn12_34_State_final.dat",
    },
    "SLAM_BEST": {
        # SLAM 128/0.65/clean1e-4 (Poincare-comparison) boundary, nElems=24, mn_max=(24,32)
        "coilset": GVEC_DIR / "coils" / "SLAM-field-09-09-26_2.h5",
        "ini": GVEC_DIR / "parameter_SLAM_128_0p65_nelems24_mn24_32_final.ini",
        "state": GVEC_DIR / "SLAM_128_0p65_nelems24_mn24_32_State_final.dat",
    },
}

DEFAULT_CASE_LABELS = ["SLNAM", "SLAM_N128_RHO065_CLEAN1E4"]

DEFAULT_RHO_SURFACES = np.array([0.2, 0.4, 0.6, 0.8, 0.92, 1.0])
DEFAULT_RHO_PROFILE = np.array([0.2, 0.4, 0.6, 0.8, 0.92, 0.95])
DEFAULT_RHO_WELL = np.linspace(0.05, 1.0, 20)
DEFAULT_ZETA_PLANES = np.array([0.0, np.pi / NFP])


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def parse_floats(text: str | None, default: np.ndarray) -> np.ndarray:
    if text is None or text.strip() == "":
        return np.asarray(default, dtype=float)
    return np.array([float(x.strip()) for x in text.split(",") if x.strip()], dtype=float)


def selected_cases(labels: list[str] | None) -> list[str]:
    if not labels:
        return DEFAULT_CASE_LABELS.copy()
    bad = sorted(set(labels) - set(CASES))
    if bad:
        raise ValueError(f"unknown case label(s): {', '.join(bad)}")
    return labels


def object_array(values) -> np.ndarray:
    return np.array(values, dtype=object)
