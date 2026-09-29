"""Shared paths for the two retained GVEC post-processing cases."""

from __future__ import annotations

from pathlib import Path


POST_DIR = Path(__file__).resolve().parent
GVEC_DIR = POST_DIR.parent
DATA_DIR = POST_DIR / "data"
FIGURE_DIR = POST_DIR / "figures"

NFP = 2

CASES = {
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
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
