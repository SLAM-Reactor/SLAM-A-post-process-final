"""Paths for GX geometry exports from retained GVEC equilibria."""

from __future__ import annotations

from pathlib import Path


GVEC_GX_DIR = Path(__file__).resolve().parent
DATA_DIR = GVEC_GX_DIR / "data"
LOG_DIR = GVEC_GX_DIR / "logs"
SOURCE_GVEC_DIR = GVEC_GX_DIR.parent / "GVEC"

NFP = 2

CASES = {
    "SLNAM": {
        "coilset": SOURCE_GVEC_DIR / "coils" / "SLNAM-field-09-09-26.h5",
        "ini": SOURCE_GVEC_DIR
        / "parameter_surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01_final.ini",
        "state": SOURCE_GVEC_DIR
        / "surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01_State_final.dat",
    },
    "SLAM_N128_RHO065_CLEAN1E4": {
        "coilset": SOURCE_GVEC_DIR / "coils" / "SLAM-field-09-09-26_2.h5",
        "ini": SOURCE_GVEC_DIR
        / "parameter_surface_2_SLAM_1p0_128_0p65_clean1em4_axis_x1m0p01_final.ini",
        "state": SOURCE_GVEC_DIR
        / "surface_2_SLAM_1p0_128_0p65_clean1em4_axis_x1m0p01_State_final.dat",
    },
    # Best GVEC equilibria from the 09-23 resolution scans (Appendix-2/*_res_scan)
    "SLNAM_BEST": {
        "coilset": SOURCE_GVEC_DIR / "coils" / "SLNAM-field-09-09-26.h5",
        "ini": SOURCE_GVEC_DIR / "parameter_SLNAM_216_0p92_nelems32_mn12_34_final.ini",
        "state": SOURCE_GVEC_DIR / "SLNAM_216_0p92_nelems32_mn12_34_State_final.dat",
    },
    "SLAM_BEST": {
        "coilset": SOURCE_GVEC_DIR / "coils" / "SLAM-field-09-09-26_2.h5",
        "ini": SOURCE_GVEC_DIR / "parameter_SLAM_128_0p65_nelems24_mn24_32_final.ini",
        "state": SOURCE_GVEC_DIR / "SLAM_128_0p65_nelems24_mn24_32_State_final.dat",
    },
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
