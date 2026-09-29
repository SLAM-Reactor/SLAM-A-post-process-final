#!/usr/bin/env python3
"""Compute GVEC magnetic-well profiles for the comparison equilibria.

The comparison plot should use ``desc_style_well``. That quantity follows the
same vacuum proxy as DESC's native ``magnetic well`` output:

    V(r) * d<|B|^2>/dr / (V_r(r) * <|B|^2>)

GVEC also exposes a different native scalar convention,
``vacuum_magnetic_well_depth``. We save the related cumulative profile for
diagnostics, but it is not the DESC-style comparison curve.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import gvec
import numpy as np

from paths import CASES, DATA_DIR, DEFAULT_RHO_WELL, ensure_dirs, object_array


# ---------------------------- calculation setup --------------------------- #

CASE_LABELS = ("SLNAM_BEST", "SLAM_BEST")
RHO_VALUES = np.array(
    [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        0.96,
        0.97,
        0.98,
        0.985,
        0.99,
        0.995,
        0.998,
        0.999,
        1.0,
    ],
    dtype=float,
)
OUTPUT = DATA_DIR / "magnetic_well_gvec_best.npz"
VOLUME_SAMPLES = 400

COMPARISON_WELL_KIND = "desc_style_well"


def cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    integral = np.zeros_like(x, dtype=float)
    integral[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return integral


def volume_profile(state, rho_targets: np.ndarray) -> np.ndarray:
    rho_grid = np.linspace(1.0e-4, 1.0, VOLUME_SAMPLES)
    data = state.evaluate(
        "dV_dPhi_n",
        "dPhi_dr",
        "Phi_edge",
        rho=rho_grid,
        theta="int",
        zeta="int",
    )

    dV_dPhi = np.asarray(data.dV_dPhi_n, dtype=float).ravel()
    dPhi_dr = np.asarray(data.dPhi_dr, dtype=float).ravel()
    phi_edge = float(np.asarray(data.Phi_edge))
    dV_dr = dV_dPhi * dPhi_dr / phi_edge

    rho_with_axis = np.concatenate([[0.0], rho_grid])
    dV_dr_with_axis = np.concatenate([[0.0], dV_dr])
    volume = cumulative_trapezoid(rho_with_axis, dV_dr_with_axis)
    return np.interp(rho_targets, rho_with_axis, volume)


def compute_well_profiles(state, rho: np.ndarray) -> dict[str, np.ndarray | float]:
    data = state.evaluate(
        "Jac",
        "dJac_dr",
        "mod_B",
        "dmod_B_dr",
        "dV_dPhi_n",
        "vacuum_magnetic_well_depth",
        rho=rho,
        theta="int",
        zeta="int",
    )

    jac = data.Jac
    d_jac_dr = data.dJac_dr
    mod_b = data.mod_B
    dmod_b_dr = data.dmod_B_dr

    area_weight = np.asarray(gvec.fluxsurface_integral(jac), dtype=float).ravel()
    d_area_weight_dr = np.asarray(gvec.fluxsurface_integral(d_jac_dr), dtype=float).ravel()
    b2_num = np.asarray(gvec.fluxsurface_integral(mod_b**2 * jac), dtype=float).ravel()
    d_b2_num_dr = np.asarray(
        gvec.fluxsurface_integral((2.0 * mod_b * dmod_b_dr * jac) + (mod_b**2 * d_jac_dr)),
        dtype=float,
    ).ravel()

    b2_avg = b2_num / area_weight
    b2_avg_r = (d_b2_num_dr * area_weight - b2_num * d_area_weight_dr) / area_weight**2

    volume = volume_profile(state, rho)
    desc_style_well = volume * b2_avg_r / (area_weight * b2_avg)

    dV_dPhi = np.asarray(data.dV_dPhi_n, dtype=float).ravel()
    dV_axis = float(
        np.asarray(
            state.evaluate(
                "dV_dPhi_n",
                rho=np.array([1.0e-4]),
                theta="int",
                zeta="int",
            ).dV_dPhi_n
        ).ravel()[0]
    )
    cumulative_vacuum_well = (dV_axis - dV_dPhi) / dV_axis
    native_depth = float(np.asarray(data.vacuum_magnetic_well_depth))

    return {
        "desc_style_well": desc_style_well,
        "cumulative_vacuum_well": cumulative_vacuum_well,
        "native_vacuum_well_depth": native_depth,
        "V": volume,
        "V_r": area_weight,
        "B2_avg": b2_avg,
        "B2_avg_r": b2_avg_r,
        "dV_dPhi_n": dV_dPhi,
    }


def load_state(label: str):
    case = CASES[label]
    print(f"[{label}] loading {case['ini'].name}", flush=True)
    return gvec.load_state(str(case["ini"]), str(case["state"]))


def main() -> None:
    ensure_dirs()
    rho = np.asarray(RHO_VALUES, dtype=float)
    rows = []

    for label in CASE_LABELS:
        profiles = compute_well_profiles(load_state(label), rho)
        rows.append(profiles)
        comparison = np.asarray(profiles[COMPARISON_WELL_KIND], dtype=float)
        print(
            f"[{label}] {COMPARISON_WELL_KIND} min={np.nanmin(comparison):+.6e} "
            f"max={np.nanmax(comparison):+.6e}; "
            f"native edge depth={profiles['native_vacuum_well_depth']:+.6e}",
            flush=True,
        )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "compute_gvec_magnetic_well.py",
        "labels": list(CASE_LABELS),
        "rho": rho.tolist(),
        "volume_samples": VOLUME_SAMPLES,
        "comparison_well_kind": COMPARISON_WELL_KIND,
        "desc_style_formula": "V(r) * d<|B|^2>/dr / (V_r(r) * <|B|^2>), with p_r=0",
        "cumulative_formula": "(dV/dPhi_n(axis) - dV/dPhi_n(rho)) / dV/dPhi_n(axis)",
    }

    np.savez(
        OUTPUT,
        labels=object_array(CASE_LABELS),
        rho=rho,
        desc_style_well=np.asarray([row["desc_style_well"] for row in rows]),
        cumulative_vacuum_well=np.asarray([row["cumulative_vacuum_well"] for row in rows]),
        native_vacuum_well_depth=np.asarray([row["native_vacuum_well_depth"] for row in rows]),
        V=np.asarray([row["V"] for row in rows]),
        V_r=np.asarray([row["V_r"] for row in rows]),
        B2_avg=np.asarray([row["B2_avg"] for row in rows]),
        B2_avg_r=np.asarray([row["B2_avg_r"] for row in rows]),
        dV_dPhi_n=np.asarray([row["dV_dPhi_n"] for row in rows]),
        metadata=json.dumps(metadata, indent=2),
    )
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
