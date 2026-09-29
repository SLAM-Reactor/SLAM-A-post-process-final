#!/usr/bin/env python3
"""Compute GVEC effective ripple for the selected comparison equilibria.

Edit the constants below to choose the case labels, rho grid, resolution, and
output cache. The default computes only the clean SLAM case; SLNAM already has
a valid per-case cache and is combined by ``combine_gvec_effective_ripple.py``.
"""

from __future__ import annotations

import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import gvec
import numpy as np

from paths import CASES, DATA_DIR, DEFAULT_RHO_PROFILE, ensure_dirs, object_array


# ---------------------------- calculation setup --------------------------- #

# case label and one rho (one surface per job; several surfaces in one job run out of memory), e.g.
#   python post-process/compute_gvec_effective_ripple.py SLNAM_BEST 0.6   (run from GVEC/)
# optional 3rd argument: Boozer MNfactor (GVEC default 5). SLAM_BEST (mn 24,32) runs out of
# memory at 5 even on one surface (job 58800587, ~490 GB), so it uses 3 (user decision 09-24):
#   python post-process/compute_gvec_effective_ripple.py SLAM_BEST 0.6 3
# A non-integer factor (e.g. 3.75, the SLAM_BEST resolution check) is passed to GVEC as
# M = int(factor*M_max), N = int(factor*N_max); for an integer factor this is the same M, N as MNfactor.
CASE_LABELS = (sys.argv[1],)
RHO_VALUES = np.array([float(sys.argv[2])])
MNFACTOR = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
MNF_TAG = f"_mnf{sys.argv[3].replace('.', 'p')}" if len(sys.argv) > 3 else ""
OUTPUT = DATA_DIR / f"effective_ripple_gvec_{sys.argv[1]}_rho{sys.argv[2].replace('.', 'p')}{MNF_TAG}.npz"

TURNS = 10.0
NZ = 751
NPITCH = 27
NUM_WELL = 250
RIPPLE_SOURCE = Path("/pscratch/sd/r/rgaur/GVEC_benchmark/GVEC_eps_eff")
sys.path.insert(0, str(RIPPLE_SOURCE))

from _bounce_points import bounce_quad_to_zeta  # noqa: E402
from _find_bounce_pairs import build_spline_and_bounce_points_rho  # noqa: E402
from _quad_utils import gc2  # noqa: E402
from ripple_calculation import (  # noqa: E402
    compute_HI_pairs,
    compute_L_rho,
    pitch_integral_simpson,
)


def scan_max_mod_b(state, rho_value: float) -> float:
    data = state.evaluate(
        "mod_B",
        rho=np.array([rho_value], dtype=float),
        theta="int",
        zeta="int",
    )
    return float(np.max(np.asarray(data.mod_B)))


def flux_surface_iota(state, rho_value: float) -> float:
    data = state.evaluate(
        "iota",
        rho=np.array([rho_value], dtype=float),
        theta=None,
        zeta=None,
    )
    return float(np.asarray(data.iota).ravel()[0])


def flux_surface_geometry(state, rho_value: float) -> tuple[float, float]:
    data = state.evaluate(
        "mod_grad_rho",
        "Jac",
        "pos",
        rho=np.array([rho_value], dtype=float),
        theta="int",
        zeta="int",
    )
    jac_integral = gvec.fluxsurface_integral(data.Jac)
    grad_rho_avg = float(
        np.asarray(gvec.fluxsurface_integral(data.mod_grad_rho * data.Jac) / jac_integral).squeeze()
    )
    major_radius = gvec.fluxsurface_integral(
        np.sqrt(data.pos[0] ** 2 + data.pos[1] ** 2) * data.Jac
    ) / jac_integral
    return grad_rho_avg, float(np.asarray(major_radius).squeeze())


def evaluate_boozer_line(state, rho_value: float, iota: float, zeta_b: np.ndarray):
    alpha = np.linspace(0.0, 2.0 * np.pi, 1, endpoint=False)
    theta_b = alpha[None, :, None] + iota * zeta_b[None, None, :]
    m_max, n_max = state.get_mn_max()
    m_booz, n_booz = int(MNFACTOR * m_max), int(MNFACTOR * n_max)
    print(f"    Boozer M, N = {m_booz}, {n_booz} (factor {MNFACTOR} x ({m_max}, {n_max}))", flush=True)

    ev = gvec.EvaluationsBoozer(
        rho=np.array([rho_value], dtype=float),
        theta_B=theta_b,
        zeta_B=zeta_b,
        state=state,
        M=m_booz,
        N=n_booz,
    )
    ev["alpha"] = ("pol", alpha)
    ev = ev.set_coords("alpha").set_xindex("alpha")
    state.compute(
        ev,
        "dmod_B_dz_B",
        "mod_B",
        "kappa_G",
        "mod_grad_rho",
        "B_contra_z_B",
    )
    return ev.sel(rho=rho_value)


def compute_ripple_at_rho(state, rho_value: float, zeta_b: np.ndarray, x_quad, w_quad) -> tuple[float, float, float]:
    b0 = scan_max_mod_b(state, rho_value)
    print(f"    B0 scan done: {b0:.5f}", flush=True)
    iota = flux_surface_iota(state, rho_value)
    print(f"    iota done: {iota:+.6f}", flush=True)
    grad_rho_avg, major_radius = flux_surface_geometry(state, rho_value)
    print(
        f"    geometry done: <|grad rho|>={grad_rho_avg:.6e}, R0={major_radius:.6e}",
        flush=True,
    )
    ev = evaluate_boozer_line(state, rho_value, iota, zeta_b)
    print("    Boozer line done", flush=True)

    b_line = np.asarray(ev.mod_B.data).squeeze()
    bz_line = np.asarray(ev.dmod_B_dz_B.data).squeeze()
    b_sup_zeta_line = np.asarray(ev.B_contra_z_B.data).squeeze()
    kappa_g_line = np.asarray(ev.kappa_G.data).squeeze()
    mod_grad_rho_line = np.asarray(ev.mod_grad_rho.data).squeeze()

    bmin = float(b_line.min())
    bmax = float(b_line.max())
    margin = 0.01 * (bmax - bmin)
    pitch_inv = np.linspace(bmin + margin, bmax - margin, NPITCH)[None, :]

    z1, z2 = build_spline_and_bounce_points_rho(
        zeta=zeta_b,
        B=b_line[None, :],
        B_z=bz_line[None, :],
        pitch_inv=pitch_inv,
        num_well=NUM_WELL,
    )
    pair_mask, pair_idx, zeta_pairs, w_pairs, _, _ = bounce_quad_to_zeta(
        z1,
        z2,
        x_quad,
        w_quad,
        return_flat=True,
    )
    del pair_mask

    h_pairs, i_pairs = compute_HI_pairs(
        B_pairs=np.interp(zeta_pairs, zeta_b, b_line),
        Bzeta_pairs=np.interp(zeta_pairs, zeta_b, b_sup_zeta_line),
        grad_rho_pairs=np.interp(zeta_pairs, zeta_b, mod_grad_rho_line),
        kappa_g_pairs=np.interp(zeta_pairs, zeta_b, kappa_g_line),
        w_pairs=w_pairs,
        pair_idx=pair_idx,
        pitch_inv=pitch_inv,
        psi_r=0.0,
    )

    s_rp = np.zeros((1, NPITCH))
    contribution = np.where(i_pairs != 0.0, h_pairs**2 / i_pairs, 0.0)
    np.add.at(s_rp, (pair_idx[:, 0], pair_idx[:, 1]), contribution)

    l_rho = compute_L_rho(zeta_b, b_sup_zeta_line[None, :])
    eps32 = (
        pitch_integral_simpson(s_rp, pitch_inv, l_rho)
        * (b0 * major_radius / grad_rho_avg) ** 2
        * (np.pi / (8.0 * np.sqrt(2.0)))
    )
    ripple = float(np.asarray(eps32 ** (2.0 / 3.0)).squeeze())
    return ripple, iota, b0


def save_cache(
    labels: tuple[str, ...],
    rho_values: np.ndarray,
    ripple_rows: list[np.ndarray],
    iota_rows: list[np.ndarray],
    b0_rows: list[np.ndarray],
    partial: bool,
) -> None:
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "compute_gvec_effective_ripple.py",
        "labels": list(labels),
        "rho": rho_values.tolist(),
        "source": str(RIPPLE_SOURCE),
        "nz": NZ,
        "npitch": NPITCH,
        "num_well": NUM_WELL,
        "turns": TURNS,
        "boozer_factor": MNFACTOR,
        "partial": partial,
        "method_note": "GVEC benchmark path with psi_r=0.0 and required Boozer fields only",
    }
    np.savez(
        OUTPUT,
        labels=object_array(labels),
        rho=rho_values,
        ripple=np.asarray(ripple_rows),
        iota=np.asarray(iota_rows),
        B0=np.asarray(b0_rows),
        metadata=json.dumps(metadata, indent=2),
    )


def compute_case(label: str, rho_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case = CASES[label]
    print(f"[{label}] loading {case['ini'].name}", flush=True)
    state = gvec.load_state(str(case["ini"]), str(case["state"]))
    print(f"[{label}] state loaded", flush=True)

    ripple = np.full(rho_values.size, np.nan)
    iota = np.full(rho_values.size, np.nan)
    b0 = np.full(rho_values.size, np.nan)

    zeta_b = np.linspace(-TURNS * np.pi, TURNS * np.pi, NZ)
    x_quad, w_quad = gc2(19)

    for index, rho_value in enumerate(rho_values):
        print(f"  rho={rho_value:.3f} starting", flush=True)
        ripple[index], iota[index], b0[index] = compute_ripple_at_rho(
            state,
            float(rho_value),
            zeta_b,
            x_quad,
            w_quad,
        )
        print(
            f"  rho={rho_value:.3f} iota={iota[index]:+.6f} "
            f"B0={b0[index]:.5f} eff ripple={ripple[index]:.6e}",
            flush=True,
        )
        save_cache((label,), rho_values, [ripple], [iota], [b0], partial=True)
        gc.collect()

    return ripple, iota, b0


def main() -> None:
    ensure_dirs()
    rho_values = np.asarray(RHO_VALUES, dtype=float)

    ripple_rows = []
    iota_rows = []
    b0_rows = []
    for label in CASE_LABELS:
        ripple, iota, b0 = compute_case(label, rho_values)
        ripple_rows.append(ripple)
        iota_rows.append(iota)
        b0_rows.append(b0)

    save_cache(CASE_LABELS, rho_values, ripple_rows, iota_rows, b0_rows, partial=False)
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
