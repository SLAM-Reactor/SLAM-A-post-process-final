#!/usr/bin/env python3
"""Q(t) plots, one figure per equilibrium: SLAM_BEST, SLNAM_BEST (gx_runs/best_gvec_t600_alpha5)
and the QI precursor OP with DESC->VMEC ns512->pyvmec geometry (gx_runs/qi_precursor_pyvmec_ns512_t600_alpha5).

Example:
    python plot_gx_heat_flux_time_best.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "gx_runs" / "best_gvec_t600_alpha5"
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

A_OVER_LT_VALUES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
ALPHAS = ("alpha0p00", "alpha0p25pi", "alpha0p50pi", "alpha0p75pi", "alpha1p00pi")
QI_ROOT = HERE / "gx_runs" / "qi_precursor_pyvmec_ns512_t600_alpha5"
CASES = ("SLAM_BEST", "SLNAM_BEST", "QI_precursor")

for case in CASES:
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharex=False)
    axes = axes.ravel()
    for ia, a_over_lt in enumerate(A_OVER_LT_VALUES):
        ax = axes[ia]
        tag = f"aLT{a_over_lt:.2f}".replace(".", "p")
        for alpha in ALPHAS:
            nc_path = (QI_ROOT if case == "QI_precursor" else RUN_ROOT) / case / alpha / tag / "gx.out.nc"
            if not nc_path.exists():
                continue
            with Dataset(nc_path) as nc:
                q = np.asarray(nc.groups["Diagnostics"].variables["HeatFlux_st"][:, 0], dtype=float)
                t = np.asarray(nc.groups["Grids"].variables["time"][:], dtype=float)
            ax.plot(t, q, lw=1.2, label=alpha.replace("alpha", r"$\alpha=$"))
            ax.axvspan(t[max(0, len(t) - 100)], t[-1], color="k", alpha=0.035, lw=0)
        ax.set_title(rf"$a/L_T={a_over_lt:g}$")
        ax.set_xlabel(r"$t v_{th}/a$")
        ax.set_ylabel(r"$Q_i/Q_{GB}$")
        ax.grid(True, alpha=0.25)
    axes[-1].axis("off")
    axes[0].legend(loc="best", fontsize=8)
    title = case
    fig.suptitle(f"{title}: ion heat flux vs time; shaded region is last 100 diagnostics")
    fig.tight_layout()
    out = FIG_DIR / f"gx_itg_heat_flux_time_rho0p650_npol5_nt641_t600_alpha5_{case}.png"
    fig.savefig(out, dpi=180)
    print(f"wrote {out}")
