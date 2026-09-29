#!/usr/bin/env python3
"""Q_i vs a/LT for the best GVEC SLAM/SLNAM equilibria (09-23 resolution scans)
and the QI precursor (OP) with DESC->VMEC ns512->pyvmec geometry (same GX settings),
following the max-alpha rule in GVEC_GX_bkp/START_HERE_NEXT_SESSION.md:

- Q_i per run = mean of the last 100 diagnostics; time std of the same samples.
- At each a/LT, plot the alpha with the largest raw Q_i. Error bar = last-100
  time std of that alpha only (alpha spread is never an error bar).
- Plotted quantity is the GX-reported Q_i as is. No <grho> re-normalization
  (Q_i * <grho>_J from standardize_heat_flux_grho.py is wrong and not used).

Example:
    python plot_gx_heat_flux_scan_best.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


HERE = Path(__file__).resolve().parent
RUN_ROOT = HERE / "gx_runs" / "best_gvec_t600_alpha5"
QI_ROOT = HERE / "gx_runs" / "qi_precursor_pyvmec_ns512_t600_alpha5"
PER_RUN_OUT = RUN_ROOT / "heat_flux_per_fieldline.csv"
MAX_OUT = RUN_ROOT / "heat_flux_max_alpha.csv"
FIG_DIR = HERE / "figures"
STEM = "gx_itg_heat_flux_scan_rho0p650_npol5_nt641_t600_alpha5_best_max_alpha"
N_AVERAGE = 100

ALPHAS = ("alpha0p00", "alpha0p25pi", "alpha0p50pi", "alpha0p75pi", "alpha1p00pi")
A_OVER_LT_VALUES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
# (case, plot label, color)
CASES = (
    ("SLAM_BEST", "SLAM (best GVEC)", "tab:orange"),
    ("SLNAM_BEST", "SLNAM (best GVEC)", "tab:green"),
    ("QI_precursor", "QI precursor OP (DESC->VMEC->pyvmec)", "tab:blue"),
)

rows = []
for case, _, _ in CASES:
    for alpha in ALPHAS:
        for a_over_lt in A_OVER_LT_VALUES:
            tag = f"aLT{a_over_lt:.2f}".replace(".", "p")
            nc_path = (QI_ROOT if case == "QI_precursor" else RUN_ROOT) / case / alpha / tag / "gx.out.nc"
            with Dataset(nc_path) as nc:
                q = np.asarray(nc.groups["Diagnostics"].variables["HeatFlux_st"][:, 0], dtype=float)[-N_AVERAGE:]
            q_mean, q_std = float(q.mean()), float(q.std())
            rows.append({"case": case, "alpha": alpha, "a_over_LT": a_over_lt,
                         "Q_i_mean": q_mean, "Q_i_std_time": q_std,
                         "path": str(nc_path)})

with PER_RUN_OUT.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

max_rows = []
for case, *_ in CASES:
    for a_over_lt in A_OVER_LT_VALUES:
        data = [r for r in rows if r["case"] == case and r["a_over_LT"] == a_over_lt]
        bad = [r["alpha"] for r in data if not np.isfinite(r["Q_i_mean"])]
        if bad:  # the max over alpha is unknown when a field line went NaN: leave the point out
            print(f"{case} a/LT={a_over_lt:.1f}: NaN run(s) {bad}, point left out")
            continue
        max_rows.append(dict(max(data, key=lambda r: r["Q_i_mean"])))

with MAX_OUT.open("w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=max_rows[0].keys())
    writer.writeheader()
    writer.writerows(max_rows)

FIG_DIR.mkdir(parents=True, exist_ok=True)
for yscale in ("linear", "log"):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for case, label, color in CASES:
        data = [r for r in max_rows if r["case"] == case]
        x = np.array([r["a_over_LT"] for r in data])
        y = np.array([r["Q_i_mean"] for r in data])
        err = np.array([r["Q_i_std_time"] for r in data])
        m = y > 0 if yscale == "log" else np.ones_like(y, bool)
        ax.errorbar(x[m], y[m], yerr=err[m], marker="o", capsize=3, color=color, label=label)
    ax.set_title("Highest-flux alpha; error is last-100 time std")
    ax.set_ylabel(r"$Q_i / Q_{GB}$")
    ax.set_yscale(yscale)
    ax.set_xlabel(r"$a/L_{T_i}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / f"{STEM}_{yscale}.png"
    fig.savefig(out, dpi=220)
    print(f"wrote {out}")

print(f"wrote {PER_RUN_OUT}")
print(f"wrote {MAX_OUT}")
for r in max_rows:
    print(f"{r['case']:26s} a/LT={r['a_over_LT']:.1f} {r['alpha']:12s} Q={r['Q_i_mean']:8.3f} +/- {r['Q_i_std_time']:6.3f}")
