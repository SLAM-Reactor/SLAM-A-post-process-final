#!/usr/bin/env python3
"""Publication-style GX heat-flux figures for SLAM, pre-SLAM, and the OP precursor.

- Q_i per run = mean of the last 100 diagnostics; time std of the same samples.
- The alpha panel uses all field lines at a/LT = 3.
- The gradient panel uses the alpha with the largest raw Q_i at each a/LT.
- Plotted quantity is the GX-reported Q_i as is. No <grho> re-normalization.

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
T1200_ROOT = HERE / "gx_runs" / "best_gvec_t1200_alpha5"
PER_RUN_OUT = RUN_ROOT / "heat_flux_per_fieldline.csv"
MAX_OUT = RUN_ROOT / "heat_flux_max_alpha.csv"
FIG_DIR = HERE / "figures"
N_AVERAGE = 100
ALPHA_SCAN_A_OVER_LT = 3.0

# Paper-style formatting, matching Figure 9 in SLAM_A_paper (6).pdf.
FIG_SIZE = (4.0, 3.0)
LABEL_SIZE = 20
TICK_SIZE = 16
LEGEND_SIZE = 14
MARKER_SIZE = 3.2
CAP_SIZE = 2.2
LINE_WIDTH = 1.25
DPI = 600

ALPHAS = {
    "alpha0p00": 0.0,
    "alpha0p25pi": 0.25,
    "alpha0p50pi": 0.5,
    "alpha0p75pi": 0.75,
    "alpha1p00pi": 1.0,
}
ALPHA_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)
ALPHA_TICK_LABELS = ("0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$")
A_OVER_LT_VALUES = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
# (case, plot label, color, marker)
CASES = (
    ("SLNAM_BEST", "pre-SLAM", "#0057B8", "o"),
    ("SLAM_BEST", "SLAM", "#D55E00", "^"),
    ("QI_precursor", "DESC OP", "#000000", "s"),
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def output_path(case, alpha, tag, a_over_lt):
    if case == "QI_precursor":
        return QI_ROOT / case / alpha / tag / "gx.out.nc"
    if a_over_lt == 1.25:
        return T1200_ROOT / case / alpha / tag / "gx.out.nc"
    return RUN_ROOT / case / alpha / tag / "gx.out.nc"


def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, labelpad=1)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, labelpad=-2)
    ax.tick_params(which="both", direction="in", top=True, right=True, width=0.7, labelsize=TICK_SIZE)
    ax.tick_params(which="major", length=4.0)
    ax.tick_params(which="minor", length=2.0)


def save(fig, stem):
    fig.tight_layout()
    for fmt in ("pdf", "png"):
        path = f"{stem}.{fmt}"
        fig.savefig(path, dpi=DPI if fmt == "png" else None, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


rows = []
for case, *_ in CASES:
    for alpha in ALPHAS:
        for a_over_lt in A_OVER_LT_VALUES:
            tag = f"aLT{a_over_lt:.2f}".replace(".", "p")
            nc_path = output_path(case, alpha, tag, a_over_lt)
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
fig_alpha, ax_alpha = plt.subplots(figsize=FIG_SIZE)
fig_grad, ax_grad = plt.subplots(figsize=FIG_SIZE)

for case, label, color, marker in CASES:
    data = [r for r in rows if r["case"] == case and r["a_over_LT"] == ALPHA_SCAN_A_OVER_LT]
    x = np.array([ALPHAS[r["alpha"]] for r in data])
    y = np.array([r["Q_i_mean"] for r in data])
    err = np.array([r["Q_i_std_time"] for r in data])
    ax_alpha.errorbar(x, y, yerr=err, marker=marker, markersize=MARKER_SIZE, lw=LINE_WIDTH,
                      capsize=CAP_SIZE, color=color, label=label)
ax_alpha.set_xticks(ALPHA_TICKS, ALPHA_TICK_LABELS)
ax_alpha.set_xlim(-0.05, 1.05)
ax_alpha.set_ylim(bottom=0.0)
style_axes(ax_alpha, r"$\alpha$", r"$Q_i/Q_{\mathrm{GB}}$")
ax_alpha.legend(loc="upper left", frameon=False, fontsize=LEGEND_SIZE, handlelength=1.8, labelspacing=0.3)

for case, label, color, marker in CASES:
    data = [r for r in max_rows if r["case"] == case]
    x = np.array([r["a_over_LT"] for r in data])
    y = np.array([r["Q_i_mean"] for r in data])
    err = np.array([r["Q_i_std_time"] for r in data])
    ax_grad.errorbar(x, y, yerr=err, marker=marker, markersize=MARKER_SIZE, lw=LINE_WIDTH,
                     capsize=CAP_SIZE, color=color, label=label)
ax_grad.set_xlim(0.9, 3.1)
ax_grad.set_ylim(bottom=0.0)
style_axes(ax_grad, r"$a/L_{T_i}$", r"$Q_i/Q_{\mathrm{GB}}$")
ax_grad.legend(loc="upper left", frameon=False, fontsize=LEGEND_SIZE, handlelength=1.8, labelspacing=0.3)

save(fig_alpha, FIG_DIR / "gx_Q_vs_alpha_aLT3p0")
save(fig_grad, FIG_DIR / "gx_Q_vs_aLT_max_alpha")

print(f"wrote {PER_RUN_OUT}")
print(f"wrote {MAX_OUT}")
for r in max_rows:
    print(f"{r['case']:26s} a/LT={r['a_over_LT']:g} {r['alpha']:12s} Q={r['Q_i_mean']:8.3f} +/- {r['Q_i_std_time']:6.3f}")
