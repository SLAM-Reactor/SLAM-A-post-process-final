#!/usr/bin/env python3
"""Publication GX figures for SLAM_BEST, SLNAM_BEST (pre-SLAM) and the QI precursor (DESC OP):

1. Q_i vs field-line label alpha at a/LT = A_OVER_LT_ALPHA_SCAN.
2. Q_i vs a/LT, highest-flux alpha at each a/LT (max-alpha rule of plot_gx_heat_flux_scan_best.py).
3. Geometric coefficients along the field line GEOMETRY_ALPHA, theta in [-2pi, 2pi].

Q_i is the time-weighted mean of HeatFlux_st over t >= T_AVG_START; the error bar is the
time-weighted std over the same window. A fixed time window is used rather than the last 100
diagnostics because GX writes every nwrite steps with an adaptive dt, so 100 samples span
t ~ 105-600 in some SLAM runs and ~500-600 in others.

Geometry "uncompression": GX stores coefficients on a compressed, equal-arc grid theta in
[-pi, pi] (constant gradpar, theta_physical ~ scale * theta with scale = npol = 5). The
.eik.nc files also store the physical field-line angle at every grid point as theta_PEST
(for the GVEC exports this is Boozer theta, non-uniform on the equal-arc grid; for the VMEC
QI export it is PEST theta = scale * theta exactly). Coefficients are plotted against
theta_PEST, which undoes the compression.

Run with the desc-env python:
    ~/anaconda3/envs/desc-env/bin/python plot_gx_publication.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"
RUN_ROOTS = {
    "SLNAM_BEST": HERE / "gx_runs" / "best_gvec_t600_alpha5",
    "SLAM_BEST": HERE / "gx_runs" / "best_gvec_t600_alpha5",
    "QI_precursor": HERE / "gx_runs" / "qi_precursor_pyvmec_ns512_t600_alpha5",
}
GEOMETRY_FILES = {
    "SLNAM_BEST": "gx_geometry_gvec_SLNAM_BEST_rho0p650_npol5_nt641_{alpha}.eik.nc",
    "SLAM_BEST": "gx_geometry_gvec_SLAM_BEST_rho0p650_npol5_nt641_{alpha}.eik.nc",
    "QI_precursor": "gx_geometry_vmec_QI_precursor_ns512_rho0p650_npol5_nt641_{alpha}.eik.nc",
}
CASE_KEY = {"SLNAM_BEST": "SLNAM", "SLAM_BEST": "SLAM", "QI_precursor": "DESC"}

ALPHAS = {"alpha0p00": 0.0, "alpha0p25pi": 0.25, "alpha0p50pi": 0.5, "alpha0p75pi": 0.75, "alpha1p00pi": 1.0}
A_OVER_LT_VALUES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
T_AVG_START = 300.0            # time-average window t >= T_AVG_START (t_max = 600)

A_OVER_LT_ALPHA_SCAN = 3.0     # figure 1
GEOMETRY_ALPHA = "alpha0p00"   # figure 3 field line; any key of ALPHAS
THETA_RANGE = (-np.pi, np.pi)
# (eik.nc variable, y label, y scale), left to right; cvdrift == gbdrift for these beta = 0 equilibria
GEOMETRY_PANELS = (
    ("bmag", r"$B/B_{\mathrm{ref}}$", "linear"),
    ("cvdrift", r"$\mathrm{cvdrift}$", "linear"),
    ("gds2", r"$\mathrm{gds2}$", "linear"),
)

# ---------------------------------------------------------------- style (edit here)
FIG_SIZE = (4., 3)             # figures 1 and 2; inches
GEOMETRY_FIG_SIZE = (13., 3.6)  # figure 3, panels side by side, one per GEOMETRY_PANELS entry
FONT_FAMILY = "DejaVu Sans"    # matplotlib default
MATHTEXT_FONTSET = "dejavusans"  # matplotlib default math font
USE_TEX = False                # True renders all text through the system LaTeX

LABEL_FONTSIZE = 20
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 14
XLABEL_PAD = 1                 # points between tick labels and axis label
YLABEL_PAD = -2
GEOMETRY_YLABEL_PAD = 2

Q_LABEL = r"$Q_i/Q_{\mathrm{GB}}$"
ALPHA_LABEL = r"$\alpha$"
A_OVER_LT_LABEL = r"$a/L_{T_i}$"
THETA_LABEL = r"$\theta$"
ALPHA_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)   # in units of pi
ALPHA_TICK_LABELS = ("0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$")
THETA_TICKS = (-1, -0.5, 0, 0.5, 1)         # in units of pi
THETA_TICK_LABELS = (r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$")
A_OVER_LT_XLIM = (0.8, 4.2)
ALPHA_SCAN_YLIM = (0.0, 95.0)                # headroom for the legend
Q_VS_A_OVER_LT_YSCALE = "linear"            # "linear" or "log" (log drops Q <= 0 points)

TICK_DIRECTION = "in"
MAJOR_TICK_LENGTH = 4.0
MINOR_TICK_LENGTH = 2.0
TICK_WIDTH = 0.7
AXES_LINEWIDTH = 0.8
SHOW_GRID = False
GRID_ALPHA = 0.25

LINE_WIDTH = 1.4
GEOMETRY_LINE_WIDTH = 1.2
MARKER_SIZE = 4.5
CAPSIZE = 2.5
COLORS = {"SLNAM": "#0057B8", "SLAM": "#D55E00", "DESC": "#000000"}
MARKERS = {"SLNAM": "o", "SLAM": "^", "DESC": "s"}
LEGEND_LABELS = {"SLNAM": "pre-SLAM", "SLAM": "SLAM", "DESC": "DESC OP"}

LEGEND_LOC_ALPHA_SCAN = "upper left"
LEGEND_LOC_A_OVER_LT = "upper left"
LEGEND_LOC_GEOMETRY = "lower center"  # anchored above the middle geometry panel by LEGEND_BBOX_GEOMETRY
LEGEND_BBOX_GEOMETRY = (0.5, 1.0)
LEGEND_NCOL_GEOMETRY = 3
LEGEND_COLUMNSPACING = 1.0
LEGEND_FRAME = False
LEGEND_HANDLELENGTH = 1.8
LEGEND_LABELSPACING = 0.3

OUTPUT_ALPHA_SCAN = FIG_DIR / "gx_Q_vs_alpha_aLT3p0"
OUTPUT_A_OVER_LT = FIG_DIR / "gx_Q_vs_aLT_max_alpha"
OUTPUT_GEOMETRY = FIG_DIR / "gx_geometry_coefficients"
SAVE_FORMATS = ("pdf", "png")
PNG_DPI = 600
# ----------------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "mathtext.fontset": MATHTEXT_FONTSET,
    "text.usetex": USE_TEX,
    "axes.linewidth": AXES_LINEWIDTH,
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


def time_averaged_q(case, alpha, a_over_lt):
    tag = f"aLT{a_over_lt:.2f}".replace(".", "p")
    with Dataset(RUN_ROOTS[case] / case / alpha / tag / "gx.out.nc") as nc:
        q = np.asarray(nc.groups["Diagnostics"].variables["HeatFlux_st"][:, 0], dtype=float)
        t = np.asarray(nc.groups["Grids"].variables["time"][:], dtype=float)
    keep = t >= T_AVG_START
    t, q = t[keep], q[keep]
    span = t[-1] - t[0]
    mean = np.trapezoid(q, t) / span
    return mean, np.sqrt(np.trapezoid((q - mean) ** 2, t) / span)


def style_axes(ax, xlabel, ylabel, ylabel_pad=YLABEL_PAD, legend_loc=None, **legend_kwargs):
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, labelpad=XLABEL_PAD)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, labelpad=ylabel_pad)
    ax.tick_params(which="both", direction=TICK_DIRECTION, top=True, right=True,
                   width=TICK_WIDTH, labelsize=TICK_FONTSIZE)
    ax.tick_params(which="major", length=MAJOR_TICK_LENGTH)
    ax.tick_params(which="minor", length=MINOR_TICK_LENGTH)
    if SHOW_GRID:
        ax.grid(alpha=GRID_ALPHA, which="major")
    if legend_loc is not None:
        ax.legend(loc=legend_loc, frameon=LEGEND_FRAME, fontsize=LEGEND_FONTSIZE,
                  handlelength=LEGEND_HANDLELENGTH, labelspacing=LEGEND_LABELSPACING,
                  columnspacing=LEGEND_COLUMNSPACING, **legend_kwargs)


def save(fig, stem):
    fig.tight_layout()
    for fmt in SAVE_FORMATS:
        fig.savefig(f"{stem}.{fmt}", dpi=PNG_DPI if fmt == "png" else None, bbox_inches="tight")
        print(f"wrote {stem}.{fmt}", flush=True)
    plt.close(fig)


FIG_DIR.mkdir(parents=True, exist_ok=True)
q_table = {(case, alpha, a): time_averaged_q(case, alpha, a)
           for case in RUN_ROOTS for alpha in ALPHAS for a in A_OVER_LT_VALUES}

# 1. Q_i vs alpha at fixed a/LT
fig, ax = plt.subplots(figsize=FIG_SIZE)
for case in RUN_ROOTS:
    key = CASE_KEY[case]
    x = np.array(list(ALPHAS.values()))
    q, err = np.array([q_table[(case, alpha, A_OVER_LT_ALPHA_SCAN)] for alpha in ALPHAS]).T
    ax.errorbar(x, q, yerr=err, marker=MARKERS[key], ms=MARKER_SIZE, lw=LINE_WIDTH, capsize=CAPSIZE,
                color=COLORS[key], label=LEGEND_LABELS[key])
    print(f"a/LT={A_OVER_LT_ALPHA_SCAN:g} {case:13s} " + " ".join(f"{v:7.2f}" for v in q), flush=True)
ax.set_xticks(ALPHA_TICKS, ALPHA_TICK_LABELS)
ax.set_ylim(*ALPHA_SCAN_YLIM)
style_axes(ax, ALPHA_LABEL, Q_LABEL, legend_loc=LEGEND_LOC_ALPHA_SCAN)
save(fig, OUTPUT_ALPHA_SCAN)

# 2. Q_i vs a/LT, highest-flux alpha at each a/LT
fig, ax = plt.subplots(figsize=FIG_SIZE)
for case in RUN_ROOTS:
    key = CASE_KEY[case]
    best = [max(ALPHAS, key=lambda alpha: q_table[(case, alpha, a)][0]) for a in A_OVER_LT_VALUES]
    q, err = np.array([q_table[(case, alpha, a)] for alpha, a in zip(best, A_OVER_LT_VALUES)]).T
    x = np.array(A_OVER_LT_VALUES)
    keep = q > 0 if Q_VS_A_OVER_LT_YSCALE == "log" else np.ones_like(q, bool)
    ax.errorbar(x[keep], q[keep], yerr=err[keep], marker=MARKERS[key], ms=MARKER_SIZE, lw=LINE_WIDTH,
                capsize=CAPSIZE, color=COLORS[key], label=LEGEND_LABELS[key])
    for a, alpha, v, e in zip(A_OVER_LT_VALUES, best, q, err):
        print(f"{case:13s} a/LT={a:.1f} {alpha:12s} Q={v:8.3f} +/- {e:6.3f}", flush=True)
ax.set_yscale(Q_VS_A_OVER_LT_YSCALE)
ax.set_xlim(*A_OVER_LT_XLIM)
style_axes(ax, A_OVER_LT_LABEL, Q_LABEL, legend_loc=LEGEND_LOC_A_OVER_LT)
save(fig, OUTPUT_A_OVER_LT)

# 3. Geometric coefficients vs physical (uncompressed) field-line angle
fig, axes = plt.subplots(1, len(GEOMETRY_PANELS), figsize=GEOMETRY_FIG_SIZE)
for case in RUN_ROOTS:
    key = CASE_KEY[case]
    with Dataset(DATA_DIR / GEOMETRY_FILES[case].format(alpha=GEOMETRY_ALPHA)) as nc:
        theta = np.asarray(nc.variables["theta_PEST"][:], dtype=float)
        shown = (theta >= THETA_RANGE[0]) & (theta <= THETA_RANGE[1])  # so y autoscale fits the plotted range
        for ax, (name, *_) in zip(axes, GEOMETRY_PANELS):
            ax.plot(theta[shown], np.asarray(nc.variables[name][:], dtype=float)[shown], lw=GEOMETRY_LINE_WIDTH,
                    color=COLORS[key], label=LEGEND_LABELS[key])
for i, (ax, (_, ylabel, yscale)) in enumerate(zip(axes, GEOMETRY_PANELS)):
    ax.set_yscale(yscale)
    ax.set_xlim(*THETA_RANGE)
    ax.set_xticks(np.array(THETA_TICKS) * np.pi, THETA_TICK_LABELS)
    legend_here = i == len(axes) // 2
    style_axes(ax, THETA_LABEL, ylabel, ylabel_pad=GEOMETRY_YLABEL_PAD,
               legend_loc=LEGEND_LOC_GEOMETRY if legend_here else None,
               **({"bbox_to_anchor": LEGEND_BBOX_GEOMETRY, "ncol": LEGEND_NCOL_GEOMETRY} if legend_here else {}))
save(fig, OUTPUT_GEOMETRY)
