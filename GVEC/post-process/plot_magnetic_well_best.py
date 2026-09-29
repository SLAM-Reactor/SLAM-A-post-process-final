#!/usr/bin/env python3
"""Plot the magnetic-well comparison from cached NPZ files.

The main comparison uses ``desc_style_well`` for GVEC and DESC's native
``magnetic well`` cache. GVEC's cumulative/native vacuum-well convention is a
different diagnostic and is only plotted when SHOW_NATIVE_GVEC_DIAGNOSTIC=True.

Run with the desc-env python (system numpy cannot unpickle the caches):
    ~/anaconda3/envs/desc-env/bin/python plot_magnetic_well_best.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FuncFormatter


# ----------------------------- input/output ------------------------------ #

POST_DIR = Path(__file__).resolve().parent
DATA_DIR = POST_DIR / "data"
FIGURE_DIR = POST_DIR / "figures"

GVEC_CACHE = DATA_DIR / "magnetic_well_gvec_best.npz"
DESC_CACHE = DATA_DIR / "magnetic_well_desc_original.npz"
OUTPUT_STEM = FIGURE_DIR / "magnetic_well_comparison_best"

WELL_KIND = "desc_style_well"
SHOW_NATIVE_GVEC_DIAGNOSTIC = False
MAX_RHO_TO_PLOT = 0.97

# ---------------------------------------------------------------- style (edit here)
FIG_SIZE = (4., 3)          # inches; ~3.4 single column, ~7.0 double column
FONT_FAMILY = "DejaVu Sans"    # matplotlib default
MATHTEXT_FONTSET = "dejavusans"  # matplotlib default math font
USE_TEX = False                # True renders all text through the system LaTeX

LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 14
XLABEL_PAD = 1                 # points between tick labels and axis label
YLABEL_PAD = 0                  # gap between y tick labels and the y label
YLABEL_ROTATION = 90           # 90 = matplotlib default (sideways, like the ripple plot); 0 = upright

Y_SCALE_EXP = -3               # y ticks in units of 10**Y_SCALE_EXP, factor shown in the y label; 0 = raw values
XLABEL = r"$\rho$"
# W = (V/<B^2>) d<2 mu0 p + B^2>/dV, Landreman & Jorge (2020) JPP 86, 905860510, eq. (3.1);
# this is DESC's "magnetic well". W > 0 is a stabilising well (equivalently V'' < 0).
YLABEL = rf"$W\ (\times 10^{{{Y_SCALE_EXP}}})$" if Y_SCALE_EXP else r"$W$"
DIAG_YLABEL = "cumulative vacuum well"
XLIM = (0.0, 1.02)
YLIM = None                    # e.g. (-14, 8); None = autoscale
XTICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
YTICKS = [-10e-3, -5e-3, 0.0, 5e-3]  # raw W values; None = auto

TICK_DIRECTION = "in"
MAJOR_TICK_LENGTH = 4.0
MINOR_TICK_LENGTH = 2.0
TICK_WIDTH = 0.7
AXES_LINEWIDTH = 0.8
SHOW_X_MINOR_TICKS = False     # ripple plot has none on x
SHOW_Y_MINOR_TICKS = True
SHOW_GRID = False
GRID_ALPHA = 0.25

LINE_WIDTH = 1.4
MARKER_SIZE = 4.5
MARKER_RHO = np.round(np.arange(0.1, 1.0, 0.1), 3)  # markers only at these rho; None = every point
ZERO_LINE = True
ZERO_LINE_COLOR = "0.6"
ZERO_LINE_WIDTH = 0.7

COLORS = {"SLNAM": "#0057B8", "SLAM": "#D55E00", "DESC": "#000000"}
MARKERS = {"SLNAM": "o", "SLAM": "^", "DESC": "s"}
LEGEND_LABELS = {"SLNAM": "pre-SLAM", "SLAM": "SLAM", "DESC": "DESC OP"}

LEGEND_LOC = "lower left"
LEGEND_BBOX = None             # e.g. (0.02, 0.05) to place by axes fraction
LEGEND_FRAME = False
LEGEND_HANDLELENGTH = 1.8
LEGEND_LABELSPACING = 0.3

SAVE_FORMATS = ("pdf", "png")
PNG_DPI = 600
# ----------------------------------------------------------------------------------

def plot_series(ax: plt.Axes, key: str, rho: np.ndarray, values: np.ndarray, label: str | None = None) -> None:
    markevery = None if MARKER_RHO is None else np.flatnonzero(np.isin(np.round(rho, 3), MARKER_RHO))
    ax.plot(rho, values, "-", marker=MARKERS[key], markevery=markevery, lw=LINE_WIDTH,
            ms=MARKER_SIZE, color=COLORS[key], label=label or LEGEND_LABELS[key])


def style_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlim(*XLIM)
    if YLIM is not None:
        ax.set_ylim(*YLIM)
    ax.set_xticks(XTICKS)
    if YTICKS is not None:
        ax.set_yticks(YTICKS)
    if SHOW_X_MINOR_TICKS:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if SHOW_Y_MINOR_TICKS:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: f"{value / 10.0**Y_SCALE_EXP:g}".replace("-", "\u2212")))
    if ZERO_LINE:
        ax.axhline(0.0, color=ZERO_LINE_COLOR, lw=ZERO_LINE_WIDTH, zorder=0)
    ax.set_xlabel(XLABEL, fontsize=LABEL_FONTSIZE, labelpad=XLABEL_PAD)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, labelpad=YLABEL_PAD, rotation=YLABEL_ROTATION,
                  ha="right" if YLABEL_ROTATION == 0 else "center",
                  va="center" if YLABEL_ROTATION == 0 else "bottom")
    ax.tick_params(which="both", direction=TICK_DIRECTION, top=True, right=True,
                   width=TICK_WIDTH, labelsize=TICK_FONTSIZE)
    ax.tick_params(which="major", length=MAJOR_TICK_LENGTH)
    ax.tick_params(which="minor", length=MINOR_TICK_LENGTH)
    if SHOW_GRID:
        ax.grid(alpha=GRID_ALPHA, which="major")
    ax.legend(loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX, frameon=LEGEND_FRAME,
              fontsize=LEGEND_FONTSIZE, handlelength=LEGEND_HANDLELENGTH,
              labelspacing=LEGEND_LABELSPACING)


def plot_desc_style_comparison(gvec, desc) -> plt.Figure:
    gvec_rho = np.asarray(gvec["rho"], dtype=float)
    desc_rho = np.asarray(desc["rho"], dtype=float)
    gvec_mask = gvec_rho <= MAX_RHO_TO_PLOT
    desc_mask = desc_rho <= MAX_RHO_TO_PLOT
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for index, label in enumerate(str(label) for label in gvec["labels"]):
        plot_series(ax, label.split("_")[0], gvec_rho[gvec_mask], gvec[WELL_KIND][index][gvec_mask])
    plot_series(ax, "DESC", desc_rho[desc_mask], desc["magnetic_well_desc_original"][desc_mask])

    style_axes(ax, YLABEL)
    fig.tight_layout()
    return fig


def plot_native_gvec_diagnostic(gvec) -> plt.Figure:
    gvec_rho = np.asarray(gvec["rho"], dtype=float)
    gvec_mask = gvec_rho <= MAX_RHO_TO_PLOT
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for index, label in enumerate(str(label) for label in gvec["labels"]):
        key = label.split("_")[0]
        plot_series(ax, key, gvec_rho[gvec_mask], gvec["cumulative_vacuum_well"][index][gvec_mask],
                    label=f"{LEGEND_LABELS[key]} cumulative")
        ax.axhline(gvec["native_vacuum_well_depth"][index], color=COLORS[key],
                   ls="--", lw=1.0, alpha=0.75)

    style_axes(ax, DIAG_YLABEL)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, stem, suffix: str = "") -> None:
    for fmt in SAVE_FORMATS:
        output = stem.parent / f"{stem.name}{suffix}.{fmt}"
        fig.savefig(output, dpi=PNG_DPI if fmt == "png" else None, bbox_inches="tight")
        print(f"wrote {output}", flush=True)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "mathtext.fontset": MATHTEXT_FONTSET,
        "text.usetex": USE_TEX,
        "axes.linewidth": AXES_LINEWIDTH,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
    })

    gvec = np.load(GVEC_CACHE, allow_pickle=True)
    desc = np.load(DESC_CACHE, allow_pickle=True)

    fig = plot_desc_style_comparison(gvec, desc)
    save_figure(fig, OUTPUT_STEM)
    plt.close(fig)

    if SHOW_NATIVE_GVEC_DIAGNOSTIC:
        fig = plot_native_gvec_diagnostic(gvec)
        save_figure(fig, OUTPUT_STEM, suffix="_gvec_native")
        plt.close(fig)


if __name__ == "__main__":
    main()
