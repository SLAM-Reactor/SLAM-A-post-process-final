#!/usr/bin/env python3
"""Effective ripple: best GVEC equilibria (09-23 scans) vs DESC original.
Surfaces that did not finish are stored as NaN and are left out of the plot.
The old 09-17 GVEC ripple (effective_ripple_gvec.npz) is not plotted: it does not
reproduce with the current script on the same equilibrium (checked 09-24, job 58824260).

Run with the desc-env python (system numpy cannot unpickle the caches):
    ~/anaconda3/envs/desc-env/bin/python plot_effective_ripple_best.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter

POST_DIR = Path(__file__).resolve().parent
DATA_DIR = POST_DIR / "data"
FIGURE_DIR = POST_DIR / "figures"
# case -> Boozer MNfactor file tag. SLNAM_BEST at the GVEC default 5 (no tag); SLAM_BEST at 3
# because it runs out of memory at 5 (job 58800587). State the MNfactor in the caption.
CASES = {"SLNAM_BEST": "", "SLAM_BEST": "_mnf3"}
DESC_CACHE = DATA_DIR / "effective_ripple_desc_original.npz"
OUTPUT_STEM = FIGURE_DIR / "effective_ripple_comparison_best"

# ---------------------------------------------------------------- style (edit here)
FIG_SIZE = (4., 3)          # inches; ~3.4 single column, ~7.0 double column
FONT_FAMILY = "DejaVu Sans"   # matplotlib default
MATHTEXT_FONTSET = "dejavusans"  # matplotlib default math font
USE_TEX = False                # True renders all text through the system LaTeX

LABEL_FONTSIZE = 18
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 14
XLABEL_PAD = 1                 # points between tick labels and axis label
YLABEL_PAD = -2

XLABEL = r"$\rho$"
YLABEL = r"$\epsilon_{\mathrm{eff}}$"
XLIM = (0.0, 1.02)
YLIM = (2e-3, 1.5e-1)          # upper limit > 0.1 so the 10^-1 tick shows
XTICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

TICK_DIRECTION = "in"
MAJOR_TICK_LENGTH = 4.0
MINOR_TICK_LENGTH = 2.0
TICK_WIDTH = 0.7
AXES_LINEWIDTH = 0.8
SHOW_GRID = False
GRID_ALPHA = 0.25

LINE_WIDTH = 1.4
MARKER_SIZE = 4.5
COLORS = {"SLNAM": "#0057B8", "SLAM": "#D55E00", "DESC": "#000000"}
MARKERS = {"SLNAM": "o", "SLAM": "^", "DESC": "s"}
LEGEND_LABELS = {"SLNAM": "pre-SLAM", "SLAM": "SLAM", "DESC": "DESC OP"}

LEGEND_LOC = "center left"
LEGEND_BBOX = None             # e.g. (0.02, 0.55) to place by axes fraction
LEGEND_FRAME = False
LEGEND_HANDLELENGTH = 1.8
LEGEND_LABELSPACING = 0.3

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


def load_gvec(label, tag):
    rho_by = {}
    for path in sorted(DATA_DIR.glob(f"effective_ripple_gvec_{label}*.npz")):
        if (tag and not path.stem.endswith(tag)) or (not tag and "_mnf" in path.stem):
            continue
        d = np.load(path, allow_pickle=True)
        for r, e in zip(d["rho"], d["ripple"][0]):
            if np.isfinite(e):
                rho_by[round(float(r), 4)] = float(e)
    rho = np.array(sorted(rho_by))
    return rho, np.array([rho_by[r] for r in rho])


def plot_series(ax, key, rho, ripple):
    ax.plot(rho, ripple, "-", marker=MARKERS[key], lw=LINE_WIDTH, ms=MARKER_SIZE,
            color=COLORS[key], label=LEGEND_LABELS[key])


fig, ax = plt.subplots(figsize=FIG_SIZE)

for label, tag in CASES.items():
    rho, ripple = load_gvec(label, tag)
    if rho.size == 0:
        print(f"{label}: no finished surfaces, skipped", flush=True)
        continue
    plot_series(ax, label.split("_")[0], rho, ripple)
    for r, e in zip(rho, ripple):
        print(f"{label:11s} rho={r:.2f} eff ripple={e:.4e}", flush=True)

desc = np.load(DESC_CACHE, allow_pickle=True)
plot_series(ax, "DESC", desc["rho"], desc["ripple_desc_original"])

ax.set_yscale("log")
ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_xticks(XTICKS)
ax.yaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
ax.yaxis.set_minor_formatter(NullFormatter())

ax.set_xlabel(XLABEL, fontsize=LABEL_FONTSIZE, labelpad=XLABEL_PAD)
ax.set_ylabel(YLABEL, fontsize=LABEL_FONTSIZE, labelpad=YLABEL_PAD)
ax.tick_params(which="both", direction=TICK_DIRECTION, top=True, right=True,
               width=TICK_WIDTH, labelsize=TICK_FONTSIZE)
ax.tick_params(which="major", length=MAJOR_TICK_LENGTH)
ax.tick_params(which="minor", length=MINOR_TICK_LENGTH)
if SHOW_GRID:
    ax.grid(alpha=GRID_ALPHA, which="major")

ax.legend(loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX, frameon=LEGEND_FRAME,
          fontsize=LEGEND_FONTSIZE, handlelength=LEGEND_HANDLELENGTH,
          labelspacing=LEGEND_LABELSPACING)

fig.tight_layout()
for fmt in SAVE_FORMATS:
    fig.savefig(f"{OUTPUT_STEM}.{fmt}", dpi=PNG_DPI if fmt == "png" else None, bbox_inches="tight")
    print(f"wrote {OUTPUT_STEM}.{fmt}", flush=True)
