#!/usr/bin/env python3
"""FTGen coordinate maps: one separate figure per quantity per chart.

Three charts, all with the toroidal angle on the x-axis:

    (phi,   alpha)        native traced grid, no spline
    (phi,   theta_PEST)
    (phi_B, theta_B)

Field lines are lines of constant alpha, so with alpha on the y-axis they are
HORIZONTAL and their k labels sit horizontally beside them.  k is the toroidal
transit number: paper eq. 10, alpha_k = 2*pi*iota*k (mod 2*pi) with alpha_0 = 0.
The data file stores alpha_sort, so the row index is not k; it is recovered by
inverting the same argsort, with an assert so a convention change fails loudly.

Everything is plotted from native tensor grids with no regridding.  X(alpha,phi)
is genuinely discontinuous across the phi seam at fixed alpha -- after one
transit the line sits at theta + 2*pi*iota, a different poloidal position -- so
nothing here wraps or smooths in phi.

No environment variables and no arguments: edit the knobs below and run
    python3 plot_FTGen_coordinate_maps.py
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ftgen")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, LinearLocator
from pathlib import Path

TWOPI = 2 * np.pi

# =============================================================================
# KNOBS
# =============================================================================

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "outfigures"

CASE = "1p0_150_0p85"

# One figure per quantity per chart.  Available: X, Y, Z, R, nu, B
QUANTITIES = ["X", "Y", "Z", "R", "nu"]

# Which charts to draw; the key is used in the output filename.
CHARTS = ["alpha", "pest", "boozer"]

# --- field lines (drawn on the (phi, alpha) chart only) ----------------------
N_FIELDLINES = 6
FIELDLINE_LINEWIDTH = 0.9
FIELDLINE_COLOR = "black"
FIELDLINE_LABEL_FONTSIZE = 20     # size of the "k = N" labels
FIELDLINE_LABEL_X = 0.015         # fraction of 2*pi, inset from the left edge
FIELDLINE_LABEL_DY = 0.012        # fraction of 2*pi, gap between label and line
FIELDLINE_LABEL_BOX = True        # white backing so labels stay readable
FIELDLINE_LABEL_BOX_ALPHA = 0.75

# --- contours / colour -------------------------------------------------------
N_CONTOUR_LEVELS = 80
USE_COMMON_COLOR_SCALE = True     # same colour scale across the three charts
CMAP = "viridis"
COLORBAR_EXTEND = "neither"

# --- type and spacing --------------------------------------------------------
N_AXIS_TICKS = 5
N_COLORBAR_TICKS = 5
TICK_FONTSIZE = 15
TICK_LENGTH = 5
TICK_WIDTH = 1.1
LABEL_FONTSIZE = 20
LABEL_PAD = 8
COLORBAR_LABEL_FONTSIZE = 20
COLORBAR_TICK_FONTSIZE = 15
COLORBAR_LABEL_PAD = 12
SHOW_COLORBAR_LABEL = False       # False -> bare colorbar, no title at all
COLORBAR_TICK_FORMAT = "%.2f"     # e.g. "%.2f", "%.3f", "%.2e", "%g"
AXES_LINEWIDTH = 1.1

FIGSIZE = (7.2, 5.4)
DPI = 300

# --- diagnostic: quantity vs phi at a few fixed alpha ------------------------
MAKE_LINECUT_FIGURE = True
LINECUT_ROWS = 4
LINECUT_LINEWIDTH = 1.3
LINECUT_LEGEND_FONTSIZE = 12
LINECUT_FIGSIZE = (7.5, 4.4)

# =============================================================================

FIG_DIR.mkdir(parents=True, exist_ok=True)

QUANTITY_LABEL = {
    "X": r"$X$ [m]", "Y": r"$Y$ [m]", "R": r"$R$ [m]", "Z": r"$Z$ [m]",
    "B": r"$|B|$ [T]", "nu": r"$\nu$ [rad]",
}

CHART_AXES = {                      # (x label, y label)
    "alpha": (r"$\phi$", r"$\alpha$"),
    "pest": (r"$\phi$", r"$\theta_{\mathrm{PEST}}$"),
    "boozer": (r"$\phi_B$", r"$\theta_B$"),
}


def pi_labels(ticks):
    out = []
    for t in ticks:
        m = t / np.pi
        if np.isclose(m, 0):
            out.append(r"$0$")
        elif np.isclose(m, 1):
            out.append(r"$\pi$")
        elif np.isclose(m, 2):
            out.append(r"$2\pi$")
        else:
            out.append(rf"${m:.2g}\pi$")
    return out


# ------------------------------------------------------------------ load once
a_npz = np.load(DATA_DIR / "alpha" / f"alpha_data_{CASE}.npz")
p_npz = np.load(DATA_DIR / "pest" / f"PEST_data_{CASE}.npz")
b_npz = np.load(DATA_DIR / "boozer" / f"Boozer_data_{CASE}.npz")

alpha = np.asarray(a_npz["alpha"]).ravel()
phi_alpha = np.asarray(a_npz["phi"]).ravel()
iota = float(a_npz["iota"])

# transit number k for each row of the sorted alpha grid
alpha_by_k = np.mod(TWOPI * iota * np.arange(len(alpha)), TWOPI)
k_of_row = np.argsort(alpha_by_k)
assert np.abs(alpha - alpha_by_k[k_of_row]).max() < 1e-9, \
    "stored alpha does not match mod(2*pi*iota*k, 2*pi)"

alpha_grid, phi_alpha_grid = np.meshgrid(alpha, phi_alpha, indexing="ij")
theta_pest_grid = np.asarray(p_npz["theta_PEST"])
phi_pest_grid = np.asarray(p_npz["phi_cyl"])
theta_booz_grid = np.asarray(b_npz["theta_booz"])
phi_booz_grid = np.asarray(b_npz["phi_booz"])

print(f"case {CASE}   iota {iota:.10f}")
print(f"  alpha {alpha_grid.shape}  PEST {theta_pest_grid.shape}"
      f"  Boozer {theta_booz_grid.shape}")


def values_for(q, chart):
    """Array for quantity q in the requested chart, or None if absent."""
    if chart == "alpha":
        key = {"X": "X_alpha", "Y": "Y_alpha", "R": "R_alpha", "Z": "Z_alpha",
               "B": "modB_alpha", "nu": "nu_alpha"}.get(q)
        return np.asarray(a_npz[key]) if key in a_npz else None
    if chart == "pest":
        if q == "X":
            return np.asarray(p_npz["R_PEST"]) * np.cos(phi_pest_grid)
        if q == "Y":
            return np.asarray(p_npz["R_PEST"]) * np.sin(phi_pest_grid)
        key = {"R": "R_PEST", "Z": "Z_PEST", "B": "modB_PEST",
               "nu": "nu_PEST"}.get(q)
        return np.asarray(p_npz[key]) if key in p_npz else None
    key = {"X": "X_booz", "Y": "Y_booz", "R": "R_booz", "Z": "Z_booz",
           "B": "modB_booz", "nu": "nu_booz"}.get(q)
    return np.asarray(b_npz[key]) if key in b_npz else None


GRIDS = {
    "alpha": (phi_alpha_grid, alpha_grid),
    "pest": (phi_pest_grid, theta_pest_grid),
    "boozer": (phi_booz_grid, theta_booz_grid),
}

for q in QUANTITIES:
    vals = {c: values_for(q, c) for c in CHARTS}
    have = {c: v for c, v in vals.items() if v is not None}
    if not have:
        print(f"  [skip] {q}: not present in any file")
        continue

    if USE_COMMON_COLOR_SCALE:
        vmin = min(np.nanmin(v) for v in have.values())
        vmax = max(np.nanmax(v) for v in have.values())
        levels = np.linspace(vmin, vmax, N_CONTOUR_LEVELS)
    else:
        levels = N_CONTOUR_LEVELS

    for chart, v in have.items():
        xg, yg = GRIDS[chart]
        xlabel, ylabel = CHART_AXES[chart]

        fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
        cf = ax.contourf(xg, yg, v, levels=levels, cmap=CMAP,
                         extend=COLORBAR_EXTEND)

        ticks = np.linspace(0, TWOPI, N_AXIS_TICKS)
        ax.set_xlim(0, TWOPI)
        ax.set_ylim(0, TWOPI)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(pi_labels(ticks))
        ax.set_yticklabels(pi_labels(ticks))
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE,
                       length=TICK_LENGTH, width=TICK_WIDTH)
        for spine in ax.spines.values():
            spine.set_linewidth(AXES_LINEWIDTH)

        # field lines: constant alpha -> horizontal, labelled by transit number
        if chart == "alpha":
            rows = np.linspace(0, len(alpha) - 1, N_FIELDLINES).astype(int)
            bbox = (dict(facecolor="white", edgecolor="none",
                         alpha=FIELDLINE_LABEL_BOX_ALPHA, pad=1.5)
                    if FIELDLINE_LABEL_BOX else None)
            for j, r in enumerate(rows):
                y = alpha[r]
                ax.plot([0, TWOPI], [y, y], color=FIELDLINE_COLOR,
                        linewidth=FIELDLINE_LINEWIDTH, solid_capstyle="butt")
                # Interior labels are centred on their line.  The two edge
                # lines would spill outside the axes that way, so the bottom one
                # is nudged above its line and the top one below.
                if j == 0:
                    dy, va = FIELDLINE_LABEL_DY, "bottom"
                elif j == len(rows) - 1:
                    dy, va = -FIELDLINE_LABEL_DY, "top"
                else:
                    dy, va = 0.0, "center"
                ax.text(FIELDLINE_LABEL_X * TWOPI, y + dy * TWOPI,
                        f"k = {k_of_row[r]}", ha="left", va=va,
                        fontsize=FIELDLINE_LABEL_FONTSIZE,
                        color=FIELDLINE_COLOR, bbox=bbox, zorder=5)

        cbar = fig.colorbar(cf, ax=ax, pad=0.02)
        if SHOW_COLORBAR_LABEL:
            cbar.set_label(QUANTITY_LABEL.get(q, q),
                           fontsize=COLORBAR_LABEL_FONTSIZE,
                           labelpad=COLORBAR_LABEL_PAD)
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE,
                            length=TICK_LENGTH, width=TICK_WIDTH)
        cbar.locator = LinearLocator(N_COLORBAR_TICKS)
        cbar.formatter = FormatStrFormatter(COLORBAR_TICK_FORMAT)
        cbar.update_ticks()

        out = FIG_DIR / f"FTGen_{CASE}_{q}_{chart}.pdf"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("  saved", out.name)

    # ------------------------------------------------ fixed-alpha line cuts
    if MAKE_LINECUT_FIGURE and vals["alpha"] is not None:
        va = vals["alpha"]
        rows = np.linspace(0, va.shape[0] - 1, LINECUT_ROWS).astype(int)
        fig, ax = plt.subplots(figsize=LINECUT_FIGSIZE, constrained_layout=True)
        for r in rows:
            ax.plot(phi_alpha, va[r], lw=LINECUT_LINEWIDTH,
                    label=f"k = {k_of_row[r]}")
        ax.set_xlabel(r"$\phi$", fontsize=LABEL_FONTSIZE, labelpad=LABEL_PAD)
        ax.set_ylabel(QUANTITY_LABEL.get(q, q), fontsize=LABEL_FONTSIZE,
                      labelpad=LABEL_PAD)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE,
                       length=TICK_LENGTH, width=TICK_WIDTH)
        for spine in ax.spines.values():
            spine.set_linewidth(AXES_LINEWIDTH)
        ax.legend(fontsize=LINECUT_LEGEND_FONTSIZE)
        out = FIG_DIR / f"FTGen_{CASE}_{q}_linecuts.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print("  saved", out.name)

print("figures in", FIG_DIR)
