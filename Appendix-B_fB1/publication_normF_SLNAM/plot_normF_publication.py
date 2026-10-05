#!/usr/bin/env python3
"""Publication contour plots of the normalized force-balance error |F| / <|grad(B^2/2mu0)|>_vol on the
cylindrical planes phi = 0 and phi = pi/2, one figure per equilibrium:
    DESC SLNAM cylindrical solve, fit 9,18   <- normF_DESC_SLNAM_cyl_f9n18_phi_planes.npz  (export_desc_normF.py)
    GVEC SLNAM_BEST                          <- normF_GVEC_SLNAM_BEST_phi_planes.npz       (export_gvec_normF.py)
Both codes use the same normalization, and both are evaluated exactly on the same planes.

Each .npz holds R, Z, F with shape (rho, theta, plane), plus rho, theta, phi and an 'info' string.
Only numpy and matplotlib are needed:  python plot_normF_publication.py
Edit the settings below; nothing else needs to change.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter, FixedLocator, FixedFormatter, MaxNLocator, MultipleLocator
from matplotlib.transforms import Bbox

# =============================== settings ===============================

# (data file, output file name without extension, figure title or None)
FIGURES = [
    ("normF_DESC_SLNAM_cyl_f9n18_phi_planes.npz", "normF_DESC_SLNAM_cyl_f9n18", "DESC"),
    ("normF_GVEC_SLNAM_BEST_phi_planes.npz", "normF_GVEC_SLNAM_BEST", "GVEC"),
]
PLANES = [0, 1]                     # which planes to draw, one panel each: 0 -> phi = 0, 1 -> phi = pi/2
PANEL_TITLES = [r"$\phi = 0$", r"$\phi = \pi/2$"]   # one per saved plane (index as in PLANES); None for no title
SHOW_MAX_MEAN_IN_TITLE = False      # append "max = ..., mean = ..." to each panel title
MAX_MEAN_FORMAT = "%.2e"

# figure
FIGSIZE = (9.0, 4.2)                # inches (width, height)
DPI = 300                           # for raster formats
FORMATS = ["pdf", "png"]            # pdf is vector
FONT_FAMILY = "sans-serif"          # matplotlib default (DejaVu Sans); "serif" for a Times-like font
MATH_FONT = "dejavusans"            # matplotlib default for $...$ text; "stix" or "cm" for LaTeX-like math
USE_TEX = False                     # True needs a LaTeX installation
FIG_TITLE_SIZE = 20
FIG_TITLE_Y = 0.93                  # height of the figure title (1 = top edge of the figure)
PANEL_TITLE_SIZE = 20
WSPACE = 0.40                       # horizontal space between panels

# contours
LEVELS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]   # color band edges (log scale); e.g. put 0.099 for 1e-1
CMAP = "jet"
EXTEND = "both"                     # arrows for values outside LEVELS: "both", "min", "max", "neither"
DRAW_CONTOUR_LINES = False          # thin lines on the band edges
CONTOUR_LINE_COLOR = "k"
CONTOUR_LINE_WIDTH = 0.3
DRAW_BOUNDARY = True                # outline of the rho = 1 surface
BOUNDARY_COLOR = "k"
BOUNDARY_WIDTH = 1.0

# axes
XLABEL = r"$R$ (m)"
YLABEL = r"$Z$ (m)"
LABEL_SIZE = 22
XLABELPAD = 4.0                     # gap (points) between x tick labels and the x label
YLABELPAD = -1.0                     # gap (points) between y tick labels and the y label
TICK_LABEL_SIZE = 20
TICK_LABEL_PAD = 3.5                # gap (points) between ticks and tick labels
# Edge-to-edge ticks: the axis limits are rounded outward to multiples of *_EDGE_ROUND (as little as needed for
# the spacing to be a whole multiple of it too), then N ticks are placed uniformly from the left edge to the right.
X_TICKS_EDGE_TO_EDGE = True
Y_TICKS_EDGE_TO_EDGE = False
X_EDGE_ROUND = 0.01
Y_EDGE_ROUND = 0.01
X_TICK_STEP = None                  # fixed tick spacing (one value, or dict per panel), used when not edge-to-edge
Y_TICK_STEP = None                  # e.g. 0.05
N_XTICKS = 4                        # edge-to-edge: exactly this many ticks; otherwise at most this many
N_YTICKS = 3
X_TICK_FORMAT = "%.2f"              # "%.2f" = 2 decimals; "%.3g" = 3 significant digits
Y_TICK_FORMAT = "%.2f"
# Hardcoded ticks: a list of values, or a dict {plane index: list} to set them per panel; None -> automatic.
# Tick labels: a list of strings (same length as the ticks) or a dict per panel; None -> use *_TICK_FORMAT.
# e.g. X_TICKS = {0: [0.8, 1.0, 1.2], 1: [0.7, 0.9, 1.1]};  X_TICKLABELS = {0: ["0.8", "1.0", "1.2"], 1: None}
X_TICKS = None
Y_TICKS = None
X_TICKLABELS = None
Y_TICKLABELS = None
TICK_DIRECTION = "in"               # "in", "out", "inout"
TICK_LENGTH = 4
TICK_WIDTH = 0.8
TICKS_ON_ALL_SIDES = True
SPINE_WIDTH = 1.0

# colorbar (right of the last panel; spans from the bottom of the x labels to the top of the panel titles,
# measured after drawing, so it follows any font-size change)
SHOW_COLORBAR = True
# COLORBAR_LABEL = r"$|\mathbf{F}| \,/\, \langle|\nabla (B^2/2\mu_0)|\rangle_\mathrm{vol}$"
COLORBAR_LABEL = None               # None -> no label
COLORBAR_LABEL_SIZE = 24
COLORBAR_LABELPAD = 8.0             # gap (points) between colorbar tick labels and the label
COLORBAR_TICK_SIZE = 18
COLORBAR_TICK_PAD = 3.5             # gap (points) between colorbar ticks and tick labels
COLORBAR_TICK_STYLE = "power"       # "power" -> 10^{-3}; otherwise a format string such as "%.0e" or "%g"
COLORBAR_TICKS = None               # hardcoded tick values, e.g. [1e-6, 1e-4, 1e-2, 1e0]; None -> LEVELS
COLORBAR_TICKLABELS = None          # hardcoded tick labels, e.g. ["1e-6", "1e-4", "0.01", "1"]; None -> TICK_STYLE
COLORBAR_SPAN = "labels"            # "labels" -> x label bottom to title top; "panel" -> same height as the panel
COLORBAR_EXTRA_BOTTOM = 0.0         # extend (+) or trim (-) the bottom end, inches
COLORBAR_EXTRA_TOP = 0.0            # extend (+) or trim (-) the top end, inches
COLORBAR_WIDTH = 0.15               # thickness, inches
COLORBAR_PAD = 0.12                 # gap between the panel and the colorbar, inches

# ========================================================================

plt.rcParams.update({"font.family": FONT_FAMILY, "mathtext.fontset": MATH_FONT, "text.usetex": USE_TEX,
                     "axes.linewidth": SPINE_WIDTH, "savefig.facecolor": "white"})
norm = LogNorm(vmin=LEVELS[0], vmax=LEVELS[-1])
cbar_ticks = COLORBAR_TICKS if COLORBAR_TICKS is not None else LEVELS
if COLORBAR_TICKLABELS is not None:
    cbar_ticklabels = COLORBAR_TICKLABELS
elif COLORBAR_TICK_STYLE == "power":
    cbar_ticklabels = [rf"$10^{{{int(round(np.log10(v)))}}}$" if np.isclose(np.log10(v), round(np.log10(v)))
                       else f"{v:g}" for v in cbar_ticks]
else:
    cbar_ticklabels = [COLORBAR_TICK_STYLE % v for v in cbar_ticks]


def per_panel(setting, k):
    """A setting given either as one value for all panels or as a dict {plane index: value}."""
    return setting.get(k) if isinstance(setting, dict) else setting


def edge_to_edge_ticks(lo, hi, n, rnd):
    """Round [lo, hi] outward to multiples of rnd so that n uniform ticks from lo to hi are also multiples of rnd."""
    i_lo = int(np.floor(lo / rnd + 1e-9))
    i_hi = int(np.ceil(hi / rnd - 1e-9))
    grow_hi = True
    while (i_hi - i_lo) % (n - 1):
        if grow_hi:
            i_hi += 1
        else:
            i_lo -= 1
        grow_hi = not grow_hi
    return i_lo * rnd, i_hi * rnd, np.linspace(i_lo, i_hi, n) * rnd


def set_axis_ticks(axis, ticks, ticklabels, step, n_auto, fmt):
    if ticks is not None:
        axis.set_major_locator(FixedLocator(ticks))
    elif step is not None:
        axis.set_major_locator(MultipleLocator(step))
    else:
        axis.set_major_locator(MaxNLocator(n_auto))
    if ticklabels is None:
        axis.set_major_formatter(FormatStrFormatter(fmt))
    else:
        axis.set_major_formatter(FixedFormatter(ticklabels))

for data_file, out_stem, fig_title in FIGURES:
    d = np.load(data_file)
    fig, axs = plt.subplots(1, len(PLANES), figsize=FIGSIZE, squeeze=False)
    fig.subplots_adjust(wspace=WSPACE)
    for n, k in enumerate(PLANES):
        ax = axs[0, n]
        R, Z, F = d["R"][:, :, k], d["Z"][:, :, k], d["F"][:, :, k]
        cf = ax.contourf(R, Z, F, levels=LEVELS, norm=norm, cmap=CMAP, extend=EXTEND)
        if DRAW_CONTOUR_LINES:
            ax.contour(R, Z, F, levels=LEVELS, colors=CONTOUR_LINE_COLOR, linewidths=CONTOUR_LINE_WIDTH)
        if DRAW_BOUNDARY:
            ax.plot(R[-1], Z[-1], color=BOUNDARY_COLOR, lw=BOUNDARY_WIDTH)
        ax.set_aspect("equal")
        ax.set_xlabel(XLABEL, fontsize=LABEL_SIZE, labelpad=XLABELPAD)
        ax.set_ylabel(YLABEL, fontsize=LABEL_SIZE, labelpad=YLABELPAD)
        xticks, yticks = per_panel(X_TICKS, k), per_panel(Y_TICKS, k)
        if xticks is None and X_TICKS_EDGE_TO_EDGE:
            x0, x1, xticks = edge_to_edge_ticks(R.min(), R.max(), N_XTICKS, X_EDGE_ROUND)
            ax.set_xlim(x0, x1)
        if yticks is None and Y_TICKS_EDGE_TO_EDGE:
            y0, y1, yticks = edge_to_edge_ticks(Z.min(), Z.max(), N_YTICKS, Y_EDGE_ROUND)
            ax.set_ylim(y0, y1)
        set_axis_ticks(ax.xaxis, xticks, per_panel(X_TICKLABELS, k), per_panel(X_TICK_STEP, k),
                       N_XTICKS, X_TICK_FORMAT)
        set_axis_ticks(ax.yaxis, yticks, per_panel(Y_TICKLABELS, k), per_panel(Y_TICK_STEP, k),
                       N_YTICKS, Y_TICK_FORMAT)
        ax.tick_params(labelsize=TICK_LABEL_SIZE, direction=TICK_DIRECTION, length=TICK_LENGTH, width=TICK_WIDTH,
                       pad=TICK_LABEL_PAD, top=TICKS_ON_ALL_SIDES, right=TICKS_ON_ALL_SIDES)
        title = PANEL_TITLES[k] if PANEL_TITLES is not None else ""
        if SHOW_MAX_MEAN_IN_TITLE:
            title += f"\nmax = {MAX_MEAN_FORMAT % F.max()}, mean = {MAX_MEAN_FORMAT % F.mean()}"
        if title:
            ax.set_title(title, fontsize=PANEL_TITLE_SIZE)
        print(f"{out_stem} plane {k}: max {F.max():.3e} mean {F.mean():.3e}")
    if SHOW_COLORBAR:
        # draw first so the equal-aspect panels and their labels have their final positions
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        to_fig = fig.transFigure.inverted()
        if COLORBAR_SPAN == "labels":
            span = Bbox.union([ax.get_tightbbox(renderer).transformed(to_fig) for ax in axs.ravel()])
        else:
            span = axs[0, -1].get_position()
        fig_w, fig_h = fig.get_size_inches()
        y0 = span.y0 - COLORBAR_EXTRA_BOTTOM / fig_h
        y1 = span.y1 + COLORBAR_EXTRA_TOP / fig_h
        # clear the panel frame and any tick label hanging past it
        x_right = max(axs[0, -1].get_position().x1, axs[0, -1].get_tightbbox(renderer).transformed(to_fig).x1)
        x0 = x_right + COLORBAR_PAD / fig_w
        cax = fig.add_axes([x0, y0, COLORBAR_WIDTH / fig_w, y1 - y0])
        cbar = fig.colorbar(cf, cax=cax)
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_ticklabels)
        cbar.minorticks_off()
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, pad=COLORBAR_TICK_PAD)
        if COLORBAR_LABEL:
            cbar.set_label(COLORBAR_LABEL, fontsize=COLORBAR_LABEL_SIZE, labelpad=COLORBAR_LABELPAD)
    # if fig_title:
    #     fig.suptitle(fig_title, fontsize=FIG_TITLE_SIZE, y=FIG_TITLE_Y)
    for fmt in FORMATS:
        fig.savefig(f"{out_stem}.{fmt}", dpi=DPI, bbox_inches="tight")
        print("wrote", f"{out_stem}.{fmt}")
    plt.close(fig)
