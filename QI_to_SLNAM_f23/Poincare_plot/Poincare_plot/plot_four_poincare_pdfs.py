#!/usr/bin/env python3
"""Make the four SLNAM Poincare plots from saved trace data."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np


# ---------------------------------------------------------------------------
# Hardcoded post-processing settings.
# ---------------------------------------------------------------------------
# This makes the script portable: copy the whole Poincare_plot directory anywhere
# and the script will read/write files next to itself, independent of your shell cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR

CASES = ("1p0", "2p0")  # the two cases in Figure 3 (0p5, 1p5 are in the backup)
NTRANSIT = 400

SAVE_PDF = True
SAVE_PNG = False
SAVE_SVG = False
CREATE_OUTPUT_DIR = True

DATA_FILENAME = "SLNAMified_L_over_R_{case}_poincare_data_ntransit{ntransit}.npz"
OUTPUT_STEM = "Poincare_plot_L_over_R_{case}_ntransit{ntransit}"

# Use None to plot every seed. Otherwise use a list such as [0, 1, 3, 5].
PLOT_SEED_INDICES = None

# Figure layout.
FIGURE_SIZE = (6.8, 3.25)
OUTPUT_DPI = 500
LEFT = 0.105
RIGHT = 0.985
BOTTOM = 0.17
TOP = 0.86
WSPACE = 0.33

# Plot appearance.
POINCARE_COLOR = "#0057B8"
MARKER_SIZE = 1.4
AXIS_MARKER_SIZE = 8.0
MARKER_ALPHA = 1.0
RASTERIZED_POINTS = True

# Text and tick sizing.
LABEL_SIZE = 20
AXIS_LABEL_SIZE = 20
TICK_SIZE = 5
MINOR_TICK_SIZE = 2.8
TITLE_SIZE = 20
TICK_FONT_SIZE = 16
LABEL_PAD = 6
TITLE_PAD = 8

# Axes and tick choices.
AXES_LINE_WIDTH = 1.1
MAJOR_TICK_WIDTH = 1.0
MINOR_TICK_WIDTH = 0.8
X_NBINS = 4
Y_LIMIT = 0.0926
Y_TICKS = np.array([-0.08, -0.04, 0.0, 0.04, 0.08])

# R limits are computed from the data with a little padding.
CLIP_PERCENTILES = (0.5, 99.5)
R_PADDING_FRACTION = 0.08


save_formats = []
if SAVE_PDF:
    save_formats.append("pdf")
if SAVE_PNG:
    save_formats.append("png")
if SAVE_SVG:
    save_formats.append("svg")
if not save_formats:
    raise ValueError("At least one of SAVE_PDF, SAVE_PNG, or SAVE_SVG must be True.")

if CREATE_OUTPUT_DIR:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": AXES_LINE_WIDTH,
        "xtick.major.width": MAJOR_TICK_WIDTH,
        "ytick.major.width": MAJOR_TICK_WIDTH,
        "xtick.minor.width": MINOR_TICK_WIDTH,
        "ytick.minor.width": MINOR_TICK_WIDTH,
        "savefig.facecolor": "white",
    }
)

for case in CASES:
    data_path = DATA_DIR / DATA_FILENAME.format(case=case, ntransit=NTRANSIT)
    output_stem = OUTPUT_DIR / OUTPUT_STEM.format(case=case, ntransit=NTRANSIT)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing Poincare data file: {data_path}")

    data = np.load(data_path)
    R_poincare = data["R"]
    Z_poincare = data["Z"]
    zeta_planes = data["phi"]
    r0 = data["r0"] if "r0" in data.files else np.arange(R_poincare.shape[2])
    axis_seed_index = int(data["axis_seed_index"]) if "axis_seed_index" in data.files else -1

    if PLOT_SEED_INDICES is None:
        seed_indices = np.arange(R_poincare.shape[2])
    else:
        seed_indices = np.array(PLOT_SEED_INDICES, dtype=int)

    R_plot = R_poincare[:, :, seed_indices]
    Z_plot = Z_poincare[:, :, seed_indices]
    r0_plot = r0[seed_indices]
    axis_local_indices = np.where(seed_indices == axis_seed_index)[0]

    plane_centers = []
    plane_R_spans = []
    for plane_index in range(zeta_planes.size):
        finite = np.isfinite(R_plot[:, plane_index, :]) & np.isfinite(
            Z_plot[:, plane_index, :]
        )
        plane_R = R_plot[:, plane_index, :][finite]
        if plane_R.size == 0:
            raise ValueError(f"No finite Poincare points for case {case}, plane {plane_index}.")
        R_min, R_max = np.percentile(plane_R, CLIP_PERCENTILES)
        plane_centers.append(0.5 * (R_min + R_max))
        plane_R_spans.append(R_max - R_min)

    R_span = max(plane_R_spans) * (1.0 + 2.0 * R_PADDING_FRACTION)

    fig, axes = plt.subplots(1, zeta_planes.size, figsize=FIGURE_SIZE, sharey=True)
    if zeta_planes.size == 1:
        axes = [axes]

    for plane_index, ax in enumerate(axes):
        finite = np.isfinite(R_plot[:, plane_index, :]) & np.isfinite(
            Z_plot[:, plane_index, :]
        )
        ax.scatter(
            R_plot[:, plane_index, :][finite],
            Z_plot[:, plane_index, :][finite],
            s=MARKER_SIZE,
            color=POINCARE_COLOR,
            alpha=MARKER_ALPHA,
            edgecolors="none",
            rasterized=RASTERIZED_POINTS,
        )

        for axis_local_index in axis_local_indices:
            finite_axis = np.isfinite(R_plot[:, plane_index, axis_local_index]) & np.isfinite(
                Z_plot[:, plane_index, axis_local_index]
            )
            ax.scatter(
                R_plot[:, plane_index, axis_local_index][finite_axis],
                Z_plot[:, plane_index, axis_local_index][finite_axis],
                s=AXIS_MARKER_SIZE,
                color=POINCARE_COLOR,
                alpha=1.0,
                edgecolors="none",
                rasterized=RASTERIZED_POINTS,
            )

        if np.isclose(zeta_planes[plane_index], 0.0):
            title = r"$\zeta = 0$"
        else:
            title = r"$\zeta = \pi/2$"

        R_center = plane_centers[plane_index]
        ax.set_xlim(R_center - 0.5 * R_span, R_center + 0.5 * R_span)
        ax.set_ylim(-Y_LIMIT, Y_LIMIT)
        ax.set_yticks(Y_TICKS)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=TITLE_SIZE, pad=TITLE_PAD)
        ax.set_xlabel(r"$R$ (m)", fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
        ax.tick_params(
            which="both",
            direction="in",
            top=True,
            right=True,
            labelsize=TICK_FONT_SIZE,
            length=TICK_SIZE,
        )
        ax.tick_params(which="minor", length=MINOR_TICK_SIZE)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=X_NBINS))
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    axes[0].set_ylabel(r"$Z$ (m)", fontsize=AXIS_LABEL_SIZE, labelpad=LABEL_PAD)
    for ax in axes:
        ax.tick_params(labelleft=True)

    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP, wspace=WSPACE)

    saved_files = []
    for save_format in save_formats:
        output_path = output_stem.with_suffix(f".{save_format}")
        fig.savefig(output_path, dpi=OUTPUT_DPI, bbox_inches="tight")
        saved_files.append(output_path)
    plt.close(fig)

    print(f"Case: {case}")
    print(f"  Data: {data_path}")
    print(f"  Shape: {R_plot.shape}")
    print(f"  Plotted seed indices: {seed_indices}")
    print(f"  Plotted seed R: {r0_plot}")
    print(f"  Finite fraction: {np.mean(np.isfinite(R_plot) & np.isfinite(Z_plot))}")
    for saved_file in saved_files:
        print(f"  Saved: {saved_file}")
