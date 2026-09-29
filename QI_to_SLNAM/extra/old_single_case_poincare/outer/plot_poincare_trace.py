#!/usr/bin/env python3
"""Plot SLNAMified-coilset Poincare points only."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import os


# ---------------------------------------------------------------------------
# Everything to change is hardcoded here.
# ---------------------------------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__))
data_filename = os.environ.get(
    "POINCARE_DATA_FILENAME",
    os.path.join(here, "SLNAMified_poincare_plot_data_ntransit400.npz"),
)
output_stem = os.environ.get("POINCARE_OUTPUT_STEM", "Poincare_plot")

figure_size = (6.8, 3.25)
output_dpi = 600
poincare_color = "#0057B8"
poincare_size = 1.15
poincare_alpha = 0.9
clip_percentiles = (0.5, 99.5)
plot_padding_fraction = 0.08
number_of_plotted_seeds = int(os.environ.get("POINCARE_N_PLOT_SEEDS", "8"))
y_limits_text = os.environ.get(
    "POINCARE_Y_LIMITS", "-0.09256116119172134,0.09256116119172135"
)


if not os.path.exists(data_filename):
    raise FileNotFoundError(
        "Poincare data file is missing. Expected: "
        f"{data_filename}."
    )

data = np.load(data_filename)

R_poincare = data["R"]
Z_poincare = data["Z"]
zeta_planes = data["phi"]
r0 = data["r0"] if "r0" in data.files else np.arange(R_poincare.shape[2])

if number_of_plotted_seeds > 0 and number_of_plotted_seeds < R_poincare.shape[2]:
    plot_seed_indices = np.unique(
        np.round(np.linspace(0, R_poincare.shape[2] - 1, number_of_plotted_seeds)).astype(int)
    )
    R_poincare = R_poincare[:, :, plot_seed_indices]
    Z_poincare = Z_poincare[:, :, plot_seed_indices]
    r0 = r0[plot_seed_indices]
else:
    plot_seed_indices = np.arange(R_poincare.shape[2])

if y_limits_text:
    y_limits = np.array(
        [
            [float(value) for value in plane_limits.split(",")]
            for plane_limits in y_limits_text.split(";")
        ]
    )
    if y_limits.shape == (1, 2):
        y_limits = np.repeat(y_limits, zeta_planes.size, axis=0)
else:
    y_limits = None

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "savefig.facecolor": "white",
    }
)

plane_centers = []
plane_spans = []
for plane_index in range(zeta_planes.size):
    finite = np.isfinite(R_poincare[:, plane_index, :]) & np.isfinite(
        Z_poincare[:, plane_index, :]
    )
    plane_R = R_poincare[:, plane_index, :][finite]
    plane_Z = Z_poincare[:, plane_index, :][finite]
    R_min, R_max = np.percentile(plane_R, clip_percentiles)
    Z_min, Z_max = np.percentile(plane_Z, clip_percentiles)
    plane_centers.append((0.5 * (R_min + R_max), 0.5 * (Z_min + Z_max)))
    plane_spans.append((R_max - R_min, Z_max - Z_min))

R_span = max(span[0] for span in plane_spans)
Z_span = max(span[1] for span in plane_spans)
R_span *= 1.0 + 2.0 * plot_padding_fraction
Z_span *= 1.0 + 2.0 * plot_padding_fraction

share_y_axes = y_limits is not None and np.allclose(y_limits, y_limits[0])

fig, axes = plt.subplots(1, 2, figsize=figure_size, sharey=share_y_axes)

for plane_index, ax in enumerate(axes):
    finite = np.isfinite(R_poincare[:, plane_index, :]) & np.isfinite(
        Z_poincare[:, plane_index, :]
    )
    ax.scatter(
        R_poincare[:, plane_index, :][finite],
        Z_poincare[:, plane_index, :][finite],
        s=poincare_size,
        color=poincare_color,
        alpha=poincare_alpha,
        edgecolors="none",
        rasterized=True,
    )

    if np.isclose(zeta_planes[plane_index], 0.0):
        title = r"$\zeta = 0$"
    else:
        title = r"$\zeta = \pi/N_{\mathrm{FP}}$"

    R_center, Z_center = plane_centers[plane_index]
    ax.set_xlim(R_center - 0.5 * R_span, R_center + 0.5 * R_span)
    if y_limits is None:
        ax.set_ylim(Z_center - 0.5 * Z_span, Z_center + 0.5 * Z_span)
    else:
        ax.set_ylim(y_limits[plane_index, 0], y_limits[plane_index, 1])
    ax.set_title(title, fontsize=18, pad=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$R$ (m)", fontsize=18)
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=14,
        length=5,
    )
    ax.tick_params(which="minor", length=2.8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

if share_y_axes:
    shared_y_limits = (float(y_limits[0, 0]), float(y_limits[0, 1]))
    shared_y_ticks = ticker.MaxNLocator(nbins=5).tick_values(*shared_y_limits)
    shared_y_ticks = shared_y_ticks[
        (shared_y_ticks > shared_y_limits[0]) & (shared_y_ticks < shared_y_limits[1])
    ]
    for ax in axes:
        ax.set_ylim(*shared_y_limits)
        ax.set_yticks(shared_y_ticks)
        ax.tick_params(labelleft=True)

axes[0].set_ylabel(r"$Z$ (m)", fontsize=18)
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.17, top=0.86, wspace=0.33)
fig.savefig(output_stem + ".png", dpi=output_dpi, bbox_inches="tight")
fig.savefig(output_stem + ".pdf", dpi=output_dpi, bbox_inches="tight")
fig.savefig(output_stem + ".svg", bbox_inches="tight")
plt.close(fig)

print("Data:", data_filename)
print("Poincare R/Z shape:", R_poincare.shape)
print("Plotted seed indices:", plot_seed_indices)
print("Plotted seed R:", r0)
print("Plotted finite fraction:", np.mean(np.isfinite(R_poincare) & np.isfinite(Z_poincare)))
print("Saved:", output_stem + ".png")
print("Saved:", output_stem + ".pdf")
print("Saved:", output_stem + ".svg")
