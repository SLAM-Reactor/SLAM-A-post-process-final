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

figure_size = tuple(
    float(value)
    for value in os.environ.get("POINCARE_FIGURE_SIZE", "6.8,3.25").split(",")
)
output_dpi = int(os.environ.get("POINCARE_OUTPUT_DPI", "500"))
save_formats = tuple(
    value.strip()
    for value in os.environ.get("POINCARE_SAVE_FORMATS", "png,pdf,svg").split(",")
    if value.strip()
)

poincare_color = os.environ.get("POINCARE_COLOR", "#0057B8")
label_size = float(os.environ.get("POINCARE_LABEL_SIZE", "18"))
marker_size = float(os.environ.get("POINCARE_MARKER_SIZE", "1.15"))
axis_label_size = float(os.environ.get("POINCARE_AXIS_LABEL_SIZE", str(label_size)))
tick_size = float(os.environ.get("POINCARE_TICK_SIZE", "5"))
minor_tick_size = float(os.environ.get("POINCARE_MINOR_TICK_SIZE", "2.8"))
title_size = float(os.environ.get("POINCARE_TITLE_SIZE", str(label_size)))
tick_font_size = float(os.environ.get("POINCARE_TICK_FONT_SIZE", "14"))
label_pad = float(os.environ.get("POINCARE_LABEL_PAD", "6"))
title_pad = float(os.environ.get("POINCARE_TITLE_PAD", "8"))
poincare_alpha = float(os.environ.get("POINCARE_MARKER_ALPHA", "0.9"))
axis_marker_size = float(os.environ.get("POINCARE_AXIS_MARKER_SIZE", "8.0"))
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
axis_seed_index = int(data["axis_seed_index"]) if "axis_seed_index" in data.files else None

if number_of_plotted_seeds > 0 and number_of_plotted_seeds < R_poincare.shape[2]:
    plot_seed_indices = np.unique(
        np.round(np.linspace(0, R_poincare.shape[2] - 1, number_of_plotted_seeds)).astype(int)
    )
    R_poincare = R_poincare[:, :, plot_seed_indices]
    Z_poincare = Z_poincare[:, :, plot_seed_indices]
    r0 = r0[plot_seed_indices]
else:
    plot_seed_indices = np.arange(R_poincare.shape[2])
axis_local_indices = (
    np.where(plot_seed_indices == axis_seed_index)[0]
    if axis_seed_index is not None
    else np.array([], dtype=int)
)

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


def symmetric_limits(limits):
    y_abs = max(abs(float(limits[0])), abs(float(limits[1])))
    return (-y_abs, y_abs)


def symmetric_ticks(limits):
    y_abs = max(abs(float(limits[0])), abs(float(limits[1])))
    positive_ticks = ticker.MaxNLocator(nbins=3).tick_values(0.0, y_abs)
    positive_ticks = positive_ticks[
        (positive_ticks > 0.0) & (positive_ticks <= y_abs * (1.0 + 1e-12))
    ]
    if positive_ticks.size == 0:
        return np.array([0.0])
    return np.concatenate((-positive_ticks[::-1], np.array([0.0]), positive_ticks))


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
        s=marker_size,
        color=poincare_color,
        alpha=poincare_alpha,
        edgecolors="none",
        rasterized=True,
    )
    for axis_local_index in axis_local_indices:
        finite_axis = np.isfinite(R_poincare[:, plane_index, axis_local_index]) & np.isfinite(
            Z_poincare[:, plane_index, axis_local_index]
        )
        ax.scatter(
            R_poincare[:, plane_index, axis_local_index][finite_axis],
            Z_poincare[:, plane_index, axis_local_index][finite_axis],
            s=axis_marker_size,
            color=poincare_color,
            alpha=1.0,
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
        ax.set_ylim(*symmetric_limits((Z_center - 0.5 * Z_span, Z_center + 0.5 * Z_span)))
    else:
        ax.set_ylim(*symmetric_limits(y_limits[plane_index]))
    ax.set_yticks(symmetric_ticks(ax.get_ylim()))
    ax.set_title(title, fontsize=title_size, pad=title_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$R$ (m)", fontsize=axis_label_size, labelpad=label_pad)
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=tick_font_size,
        length=tick_size,
    )
    ax.tick_params(which="minor", length=minor_tick_size)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

if share_y_axes:
    shared_y_limits = symmetric_limits(y_limits[0])
    shared_y_ticks = symmetric_ticks(shared_y_limits)
    for ax in axes:
        ax.set_ylim(*shared_y_limits)
        ax.set_yticks(shared_y_ticks)
        ax.tick_params(labelleft=True)

axes[0].set_ylabel(r"$Z$ (m)", fontsize=axis_label_size, labelpad=label_pad)
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.17, top=0.86, wspace=0.33)
saved_files = []
for save_format in save_formats:
    output_filename = f"{output_stem}.{save_format}"
    save_kwargs = {"bbox_inches": "tight"}
    if save_format.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
        save_kwargs["dpi"] = output_dpi
    fig.savefig(output_filename, **save_kwargs)
    saved_files.append(output_filename)
plt.close(fig)

print("Data:", data_filename)
print("Poincare R/Z shape:", R_poincare.shape)
print("Plotted seed indices:", plot_seed_indices)
print("Plotted seed R:", r0)
print("Plotted finite fraction:", np.mean(np.isfinite(R_poincare) & np.isfinite(Z_poincare)))
for saved_file in saved_files:
    print("Saved:", saved_file)
