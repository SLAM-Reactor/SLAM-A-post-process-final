#!/usr/bin/env python3
"""Plot equilibrium surfaces against coil-field Poincare points up to rho=0.95."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np


# ---------------------------------------------------------------------------
# Everything to change is hardcoded here.
# ---------------------------------------------------------------------------
data_filename = "QI_poincare_rhop95_data.npz"
output_stem = "QI_poincare_rhop95_comparison"

figure_size = (6.8, 3.25)
output_dpi = 500
surface_color = "black"
poincare_color = "#0057B8"
surface_line_width = 0.8
lcfs_line_width = 1.2
poincare_size = 1.15
poincare_alpha = 0.9
number_of_plotted_surfaces = 8
x_major_ticks = (
    np.array([0.9, 1.0, 1.1]),
    np.array([0.75, 0.90, 1.05]),
)


data = np.load(data_filename)

R_poincare = data["R_poincare"]
Z_poincare = data["Z_poincare"]
R_eq = data["R_eq"]
Z_eq = data["Z_eq"]
zeta_planes = data["zeta_planes"]
rho_start = data["rho_start"]
rho_surfaces = data["rho_surfaces"]
target_rho = float(data["target_rho"])
NFP = int(data["NFP"])

plot_rho = np.linspace(0.0, target_rho, number_of_plotted_surfaces)
plot_trace_indices = np.array(
    [np.argmin(np.abs(rho_start - rho_value)) for rho_value in plot_rho]
)
plot_surface_indices = np.array(
    [np.argmin(np.abs(rho_surfaces - rho_value)) for rho_value in plot_rho]
)

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

plane_limits = []
plane_width_ratios = []
for plane_index in range(zeta_planes.size):
    plane_R = np.concatenate(
        (
            R_eq[:, plot_surface_indices, plane_index].ravel(),
            R_poincare[:, plane_index, plot_trace_indices].ravel(),
        )
    )
    plane_Z = np.concatenate(
        (
            Z_eq[:, plot_surface_indices, plane_index].ravel(),
            Z_poincare[:, plane_index, plot_trace_indices].ravel(),
        )
    )
    finite = np.isfinite(plane_R) & np.isfinite(plane_Z)
    plane_R = plane_R[finite]
    plane_Z = plane_Z[finite]
    R_min = np.min(plane_R)
    R_max = np.max(plane_R)
    Z_min = np.min(plane_Z)
    Z_max = np.max(plane_Z)
    plot_width = max(R_max - R_min, Z_max - Z_min)
    padding = 0.035 * plot_width
    plane_limits.append(
        (R_min - padding, R_max + padding, Z_min - padding, Z_max + padding)
    )
    plane_width_ratios.append((R_max - R_min + 2 * padding) / (Z_max - Z_min + 2 * padding))

fig, axes = plt.subplots(
    1,
    2,
    figsize=figure_size,
    gridspec_kw={"width_ratios": plane_width_ratios},
)

for plane_index, ax in enumerate(axes):
    for plotted_index, surface_index in enumerate(plot_surface_indices):
        line_width = lcfs_line_width
        if plotted_index < plot_surface_indices.size - 1:
            line_width = surface_line_width

        ax.plot(
            R_eq[:, surface_index, plane_index],
            Z_eq[:, surface_index, plane_index],
            color=surface_color,
            lw=line_width,
            solid_capstyle="round",
        )

    ax.scatter(
        R_poincare[:, plane_index, plot_trace_indices].ravel(),
        Z_poincare[:, plane_index, plot_trace_indices].ravel(),
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
    ax.set_xticks(x_major_ticks[plane_index])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.set_xlim(plane_limits[plane_index][0], plane_limits[plane_index][1])
    ax.set_ylim(plane_limits[plane_index][2], plane_limits[plane_index][3])

axes[0].set_ylabel(r"$Z$ (m)", fontsize=18)
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.17, top=0.86, wspace=0.22)
fig.savefig(output_stem + ".png", dpi=output_dpi, bbox_inches="tight")
fig.savefig(output_stem + ".pdf", bbox_inches="tight")
fig.savefig(output_stem + ".svg", bbox_inches="tight")
plt.close(fig)

print("Data:", data_filename)
print("NFP:", NFP)
print("Target rho:", target_rho)
print("Saved dense radial seeds:", rho_start)
print("Plotted surfaces:", rho_surfaces[plot_surface_indices])
print("Plotted radial seeds:", rho_start[plot_trace_indices])
print("Saved:", output_stem + ".png")
print("Saved:", output_stem + ".pdf")
print("Saved:", output_stem + ".svg")
