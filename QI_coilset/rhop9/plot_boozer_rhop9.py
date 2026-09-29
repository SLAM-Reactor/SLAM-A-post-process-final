#!/usr/bin/env python3
"""DESC Boozer plot for the QI equilibrium at rho=0.9."""

import os
import sys
import warnings

DESC_SOURCE = os.environ.get("DESC_SOURCE", "/pscratch/sd/r/rgaur/DESC3/DESC")
sys.path.insert(0, DESC_SOURCE)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.plotting import plot_boozer_surface


# ---------------------------------------------------------------------------
# Everything to change is hardcoded here.
# ---------------------------------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__))
qi_dir = os.path.abspath(os.path.join(here, ".."))

equilibrium_filename = os.path.join(
    qi_dir,
    "eq_NFP2_A+9.0_tor+0.0_mr+0.10_shift+0.00_wd+0.00_elong+3.00"
    "_wellwt+2.0_vacuum_final2.h5",
)
rho = 0.9
output_stem = "Boozer_DESC_rhop9"
number_of_contours = 30
figure_size = (6, 5)
color_map = "jet"
output_dpi = 400


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"desc\.io\.hdf5_io",
    )
    eq = Equilibrium.load(equilibrium_filename)

boozer_compute_grid = LinearGrid(
    rho=rho,
    M=2 * eq.M,
    N=2 * eq.N,
    NFP=eq.NFP,
    endpoint=False,
)

desc_fig, desc_ax, boozer_data = plot_boozer_surface(
    eq,
    grid_compute=boozer_compute_grid,
    rho=rho,
    M_booz=eq.M,
    N_booz=eq.N,
    return_data=True,
)
plt.close(desc_fig)

theta_B = np.asarray(boozer_data["theta_B"])
zeta_B = np.asarray(boozer_data["zeta_B"])
mod_B = np.asarray(boozer_data["|B|"])

np.savez(
    output_stem + ".npz",
    theta_B=theta_B,
    zeta_B=zeta_B,
    mod_B=mod_B,
    rho=rho,
    NFP=eq.NFP,
    equilibrium_filename=equilibrium_filename,
)

fig, ax = plt.subplots(figsize=figure_size)

contour = ax.contour(
    zeta_B,
    theta_B,
    mod_B,
    levels=np.linspace(np.min(mod_B), np.max(mod_B), number_of_contours),
    cmap=color_map,
)

colorbar = fig.colorbar(contour, ax=ax, orientation="vertical")
colorbar.locator = ticker.MaxNLocator(nbins=7)
colorbar.update_ticks()
colorbar.ax.tick_params(labelsize=18)

x_ticks = np.array(
    [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
) / eq.NFP
y_ticks = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
angle_labels = [
    "0",
    r"$\pi/2$",
    r"$\pi$",
    r"$3\pi/2$",
    r"$2\pi$",
]

ax.set_xticks(x_ticks)
ax.set_xticklabels(angle_labels, fontsize=28)
ax.set_yticks(y_ticks)
ax.set_yticklabels(angle_labels, fontsize=28)
ax.set_xlabel(r"$\zeta_{\mathrm{Boozer}}$", fontsize=34, labelpad=-3)
ax.set_ylabel(r"$\theta_{\mathrm{Boozer}}$", fontsize=34, labelpad=-1)

plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.9)
plt.savefig(output_stem + ".png", dpi=output_dpi, bbox_inches="tight")
plt.savefig(output_stem + ".svg", bbox_inches="tight")
plt.close(fig)

print("Equilibrium:", equilibrium_filename)
print("Boozer surface rho:", rho)
print("Saved:", output_stem + ".npz")
print("Saved:", output_stem + ".png")
print("Saved:", output_stem + ".svg")
