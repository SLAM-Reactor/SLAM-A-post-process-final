#!/usr/bin/env python
"""Compare SLNAM and SLAM QFM surface cross-sections in one frame."""

import os

os.makedirs(".mplconfig", exist_ok=True)
os.environ["MPLCONFIGDIR"] = ".mplconfig"

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from desc.io import load
from desc.grid import Grid


SURFACES = [
    "SLNAM_QFM_surface_09-09-26_3.h5",
    "SLAM_QFM_surface_09-09-26_2.h5",
]
LABELS = ["SLNAM QFM", "SLAM QFM"]
PHI = 12
N_THETA = 1000
FIGSIZE = (10.0, 10.0)
LINE_WIDTHS = [2.0, 2.0]
LINE_STYLES = ["-", "--"]
COLORS = ["tab:red", "tab:blue"]
TICK_SIZE = 13
LABEL_SIZE = 15


surfaces = [load(name) for name in SURFACES]

fig, ax = plt.subplots(figsize=FIGSIZE)
phis = np.linspace(0.0, 2.0 * np.pi / surfaces[0].NFP, PHI, endpoint=False)
theta = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=True)

for i, surface in enumerate(surfaces):
    for j, phi in enumerate(phis):
        nodes = np.column_stack([np.ones_like(theta), theta, np.full_like(theta, phi)])
        grid = Grid(nodes, sort=False)
        data = surface.compute(["R", "Z"], grid=grid)
        ax.plot(
            data["R"],
            data["Z"],
            color=COLORS[i],
            lw=LINE_WIDTHS[i],
            ls=LINE_STYLES[i],
            label=LABELS[i] if j == 0 else None,
        )

ax.tick_params(labelsize=TICK_SIZE)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(r"$R$ (m)", fontsize=LABEL_SIZE)
ax.set_ylabel(r"$Z$ (m)", fontsize=LABEL_SIZE)
ax.legend(loc="upper center", ncol=2, frameon=False)
fig.tight_layout()

fig.savefig("SLAM_QFM_surface_cross_section_comparison.png", dpi=400, bbox_inches="tight")
fig.savefig("SLAM_QFM_surface_cross_section_comparison.pdf", bbox_inches="tight")
plt.close(fig)

print("wrote SLAM_QFM_surface_cross_section_comparison.png")
print("wrote SLAM_QFM_surface_cross_section_comparison.pdf")
