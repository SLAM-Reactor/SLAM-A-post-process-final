#!/usr/bin/env python3
"""Overlay the DESC NFP=3 coilset Poincare trace on the GVEC flux surfaces,
styled the same as GVEC_f67/post-process/plot_poincare_vs_gvec.py
(figures/poincare_vs_gvec_SLNAM.png): black GVEC curves, blue Poincare dots,
one panel per phi plane, equal aspect."""

import matplotlib

matplotlib.use("Agg")
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

sections = np.load("gvec_sections_nfp3.npz")
traces = np.load("desc_coil_poincare_nfp3.npz")

rho = sections["rho"]
phi_planes = sections["phi_planes"]
R_gvec = sections["R_phi"]
Z_gvec = sections["Z_phi"]
R_pts = traces["R_poincare"]
Z_pts = traces["Z_poincare"]

FIGSIZE = (7.2, 3.6)
PAD_FRACTION = 0.04

plt.rcParams.update({"font.size": 11, "savefig.facecolor": "white"})

fig, axes = plt.subplots(1, phi_planes.size, figsize=FIGSIZE)
axes = np.atleast_1d(axes)

for ip, ax in enumerate(axes):
    boundary_index = R_gvec.shape[0] - 1
    for ir in range(R_gvec.shape[0]):
        is_boundary = ir == boundary_index
        ax.plot(
            np.append(R_gvec[ir, :, ip], R_gvec[ir, 0, ip]),
            np.append(Z_gvec[ir, :, ip], Z_gvec[ir, 0, ip]),
            color="black",
            lw=1.25 if is_boundary else 0.7,
            alpha=0.95 if is_boundary else 0.65,
        )
    ax.scatter(
        R_pts[:, ip, :].ravel(), Z_pts[:, ip, :].ravel(),
        s=1.0, color="#0057B8", alpha=0.75, edgecolors="none", rasterized=True,
    )

    plane_r = np.concatenate((R_gvec[:, :, ip].ravel(), R_pts[:, ip, :].ravel()))
    plane_z = np.concatenate((Z_gvec[:, :, ip].ravel(), Z_pts[:, ip, :].ravel()))
    width = max(float(plane_r.max() - plane_r.min()), float(plane_z.max() - plane_z.min()))
    pad = PAD_FRACTION * width
    ax.set_xlim(plane_r.min() - pad, plane_r.max() + pad)
    ax.set_ylim(plane_z.min() - pad, plane_z.max() + pad)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R (m)")
    if ip == 0:
        ax.set_ylabel("Z (m)")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    title = r"$\phi = 0$" if np.isclose(phi_planes[ip], 0.0) else r"$\phi = \pi/\mathrm{NFP}$"
    ax.set_title(title, fontsize=13)

fig.subplots_adjust(wspace=0.28)
fig.savefig("poincare_vs_gvec_nfp3.png", dpi=450, bbox_inches="tight")
fig.savefig("poincare_vs_gvec_nfp3.pdf", bbox_inches="tight")
print("wrote poincare_vs_gvec_nfp3.png / .pdf", flush=True)
