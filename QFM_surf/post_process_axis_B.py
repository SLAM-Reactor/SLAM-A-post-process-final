#!/usr/bin/env python
"""Compare saved SLNAM/SLAM |B| on-axis data.

Run from QFM_surf after trace_axis_B.py:
    python post_process_axis_B.py
"""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


DATA_FILE = "axis_B_data.npz"
METRICS_FILE = "axis_B_metrics.json"
PNG_OUT = "axis_B_comparison.png"
PDF_OUT = "axis_B_comparison.pdf"

FONT_FAMILY = "DejaVu Sans"
MPL_FIGSIZE = (5.0, 4.4)
LINE_WIDTH = 1.5
ADD_MARKERS = True
MARKER_SIZE = 3.5
MARKER_EVERY = 5
MARKER_STYLE = "o"
TICK_SIZE =  24
LABEL_SIZE = 24
LABELPAD_X = -2
LABELPAD_Y = -2
LEGEND_SIZE = 20
DPI = 350
X_LABEL = r"$\zeta$"
Y_LABEL = r"$|B|$ on axis (T)"
X_TICKS = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi]
X_TICK_LABELS = [r"$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"]


data = np.load(DATA_FILE)
with open(METRICS_FILE, "r", encoding="utf-8") as f:
    metrics = json.load(f)

phi_slnam = data["slnam_phi_axis"]
b_slnam = data["slnam_B_axis"]
phi_slam = data["slam_phi_axis"]
b_slam = data["slam_B_axis"]

plt.rcParams.update({"font.family": FONT_FAMILY})
fig, ax = plt.subplots(figsize=MPL_FIGSIZE)
marker = MARKER_STYLE if ADD_MARKERS else None
ax.plot(phi_slnam, b_slnam, lw=LINE_WIDTH / 1.2, color="blue", label="pre-SLAM",)
        #marker=marker, markersize=MARKER_SIZE, markevery=MARKER_EVERY)
ax.plot(phi_slam, b_slam, lw=LINE_WIDTH, color="red", label="SLAM",)
        #marker=marker, markersize=MARKER_SIZE, markevery=MARKER_EVERY)
#ax.plot(phi_slnam, b_slnam, lw=LINE_WIDTH / 2, color="blue", label=r"pre-SLAM $\mathcal{S}_1$",
#        marker=marker, markersize=MARKER_SIZE, markevery=MARKER_EVERY)
#ax.plot(phi_slam, b_slam, lw=LINE_WIDTH, color="red", label=r"SLAM $\mathcal{S}_2$",
#        marker=marker, markersize=MARKER_SIZE, markevery=MARKER_EVERY)
ax.set_xlabel(X_LABEL, fontsize=LABEL_SIZE, labelpad=LABELPAD_X)
ax.set_ylabel(Y_LABEL, fontsize=LABEL_SIZE, labelpad=LABELPAD_Y)
ax.set_xticks(X_TICKS)
ax.set_xticklabels(X_TICK_LABELS)
ax.tick_params(axis="both", labelsize=TICK_SIZE, direction="in", top=True, right=True)
ax.grid(True, alpha=0.25)
ax.legend(frameon=False, fontsize=LEGEND_SIZE)
fig.tight_layout()
fig.savefig(PNG_OUT, dpi=DPI)
fig.savefig(PDF_OUT, bbox_inches="tight")
plt.close(fig)

print("wrote", PNG_OUT)
print("wrote", PDF_OUT)
for label in ("SLNAM", "SLAM"):
    print(label, metrics[label])
