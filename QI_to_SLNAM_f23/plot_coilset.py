#!/usr/bin/env python3
"""Plot the SLNAMified QI coilset with DESC/Plotly."""

import sys

# Use the clean DESC checkout when this file is run directly in desc-env.
DESC_SOURCE = "/home/rgaur/DESC3/DESC"
sys.path.insert(0, DESC_SOURCE)

from desc.grid import LinearGrid
from desc.io import load
from desc.plotting import plot_coils


# ---------------------------------------------------------------------------
# Everything to change is hardcoded here.
# ---------------------------------------------------------------------------
coilset_filename = (
    "SLNAMified_NFP2_A8p7_coilset_L_over_R_1p0_SLAM-A.h5"
)
output_filename = "SLNAMified_QI_L_over_R_1p0_SLAM-A_coils_plotly.html"

coil_points = 500
coil_color = "black"
coil_line_width = 8


# Load and plot only the coils. No equilibrium or plasma surface is used.
coilset = load(coilset_filename)
coil_grid = LinearGrid(N=coil_points, endpoint=True)

fig = plot_coils(
    coilset,
    grid=coil_grid,
    color=coil_color,
    lw=coil_line_width,
    showgrid=False,
    showticklabels=False,
    showaxislabels=False,
    zeroline=False,
)


# White background, equal physical scale, no title, axes, or legend.
fig.update_layout(
    width=1100,
    height=850,
    paper_bgcolor="white",
    plot_bgcolor="white",
    showlegend=False,
    title=None,
    margin=dict(l=0, r=0, b=0, t=0),
    scene=dict(
        aspectmode="data",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        bgcolor="white",
        camera=dict(eye=dict(x=1.55, y=-1.55, z=1.05)),
    ),
)


plotly_config = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "filename": "SLNAMified_QI_L_over_R_1p0_SLAM-A_coils",
        "format": "svg",
        "scale": 2,
    },
}

fig.write_html(
    output_filename,
    config=plotly_config,
    include_plotlyjs=True,
    full_html=True,
)

print("Coilset:", coilset_filename)
print("Stored top-level objects:", len(coilset))
print("Physical coils:", coilset.num_coils)
print("Plotly traces:", len(fig.data))
print("Saved:", output_filename)
