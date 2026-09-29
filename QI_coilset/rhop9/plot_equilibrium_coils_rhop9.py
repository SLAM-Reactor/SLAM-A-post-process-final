#!/usr/bin/env python3
"""Plotly plot of the QI rho=0.9 surface, coilset, and coil B*n."""

import os
import sys
import warnings

DESC_SOURCE = os.environ.get("DESC_SOURCE", "/pscratch/sd/r/rgaur/DESC3/DESC")
sys.path.insert(0, DESC_SOURCE)

import numpy as np

from desc.coils import CoilSet
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.plotting import plot_3d, plot_coils


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
coilset_filename = os.path.join(qi_dir, "NFP2_A8p7_coilset_40_coils_Bn_0p015_N9.h5")
output_filename = "QI_rhop9_surface_and_40_coils_Bn_plotly.html"

rho = 0.9
surface_M = 60
surface_N = 120
coil_points = 400
biot_savart_chunk_size = 256

coil_color = "black"
coil_line_width = 7
surface_color_map = "RdBu_r"
colorbar_tick_font_size = 58
colorbar_title_font_size = 58
colorbar_length = 0.8


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module=r"desc\.io\.hdf5_io",
    )
    eq = Equilibrium.load(equilibrium_filename)

field = CoilSet.load(coilset_filename)
surf = eq.get_surface_at(rho=rho)

surface_grid = LinearGrid(
    rho=1.0,
    M=surface_M,
    N=surface_N,
    NFP=1,
    endpoint=True,
)
coil_plot_grid = LinearGrid(N=coil_points, endpoint=True)
field_grid = LinearGrid(N=coil_points, endpoint=False)

fig = plot_coils(
    field,
    grid=coil_plot_grid,
    lw=coil_line_width,
    color=coil_color,
    showgrid=False,
    showticklabels=False,
    showaxislabels=False,
    zeroline=False,
)

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Unequal number of field periods for grid.*",
    )
    fig = plot_3d(
        surf,
        name="B*n",
        grid=surface_grid,
        fig=fig,
        field=field,
        field_grid=field_grid,
        chunk_size=biot_savart_chunk_size,
        cmap=surface_color_map,
        showgrid=False,
        showticklabels=False,
        showaxislabels=False,
        zeroline=False,
    )

surface_trace = fig.data[-1]
surface_trace.update(
    flatshading=False,
    intensitymode="vertex",
    lighting=dict(
        ambient=0.90,
        diffuse=0.45,
        roughness=1.0,
        specular=0.0,
        fresnel=0.0,
        facenormalsepsilon=1e-6,
        vertexnormalsepsilon=1e-6,
    ),
    colorbar=dict(
        title=dict(font=dict(size=colorbar_title_font_size)),
        tickfont=dict(size=colorbar_tick_font_size),
        len=colorbar_length,
    ),
)

fig.update_layout(
    width=1100,
    height=850,
    font=dict(size=18, color="black", family="Arial, sans-serif"),
    paper_bgcolor="white",
    plot_bgcolor="white",
    showlegend=False,
    title=None,
    margin=dict(l=0, r=20, b=0, t=10),
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
        "filename": "QI_rhop9_surface_and_40_coils_Bn",
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

Bn = np.asarray(surface_trace.intensity, dtype=float)
print("Equilibrium:", equilibrium_filename)
print("Coilset:", coilset_filename)
print("Surface rho:", rho)
print("Stored unique coils:", len(field))
print("Physical coils:", field.num_coils)
print("B*n minimum [T]:", np.nanmin(Bn))
print("B*n maximum [T]:", np.nanmax(Bn))
print("Saved:", output_filename)
