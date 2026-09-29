#!/usr/bin/env python
"""Make the two QFM surface + coil HTML figures.

Run from QFM_surf:
    python plot_qfm_surfaces.py
"""

import json
import os

import numpy as np
import plotly.graph_objects as go

from desc.grid import Grid, LinearGrid
from desc.io import load


CASES = [
    ("SLNAM", "SLNAM-field-09-09-26.h5", "SLNAM_QFM_surface_09-09-26_3.h5",
     "SLNAM_QFM_surface_and_coils_Bn.html", 32, 40, False, 32, 40, False),
    ("SLAM", "SLAM-field-09-09-26_2.h5", "SLAM_QFM_surface_09-09-26_2.h5",
     "SLAM_QFM_surface_and_coils_Bn.html", 20, 24, True, 20, 24, True),
]

USE_CACHED_DATA = True
RECOMPUTE_LABELS = []
USE_DIRECT_SURFACE_GEOMETRY = True
WRITE_HTML = True
PLOTLY_EXPORT_FORMAT = "svg"
PLOTLY_EXPORT_SCALE = 2
DATA_FILE = "qfm_surface_Bn_data.npz"
METRICS_FILE = "qfm_surface_Bn_metrics.json"
COIL_NPTS = 800
PLOT_WHOLE_TORUS = False
CHUNK = 500
CLIP_PERCENTILE = 100.0
DISPLAY_REFINE = {"SLNAM": (1, 1), "SLAM": (4, 4)}

FONT_FAMILY = "Arial"
FIG_WIDTH = 1000
FIG_HEIGHT = 1000
AXIS_TICK_SIZE = 14
AXIS_LABEL_SIZE = 18
COIL_WIDTH = 6
COLORBAR_TICK_SIZE = 48
COLORBAR_THICKNESS = 26
COLORBAR_LENGTH = 0.70
COLORBAR_DECIMALS = 1
COLORBAR_X = 1.03
COLORBAR_Y = 0.50
COLORBAR_XANCHOR = "left"
COLORBAR_YANCHOR = "middle"
MARGIN_LEFT = 0
MARGIN_RIGHT = 170
MARGIN_TOP = 65
MARGIN_BOTTOM = 0

if not USE_CACHED_DATA or RECOMPUTE_LABELS:
    from desc import set_device

    set_device("cpu")

    from desc.coils import CoilSet


def flatten(coilset):
    if not hasattr(coilset, "__len__"):
        return [coilset]
    if hasattr(coilset, "_NFP") and hasattr(coilset, "_sym") and (coilset.NFP > 1 or coilset.sym):
        coilset = CoilSet.from_symmetry(
            coilset, NFP=coilset.NFP, sym=coilset.sym, check_intersection=False
        )
    coils = []
    for sub in coilset:
        coils.extend(flatten(sub))
    return coils


def coil_xyz_and_group(field):
    grid = LinearGrid(N=COIL_NPTS, endpoint=True)
    coils = flatten(field)
    xyz = np.array([np.asarray(c.compute("x", grid=grid, basis="xyz")["x"]) for c in coils])
    planar = np.array([type(c).__name__ == "FourierPlanarCoil" for c in coils])
    group = np.where(planar, 2, 0)  # 0 modular, 1 end mirror, 2 mirror
    span_z = np.ptp(xyz, axis=1)[:, 2]
    center_x = np.abs(xyz.mean(axis=1)[:, 0])
    if int((span_z[~planar] > 1.0).sum()) == 4:
        idx = np.where(~planar)[0]
        group[idx[np.argsort(span_z[idx])[-4:]]] = 1
    else:
        idx = np.where(planar)[0]
        group[idx[np.argsort(center_x[idx])[-4:]]] = 1
    return xyz, group, np.array([float(c.current) for c in coils])


def mesh_and_metrics(field, surface, metric_m, metric_n, metric_sym, plot_m, plot_n, plot_sym):
    if PLOT_WHOLE_TORUS:
        plot_grid = LinearGrid(
            rho=np.array([1.0]),
            M=plot_m,
            N=plot_n * surface.NFP,
            NFP=1,
            sym=False,
            endpoint=False,
        )
    else:
        plot_grid = LinearGrid(
            rho=np.array([1.0]),
            M=plot_m,
            N=plot_n,
            NFP=surface.NFP,
            sym=plot_sym,
            endpoint=False,
        )
    coords = surface.compute(["X", "Y", "Z"], grid=plot_grid)
    bn, xyz = field.compute_Bnormal(surface, eval_grid=plot_grid, chunk_size=CHUNK)
    bvec = field.compute_magnetic_field(xyz)
    shape = (plot_grid.num_theta, plot_grid.num_zeta)
    mesh = {
        "X": np.asarray(coords["X"]).reshape(shape, order="F"),
        "Y": np.asarray(coords["Y"]).reshape(shape, order="F"),
        "Z": np.asarray(coords["Z"]).reshape(shape, order="F"),
        "Bn": np.asarray(bn).reshape(shape, order="F"),
        "Bmag": np.linalg.norm(np.asarray(bvec), axis=1).reshape(shape, order="F"),
    }
    mesh["Bn_over_meanB"] = mesh["Bn"] / float(np.mean(mesh["Bmag"]))

    metric_grid = LinearGrid(
        rho=np.array([1.0]),
        M=metric_m,
        N=metric_n,
        NFP=surface.NFP,
        sym=metric_sym,
        endpoint=False,
    )
    bn_metric, xyz_metric = field.compute_Bnormal(surface, eval_grid=metric_grid, chunk_size=CHUNK)
    bn_metric = np.asarray(bn_metric)
    bmag_metric = np.linalg.norm(np.asarray(field.compute_magnetic_field(xyz_metric)), axis=1)
    mean_abs = float(np.mean(np.abs(bn_metric)))
    rms = float(np.sqrt(np.mean(bn_metric * bn_metric)))
    max_abs = float(np.max(np.abs(bn_metric)))
    mean_b = float(np.mean(bmag_metric))
    metrics = {
        "nodes": int(metric_grid.num_nodes),
        "grid_M": int(metric_m),
        "grid_N": int(metric_n),
        "grid_NFP": int(surface.NFP),
        "grid_sym": bool(metric_sym),
        "plot_nodes": int(plot_grid.num_nodes),
        "plot_grid_M": int(plot_m),
        "plot_grid_N": int(plot_n * surface.NFP if PLOT_WHOLE_TORUS else plot_n),
        "plot_grid_NFP": int(1 if PLOT_WHOLE_TORUS else surface.NFP),
        "plot_grid_sym": False if PLOT_WHOLE_TORUS else bool(plot_sym),
        "mean_abs_Bn": mean_abs,
        "rms_Bn": rms,
        "max_abs_Bn": max_abs,
        "mean_B": mean_b,
        "mean_abs_Bn_over_mean_B": mean_abs / mean_b,
        "rms_Bn_over_mean_B": rms / mean_b,
        "max_abs_Bn_over_mean_B": max_abs / mean_b,
        "plot_mean_abs_Bn": float(np.mean(np.abs(mesh["Bn"]))),
        "plot_max_abs_Bn": float(np.max(np.abs(mesh["Bn"]))),
        "plot_mean_B": float(np.mean(mesh["Bmag"])),
        "plot_mean_abs_Bn_over_mean_B": float(np.mean(np.abs(mesh["Bn_over_meanB"]))),
        "plot_max_abs_Bn_over_mean_B": float(np.max(np.abs(mesh["Bn_over_meanB"]))),
    }
    return mesh, metrics


def coil_color(xyz, group):
    if int(group) == 1:
        return "green"
    if int(group) == 2:
        return "black"
    return "blue" if xyz[:, 0].mean() < 0 else "red"


def display_mesh(mesh, nfp, grid_sym):
    out = {key: np.array(value, copy=True) for key, value in mesh.items()}
    if grid_sym:
        phi = np.unwrap(np.arctan2(out["Y"][0], out["X"][0]))
        period = 2.0 * np.pi / nfp
        phi = np.mod(phi, period)
        j_neg = np.array([np.argmin(np.abs(np.mod(phi + ph, period))) for ph in phi])
        src = np.arange(out["X"].shape[0] - 1, -1, -1)
        R = np.sqrt(out["X"][src][:, j_neg] ** 2 + out["Y"][src][:, j_neg] ** 2)
        X = R * np.cos(phi)
        Y = R * np.sin(phi)
        out["X"] = np.vstack([out["X"], X])
        out["Y"] = np.vstack([out["Y"], Y])
        out["Z"] = np.vstack([out["Z"], -out["Z"][src][:, j_neg]])
        out["Bn"] = np.vstack([out["Bn"], out["Bn"][src][:, j_neg]])
        out["Bmag"] = np.vstack([out["Bmag"], out["Bmag"][src][:, j_neg]])
        out["Bn_over_meanB"] = np.vstack(
            [out["Bn_over_meanB"], out["Bn_over_meanB"][src][:, j_neg]]
        )
    pieces = []
    for k in range(nfp):
        angle = 2.0 * np.pi * k / nfp
        ca = np.cos(angle)
        sa = np.sin(angle)
        piece = {key: np.array(value, copy=True) for key, value in out.items()}
        piece["X"] = out["X"] * ca - out["Y"] * sa
        piece["Y"] = out["X"] * sa + out["Y"] * ca
        pieces.append(piece)
    return {key: np.concatenate([piece[key] for piece in pieces], axis=1) for key in out}


def refine_mesh(mesh, theta_factor, zeta_factor):
    if theta_factor == 1 and zeta_factor == 1:
        return mesh
    refined = {}
    for key, value in mesh.items():
        nt, nz = value.shape
        theta_new = np.linspace(0.0, nt, nt * theta_factor, endpoint=False)
        zeta_new = np.linspace(0.0, nz, nz * zeta_factor, endpoint=False)
        value_theta = np.vstack([value, value[:1]])
        tmp = np.empty((theta_new.size, nz))
        for j in range(nz):
            tmp[:, j] = np.interp(theta_new, np.arange(nt + 1), value_theta[:, j])
        tmp_zeta = np.hstack([tmp, tmp[:, :1]])
        arr = np.empty((theta_new.size, zeta_new.size))
        for i in range(theta_new.size):
            arr[i] = np.interp(zeta_new, np.arange(nz + 1), tmp_zeta[i])
        refined[key] = arr
    return refined


def replace_with_direct_surface_geometry(surface_file, mesh):
    surface = load(surface_file)
    ntheta, nzeta = mesh["X"].shape
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    nodes = np.column_stack(
        [
            np.ones(theta_grid.size),
            theta_grid.ravel(order="F"),
            zeta_grid.ravel(order="F"),
        ]
    )
    grid = Grid(nodes, sort=False)
    coords = surface.compute(["X", "Y", "Z"], grid=grid)
    out = {key: np.array(value, copy=True) for key, value in mesh.items()}
    shape = (grid.num_theta, grid.num_zeta)
    out["X"] = np.asarray(coords["X"]).reshape(shape, order="F")
    out["Y"] = np.asarray(coords["Y"]).reshape(shape, order="F")
    out["Z"] = np.asarray(coords["Z"]).reshape(shape, order="F")
    return out


def write_outputs(html_file, mesh, coil_xyz, coil_group, cmin, cmax, label):
    fig = go.Figure()
    for xyz, group in zip(coil_xyz, coil_group):
        color = coil_color(xyz, group)
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                                   mode="lines", line=dict(color=color, width=COIL_WIDTH),
                                   showlegend=False, hoverinfo="skip"))

    ntheta, nzeta = mesh["X"].shape
    a, b = np.meshgrid(np.arange(ntheta), np.arange(nzeta), indexing="ij")
    p00 = (a * nzeta + b).ravel()
    p10 = (((a + 1) % ntheta) * nzeta + b).ravel()
    p01 = (a * nzeta + ((b + 1) % nzeta)).ravel()
    p11 = (((a + 1) % ntheta) * nzeta + ((b + 1) % nzeta)).ravel()
    i = np.concatenate([p00, p00])
    j = np.concatenate([p10, p11])
    k = np.concatenate([p11, p01])
    ticks = np.linspace(cmin, cmax, 5)
    peak = max(abs(cmin), abs(cmax))
    exp = int(np.floor(np.log10(peak))) if peak > 0 else 0
    scale = 10.0**exp

    fig.add_trace(go.Mesh3d(x=mesh["X"].ravel(), y=mesh["Y"].ravel(), z=mesh["Z"].ravel(),
                            i=i, j=j, k=k, intensity=mesh["Bn_over_meanB"].ravel(),
                            cmin=cmin, cmax=cmax, colorscale="RdBu_r",
                            opacity=1.0, flatshading=True, showscale=True,
                            name=f"{label} normalized normal field",
                            colorbar=dict(
                                tickmode="array", tickvals=ticks,
                                ticktext=[f"{v / scale:.{COLORBAR_DECIMALS}f}" for v in ticks],
                                tickfont=dict(size=COLORBAR_TICK_SIZE),
                                len=COLORBAR_LENGTH, thickness=COLORBAR_THICKNESS,
                                x=COLORBAR_X, y=COLORBAR_Y,
                                xanchor=COLORBAR_XANCHOR, yanchor=COLORBAR_YANCHOR,
                            )))

    axis = dict(backgroundcolor="white", gridcolor="darkgrey", showbackground=False,
                zerolinecolor="darkgrey", showgrid=True, zeroline=True,
                showticklabels=True, tickfont=dict(size=AXIS_TICK_SIZE))
    fig.update_layout(scene=dict(
        aspectmode="data",
        xaxis=dict(title=dict(text="X (m)", font=dict(size=AXIS_LABEL_SIZE)), **axis),
        yaxis=dict(title=dict(text="Y (m)", font=dict(size=AXIS_LABEL_SIZE)), **axis),
        zaxis=dict(title=dict(text="Z (m)", font=dict(size=AXIS_LABEL_SIZE)), **axis),
    ), width=FIG_WIDTH, height=FIG_HEIGHT, font=dict(family=FONT_FAMILY),
        paper_bgcolor="white",
        margin=dict(l=MARGIN_LEFT, r=MARGIN_RIGHT, t=MARGIN_TOP, b=MARGIN_BOTTOM))
    if WRITE_HTML:
        plotly_config = {
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "filename": html_file.replace(".html", ""),
                "format": PLOTLY_EXPORT_FORMAT,
                "scale": PLOTLY_EXPORT_SCALE,
            },
        }
        fig.write_html(html_file, config=plotly_config, include_plotlyjs=True, full_html=True)


meshes = {}
coil_data = {}
metrics = {}
npz = {}
surface_files = {}
old_data = np.load(DATA_FILE) if os.path.exists(DATA_FILE) else None
old_metrics = None
if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        old_metrics = json.load(f)
        metrics = dict(old_metrics)

for label, field_file, surface_file, html_file, metric_m, metric_n, metric_sym, plot_m, plot_n, plot_sym in CASES:
    prefix = label.lower()
    surface_files[label] = surface_file
    if USE_CACHED_DATA and label not in RECOMPUTE_LABELS:
        data = old_data
        meshes[label] = {
            "X": data[f"{prefix}_X"],
            "Y": data[f"{prefix}_Y"],
            "Z": data[f"{prefix}_Z"],
            "Bn": data[f"{prefix}_Bn"],
            "Bmag": data[f"{prefix}_Bmag"],
            "Bn_over_meanB": data[f"{prefix}_Bn_over_meanB"],
        }
        coil_data[label] = (
            data[f"{prefix}_coil_xyz"],
            data[f"{prefix}_coil_group"],
            data[f"{prefix}_coil_current"],
            html_file,
        )
        print(label, "cached", DATA_FILE)
        print("  <|B.n|>/<|B|> =", f"{metrics[label]['mean_abs_Bn_over_mean_B']:.6e}")
    else:
        print(label, field_file, surface_file)
        field = load(field_file)
        surface = load(surface_file)
        if old_data is not None:
            coil_xyz = old_data[f"{prefix}_coil_xyz"]
            coil_group = old_data[f"{prefix}_coil_group"]
            coil_current = old_data[f"{prefix}_coil_current"]
        else:
            coil_xyz, coil_group, coil_current = coil_xyz_and_group(field)
        mesh, metric = mesh_and_metrics(
            field, surface, metric_m, metric_n, metric_sym, plot_m, plot_n, plot_sym
        )
        metric["field"] = field_file
        metric["surface"] = surface_file
        metric["n_coils"] = int(len(coil_xyz))
        metric["coil_groups"] = [int(x) for x in np.bincount(coil_group, minlength=3)]
        meshes[label] = mesh
        coil_data[label] = (coil_xyz, coil_group, coil_current, html_file)
        metrics[label] = metric
        print("  <|B.n|>/<|B|> =", f"{metric['mean_abs_Bn_over_mean_B']:.6e}")

for label, mesh in meshes.items():
    coil_xyz, coil_group, coil_current, html_file = coil_data[label]
    mesh_for_plot = display_mesh(mesh, metrics[label]["grid_NFP"], metrics[label]["grid_sym"])
    mesh_for_plot = refine_mesh(mesh_for_plot, *DISPLAY_REFINE[label])
    if USE_DIRECT_SURFACE_GEOMETRY:
        mesh_for_plot = replace_with_direct_surface_geometry(surface_files[label], mesh_for_plot)
    cmax = float(np.percentile(np.abs(mesh["Bn_over_meanB"].ravel()), CLIP_PERCENTILE))
    cmin = -cmax
    write_outputs(html_file, mesh_for_plot, coil_xyz, coil_group, cmin, cmax, label)
    prefix = label.lower()
    if not USE_CACHED_DATA or RECOMPUTE_LABELS:
        for key, value in mesh.items():
            npz[f"{prefix}_{key}"] = value
        npz[f"{prefix}_coil_xyz"] = coil_xyz
        npz[f"{prefix}_coil_group"] = coil_group
        npz[f"{prefix}_coil_current"] = coil_current
    if WRITE_HTML:
        print("  wrote", html_file)

if not USE_CACHED_DATA or RECOMPUTE_LABELS:
    np.savez_compressed(DATA_FILE, **npz)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("wrote", DATA_FILE)
    print("wrote", METRICS_FILE)
