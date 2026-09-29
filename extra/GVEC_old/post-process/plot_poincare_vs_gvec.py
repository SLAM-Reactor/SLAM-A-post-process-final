#!/usr/bin/env python3
"""Plot DESC-coil Poincare points against GVEC flux-surface sections.

Edit the constants below to control the figure appearance. Running this script
regenerates the PNG and PDF files for every entry in CASES.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

from paths import DATA_DIR, FIGURE_DIR, ensure_dirs


# ----------------------------- cases to plot ----------------------------- #

OUTPUT_STEM = FIGURE_DIR / "poincare_vs_gvec"
FORMATS = ("png", "pdf")

CASES = (
    {
        "label": "SLNAM",
        "sections": DATA_DIR / "gvec_sections_for_poincare.npz",
        "traces": DATA_DIR / "desc_coil_poincare_traces.npz",
        "y_ticks_by_plane": None,
    },
    {
        "label": "SLAM_N128_RHO065_CLEAN1E4",
        "sections": DATA_DIR / "gvec_sections_for_poincare_SLAM_N128_RHO065_CLEAN1E4.npz",
        "traces": DATA_DIR / "desc_coil_poincare_traces_SLAM_N128_RHO065_CLEAN1E4.npz",
        "y_ticks_by_plane": (
            (-0.04, 0.0, 0.04),
            (-0.08, -0.04, 0.0, 0.04, 0.08),
        ),
    },
)


# ----------------------------- data switches ----------------------------- #

USE_LOGICAL_ZETA = False
MIRROR_GVEC_Z = False
MIRROR_POINCARE_Z = False


# ----------------------------- figure style ------------------------------ #

FIGSIZE = (7.2, 3.35)
PANEL_GAP = 0.40
PAD_FRACTION = 0.04

FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 24.0
TICK_SIZE = 20.0
LABEL_SIZE = 24.0
PLANE_LABEL_SIZE = 24.0

X_TICK_COUNT = 3
Y_TICK_COUNT = 4
TICK_WIDTH = 1.0
AXIS_WIDTH = 1.0

POINCARE_MARKER_SIZE = 2.1
POINCARE_ALPHA = 0.75

GVEC_LINEWIDTH = 0.7
GVEC_BOUNDARY_LINEWIDTH = 1.25
GVEC_ALPHA = 0.65
GVEC_BOUNDARY_ALPHA = 0.95


def label_index(labels: list[str], label: str, source: Path) -> int:
    if label in labels:
        return labels.index(label)
    choices = ", ".join(labels)
    raise ValueError(f"{label!r} is not in {source}; available labels: {choices}")


def plane_title(phi: float) -> str:
    if np.isclose(phi, 0.0):
        return r"$\phi = 0$"
    return r"$\phi = \pi/2$"


def set_axis_limits(ax: plt.Axes, r_values: np.ndarray, z_values: np.ndarray) -> None:
    finite = np.isfinite(r_values) & np.isfinite(z_values)
    r_values = r_values[finite]
    z_values = z_values[finite]

    width = max(float(r_values.max() - r_values.min()), float(z_values.max() - z_values.min()))
    pad = PAD_FRACTION * width
    ax.set_xlim(r_values.min() - pad, r_values.max() + pad)
    ax.set_ylim(z_values.min() - pad, z_values.max() + pad)


def style_axis(ax: plt.Axes, y_ticks: tuple[float, ...] | None, show_ylabel: bool) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("R (m)", fontsize=LABEL_SIZE)
    if show_ylabel:
        ax.set_ylabel("Z (m)", fontsize=LABEL_SIZE)

    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=TICK_SIZE,
        width=TICK_WIDTH,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_WIDTH)

    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=X_TICK_COUNT))
    if y_ticks is None:
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=Y_TICK_COUNT))
    else:
        ax.set_yticks(y_ticks)


def draw_gvec_surfaces(
    ax: plt.Axes,
    r_gvec: np.ndarray,
    z_gvec: np.ndarray,
    plane_index: int,
) -> None:
    boundary_index = r_gvec.shape[0] - 1
    for surface_index in range(r_gvec.shape[0]):
        is_boundary = surface_index == boundary_index
        ax.plot(
            r_gvec[surface_index, :, plane_index],
            z_gvec[surface_index, :, plane_index],
            color="black",
            lw=GVEC_BOUNDARY_LINEWIDTH if is_boundary else GVEC_LINEWIDTH,
            alpha=GVEC_BOUNDARY_ALPHA if is_boundary else GVEC_ALPHA,
        )


def draw_poincare_points(
    ax: plt.Axes,
    r_poincare: np.ndarray,
    z_poincare: np.ndarray,
    plane_index: int,
) -> None:
    ax.scatter(
        r_poincare[:, plane_index, :].ravel(),
        z_poincare[:, plane_index, :].ravel(),
        s=POINCARE_MARKER_SIZE,
        color="#0057B8",
        alpha=POINCARE_ALPHA,
        edgecolors="none",
        rasterized=True,
    )


def save_figure(fig: plt.Figure, label: str) -> None:
    suffix = []
    if USE_LOGICAL_ZETA:
        suffix.append("logical")
    if MIRROR_GVEC_Z:
        suffix.append("mirror-gvec-z")
    if MIRROR_POINCARE_Z:
        suffix.append("mirror-poincare-z")

    suffix_text = "" if not suffix else "_" + "_".join(suffix)
    for fmt in FORMATS:
        output = OUTPUT_STEM.parent / f"{OUTPUT_STEM.name}_{label}{suffix_text}.{fmt}"
        fig.savefig(output, dpi=450 if fmt == "png" else None, bbox_inches="tight")
        print(f"wrote {output}", flush=True)


def plot_case(case: dict) -> None:
    sections_path = Path(case["sections"])
    traces_path = Path(case["traces"])
    label = case["label"]

    sections = np.load(sections_path, allow_pickle=True)
    traces = np.load(traces_path, allow_pickle=True)

    section_labels = [str(x) for x in sections["labels"]]
    trace_labels = [str(x) for x in traces["labels"]]
    section_index = label_index(section_labels, label, sections_path)
    trace_index = label_index(trace_labels, label, traces_path)

    r_key, z_key = ("R_logical", "Z_logical") if USE_LOGICAL_ZETA else ("R_phi", "Z_phi")
    r_gvec = np.asarray(sections[r_key][section_index], dtype=float)
    z_gvec = np.asarray(sections[z_key][section_index], dtype=float)
    r_poincare = np.asarray(traces["R_poincare"][trace_index], dtype=float)
    z_poincare = np.asarray(traces["Z_poincare"][trace_index], dtype=float)
    phi_planes = np.asarray(sections["phi_planes"], dtype=float)

    if MIRROR_GVEC_Z:
        z_gvec = -z_gvec
    if MIRROR_POINCARE_Z:
        z_poincare = -z_poincare

    y_ticks_by_plane = case["y_ticks_by_plane"]
    if y_ticks_by_plane is None:
        y_ticks_by_plane = tuple(None for _ in phi_planes)
    if len(y_ticks_by_plane) != len(phi_planes):
        raise ValueError(f"{label}: expected {len(phi_planes)} Y tick groups")

    fig, axes = plt.subplots(1, len(phi_planes), figsize=FIGSIZE)
    axes = np.atleast_1d(axes)

    for plane_index, ax in enumerate(axes):
        draw_gvec_surfaces(ax, r_gvec, z_gvec, plane_index)
        draw_poincare_points(ax, r_poincare, z_poincare, plane_index)

        plane_r = np.concatenate(
            (
                r_gvec[:, :, plane_index].ravel(),
                r_poincare[:, plane_index, :].ravel(),
            )
        )
        plane_z = np.concatenate(
            (
                z_gvec[:, :, plane_index].ravel(),
                z_poincare[:, plane_index, :].ravel(),
            )
        )
        set_axis_limits(ax, plane_r, plane_z)
        style_axis(ax, y_ticks_by_plane[plane_index], show_ylabel=plane_index == 0)
        ax.set_title(plane_title(phi_planes[plane_index]), fontsize=PLANE_LABEL_SIZE)

    fig.subplots_adjust(wspace=PANEL_GAP)
    save_figure(fig, label)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": AXIS_WIDTH,
            "savefig.facecolor": "white",
        }
    )

    for case in CASES:
        plot_case(case)


if __name__ == "__main__":
    main()
