#!/usr/bin/env python3
"""Plot the magnetic-well comparison from cached NPZ files.

The main comparison uses ``desc_style_well`` for GVEC and DESC's native
``magnetic well`` cache. GVEC's cumulative/native vacuum-well convention is a
different diagnostic and is only plotted when SHOW_NATIVE_GVEC_DIAGNOSTIC=True.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------- input/output ------------------------------ #

POST_DIR = Path(__file__).resolve().parent
DATA_DIR = POST_DIR / "data"
FIGURE_DIR = POST_DIR / "figures"

GVEC_CACHE = DATA_DIR / "magnetic_well_gvec.npz"
DESC_CACHE = DATA_DIR / "magnetic_well_desc_original.npz"
OUTPUT_STEM = FIGURE_DIR / "magnetic_well_comparison"
FORMATS = ("png", "pdf")

WELL_KIND = "desc_style_well"
SHOW_NATIVE_GVEC_DIAGNOSTIC = False
MAX_RHO_TO_PLOT = 0.97


# ----------------------------- figure style ------------------------------ #

FIGSIZE = (5.2, 3.7)
FONT_FAMILY = "DejaVu Sans"
FONT_SIZE = 11.0
TICK_SIZE = 11.0
LABEL_SIZE = 12.0
LINEWIDTH = 1.6
DESC_LINEWIDTH = 1.9
MARKER_SIZE = 4.0
GRID_ALPHA = 0.28
LEGEND_SIZE = 8.5

GVEC_STYLES = {
    "SLNAM": {"color": "#0057B8", "marker": "o"},
    "SLAM_N128_RHO065_CLEAN1E4": {"color": "#D55E00", "marker": "^"},
}
DESC_STYLE = {"color": "black", "marker": "s"}


def style_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel(r"$\rho$", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=TICK_SIZE)
    ax.grid(alpha=GRID_ALPHA)


def plot_desc_style_comparison(gvec, desc) -> plt.Figure:
    labels = [str(label) for label in gvec["labels"]]
    gvec_mask = np.asarray(gvec["rho"], dtype=float) <= MAX_RHO_TO_PLOT
    desc_mask = np.asarray(desc["rho"], dtype=float) <= MAX_RHO_TO_PLOT
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for index, label in enumerate(labels):
        style = GVEC_STYLES.get(label, {})
        ax.plot(
            gvec["rho"][gvec_mask],
            gvec[WELL_KIND][index][gvec_mask],
            lw=LINEWIDTH,
            ms=MARKER_SIZE,
            label=f"{label} GVEC",
            **style,
        )

    ax.plot(
        desc["rho"][desc_mask],
        desc["magnetic_well_desc_original"][desc_mask],
        lw=DESC_LINEWIDTH,
        ms=MARKER_SIZE,
        label="DESC original",
        **DESC_STYLE,
    )
    ax.axhline(0.0, color="0.55", lw=0.8)
    style_axes(ax, "magnetic well")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    fig.tight_layout()
    return fig


def plot_native_gvec_diagnostic(gvec) -> plt.Figure:
    labels = [str(label) for label in gvec["labels"]]
    gvec_mask = np.asarray(gvec["rho"], dtype=float) <= MAX_RHO_TO_PLOT
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for index, label in enumerate(labels):
        style = GVEC_STYLES.get(label, {})
        ax.plot(
            gvec["rho"][gvec_mask],
            gvec["cumulative_vacuum_well"][index][gvec_mask],
            lw=LINEWIDTH,
            ms=MARKER_SIZE,
            label=f"{label} cumulative",
            **style,
        )
        ax.axhline(
            gvec["native_vacuum_well_depth"][index],
            color=style.get("color"),
            ls="--",
            lw=1.0,
            alpha=0.75,
        )

    ax.axhline(0.0, color="0.55", lw=0.8)
    style_axes(ax, "cumulative vacuum well depth")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, stem, suffix: str = "") -> None:
    for fmt in FORMATS:
        output = stem.parent / f"{stem.name}{suffix}.{fmt}"
        fig.savefig(output, dpi=450 if fmt == "png" else None, bbox_inches="tight")
        print(f"wrote {output}", flush=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "mathtext.fontset": "dejavusans",
            "savefig.facecolor": "white",
        }
    )

    gvec = np.load(GVEC_CACHE, allow_pickle=True)
    desc = np.load(DESC_CACHE, allow_pickle=True)

    fig = plot_desc_style_comparison(gvec, desc)
    save_figure(fig, OUTPUT_STEM)
    plt.close(fig)

    if SHOW_NATIVE_GVEC_DIAGNOSTIC:
        fig = plot_native_gvec_diagnostic(gvec)
        save_figure(fig, OUTPUT_STEM, suffix="_gvec_native")
        plt.close(fig)


if __name__ == "__main__":
    main()
