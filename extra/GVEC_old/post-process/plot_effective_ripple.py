#!/usr/bin/env python3
"""Plot effective-ripple comparison from cached NPZ files."""

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

GVEC_CACHE = DATA_DIR / "effective_ripple_gvec.npz"
DESC_CACHE = DATA_DIR / "effective_ripple_desc_original.npz"
OUTPUT_STEM = FIGURE_DIR / "effective_ripple_comparison"
FORMATS = ("png", "pdf")


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
Y_SCALE = "linear"

GVEC_STYLES = {
    "SLNAM": {"color": "#0057B8", "marker": "o"},
    "SLAM_N128_RHO065_CLEAN1E4": {"color": "#D55E00", "marker": "^"},
}
DESC_STYLE = {"color": "black", "marker": "s"}


def style_axes(ax: plt.Axes) -> None:
    ax.set_xlabel(r"$\rho$", fontsize=LABEL_SIZE)
    ax.set_ylabel("effective ripple", fontsize=LABEL_SIZE)
    ax.set_yscale(Y_SCALE)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=TICK_SIZE)
    ax.grid(alpha=GRID_ALPHA, which="both")


def save_figure(fig: plt.Figure) -> None:
    for fmt in FORMATS:
        output = OUTPUT_STEM.parent / f"{OUTPUT_STEM.name}.{fmt}"
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
    labels = [str(label) for label in gvec["labels"]]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for index, label in enumerate(labels):
        style = GVEC_STYLES.get(label, {})
        ax.plot(
            gvec["rho"],
            gvec["ripple"][index],
            lw=LINEWIDTH,
            ms=MARKER_SIZE,
            label=f"{label} GVEC",
            **style,
        )

    ax.plot(
        desc["rho"],
        desc["ripple_desc_original"],
        lw=DESC_LINEWIDTH,
        ms=MARKER_SIZE,
        label="DESC original",
        **DESC_STYLE,
    )
    style_axes(ax)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE)
    fig.tight_layout()
    save_figure(fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
