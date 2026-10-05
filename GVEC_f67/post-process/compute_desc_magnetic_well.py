#!/usr/bin/env python3
"""Compute DESC magnetic-well profile for the original DESC equilibrium.

Run in desc-env. This saves DESC's native `magnetic well` quantity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np

from paths import DATA_DIR, DEFAULT_RHO_WELL, DESC_EQ, ensure_dirs, parse_floats


DESC_REPO = os.environ.get("DESC_REPO", "/pscratch/sd/r/rgaur/DESC")
sys.path.insert(0, DESC_REPO)

from desc.equilibrium import Equilibrium  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", default=None, help="comma-separated rho values")
    parser.add_argument("--eq", default=str(DESC_EQ))
    parser.add_argument("--output", default=str(DATA_DIR / "magnetic_well_desc_original.npz"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    rho = parse_floats(args.rho, DEFAULT_RHO_WELL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        eq = Equilibrium.load(args.eq)

    grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)
    data = eq.compute(["magnetic well", "iota"], grid=grid)
    magnetic_well = np.asarray(grid.compress(data["magnetic well"]), dtype=float)
    iota = np.asarray(grid.compress(data["iota"]), dtype=float)

    for r, w in zip(rho, magnetic_well):
        print(f"rho={r:.3f} DESC magnetic well = {w:+.6e}", flush=True)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "compute_desc_magnetic_well.py",
        "desc_repo": DESC_REPO,
        "equilibrium": args.eq,
        "rho": rho.tolist(),
        "quantity": "DESC native `magnetic well`",
    }
    np.savez(
        args.output,
        rho=rho,
        magnetic_well_desc_original=magnetic_well,
        iota_desc_original=iota,
        metadata=json.dumps(metadata, indent=2),
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
