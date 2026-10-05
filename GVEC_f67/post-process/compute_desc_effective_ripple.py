#!/usr/bin/env python3
"""Compute DESC effective ripple for the original DESC equilibrium.

Run in desc-env. This uses DESC's field-line/Bounce1D style
`old effective ripple`, matching the comparison path used in
generalized_toroidal/stages/stage25_desc_ripple_1d.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np

from paths import DATA_DIR, DEFAULT_RHO_PROFILE, DESC_EQ, ensure_dirs, parse_floats


DESC_REPO = os.environ.get("DESC_REPO", "/pscratch/sd/r/rgaur/DESC")
sys.path.insert(0, DESC_REPO)

if os.environ.get("DESC_DEVICE", "gpu").lower() == "gpu":
    import desc  # noqa: E402

    desc.set_device("gpu")

from desc.equilibrium import Equilibrium  # noqa: E402
from desc.grid import Grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", default=None, help="comma-separated rho values")
    parser.add_argument("--output", default=str(DATA_DIR / "effective_ripple_desc_original.npz"))
    parser.add_argument("--eq", default=str(DESC_EQ))
    parser.add_argument("--num-transit", type=int, default=10)
    parser.add_argument("--num-well", type=int, default=250)
    parser.add_argument("--num-quad", type=int, default=19)
    parser.add_argument("--tol", type=float, default=1e-10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    rho = parse_floats(args.rho, DEFAULT_RHO_PROFILE)
    eq = Equilibrium.load(args.eq)
    y_b = eq.N_grid * 2 * eq.NFP
    zeta = np.linspace(0.0, args.num_transit * 2.0 * np.pi, args.num_transit * y_b)
    grid = Grid.create_meshgrid([rho, np.array([0.0]), zeta], coordinates="raz")

    try:
        data = eq.compute(
            "old effective ripple",
            grid,
            num_well=args.num_well,
            num_quad=args.num_quad,
            tol=args.tol,
        )
        ripple = np.asarray(grid.compress(data["old effective ripple"]))
        status = "ok"
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        ripple = np.full(rho.size, np.nan)
        status = f"failed: {type(exc).__name__}: {str(exc)[:180]}"

    for r, v in zip(rho, ripple):
        print(f"rho={r:.3f} eff ripple DESC original = {v:.6e}", flush=True)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "compute_desc_effective_ripple.py",
        "desc_repo": DESC_REPO,
        "equilibrium": args.eq,
        "rho": rho.tolist(),
        "num_transit": args.num_transit,
        "num_well": args.num_well,
        "num_quad": args.num_quad,
        "tol": args.tol,
        "status": status,
    }
    np.savez(
        args.output,
        rho=rho,
        ripple_desc_original=ripple,
        metadata=json.dumps(metadata, indent=2),
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
