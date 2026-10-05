#!/usr/bin/env python3
"""Cheap local GVEC initialization check before submitting a full solve.

This intentionally runs on the login node with totalIter=1. Its job is to catch
bad theta orientation, insufficient smoothing, or a bad axis guess before a
multi-hour Slurm job is submitted.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=("SLAM", "SLNAM"))
    parser.add_argument("--clean", required=True, help="pygvec load-quasr --clean value, e.g. 3e-3")
    parser.add_argument("--axis-move-x1", type=float, required=True)
    parser.add_argument("--axis-move-x2", type=float, default=0.0)
    parser.add_argument("--fname1", default="1p0")
    parser.add_argument("--nturns", default="216")
    parser.add_argument("--fname2", default="0p92")
    parser.add_argument("--nfp", type=int, default=2)
    parser.add_argument("--n-elems", type=int, default=8)
    parser.add_argument("--timeout", default="10m")
    parser.add_argument("--keep", action="store_true", help="keep any previous smoke directory/files")
    parser.add_argument("--no-reverse-theta", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def tag_value(value: str | float) -> str:
    return str(value).replace("+", "p").replace("-", "m").replace(".", "p")


def main() -> int:
    args = parse_args()
    clean_tag = tag_value(args.clean)
    x1_tag = tag_value(f"{args.axis_move_x1:+.3f}")
    x2_tag = tag_value(f"{args.axis_move_x2:+.3f}")
    base = (
        f"smoke_{args.label}_{args.fname1}_{args.nturns}_{args.fname2}_"
        f"clean{clean_tag}_axis_x1{x1_tag}_x2{x2_tag}"
    )

    if not args.keep:
        for path in [
            Path(f"{base}_gvec_stages"),
            Path(f"{base}.nc"),
            Path(f"{base}.png"),
            Path(f"{base}-Gframe.nc"),
            Path(f"{base}-parameters.toml"),
            Path(f"{base}_State_final.dat"),
            Path(f"parameter_{base}_final.ini"),
        ]:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "12"
    env["OM_NUM_THREADS"] = "12"

    sym_cmd = [
        sys.executable,
        "-u",
        "GVEC_Boozer_spline_symmetrize.py",
        "--label",
        args.label,
        "--fname1",
        args.fname1,
        "--nturns",
        args.nturns,
        "--fname2",
        args.fname2,
        "--nfp",
        str(args.nfp),
        "--output-name",
        base,
    ]
    if not args.no_reverse_theta:
        sym_cmd.append("--reverse-theta")
    run(sym_cmd, env)

    run(
        [
            "pygvec",
            "load-quasr",
            "-f",
            f"{base}.nc",
            "-v",
            "--stellsym",
            "--tol",
            "1e-4",
            "--clean",
            args.clean,
            "--name",
            base,
        ],
        env,
    )

    run(
        [
            sys.executable,
            "-u",
            "patch_gvec_params.py",
            f"{base}-parameters.toml",
            "--n-elems",
            str(args.n_elems),
            "--phi-edge",
            "0.01",
            "--total-iter",
            "1",
            "--init-average-axis",
            "true",
            "--axis-move-x1",
            str(args.axis_move_x1),
            "--axis-move-x2",
            str(args.axis_move_x2),
        ],
        env,
    )

    run(["timeout", args.timeout, "pygvec", "run", f"{base}-parameters.toml"], env)
    print(f"SMOKE PASS: {base}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
