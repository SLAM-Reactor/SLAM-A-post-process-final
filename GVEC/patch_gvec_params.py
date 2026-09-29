#!/usr/bin/env python3
"""Patch pyGVEC TOML files with the local SLAM settings."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("toml")
    parser.add_argument("--n-elems", type=int, default=8)
    parser.add_argument("--phi-edge", default="0.01")
    parser.add_argument("--minimize-tol", default=None)
    parser.add_argument("--total-iter", type=int, default=None)
    parser.add_argument("--init-average-axis", choices=("true", "false", "keep"), default="true")
    parser.add_argument("--init-la", choices=("true", "false", "keep"), default="keep")
    parser.add_argument("--axis-move-x1", type=float, default=None)
    parser.add_argument("--axis-move-x2", type=float, default=None)
    return parser.parse_args()


def upsert(lines: list[str], pattern: str, value: str, insert_after: str | None = None) -> list[str]:
    rx = re.compile(pattern)
    for i, line in enumerate(lines):
        if rx.match(line):
            lines[i] = value
            return lines
    if insert_after is not None:
        after_rx = re.compile(insert_after)
        for i, line in enumerate(lines):
            if after_rx.match(line):
                lines.insert(i + 1, value)
                return lines
    lines.append(value)
    return lines


def main() -> None:
    args = parse_args()
    path = Path(args.toml)
    lines = path.read_text().splitlines()

    lines = [line for line in lines if not re.match(r"^\s*PHIEDGE\s*=", line)]
    if args.init_average_axis != "keep":
        lines = upsert(lines, r"^\s*init_average_axis\s*=", f"init_average_axis = {args.init_average_axis}")
    lines = upsert(lines, r"^\s*PHIEDGE\s*=", f"PHIEDGE = {args.phi_edge}", r"^\s*init_average_axis\s*=")
    if args.total_iter is not None:
        lines = upsert(lines, r"^\s*totalIter\s*=", f"totalIter = {args.total_iter}")
    if args.minimize_tol is not None:
        lines = upsert(lines, r"^\s*minimize_tol\s*=", f"minimize_tol = {args.minimize_tol}")
    if args.init_la != "keep":
        lines = upsert(lines, r"^\s*init_LA\s*=", f"init_LA = {args.init_la}", r"^\s*PHIEDGE\s*=")
    if args.axis_move_x1 is not None:
        lines = upsert(
            lines,
            r"^\s*average_axis_move_X1\s*=",
            f"average_axis_move_X1 = {args.axis_move_x1:.12g}",
            r"^\s*init_average_axis\s*=",
        )
    if args.axis_move_x2 is not None:
        lines = upsert(
            lines,
            r"^\s*average_axis_move_X2\s*=",
            f"average_axis_move_X2 = {args.axis_move_x2:.12g}",
            r"^\s*average_axis_move_X1\s*=",
        )
    lines = upsert(lines, r"^\s*nElems\s*=", f"nElems = {args.n_elems}", r"^\s*grid_type\s*=")

    path.write_text("\n".join(lines) + "\n")
    suffix = "" if args.total_iter is None else f", totalIter={args.total_iter}"
    if args.minimize_tol is not None:
        suffix += f", minimize_tol={args.minimize_tol}"
    if args.init_la != "keep":
        suffix += f", init_LA={args.init_la}"
    if args.axis_move_x1 is not None or args.axis_move_x2 is not None:
        suffix += f", axis_move=({args.axis_move_x1}, {args.axis_move_x2})"
    print(
        f"Patched {path}: nElems={args.n_elems}, PHIEDGE={args.phi_edge}, "
        f"init_average_axis={args.init_average_axis}{suffix}"
    )


if __name__ == "__main__":
    main()
