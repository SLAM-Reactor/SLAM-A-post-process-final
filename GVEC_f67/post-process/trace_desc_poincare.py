#!/usr/bin/env python3
"""Trace DESC coil-field Poincare data from GVEC-derived seed points.

Run in desc-env2. This script does not import GVEC; it consumes the `.npz`
written by export_gvec_sections.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone

import numpy as np

from paths import CASES, DATA_DIR, NFP, TWOPI, ensure_dirs, object_array, selected_cases


DESC_SOURCE = os.environ.get("DESC_SOURCE", "/pscratch/sd/r/rgaur/DESC3/DESC")
sys.path.insert(0, DESC_SOURCE)

import desc  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402
from desc.io import load  # noqa: E402
from desc.magnetic_fields import field_line_integrate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", nargs="+", choices=sorted(CASES), default=None)
    parser.add_argument(
        "--sections",
        default=str(DATA_DIR / "gvec_sections_for_poincare.npz"),
        help="GVEC section cache from export_gvec_sections.py",
    )
    parser.add_argument("--output", default=str(DATA_DIR / "desc_coil_poincare_traces.npz"))
    parser.add_argument("--ntransit", type=int, default=500)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means DESC default")
    parser.add_argument("--chunk-size", type=int, default=0, help="0 means DESC default")
    parser.add_argument("--bs-chunk-size", type=int, default=0, help="0 means DESC default")
    parser.add_argument("--source-grid-n", type=int, default=32)
    parser.add_argument("--bounds-r", default=None, help="override R bounds as low,high")
    parser.add_argument("--bounds-z", default=None, help="override Z bounds as low,high")
    parser.add_argument("--min-finite-fraction", type=float, default=0.95)
    parser.add_argument("--device", choices=("gpu", "cpu"), default=os.environ.get("DESC_DEVICE", "gpu"))
    return parser.parse_args()


def load_one(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        obj = load(path)
    if isinstance(obj, (tuple, list)) and len(obj) == 1:
        return obj[0]
    return obj


def optional_positive_int(value: int):
    return None if value <= 0 else value


def parse_pair(text: str | None):
    if text is None or text.strip() == "":
        return None
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError(f"expected low,high pair, got {text!r}")
    return tuple(vals)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if args.device == "gpu":
        desc.set_device("gpu")
    else:
        desc.set_device("cpu")

    sections = np.load(args.sections, allow_pickle=True)
    section_labels = [str(x) for x in sections["labels"]]
    labels = selected_cases(args.labels) if args.labels else section_labels
    label_to_section = {label: i for i, label in enumerate(section_labels)}

    phi_planes = np.asarray(sections["phi_planes"], dtype=float)
    rho = np.asarray(sections["rho"], dtype=float)
    phis = (
        phi_planes[None, :]
        + np.arange(args.ntransit, dtype=float)[:, None] * TWOPI / NFP
    ).ravel()

    all_r = []
    all_z = []
    used_bounds_r = []
    used_bounds_z = []
    used_r0 = []
    used_z0 = []

    for label in labels:
        idx = label_to_section[label]
        case = CASES[label]
        r0 = np.asarray(sections["seed_R0"][idx], dtype=float)
        z0 = np.asarray(sections["seed_Z0"][idx], dtype=float)
        bounds_r = parse_pair(args.bounds_r) or tuple(float(x) for x in sections["bounds_R"][idx])
        bounds_z = parse_pair(args.bounds_z) or tuple(float(x) for x in sections["bounds_Z"][idx])

        print(f"[{label}] loading coilset {case['coilset']}", flush=True)
        field = load_one(case["coilset"])
        print(
            f"[{label}] tracing {r0.size} seeds for {args.ntransit} transits; "
            f"R bounds={bounds_r}, Z bounds={bounds_z}",
            flush=True,
        )

        field_r, field_z = field_line_integrate(
            r0,
            z0,
            phis,
            field,
            source_grid=LinearGrid(N=args.source_grid_n),
            rtol=args.rtol,
            atol=args.atol,
            max_steps=optional_positive_int(args.max_steps),
            bounds_R=bounds_r,
            bounds_Z=bounds_z,
            chunk_size=optional_positive_int(args.chunk_size),
            bs_chunk_size=optional_positive_int(args.bs_chunk_size),
            options={"throw": False},
        )
        r_trace = np.asarray(field_r).reshape((args.ntransit, phi_planes.size, r0.size))
        z_trace = np.asarray(field_z).reshape((args.ntransit, phi_planes.size, z0.size))

        finite = np.isfinite(r_trace) & np.isfinite(z_trace)
        finite_fraction_by_seed = np.mean(finite, axis=(0, 1))
        finite_count_by_seed = np.sum(finite, axis=(0, 1))
        finite_fraction_by_plane = np.mean(finite, axis=(0, 2))
        last_finite_transit = np.full(r0.size, -1, dtype=int)
        for seed_index in range(r0.size):
            finite_transits = np.where(np.any(finite[:, :, seed_index], axis=1))[0]
            if finite_transits.size:
                last_finite_transit[seed_index] = finite_transits[-1]

        all_r.append(r_trace)
        all_z.append(z_trace)
        used_bounds_r.append(bounds_r)
        used_bounds_z.append(bounds_z)
        used_r0.append(r0)
        used_z0.append(z0)
        print(f"[{label}] trace shape {r_trace.shape}", flush=True)
        print(
            f"[{label}] finite fraction by plane={finite_fraction_by_plane}; "
            f"by seed={finite_fraction_by_seed}",
            flush=True,
        )
        print(
            f"[{label}] finite count by seed={finite_count_by_seed}; "
            f"last finite transit={last_finite_transit}",
            flush=True,
        )
        if np.min(finite_fraction_by_seed) < args.min_finite_fraction:
            raise RuntimeError(
                f"{label} trace finite fraction below threshold "
                f"{args.min_finite_fraction}: {finite_fraction_by_seed}"
            )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "trace_desc_poincare.py",
        "desc_source": DESC_SOURCE,
        "device": args.device,
        "sections": args.sections,
        "labels": labels,
        "nfp": NFP,
        "ntransit": args.ntransit,
        "rtol": args.rtol,
        "atol": args.atol,
        "source_grid_n": args.source_grid_n,
        "bounds_r_override": args.bounds_r,
        "bounds_z_override": args.bounds_z,
        "note": "Seeds and integration bounds came from the GVEC section cache.",
    }

    np.savez(
        args.output,
        labels=object_array(labels),
        rho=rho,
        phi_planes=phi_planes,
        phis=phis,
        R_poincare=np.asarray(all_r),
        Z_poincare=np.asarray(all_z),
        seed_R0=np.asarray(used_r0),
        seed_Z0=np.asarray(used_z0),
        bounds_R=np.asarray(used_bounds_r),
        bounds_Z=np.asarray(used_bounds_z),
        metadata=json.dumps(metadata, indent=2),
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
