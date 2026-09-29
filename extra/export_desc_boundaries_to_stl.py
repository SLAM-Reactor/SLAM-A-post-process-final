#!/usr/bin/env python3
"""Export DESC equilibrium boundaries to binary STL.

The saved SLAM-A equilibria store the boundary in generalized toroidal
coordinates.  The lab-frame cylindrical angle is phi = zeta + omega, where
omega is stored in the saved _surface/_W_lmn coefficients.  DESC 0.17's public
surface compute path returns omega=0 here, so this exporter evaluates the saved
Fourier bases directly.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import h5py
import numpy as np


def fourier_1d(angle: np.ndarray, mode: np.ndarray, nfp: int = 1) -> np.ndarray:
    """Match DESC's Fourier basis: negative modes are sin, nonnegative are cos."""
    argument = np.abs(mode)[None, :] * nfp * angle[:, None]
    return np.where(mode[None, :] >= 0, np.cos(argument), np.sin(argument))


def double_fourier(
    theta: np.ndarray,
    zeta: np.ndarray,
    modes: np.ndarray,
    coeffs: np.ndarray,
    nfp: int,
) -> np.ndarray:
    """Evaluate a DESC DoubleFourierSeries from raw modes and coefficients."""
    theta_basis = fourier_1d(theta.ravel(), modes[:, 1], nfp=1)
    zeta_basis = fourier_1d(zeta.ravel(), modes[:, 2], nfp=nfp)
    return np.sum(theta_basis * zeta_basis * coeffs[None, :], axis=1)


def read_surface_series(h5: h5py.File, quantity: str) -> tuple[np.ndarray, np.ndarray, int]:
    group = h5["_surface"]
    coeffs = group[f"_{quantity}_lmn"][()]
    basis = group[f"_{quantity}_basis"]
    modes = basis["_modes"][()]
    nfp = int(basis["_NFP"][()])
    return coeffs, modes, nfp


def sample_boundary(path: Path, ntheta: int, nzeta: int | None) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        eq_nfp = int(h5["_NFP"][()])
        nzeta = nzeta or ntheta * eq_nfp

        theta_values = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
        zeta_values = np.linspace(0.0, 2.0 * np.pi, nzeta, endpoint=False)
        zeta, theta = np.meshgrid(zeta_values, theta_values, indexing="ij")

        r_coeffs, r_modes, r_nfp = read_surface_series(h5, "R")
        z_coeffs, z_modes, z_nfp = read_surface_series(h5, "Z")
        w_coeffs, w_modes, w_nfp = read_surface_series(h5, "W")

    r = double_fourier(theta, zeta, r_modes, r_coeffs, r_nfp).reshape(nzeta, ntheta)
    z = double_fourier(theta, zeta, z_modes, z_coeffs, z_nfp).reshape(nzeta, ntheta)
    omega = double_fourier(theta, zeta, w_modes, w_coeffs, w_nfp).reshape(
        nzeta, ntheta
    )
    phi = zeta + omega

    vertices = np.empty((nzeta, ntheta, 3), dtype=np.float64)
    vertices[:, :, 0] = r * np.cos(phi)
    vertices[:, :, 1] = r * np.sin(phi)
    vertices[:, :, 2] = z
    return vertices


def triangulate_periodic(vertices: np.ndarray) -> np.ndarray:
    nzeta, ntheta, _ = vertices.shape
    triangles = np.empty((2 * nzeta * ntheta, 3, 3), dtype=np.float32)
    k = 0
    for j in range(nzeta):
        jp = (j + 1) % nzeta
        for i in range(ntheta):
            ip = (i + 1) % ntheta
            v00 = vertices[j, i]
            v01 = vertices[j, ip]
            v10 = vertices[jp, i]
            v11 = vertices[jp, ip]
            triangles[k] = (v00, v01, v11)
            triangles[k + 1] = (v00, v11, v10)
            k += 2
    return triangles


def write_binary_stl(path: Path, triangles: np.ndarray, name: str) -> None:
    header = f"DESC lab-frame boundary {name}".encode("ascii", errors="ignore")[:80]
    header = header + b" " * (80 - len(header))

    with path.open("wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 0 else np.zeros(3, dtype=np.float32)
            f.write(struct.pack("<3f", *normal.astype(np.float32)))
            f.write(struct.pack("<9f", *triangle.reshape(-1).astype(np.float32)))
            f.write(struct.pack("<H", 0))


def validate_stl(path: Path) -> tuple[int, bool]:
    with path.open("rb") as f:
        f.seek(80)
        facets = struct.unpack("<I", f.read(4))[0]
    return facets, path.stat().st_size == 84 + 50 * facets


def export_file(path: Path, ntheta: int, nzeta: int | None) -> Path:
    vertices = sample_boundary(path, ntheta=ntheta, nzeta=nzeta)
    triangles = triangulate_periodic(vertices)
    output = path.with_name(f"{path.stem}_boundary.stl")
    write_binary_stl(output, triangles, path.stem)
    facets, valid = validate_stl(output)
    bounds_min = vertices.reshape(-1, 3).min(axis=0)
    bounds_max = vertices.reshape(-1, 3).max(axis=0)
    print(f"{path.name} -> {output.name}")
    print(f"  vertices={vertices.shape[0] * vertices.shape[1]} facets={facets}")
    print(f"  bbox min={bounds_min} max={bounds_max} valid={valid}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path, default=sorted(Path(".").glob("eq_*_solved.h5")))
    parser.add_argument("--ntheta", type=int, default=216)
    parser.add_argument("--nzeta", type=int, default=None)
    args = parser.parse_args()

    if not args.files:
        raise SystemExit("No equilibrium files found.")
    for path in args.files:
        export_file(path, ntheta=args.ntheta, nzeta=args.nzeta)


if __name__ == "__main__":
    main()
