#!/usr/bin/env python3
"""Build a stellarator-symmetric GVEC surface from FTGen Boozer splines."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import gvec
import numpy as np
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Case label, e.g. SLAM or SLNAM")
    parser.add_argument("--fname1", default="1p0")
    parser.add_argument("--nturns", default="216")
    parser.add_argument("--fname2", default="0p92")
    parser.add_argument("--nfp", type=int, default=2)
    parser.add_argument("--reverse-theta", action="store_true")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--m-max", type=int, default=14)
    parser.add_argument("--n-max", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_root) / args.label
    spline_file = data_dir / f"Boozer_splines_{args.fname1}_{args.nturns}_{args.fname2}.pkl"
    if not spline_file.is_file():
        raise FileNotFoundError(spline_file)

    with spline_file.open("rb") as f:
        splines = pickle.load(f)

    ntheta = 129
    nzeta = 197
    nzeta_by_two = (nzeta - 1) // 2

    if args.reverse_theta:
        theta = np.linspace(2 * np.pi, 0, ntheta, endpoint=True)
    else:
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=True)

    zeta = np.concatenate(
        (
            np.linspace(-np.pi / args.nfp, 0, nzeta_by_two, endpoint=False),
            np.linspace(0, np.pi / args.nfp, nzeta_by_two + 1, endpoint=True),
        )
    )
    theta_mesh, zeta_mesh = np.meshgrid(theta, zeta, indexing="ij")

    x = splines["X"].ev(theta_mesh, zeta_mesh)
    y = splines["Y"].ev(theta_mesh, zeta_mesh)
    z = splines["Z"].ev(theta_mesh, zeta_mesh)

    # Enforce the same single-field-period stellarator symmetry as the old TQ workflow.
    x[:, 0:nzeta_by_two] = x[:, nzeta_by_two + 1 :][::-1, ::-1]
    y[:, 0:nzeta_by_two] = -y[:, nzeta_by_two + 1 :][::-1, ::-1]
    z[:, 0:nzeta_by_two] = -z[:, nzeta_by_two + 1 :][::-1, ::-1]

    m_max = args.m_max
    n_max = args.n_max
    theta_mesh = theta_mesh[1:-1, 1:-1]
    zeta_mesh = zeta_mesh[1:-1, 1:-1]
    x = x[1:-1, 1:-1]
    y = y[1:-1, 1:-1]
    z = z[1:-1, 1:-1]

    ntheta_fit, nzeta_fit = x.shape
    field = x + 1j * y
    theta_flat = theta_mesh.ravel()
    zeta_flat = zeta_mesh.ravel()
    field_flat = field.ravel()
    z_flat = z.ravel()
    npts = theta_flat.size

    m_vals = np.arange(-m_max, m_max + 1)
    k_vals = np.arange(-n_max, n_max + 1)
    m_arr = m_vals[None, :, None]

    n_odd = 2 * k_vals + 1
    n_arr_field = n_odd[None, None, :]
    phase_field = theta_flat[:, None, None] * m_arr + zeta_flat[:, None, None] * n_arr_field
    matrix_field = np.exp(1j * phase_field).reshape(npts, -1)
    coef_field, *_ = np.linalg.lstsq(matrix_field, field_flat, rcond=None)
    field_fit = (matrix_field @ coef_field).reshape(ntheta_fit, nzeta_fit)
    x_fit = field_fit.real
    y_fit = field_fit.imag

    n_even = 2 * k_vals
    n_arr_z = n_even[None, None, :]
    phase_z = theta_flat[:, None, None] * m_arr + zeta_flat[:, None, None] * n_arr_z
    matrix_z = np.exp(1j * phase_z).reshape(npts, -1)
    coef_z, *_ = np.linalg.lstsq(matrix_z, z_flat, rcond=None)
    z_fit = (matrix_z @ coef_z).reshape(ntheta_fit, nzeta_fit)

    print(f"{args.label}: X RMS {np.sqrt(np.mean((x_fit - x) ** 2)):.6e}")
    print(f"{args.label}: Y RMS {np.sqrt(np.mean((y_fit - y) ** 2)):.6e}")
    print(f"{args.label}: Z RMS {np.sqrt(np.mean((z_fit - z) ** 2)):.6e}")
    print(f"{args.label}: X rel {np.linalg.norm(x_fit - x) / np.linalg.norm(x):.6e}")
    print(f"{args.label}: Y rel {np.linalg.norm(y_fit - y) / np.linalg.norm(y):.6e}")
    print(f"{args.label}: Z rel {np.linalg.norm(z_fit - z) / np.linalg.norm(z):.6e}")

    ntheta_out = 48
    nzeta_per_fp = 64
    if args.reverse_theta:
        theta = np.linspace(2 * np.pi, 0, ntheta_out, endpoint=False)
    else:
        theta = np.linspace(0, 2 * np.pi, ntheta_out, endpoint=False)
    zeta = np.linspace(-np.pi, np.pi, nzeta_per_fp * args.nfp, endpoint=False)
    theta_mesh, zeta_mesh = np.meshgrid(theta, zeta, indexing="ij")
    theta_flat = theta_mesh.ravel()
    zeta_flat = zeta_mesh.ravel()
    npts = theta_flat.size

    phase_field = theta_flat[:, None, None] * m_arr + zeta_flat[:, None, None] * n_arr_field
    matrix_field = np.exp(1j * phase_field).reshape(npts, -1)
    field_fit = (matrix_field @ coef_field).reshape(ntheta_out, nzeta_per_fp * args.nfp)
    x_fit = field_fit.real
    y_fit = field_fit.imag

    phase_z = theta_flat[:, None, None] * m_arr + zeta_flat[:, None, None] * n_arr_z
    matrix_z = np.exp(1j * phase_z).reshape(npts, -1)
    z_fit = (matrix_z @ coef_z).reshape(ntheta_out, nzeta_per_fp * args.nfp).real

    output_name = args.output_name or f"surface_{args.nfp}_{args.label}_{args.fname1}_{args.nturns}_{args.fname2}"
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(x_fit, y_fit, z_fit, alpha=0.8)
    ax.set_box_aspect([4, 2, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.savefig(f"{output_name}.png", dpi=160)
    plt.close(fig)

    xyz = np.zeros((nzeta_per_fp * args.nfp, ntheta_out, 3))
    xyz[:, :, 0] = x_fit.T
    xyz[:, :, 1] = y_fit.T
    xyz[:, :, 2] = z_fit.T
    gvec.scripts.quasr.save_xyz(xyz, args.nfp, f"{output_name}.nc")
    print(f"Wrote {output_name}.nc")


if __name__ == "__main__":
    main()
