#!/usr/bin/env python3
"""QFM surface optimization for SLNAM0 in the GENERALIZED toroidal angle.

Follows QFM_SLNAM_to_SLAM_optimization.ipynb, with one substitution: instead of
building the seed surface as a circular tube around the traced magnetic axis
(which has no omega, so phi == zeta), the seed IS the SLNAM0 equilibrium boundary
-- a FourierRZToroidalSurface carrying a W_basis, i.e. phi = zeta + omega.

Both QFM objectives evaluate on the data key "phi" rather than on zeta, so the
generalized angle is carried through the Biot-Savart evaluation points; DRY_RUN
checks that claim by building and evaluating them before any optimization runs.

Toroidal-flux target: measured from the coil field through the seed surface, NOT
inherited from eq.Psi.  eq.Psi came from the reference rot_ellipse equilibrium
and is ~28% low against this coilset (see the |B| check in the vacuum driver);
using the measured flux sidesteps that error entirely.

Outputs land flat in QFM_optimization/.
"""

import json
import os
import sys
import traceback
from pathlib import Path

DESC_REPO = Path(os.environ.get("DESC_REPO", "/pscratch/sd/r/rgaur/DESC"))
sys.path.insert(0, str(DESC_REPO))

if os.environ.get("DESC_DEVICE", "gpu").lower() == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

from desc import set_device

set_device(os.environ.get("DESC_DEVICE", "gpu"))

try:
    import jax

    cache_dir = Path(os.environ.get("JAX_CACHE_DIR", "jax-caches")).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
except Exception:
    jax = None

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from desc.coils import CoilSet
from desc.equilibrium import Equilibrium
from desc.grid import LinearGrid
from desc.objectives import ObjectiveFunction, SurfaceQuadraticFlux, ToroidalFlux
from desc.optimize import Optimizer
from desc.plotting import plot_boundaries

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent

COILSET = Path(os.environ.get("COILSET", PARENT / "SLNAM0_for_C5.h5"))
INPUT_EQ = Path(
    os.environ.get("INPUT_EQ", PARENT / "eq_slnam0_1p5_196_0p82_m4n8_omega_m2n8_solved.h5")
)
TAG = os.environ.get("TAG", "qfm_slnam0_196_0p82")

M_SURF = int(os.environ.get("M_SURF", 8))
N_SURF = int(os.environ.get("N_SURF", 16))
EVAL_M = int(os.environ.get("EVAL_M", 24))
EVAL_N = int(os.environ.get("EVAL_N", 32))
BS_CHUNK = int(os.environ.get("BS_CHUNK_SIZE", 10))

QFLUX_WEIGHT = float(os.environ.get("QFLUX_WEIGHT", 1e-1))
TFLUX_WEIGHT = float(os.environ.get("TFLUX_WEIGHT", 1e3))
MAXITER = int(os.environ.get("MAXITER", 100))
FTOL = float(os.environ.get("FTOL", 5e-4))

DRY_RUN = os.environ.get("DRY_RUN", "0").lower() not in ("0", "false", "no")


def scalar(value):
    return float(np.ravel(np.asarray(value))[0])


def average_normalized_field(field, surface, chunk_size=None):
    """<|B.n|>/<|B|> on the surface -- the QFM quality figure of merit.

    Passing the SURFACE (not an Equilibrium) skips the plasma-current
    contribution, which is what we want for a vacuum QFM.
    """
    grid = LinearGrid(M=80, N=80, NFP=surface.NFP)
    Bn, coords = field.compute_Bnormal(surface, eval_grid=grid, chunk_size=chunk_size)
    Bvec = field.compute_magnetic_field(coords)
    return float(np.mean(np.abs(np.asarray(Bn))) / np.mean(np.linalg.norm(np.asarray(Bvec), axis=1)))


def main():
    print("coilset:", COILSET)
    print("input eq:", INPUT_EQ)
    print("output dir:", HERE)

    field = CoilSet.load(str(COILSET))
    eq = Equilibrium.load(str(INPUT_EQ))
    print(
        "eq: L=%d M=%d N=%d Lz=%d Mz=%d Nz=%d NFP=%d sym=%s Psi=%.6e"
        % (eq.L, eq.M, eq.N, eq.Lz, eq.Mz, eq.Nz, eq.NFP, eq.sym, eq.Psi)
    )

    # ---- the seed IS the equilibrium boundary, in generalized angle ----
    surf0 = eq.surface.copy()
    n_omega = surf0.W_basis.num_modes if hasattr(surf0, "W_basis") else 0
    print("seed surface omega modes:", n_omega)
    if n_omega == 0:
        print("[warn] seed surface carries NO omega -- this is not a generalized-angle run")
    w_max = float(np.max(np.abs(np.asarray(surf0.W_lmn)))) if n_omega else 0.0
    print("max |W_lmn| = %.6e rad" % w_max)

    surf0.change_resolution(M=M_SURF, N=N_SURF)
    print("seed surface resolution: M=%d N=%d" % (surf0.M, surf0.N))

    # QFM bookkeeping attributes, as in the reference notebook
    gi = LinearGrid(rho=np.array([1.0]), M=2 * eq.M, N=2 * eq.N, NFP=eq.NFP, sym=False)
    iota_edge = float(np.mean(np.asarray(eq.compute("iota", grid=gi)["iota"])))
    G_edge = float(np.mean(np.asarray(eq.compute("G", grid=gi)["G"])))
    surf0.rho = 1.0
    surf0.iota = iota_edge
    surf0.G = G_edge
    print("seed iota=%.6f  G=%.6f" % (iota_edge, G_edge))

    eval_grid = LinearGrid(M=EVAL_M, N=EVAL_N, NFP=surf0.NFP)

    # ---- measure the flux the COILS actually put through the seed surface ----
    # eval_grid above spans zeta for SurfaceQuadraticFlux's whole-surface
    # integral.  ToroidalFlux integrates ONE constant-zeta cut, so it must build
    # its own grid -- passing the zeta-spanning one raises
    # "Evaluation grid should be at constant zeta".
    tflux_probe = ToroidalFlux(surf0, field, eq_fixed=False, field_fixed=True)
    tflux_probe.build(verbose=0)
    measured_psi = scalar(tflux_probe.compute(surf0.params_dict))
    print("measured toroidal flux through seed surface: %.8e Wb" % measured_psi)
    print("eq.Psi (inherited, known ~28%% low): %.8e Wb" % eq.Psi)

    target_psi = float(os.environ.get("TARGET_PSI", measured_psi))
    print("ToroidalFlux target: %.8e Wb" % target_psi)

    bn0 = average_normalized_field(field, surf0, chunk_size=BS_CHUNK)
    print("seed <|B.n|>/<|B|> = %.6e" % bn0)

    qflux = SurfaceQuadraticFlux(
        surf0,
        field,
        eval_grid=eval_grid,
        field_fixed=True,
        bs_chunk_size=BS_CHUNK,
        weight=QFLUX_WEIGHT,
    )
    tflux = ToroidalFlux(
        surf0,
        field,
        target=target_psi,
        eq_fixed=False,
        field_fixed=True,
        weight=TFLUX_WEIGHT,
    )
    obj = ObjectiveFunction((qflux, tflux))
    obj.build(verbose=1)
    r0 = np.asarray(obj.compute_scaled_error(obj.x(surf0)))
    print("initial objective: dim_f=%d  |r|=%.6e  max|r|=%.6e"
          % (r0.size, float(np.linalg.norm(r0)), float(np.max(np.abs(r0)))))

    if DRY_RUN:
        print("DRY_RUN: objectives built and evaluated on the generalized-angle "
              "surface without error; exiting before optimization.")
        return

    opt = Optimizer("lsq-exact")
    (qfm_surf,), result = opt.optimize(
        surf0, objective=obj, ftol=FTOL, maxiter=MAXITER, verbose=3, copy=True
    )
    print("optimizer exit: success=%s message=%s"
          % (getattr(result, "success", None), getattr(result, "message", "")))

    qfm_path = HERE / f"{TAG}_surface.h5"
    qfm_surf.save(str(qfm_path))
    print("saved:", qfm_path)

    # ------------------------------------------------------------ metrics
    bn1 = average_normalized_field(field, qfm_surf, chunk_size=BS_CHUNK)
    tflux_probe2 = ToroidalFlux(qfm_surf, field, eq_fixed=False, field_fixed=True)
    tflux_probe2.build(verbose=0)
    psi1 = scalar(tflux_probe2.compute(qfm_surf.params_dict))

    elong_grid = LinearGrid(
        M=2 * int(surf0.M), zeta=np.array([0.0]), rho=np.array([1.0]),
        NFP=eq.NFP, sym=False, axis=False,
    )
    metrics = {
        "tag": TAG,
        "coilset": str(COILSET),
        "input_eq": str(INPUT_EQ),
        "omega_modes_seed": int(n_omega),
        "max_abs_W_lmn_seed": w_max,
        "seed_iota": iota_edge,
        "seed_G": G_edge,
        "target_psi": target_psi,
        "measured_psi_seed": measured_psi,
        "measured_psi_qfm": psi1,
        "eq_Psi_inherited": float(eq.Psi),
        "Bn_over_B_seed": bn0,
        "Bn_over_B_qfm": bn1,
        "qfm_surface": str(qfm_path),
    }
    for name, surface in (("seed", surf0), ("qfm", qfm_surf)):
        try:
            metrics[f"elongation_{name}"] = scalar(
                surface.compute("a_major/a_minor", grid=elong_grid)["a_major/a_minor"]
            )
            metrics[f"volume_{name}"] = scalar(surface.compute("V")["V"])
        except Exception as err:
            print(f"[warn] geometry for {name} failed: {err}")

    with open(HERE / f"{TAG}_metrics.json", "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)

    print("\n<|B.n|>/<|B|>: seed %.6e -> qfm %.6e  (factor %.3f)"
          % (bn0, bn1, bn1 / bn0 if bn0 else float("nan")))
    print("toroidal flux: target %.6e  achieved %.6e" % (target_psi, psi1))
    for key in ("elongation_seed", "elongation_qfm", "volume_seed", "volume_qfm"):
        if key in metrics:
            print(f"{key}: {metrics[key]}")

    # ------------------------------------------------------------- plots
    try:
        plt.figure()
        plot_boundaries([surf0, qfm_surf], phi=6, labels=["seed (eq boundary)", "QFM"])
        plt.savefig(HERE / f"{TAG}_boundaries.png", dpi=300, bbox_inches="tight")
        plt.close("all")
        print("wrote", HERE / f"{TAG}_boundaries.png")
    except Exception as err:
        print(f"[plot warning] plot_boundaries failed: {err}")
        traceback.print_exc()

    if jax is not None:
        jax.clear_caches()


if __name__ == "__main__":
    main()
