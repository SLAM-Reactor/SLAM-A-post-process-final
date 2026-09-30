#!/usr/bin/env python3
"""Boozer |B| contours and theta_B closure of the BEST GVEC SLAM and SLNAM equilibria (user 09-30 00:05: "repeat the same
Boozer analysis for the BEST GVEC SLAM and SLNAM equilibria"), as boozer_theta_closure.py did for the DESC states.

States (read only; SLAM-A-post-process-final/GVEC/post-process/paths.py, "best GVEC equilibria from the 09-23 resolution
scans, lowest max |F|_normalized"): SLAM_BEST = SLAM 128/0.65/clean1e-4 boundary, nElems 24, mn_max (24, 32); SLNAM_BEST =
SLNAM 216/0.92 boundary, nElems 32, mn_max (12, 34). Boozer transform resolution as the GVEC ripple runs: MNfactor 3 for
SLAM_BEST (user 09-24; 5 needs ~490 GB), 5 for SLNAM_BEST. Measured memory of one surface: 150 GB (SLAM_BEST, job 58849241)
and 320 GB (SLNAM_BEST, 58849240) -> CPU compute nodes only, one surface per process (memory builds up across surfaces).

Task pool: every (state, rho) is one task, run in its own child process; a job claims the next free task with an
exclusive lock file (data/conclusion/gvec_boozer_locks/) and saves data/conclusion/gvec_boozer_<state>_rho<r>.npz, so several
identical jobs share the work. Saved per surface (user 09-30 00:15): the whole GVEC Boozer dataset as netCDF (.nc) and |B| on a grid at
the Boozer transform's resolution with its 2D Fourier coefficients (.npz). When all tasks are saved, the figure figures/conclusion/boozer_lowfield_gvec.png (rows =
states, columns rho 0.5 / 0.85 / 1.0; contour lines, jet, 20 even levels + 30 in [Bmin, Bmin + 0.08 (Bmax - Bmin)], 4 field
lines, as plot_boozer_surface) and data/conclusion/gvec_boozer_theta_closure.json (closed_frac, theta_var as
boozer_theta_closure.py) are written.
    (gvec-env)  python -u gvec_boozer_plane.py     - from this directory on a CPU compute node (sbatch job_gvec_boozer_cpu.sl);
    with all data/*.npz present it only redraws the figure and the json (cheap; the login node is fine)
"""
import json
import multiprocessing as mp
import os
import time

import numpy as np

# ---------------- settings ----------------
GV = "/pscratch/sd/r/rgaur/SLAM-A-post-process-final/GVEC"
states = {  # label: (ini, state file, Boozer MNfactor)
    "SLNAM_BEST": (GV + "/parameter_SLNAM_216_0p92_nelems32_mn12_34_final.ini",
                   GV + "/SLNAM_216_0p92_nelems32_mn12_34_State_final.dat", 5.0),
    "SLAM_BEST": (GV + "/parameter_SLAM_128_0p65_nelems24_mn24_32_final.ini",
                  GV + "/SLAM_128_0p65_nelems24_mn24_32_State_final.dat", 3.0),
}
rho_boozer = [0.5, 0.85, 1.0]
tasks = [(s, r) for s in ("SLNAM_BEST", "SLAM_BEST") for r in rho_boozer]  # the long SLNAM surfaces first
NFP = 2
n_theta, n_zeta = 90, 90  # minimum grid per field period (raised to the Boozer resolution in boozer_plane)
fieldlines = 4
t_levels = [round(0.02 * k, 2) for k in range(1, 50)]
min_cells = 4
out_dir, fig_dir = "data", "figures"  # copy in SLAM-A-post-process-final/SLAM-SLNAM-Boozer (09-30 01:00); the original in
#   generalized_toroidal/SLAM_LT3_09-27-26 wrote to data/conclusion, figures/conclusion
lock_dir = out_dir + "/gvec_boozer_locks"
# ------------------------------------------


def npz_path(label, rho):
    return "%s/gvec_boozer_%s_rho%s.npz" % (out_dir, label, ("%.2f" % rho).replace(".", "p"))


def boozer_plane(label, rho):
    """Child process: |B| on the (theta_B, zeta_B) grid of one surface of one GVEC state."""
    import gvec

    t0 = time.time()
    ini, dat, mnf = states[label]
    here = os.getcwd()
    os.chdir(GV)  # the parameter files name their inputs relative to GVEC/
    state = gvec.load_state(ini, dat)
    m_max, n_max = state.get_mn_max()
    M, N = int(mnf * m_max), int(mnf * n_max)
    # high-resolution grid (user 09-30 00:15: "save the high-res Boozer data from GVEC so we can make plots later"): at least
    # the Boozer transform's own resolution, 2 M + 2 poloidal and 2 N + 2 toroidal points per field period (N counts field-
    # period harmonics), rounded up to a multiple of 32
    nth = max(n_theta, 32 * int(np.ceil((2 * M + 2) / 32)))
    nze = max(n_zeta, 32 * int(np.ceil((2 * N + 2) / 32)))
    theta = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / NFP, nze, endpoint=False)
    print("[%s rho %.2f] loaded (%.0f s); Boozer M, N = %d, %d (MNfactor %g x (%d, %d)); grid %d x %d"
          % (label, rho, time.time() - t0, M, N, mnf, m_max, n_max, nth, nze), flush=True)
    ev = gvec.EvaluationsBoozer(rho=np.array([rho]), theta_B=theta, zeta_B=zeta, state=state, radial_derivative=False,
                                M=M, N=N)
    state.compute(ev, "mod_B", "iota", "pos")
    B = np.asarray(ev.mod_B.sel(rho=rho).transpose("pol", "tor"))
    iota = float(np.asarray(ev.iota.sel(rho=rho)).ravel()[0])
    os.chdir(here)
    nc = npz_path(label, rho).replace(".npz", ".nc")
    try:  # the whole GVEC Boozer dataset: theta_B, zeta_B, logical theta / zeta, LA, NU_B, mod_B, iota, pos
        ev.to_netcdf(nc)
    except Exception as err:  # noqa: BLE001 - keep the plane even if netCDF writing fails
        print("[%s rho %.2f] netCDF not written (%s); variables go to the npz" % (label, rho, err), flush=True)
        np.savez(nc.replace(".nc", "_dataset.npz"), **{k: np.asarray(ev[k]) for k in list(ev.data_vars) + list(ev.coords)})
    B_mn = np.fft.fft2(B) / B.size  # |B| = sum B_mn exp(i (m theta_B + n NFP zeta_B)), numpy fft ordering
    np.savez(npz_path(label, rho), B=B, B_mn=B_mn, theta=theta, zeta=zeta, iota=iota, rho=rho, M=M, N=N, mnfactor=mnf,
             NFP=NFP, seconds=time.time() - t0)
    print("[%s rho %.2f] saved (%s + npz), Bmin %.4f Bmax %.4f iota %+.4f (%.0f s)"
          % (label, rho, nc, B.min(), B.max(), iota, time.time() - t0), flush=True)


def theta_closure(B):
    """As boozer_theta_closure.py: per level, all regions of {B < B*} and {B > B*} are bands winding around theta_B only."""
    from scipy import ndimage

    def winding(mask):
        nt, nz = mask.shape
        lab, _ = ndimage.label(np.tile(mask, (3, 3)))
        c = lab[nt:2 * nt, nz:2 * nz]
        st, sz = lab[2 * nt:3 * nt, nz:2 * nz], lab[nt:2 * nt, 2 * nz:3 * nz]
        return [(bool(np.any(st[c == l] == l)), bool(np.any(sz[c == l] == l)), int((c == l).sum()))
                for l in np.unique(c[c > 0])]

    bmin, bmax = B.min(), B.max()
    closed = [all(wt and not wz for wt, wz, n in winding(B < bmin + t * (bmax - bmin)) + winding(B > bmin + t * (bmax - bmin))
                  if n >= min_cells) for t in t_levels]
    dB = B - B.mean(axis=0, keepdims=True)
    return closed, float(np.sqrt(np.mean(dB**2)) / (bmax - bmin)), float(np.abs(dB).max() / (bmax - bmin))


def panel(ax, d, title):
    """Contour lines + field lines as desc.plotting.plot_boozer_surface(fill=False)."""
    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    B, th, ze, iota = d["B"], d["theta"], d["zeta"], float(d["iota"])
    Bp = np.concatenate([B, B[:1]], axis=0)  # close theta = 2 pi and zeta = 2 pi / NFP for drawing
    Bp = np.concatenate([Bp, Bp[:, :1]], axis=1)
    thp, zep = np.append(th, 2 * np.pi), np.append(ze, 2 * np.pi / NFP)
    bmin, bmax = B.min(), B.max()
    levels = np.unique(np.concatenate([np.linspace(bmin, bmax, 20), np.linspace(bmin, bmin + 0.08 * (bmax - bmin), 30)]))
    im = ax.contour(zep, thp, Bp, levels=levels, cmap="jet", norm=matplotlib.colors.Normalize(), extend="both")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax)
    z = np.linspace(0, 2 * np.pi / NFP, 100)
    for a0 in np.linspace(0, 2 * np.pi, fieldlines, endpoint=False):
        a = a0 + iota * z
        ax.plot(z, np.where((a >= 0) & (a <= 2 * np.pi), a, np.nan), color="k", lw=2)
        ax.plot(z, np.where((a < 0) | (a > 2 * np.pi), np.mod(a, 2 * np.pi), np.nan), color="k", lw=2)
    ax.set_xlim([0, 2 * np.pi / NFP])
    ax.set_ylim([0, 2 * np.pi])
    ax.set_xlabel(r"$\zeta_{Boozer}$")
    ax.set_ylabel(r"$\theta_{Boozer}$")
    ax.set_title(title, fontsize=9)


if __name__ == "__main__":
    os.makedirs(lock_dir, exist_ok=True)
    ctx = mp.get_context("spawn")
    for label, rho in tasks:
        if os.path.exists(npz_path(label, rho)):
            continue
        try:  # claim the task
            os.close(os.open("%s/%s_rho%.2f.lock" % (lock_dir, label, rho), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        print("%s claimed %s rho %.2f (job %s)" % (time.strftime("%H:%M:%S"), label, rho, os.environ.get("SLURM_JOB_ID")),
              flush=True)
        p = ctx.Process(target=boozer_plane, args=(label, rho))
        p.start()
        p.join()
        print("%s %s rho %.2f child exit %s" % (time.strftime("%H:%M:%S"), label, rho, p.exitcode), flush=True)

    if all(os.path.exists(npz_path(s, r)) for s, r in tasks):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        labels = list(states)[::-1]  # SLAM_BEST row first
        fig, axes = plt.subplots(len(labels), len(rho_boozer), figsize=(5.2 * len(rho_boozer), 4.4 * len(labels)),
                                 squeeze=False)
        results = []
        for i, label in enumerate(labels):
            res = dict(label=label, ini=states[label][0], state=states[label][1], mnfactor=states[label][2], surfaces=[])
            for j, r in enumerate(rho_boozer):
                d = dict(np.load(npz_path(label, r)))
                closed, tv, tvmax = theta_closure(d["B"])
                s = dict(rho=r, closed_frac=float(np.mean(closed)), t_open=[t for t, c in zip(t_levels, closed) if not c],
                         theta_var=tv, theta_var_max=tvmax, Bmin_booz=float(d["B"].min()), Bmax_booz=float(d["B"].max()),
                         iota=float(d["iota"]), M_booz=int(d["M"]), N_booz=int(d["N"]))
                res["surfaces"].append(s)
                panel(axes[i, j], d, "GVEC %s  rho %.2f\nclosed in theta_B: %.0f %% of trapped levels, theta var %.3f"
                      % (label, r, 100 * s["closed_frac"], tv))
            axes[i, 0].text(-0.28, 0.5, "GVEC %s\nMNfactor %g\nnot traced" % (label, states[label][2]),
                            transform=axes[i, 0].transAxes, rotation=90, va="center", ha="center", fontsize=9)
            results.append(res)
            print("DONE %s: %s" % (label, "; ".join("rho %.2f closed %.2f theta_var %.3f" % (s["rho"], s["closed_frac"],
                                                                                           s["theta_var"])
                                                     for s in res["surfaces"])), flush=True)
        fig.tight_layout()
        fig.savefig(fig_dir + "/boozer_lowfield_gvec.png", dpi=110)
        json.dump(results, open(out_dir + "/gvec_boozer_theta_closure.json", "w"), indent=1)
        print("ALL DONE", flush=True)
    else:
        print("not all surfaces saved yet: %s" % [t for t in tasks if not os.path.exists(npz_path(*t))], flush=True)
