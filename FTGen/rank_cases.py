#!/usr/bin/env python3
"""Rank FTGen cases by how well conditioned and how smooth they are.

No judgement about periodicity here -- this only answers "which case is a clean
dataset worth analysing?".  Periodicity is assessed separately, on the winner.

Two independent groups of metrics:

  CONDITIONING of the alpha grid (raw traces, before any spline)
    alpha = mod(2*pi*iota*i, 2*pi) is an irrational rotation, so the spacing is
    uneven by construction and the largest alpha lands a case-dependent
    distance below 2*pi.  A near-degenerate spacing makes every downstream
    RectBivariateSpline ill-conditioned regardless of the physics.

  SMOOTHNESS of the traced quantities along phi at fixed alpha
    second differences relative to first differences; a clean trace has a
    small, uniform ratio and no outlier rows.

Everything is measured on data/alpha/*.npz, the native arrays, so nothing here
depends on the spline chain that the PEST and Boozer products go through.
"""

import glob
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FTGEN_DATA_DIR", HERE / "data"))
TWOPI = 2 * np.pi

FIELDS = ("R_alpha", "Z_alpha", "modB_alpha")


def case_name(path):
    return os.path.basename(path)[len("alpha_data_"):-len(".npz")]


rows = []
for f in sorted(glob.glob(str(DATA_DIR / "alpha" / "alpha_data_*.npz"))):
    d = np.load(f)
    a = np.asarray(d["alpha"]).ravel()
    g = np.diff(a)
    gap = TWOPI - a[-1]
    rec = {
        "case": case_name(f),
        "n_alpha": len(a),
        "iota": float(d["iota"]),
        "min_gap": g.min(),
        "med_gap": float(np.median(g)),
        "gap_ratio": g.max() / g.min(),          # spline knot conditioning
        "seam_gap": gap,
        "seam_over_med": gap / float(np.median(g)),
    }
    # smoothness along phi at fixed alpha
    worst_ratio = 0.0
    worst_rows = 0
    for k in FIELDS:
        v = np.asarray(d[k])
        d1 = np.abs(np.diff(v, axis=1))
        d2 = np.abs(np.diff(v, n=2, axis=1))
        r = np.median(d2) / np.median(d1)
        rec[f"smooth_{k.split('_')[0]}"] = r
        worst_ratio = max(worst_ratio, r)
        worst_rows += int((d2.max(axis=1) > 20 * np.median(d2)).sum())
    rec["worst_smooth"] = worst_ratio
    rec["rough_rows"] = worst_rows
    rows.append(rec)

if not rows:
    raise SystemExit(f"no alpha_data_*.npz under {DATA_DIR/'alpha'}")

print("CONDITIONING of the alpha grid (raw traces)")
print("%-16s %6s %10s %10s %10s %10s %9s" % (
    "case", "N", "iota", "min_gap", "gap_ratio", "seam_gap", "seam/med"))
for r in rows:
    flag = ""
    if r["seam_over_med"] < 0.25:
        flag = "  DEGENERATE seam knot"
    elif r["gap_ratio"] > 20:
        flag = "  uneven spacing"
    print("%-16s %6d %10.6f %10.3e %10.1f %10.3e %9.2f%s" % (
        r["case"], r["n_alpha"], r["iota"], r["min_gap"], r["gap_ratio"],
        r["seam_gap"], r["seam_over_med"], flag))

print()
print("SMOOTHNESS along phi at fixed alpha  (median|d2| / median|d1|, lower is smoother)")
print("%-16s %10s %10s %10s %12s" % ("case", "R", "Z", "|B|", "rough rows"))
for r in rows:
    print("%-16s %10.4f %10.4f %10.4f %12d" % (
        r["case"], r["smooth_R"], r["smooth_Z"], r["smooth_modB"], r["rough_rows"]))

# ---- pick the cleanest case ------------------------------------------------
ok = [r for r in rows if r["seam_over_med"] >= 0.25 and r["gap_ratio"] <= 20
      and r["rough_rows"] == 0]
print()
if not ok:
    print("no case passes the conditioning + smoothness filter")
else:
    # among well-conditioned cases prefer the most alpha lines, then the most
    # even spacing -- more field lines is a better sampled surface.
    ok.sort(key=lambda r: (-r["n_alpha"], r["gap_ratio"]))
    print("well-conditioned and smooth, best first:")
    for r in ok:
        print("   %-16s N=%-5d gap_ratio=%5.1f  seam/med=%.2f  worst_smooth=%.4f" % (
            r["case"], r["n_alpha"], r["gap_ratio"], r["seam_over_med"], r["worst_smooth"]))
    print()
    print("RECOMMENDED CASE:", ok[0]["case"])
