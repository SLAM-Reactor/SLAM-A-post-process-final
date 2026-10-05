#!/usr/bin/env python3
"""Periodicity analysis for one FTGen case, in all three coordinate systems.

usage:  python3 check_periodicity.py [case]        e.g. 1p0_216_0p85
        (defaults to $FTGEN_CASE, else the first case found)

Three questions, kept separate because they fail for different reasons:

  1. SEAM CONTINUITY.  On a grid built with endpoint=False, f[0] and f[-1] are
     one step apart, so |f[0]-f[-1]| should be comparable to an ordinary step.
     Compared median-to-median and max-to-max: mixing the two (max seam vs
     median step) inflates the ratio and is not a valid test.

  2. FIELD-PERIOD SYMMETRY.  For NFP=2, R, Z and |B| must satisfy
     f(theta, zeta + pi) = f(theta, zeta).  X and Y are lab Cartesian
     coordinates and are NOT invariant under a half-period rotation, so they
     are excluded -- reporting them as failures would be wrong.

  3. CHART CONSISTENCY.  Does the Newton solve actually land on the grid it
     claims?  zeta_B = phi_cyl + nu and theta_B = theta_PEST + iota*nu.

The alpha grid is the control: it is the raw traced data with no spline in it.
alpha is periodic (alpha = 2pi is the same field line as alpha = 0), but phi is
NOT -- at fixed alpha, theta = alpha + iota*phi, so advancing phi by 2pi lands
at a different poloidal position.  A phi seam jump there is correct, not a bug.
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FTGEN_DATA_DIR", HERE / "data"))
TWOPI = 2 * np.pi

if len(sys.argv) > 1:
    CASE = sys.argv[1]
elif os.environ.get("FTGEN_CASE"):
    CASE = os.environ["FTGEN_CASE"]
else:
    hits = sorted(glob.glob(str(DATA_DIR / "boozer" / "Boozer_data_*.npz")))
    if not hits:
        raise SystemExit("no Boozer_data_*.npz found")
    CASE = os.path.basename(hits[0])[len("Boozer_data_"):-len(".npz")]

print(f"case {CASE}")
print(f"data {DATA_DIR}")


def seam(f, axis):
    """median-to-median and max-to-max seam ratios across one axis."""
    if axis == 0:
        s = np.abs(f[0, :] - f[-1, :])
        d = np.abs(np.diff(f, axis=0))
    else:
        s = np.abs(f[:, 0] - f[:, -1])
        d = np.abs(np.diff(f, axis=1))
    return np.median(s) / np.median(d), s.max() / d.max()


def nfp_dev(f, nfp=2):
    """max |f(theta, zeta + 2pi/nfp) - f(theta, zeta)|, relative to the range."""
    n = f.shape[1]
    if n % nfp:
        return float("nan")
    shift = n // nfp
    rng = f.max() - f.min()
    return np.abs(f - np.roll(f, shift, axis=1)).max() / rng if rng else float("nan")


# ------------------------------------------------------------ alpha (control)
ap = DATA_DIR / "alpha" / f"alpha_data_{CASE}.npz"
if ap.exists():
    a = np.load(ap)
    print("\n--- (alpha, phi): raw traces, no spline.  CONTROL ---")
    print("    alpha is periodic; phi is NOT (theta = alpha + iota*phi)")
    print("    %-8s %12s %12s | %12s %12s" % (
        "field", "alpha med", "alpha max", "phi med", "phi max"))
    for k, lbl in (("R_alpha", "R"), ("Z_alpha", "Z"), ("modB_alpha", "|B|")):
        f = np.asarray(a[k])
        am, ax_ = seam(f, 0)
        pm, px = seam(f, 1)
        print("    %-8s %12.2f %12.2f | %12.2f %12.2f" % (lbl, am, ax_, pm, px))
    print("    (phi ratios are EXPECTED to be large -- that is the real"
          " discontinuity)")

# -------------------------------------------------------------------- PEST
pp = DATA_DIR / "pest" / f"PEST_data_{CASE}.npz"
if pp.exists():
    p = np.load(pp)
    print("\n--- (theta_PEST, zeta) ---")
    print("    %-8s %10s %10s %10s %10s %10s" % (
        "field", "th med", "th max", "ze med", "ze max", "NFP dev"))
    for k, lbl in (("R_PEST", "R"), ("Z_PEST", "Z"), ("modB_PEST", "|B|")):
        f = np.asarray(p[k])
        tm, tx = seam(f, 0)
        zm, zx = seam(f, 1)
        print("    %-8s %10.2f %10.2f %10.2f %10.2f %9.2f%%" % (
            lbl, tm, tx, zm, zx, 100 * nfp_dev(f)))

# ------------------------------------------------------------------ Boozer
bp = DATA_DIR / "boozer" / f"Boozer_data_{CASE}.npz"
if bp.exists():
    b = np.load(bp)
    print("\n--- (theta_B, zeta_B) ---")
    print("    %-8s %10s %10s %10s %10s %10s" % (
        "field", "th med", "th max", "ze med", "ze max", "NFP dev"))
    for k, lbl in (("R_booz", "R"), ("Z_booz", "Z"), ("modB_booz", "|B|")):
        if k not in b.files:
            continue
        f = np.asarray(b[k])
        tm, tx = seam(f, 0)
        zm, zx = seam(f, 1)
        print("    %-8s %10.2f %10.2f %10.2f %10.2f %9.2f%%" % (
            lbl, tm, tx, zm, zx, 100 * nfp_dev(f)))
    print("    X_booz / Y_booz omitted: lab Cartesian coords are not invariant")
    print("    under a half-period rotation, so an NFP test on them is invalid.")

    # chart consistency
    if all(k in b.files for k in ("theta_PEST", "phi_cyl", "nu_booz", "iota")):
        iota = float(b["iota"])
        nu = np.asarray(b["nu_booz"])
        tb = np.asarray(b["theta_booz"])
        pb = np.asarray(b["phi_booz"])
        ez = np.abs(((np.asarray(b["phi_cyl"]) + nu - pb + np.pi) % TWOPI) - np.pi)
        et = np.abs(((np.asarray(b["theta_PEST"]) + iota * nu - tb + np.pi) % TWOPI) - np.pi)
        print("\n--- chart consistency (Newton solve) ---")
        print("    zeta_B  = phi_cyl + nu     max %.3e  mean %.3e" % (ez.max(), ez.mean()))
        print("    theta_B = theta_PEST+i*nu  max %.3e  mean %.3e" % (et.max(), et.mean()))
        bad = (ez > 1e-4) | (et > 1e-4)
        print("    points off by >1e-4: %d of %d (%.3f%%)" % (
            bad.sum(), bad.size, 100 * bad.mean()))

print("\nreading: seam ratios near 1 are periodic; NFP dev is the physical test")
print("and should be small for R, Z and |B| in a genuinely NFP=2 device.")
