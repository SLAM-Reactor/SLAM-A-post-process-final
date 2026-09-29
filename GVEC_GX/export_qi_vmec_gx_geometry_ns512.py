#!/usr/bin/env python3
"""Use gx_geo_vmec.py to make QI precursor geometry from the ns512 VMEC wout file.

Example:
    python export_qi_vmec_gx_geometry_ns512.py
"""

import os
import subprocess
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DESC_DATA_DIR = HERE.parent / "DESC_GX" / "data"
GX_GEO_VMEC = Path(
    os.environ.get(
        "GX_GEO_VMEC",
        "/global/homes/r/rgaur/gx/geometry_modules/pyvmec/gx_geo_vmec.py",
    )
)
PYVMEC_DEPS = HERE / ".pyvmec_deps"

WOUT_FILE = DESC_DATA_DIR / "wout_QI_precursor_OP_NFP2_A9_vacuum_final2_ns512.nc"
RHO = 0.65
TORFLUX = RHO * RHO
NPOL = 5.0
NTHETA = 640
A_OVER_LN = 1.0

ALPHAS = (
    ("alpha0p00", 0.0),
    ("alpha0p25pi", 0.25),
    ("alpha0p50pi", 0.50),
    ("alpha0p75pi", 0.75),
    ("alpha1p00pi", 1.00),
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
if not WOUT_FILE.exists():
    raise FileNotFoundError(WOUT_FILE)
if not GX_GEO_VMEC.exists():
    raise FileNotFoundError(GX_GEO_VMEC)

for alpha_label, alpha_over_pi in ALPHAS:
    alpha = alpha_over_pi * np.pi
    toml_file = DATA_DIR / f"qi_precursor_vmec_ns512_rho0p650_npol5_nt641_{alpha_label}.toml"
    eik_file = DATA_DIR / f"gx_geometry_vmec_QI_precursor_ns512_rho0p650_npol5_nt641_{alpha_label}.eik.nc"
    log_file = DATA_DIR / f"gx_geometry_vmec_QI_precursor_ns512_rho0p650_npol5_nt641_{alpha_label}.log"
    text = f"""[Dimensions]
ntheta = {NTHETA}

[Domain]
boundary = "linked"
y0 = 21.0

[Physics]
beta = 0.0

[Geometry]
geo_option = "vmec"
vmec_file = "{WOUT_FILE}"
alpha = {alpha:.16e}
zeta_center = 0.0
npol = {NPOL}
torflux = {TORFLUX}
betaprim = 0.0
isaxisym = false

[species]
dens = [1.0, 1.0]
temp = [1.0, 1.0]
tprim = [0.0, 0.0]
fprim = [{A_OVER_LN}, 0.0]
"""
    toml_file.write_text(text)
    print(f"alpha={alpha_over_pi:.2f} pi: {eik_file}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PYVMEC_DEPS}:{env.get('PYTHONPATH', '')}"
    with log_file.open("w") as fp:
        subprocess.run(
            ["python", str(GX_GEO_VMEC), str(toml_file), str(eik_file)],
            check=True,
            stdout=fp,
            stderr=subprocess.STDOUT,
            env=env,
        )
    print(f"wrote {eik_file}")
