#!/usr/bin/env python3
"""Export the precursor QI DESC equilibrium to a VMEC wout file with ns=512."""

import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
DESC_SOURCE = Path("/pscratch/sd/r/rgaur/DESC3/DESC")

DESC_FILE = Path(
    "/pscratch/sd/r/rgaur/SLAM-A-post-process-final/QI_coilset/"
    "eq_NFP2_A+9.0_tor+0.0_mr+0.10_shift+0.00_wd+0.00_elong+3.00_wellwt+2.0_vacuum_final2.h5"
)
WOUT_FILE = DATA_DIR / "wout_QI_precursor_OP_NFP2_A9_vacuum_final2_ns512.nc"

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, str(DESC_SOURCE))

import desc.io
from desc import set_device
from desc.vmec import VMECIO


set_device("cpu")
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"loading {DESC_FILE}")
eq = desc.io.load(str(DESC_FILE))
try:
    eq = eq[-1]
except TypeError:
    pass

print(f"writing {WOUT_FILE}")
VMECIO.save(eq, str(WOUT_FILE), surfs=512, verbose=1)
print(f"wrote {WOUT_FILE}")
