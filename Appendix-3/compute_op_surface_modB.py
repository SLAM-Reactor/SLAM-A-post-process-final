#!/usr/bin/env python3
"""Compute the NFP=3 precursor OP equilibrium's rho=1 surface (X,Y,Z,|B|) so
the pyvista render script (which needs pyvista/netCDF4, not available in the
same env as DESC here) can build the mesh without importing DESC. Run in
desc-env3 (desc-env's DESC checkout is mid-edit and needs a module that is not
installed).

Node ordering matters and is easy to get wrong: DESC's LinearGrid returns
nodes with theta varying fastest and zeta slowest, so the flat arrays are
indexed ``zeta * THETA_COUNT + theta``. They are cached flat, exactly as DESC
returns them, and the render script declares the structured-grid dimensions to
match. Nothing here reshapes - a (THETA_COUNT, ZETA_COUNT) reshape silently
scrambles the surface instead of failing, because both counts multiply to the
same total.
"""

import os
import sys

# The DESC3 checkout this was developed against: on NERSC it lives on scratch,
# on the local machine it is mirrored under $HOME. Use whichever exists, and
# otherwise fall back to whatever the active env already resolves.
DESC_SOURCES = [
    "/pscratch/sd/r/rgaur/DESC3/DESC",
    os.path.expanduser("~/DESC3/DESC"),
]
for _source in DESC_SOURCES:
    if os.path.isdir(_source):
        sys.path.insert(0, _source)
        break

from desc import set_device

set_device("cpu")

import desc
import numpy as np
from desc.io import load
from desc.grid import LinearGrid

EQ_FILE = "eq_NFP3_A+10.0_tor+0.0_mr+0.30_shift+0.00_wd+0.00_elong+1.50_wellwt+2.0_vacuum_final2_resolve.h5"
RHO_SURFACE = 1.0
THETA_COUNT = 50
ZETA_COUNT = 128

print(f"using DESC {desc.__version__} from {os.path.dirname(desc.__file__)}", flush=True)

eq = load(EQ_FILE)
grid = LinearGrid(
    rho=np.array([RHO_SURFACE]),
    theta=np.linspace(0, 2 * np.pi, THETA_COUNT, endpoint=True),
    zeta=np.linspace(0, 2 * np.pi, ZETA_COUNT, endpoint=True),
    NFP=1,
)

expected = THETA_COUNT * ZETA_COUNT
if grid.num_nodes != expected:
    raise RuntimeError(f"grid has {grid.num_nodes} nodes, expected {expected}")

# Confirm the theta-fastest/zeta-slowest ordering this cache format assumes,
# rather than trusting it to hold across DESC versions.
nodes = np.asarray(grid.nodes)
theta_grid = nodes[:, 1].reshape(ZETA_COUNT, THETA_COUNT)
zeta_grid = nodes[:, 2].reshape(ZETA_COUNT, THETA_COUNT)
if not (np.allclose(theta_grid, theta_grid[0][None, :])
        and np.allclose(zeta_grid, zeta_grid[:, 0][:, None])):
    raise RuntimeError(
        "LinearGrid node ordering is not theta-fastest/zeta-slowest; "
        "the cache layout and the render script's dimensions need updating"
    )

data = eq.compute(["X", "Y", "Z", "|B|"], grid=grid)
x = np.asarray(data["X"]).ravel()
y = np.asarray(data["Y"]).ravel()
z = np.asarray(data["Z"]).ravel()
modB = np.asarray(data["|B|"]).ravel()

np.savez("op_surface_modB_nfp3.npz", X=x, Y=y, Z=z, mod_B=modB,
         theta_count=THETA_COUNT, zeta_count=ZETA_COUNT, layout="theta_fastest")
print(f"wrote op_surface_modB_nfp3.npz ({expected} nodes, theta fastest)", flush=True)
