#!/usr/bin/env python3
"""Make GX inputs for the QI precursor (OP) with DESC -> VMEC ns512 -> pyvmec geometry.

Each input is the DESC-direct QI input (same physics settings as SLAM/SLNAM_BEST)
with only the geometry line changed to the pyvmec ns512 .eik.nc file (geo_option = "nc").
The DESC-direct geometry is wrong on the hot field lines (alpha=0.5pi, a/LT=3:
Q=1239 DESC-direct vs 41.9 pyvmec ns512, test of 09-24).

Example:
    python make_qi_pyvmec_ns512_inputs.py
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
# DESC-direct QI runs were moved to the backup tree on 09-24; their inputs are the template
DIRECT_INPUTS = HERE.parent / "GVEC_GX_bkp" / "moved_old_equilibria_09-24" / "gx_runs" / "qi_precursor_desc_direct_t600_alpha5" / "inputs"
RUN_ROOT = HERE / "gx_runs" / "qi_precursor_pyvmec_ns512_t600_alpha5"
GEO = HERE / "data" / "gx_geometry_vmec_QI_precursor_ns512_rho0p650_npol5_nt641_{alpha}.eik.nc"

ALPHAS = ("alpha0p00", "alpha0p25pi", "alpha0p50pi", "alpha0p75pi", "alpha1p00pi")
A_OVER_LT_VALUES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)

manifest = []
for alpha in ALPHAS:
    geo = Path(str(GEO).format(alpha=alpha))
    if not geo.exists():
        raise FileNotFoundError(geo)
    for a_over_lt in A_OVER_LT_VALUES:
        tag = f"aLT{a_over_lt:.2f}".replace(".", "p")
        text = (DIRECT_INPUTS / f"QI_precursor__{alpha}__{tag}.gx.in").read_text()
        lines = text.splitlines()
        geo_lines = [i for i, line in enumerate(lines) if line.startswith("geo_file")]
        assert len(geo_lines) == 1
        lines[geo_lines[0]] = f'geo_file = "{geo}"'
        text = "\n".join(lines) + "\n"
        text = text.replace("# Geometry scan: case=QI_precursor,", "# Geometry scan: case=QI_precursor (DESC->VMEC ns512->pyvmec),")
        run_dir = RUN_ROOT / "QI_precursor" / alpha / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "gx.in").write_text(text)
        manifest.append(run_dir)

(RUN_ROOT / "manifest.txt").write_text("\n".join(str(p) for p in manifest) + "\n")
print(f"wrote {len(manifest)} GX inputs, manifest {RUN_ROOT / 'manifest.txt'}")
