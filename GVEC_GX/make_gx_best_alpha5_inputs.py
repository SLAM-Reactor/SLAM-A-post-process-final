#!/usr/bin/env python3
"""Make GX ITG scan inputs for the best GVEC SLAM/SLNAM equilibria (09-23 resolution scans).
Same physics settings as make_gx_itg_scan_alpha5_inputs.py.

Example:
    python make_gx_best_alpha5_inputs.py
"""

from pathlib import Path


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RUN_ROOT = HERE / "gx_runs" / "best_gvec_t600_alpha5"

A_OVER_LT_VALUES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
A_OVER_LN = 1.0

ALPHAS = (
    ("alpha0p00", 0.0),
    ("alpha0p25pi", 0.25),
    ("alpha0p50pi", 0.50),
    ("alpha0p75pi", 0.75),
    ("alpha1p00pi", 1.00),
)

GEOMETRY_TEMPLATES = {
    "SLAM_BEST": DATA_DIR / "gx_geometry_gvec_SLAM_BEST_rho0p650_npol5_nt641_{alpha_label}.eik.nc",
    "SLNAM_BEST": DATA_DIR / "gx_geometry_gvec_SLNAM_BEST_rho0p650_npol5_nt641_{alpha_label}.eik.nc",
}

BASE_INPUT = """# Nonlinear ITG turbulence with adiabatic electrons.
# Geometry scan: case={case_label}, alpha={alpha_over_pi:.2f} pi.

debug = false

[Dimensions]
ntheta = 640
nperiod = 1
ny = 96
nx = 96
nhermite = 8
nlaguerre = 4
nspecies = 1

[Domain]
y0 = 21.0
boundary = "linked"

[Physics]
beta = 0.0
nonlinear_mode = true

[Time]
t_max = 600.0
dt = 0.1
cfl = 1.0
scheme = "rk3"

[Initialization]
ikpar_init = 0
init_field = "density"
init_amp = 1.0e-3

[Geometry]
geo_option = "nc"
geo_file = "{geo_file}"

[species]
z     = [ 1.0,      -1.0    ]
mass  = [ 1.0,       2.7e-4 ]
dens  = [ 1.0,       1.0    ]
temp  = [ 1.0,       1.0    ]
tprim = [ {a_over_lt:.6f},       0.0    ]
fprim = [ {a_over_ln:.6f},       0.0    ]
vnewk = [ 0.01,      0.0    ]
type  = [ "ion",  "electron" ]

[Boltzmann]
add_Boltzmann_species = true
Boltzmann_type = "electrons"
tau_fac = 1.0

[Dissipation]
closure_model = "none"
hypercollisions = true
hyper = true
D_hyper = 0.05

[Restart]
restart = false
save_for_restart = true

[Diagnostics]
nwrite = 50
free_energy = true
fluxes = true
fields = true
moments = true
"""


RUN_ROOT.mkdir(parents=True, exist_ok=True)
manifest = []

for case_label, template in GEOMETRY_TEMPLATES.items():
    for alpha_label, alpha_over_pi in ALPHAS:
        geo_file = Path(str(template).format(alpha_label=alpha_label))
        if not geo_file.exists():
            raise FileNotFoundError(geo_file)

        for a_over_lt in A_OVER_LT_VALUES:
            tag = f"aLT{a_over_lt:.2f}".replace(".", "p")
            run_dir = RUN_ROOT / case_label / alpha_label / tag
            run_dir.mkdir(parents=True, exist_ok=True)
            text = BASE_INPUT.format(
                case_label=case_label,
                alpha_over_pi=alpha_over_pi,
                geo_file=geo_file,
                a_over_lt=a_over_lt,
                a_over_ln=A_OVER_LN,
            )
            (run_dir / "gx.in").write_text(text)
            manifest.append(run_dir)

manifest_path = RUN_ROOT / "manifest.txt"
manifest_path.write_text("\n".join(str(path) for path in manifest) + "\n")

print(f"wrote {len(manifest)} GX inputs")
print(f"manifest: {manifest_path}")
