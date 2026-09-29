# QI coilset post-processing

rho0p9 is the directory with the main results!

Both files are deliberately flat, hardcoded scripts with no argument parser,
`main()` function, or helper functions. Change filenames and plotting settings
in the variable block near the top of each file.

- `plot_boozer.py` gets `theta_B`, `zeta_B`, and `|B|` from DESC's
  `plot_boozer_surface(..., return_data=True)`, saves the arrays, and replots
  them using the fonts and pi ticks from the omnigenous-umbilic scripts.
- `plot_equilibrium_coils.py` follows the TQ notebook pattern
  `plot_coils(field)` → `plot_3d(surf, "B*n", field=field, fig=fig)`.
  Here `surf = eq.get_surface_at(rho=0.9)`. It uses DESC's `RdBu_r` scale
  and writes title-free Plotly HTML.

Run both:

```bash
cd /home/rgaur/SLAM-A-post-process/QI_coilset
bash run_all.sh
```

Outputs:

- `Boozer_DESC_rho1.npz`, `.png`, and `.svg`
- `QI_rho0p9_surface_and_40_coils_Bn_plotly.html`

The HTML is self-contained and its Plotly modebar exports an SVG view.

`run_all.sh` activates `desc-env` and prepends `/home/rgaur/DESC3/DESC`, the
clean compatible checkout. The editable `/home/rgaur/DESC` currently contains
unresolved conflict markers and is not modified.
