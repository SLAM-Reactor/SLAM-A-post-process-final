# Appendix B (Figure B1): Force-Balance Data

This directory is intentionally small. It contains the exact data needed for
Appendix B / Figure B1 of the paper.

## Retained Equilibria

GVEC: the pre-SLAM GVEC equilibrium (`SLNAM_BEST`) is not duplicated here. It is
read from `../GVEC_f67/`:

```text
../GVEC_f67/parameter_SLNAM_216_0p92_nelems32_mn12_34_final.ini
../GVEC_f67/SLNAM_216_0p92_nelems32_mn12_34_State_final.dat
../GVEC_f67/surface_2_SLNAM_1p0_216_0p92_clean1e-3_axis_x1m0p01-Gframe.nc
```

DESC:

```text
publication_normF_SLNAM/desc_slnam_216_0p92/eq_slnam216_0p92_cyl_f9n18_L20M10N20.h5
```

This is the single DESC cylindrical fit used in Figure B1.

## Figure B1

The publication-ready files are:

```text
publication_normF_SLNAM/normF_DESC_SLNAM_cyl_f9n18.pdf
publication_normF_SLNAM/normF_DESC_SLNAM_cyl_f9n18.png
publication_normF_SLNAM/normF_GVEC_SLNAM_BEST.pdf
publication_normF_SLNAM/normF_GVEC_SLNAM_BEST.png
```

Redraw them from cached plane data with:

```bash
cd Appendix-B_fB1/publication_normF_SLNAM
python plot_normF_publication.py
```

The cached plane data are:

```text
publication_normF_SLNAM/normF_DESC_SLNAM_cyl_f9n18_phi_planes.npz
publication_normF_SLNAM/normF_GVEC_SLNAM_BEST_phi_planes.npz
```

Regenerate those caches, if needed, with:

```text
publication_normF_SLNAM/export_desc_normF.py
publication_normF_SLNAM/export_gvec_normF.py
```

`export_gvec_normF.py` reads the SLNAM_BEST equilibrium from `../GVEC_f67/`.
