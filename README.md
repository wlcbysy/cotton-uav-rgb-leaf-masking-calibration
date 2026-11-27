# Leaf-masked UAV RGB for cotton diagnosis and yield prediction

Code and processed source data for **“Leaf-masked UAV RGB decouples nitrogen deficit from canopy openness and stabilizes few-shot cross-environment yield prediction in mulched cotton”**, prepared for *Artificial Intelligence in Agriculture*.

## Repository contents

- `01_Data/processed/`: model-ready observations (`cotton_rgb_formal_714_slots.csv`), plot yield mappings, and plot boundary geometries (`field_authority.gpkg` and CSV mirrors).
- `02_Code/src/upgrade_q1/`: multiscale coarsening sensitivity, few-shot calibration, orthogonal ablation, supplementary audit, and figure-plotting scripts. `02_Code/src/data/`, `02_Code/src/modeling/`, `02_Code/config/`, and `02_Code/utilities/` contain supporting modules.
- `10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/analysis/q1_spatial_calibration/`: multiscale features, predictions, photometric stress-test outputs (all 64 complete-case plot ROIs), figure inputs, and ablation tables.
- `10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/analysis/frozen_baseline/`: baseline source tables required by the reproduction scripts (with local workstation paths removed).
- `10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/tables/q1_current/`: machine-readable CSV mirrors of Tables 1–3 and Supplementary Tables S1–S7.

The two supplementary figures are **Fig. S1** (forward-only recursive forecast trajectories across calendar and thermal time) and **Fig. S2** (2024 Field 1 plot-level spatial residual maps and Moran's *I*).

Raw GeoTIFF orthomosaics are available from the corresponding author upon reasonable request and are not distributed here. Paths in released tables referring to raw imagery are project-relative. The external China basemap and daily weather table are excluded from public redistribution.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Core commands were tested with Python 3.14, NumPy 2.4, pandas 3.0, SciPy 1.16, Matplotlib 3.11, scikit-learn 1.9, GeoPandas 1.1, and Rasterio 1.5. ResNet-50 image-feature extraction additionally requires compatible `torch` and `torchvision` builds and pretrained ImageNet weights.

## Reproduce selected results

Run from the repository root. Outputs are written inside the `10_Results/.../` tree; work on a copy to preserve the released input files.

```bash
python reproduce.py fig3
python reproduce.py fig4
python reproduce.py fig5
python reproduce.py fig6
python reproduce.py figs1
python reproduce.py ablation-2x2
```

These commands use the distributed processed inputs: `fig3`–`fig6` generate Main Figures 3–6, `figs1` generates Supplementary Figure S1, and `ablation-2x2` reruns the 2 × 2 masking × index comparison across spatial resolutions.

To reproduce **Fig. S2**, place the 2024 Field 1 mid-boll-filling orthomosaic (available from the corresponding author) at `01_Data/raw/remote_sensing/orthomosaics/2024/2024-08-21_f01_visible_orthomosaic.tif`, then run:

```bash
python reproduce.py figs2
```

All 181 DAS boll analyses use the fixed white-pixel threshold rule `white_b140_s30`: mean RGB brightness (*R* + *G* + *B*)/3 ≥ 140 on the 8-bit scale and saturation (max(*R*,*G*,*B*) − min(*R*,*G*,*B*))/max(*R*,*G*,*B*) ≤ 0.30, with connected-component morphological filtering retaining 0.5–120 cm² components.

## License

The authors' code is licensed under the MIT License in [LICENSE](LICENSE). The authors' processed data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party software, pretrained weights, and geographic resources retain their respective licenses. Please cite the article and repository when using these materials.
