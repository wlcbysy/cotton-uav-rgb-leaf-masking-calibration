# Leaf-masked UAV RGB for cotton diagnosis and yield prediction

Code and processed source data for **“Leaf-masked UAV RGB decouples nitrogen deficit from canopy openness and stabilizes few-shot cross-environment yield prediction in mulched cotton”**, prepared for *Artificial Intelligence in Agriculture*. This title matches the current manuscript.

## Repository contents

- `01_Data/processed/`: model-ready observations, plot boundaries, and small reconstructed image patches; the 15 TIFF files are reconstructed plot patches, not original orthomosaics.
- `02_Code/src/upgrade_q1/`: analysis, ablation and plotting scripts. `02_Code/src/data/`, `02_Code/src/modeling/`, and `02_Code/utilities/` contain supporting code.
- `10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/analysis/q1_spatial_calibration/`: multiscale features, predictions, figure inputs, and analysis tables.
- `10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/analysis/frozen_baseline/`: source tables and earlier outputs needed by current analyses. This public copy has identifying paths and names removed and is not byte-identical to the internal baseline.
- `10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/tables/q1_current/`: machine-readable manuscript tables.

The two supplementary figures are **Fig. S1**, forward-only recursive forecast trajectories across calendar and thermal time, and **Fig. S2**, 2024 Field 1 plot-level spatial residual maps and Moran's I.

Raw GeoTIFF orthomosaics are available from the corresponding author upon reasonable request and are not distributed here. Paths in released tables referring to raw imagery are project-relative. The external China basemap and the daily weather table are excluded; the latter's original acquisition and redistribution terms have not been established.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Core commands were tested with Python 3.14, NumPy 2.4.6, pandas 3.0.3, SciPy 1.16.3, Matplotlib 3.11.0, scikit-learn 1.9.0, GeoPandas 1.1.3, and Rasterio 1.5.0. ResNet-50 image-feature extraction additionally requires compatible `torch` and `torchvision` builds and pretrained ImageNet weights.

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

These commands use the distributed processed inputs. `fig3`–`fig6` generate the corresponding main figures, `figs1` generates supplementary Fig. S1, and `ablation-2x2` reruns the masking × index comparison across spatial resolutions. Fonts may affect figure appearance.

To reproduce **Fig. S2**, first obtain the 2024 Field 1 orthomosaic from the corresponding author and place it at `01_Data/raw/remote_sensing/orthomosaics/2024/2024-08-21_f01_visible_orthomosaic.tif`, then run:

```bash
python reproduce.py figs2
```

The spatial residual and Moran's I tables used by this figure are included. Full raw-image processing, other analyses, Fig. 1, and Fig. 2 may require additional orthomosaics, pretrained weights, external basemaps, the weather table, and original project resources; those workflows are not validated as self-contained from this release.

The released source-specific and held-out-block boll mask selections are in `source_boll_domain_rules.csv` and `source_boll_mask_rules.csv` in the `q1_spatial_calibration` directory. These selections retain the rules used by the existing analysis scripts; they vary by source and fold. They are supplied to make the existing analysis traceable and are not evidence that every model used one fixed threshold.

## License

The authors' code is licensed under the MIT License in [LICENSE](LICENSE). The authors' processed data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party software, pretrained weights, and geographic resources retain their respective licenses. Please cite the article and repository when using these materials.
