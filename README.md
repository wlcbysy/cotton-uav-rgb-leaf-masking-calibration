# Leaf-masked UAV RGB for cotton diagnosis and yield prediction

Code and source data for **“Leaf-masked UAV RGB decouples nitrogen deficit from canopy openness and stabilizes few-shot cross-environment yield prediction in mulched cotton”**, prepared for *Artificial Intelligence in Agriculture*.

## Contents

- `01_Data/processed/`: processed model-ready observations, spatial plot boundaries, and small reconstructed image patches. The 15 TIFFs here are reconstructed plot patches, not the original orthomosaics.
- `02_Code/src/upgrade_q1/`: analysis, ablation, and plotting scripts; `02_Code/src/data/`, `02_Code/src/modeling/`, and `02_Code/utilities/` contain supporting code.
- `10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/analysis/q1_spatial_calibration/`: multi-scale features, predictions, precomputed inputs, and ablation results.
- `10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/analysis/frozen_baseline/`: source tables and prior analysis outputs used by the current scripts. The published copy has identifying paths and annotator names removed; it is not a byte-identical copy of the internal frozen baseline.
- `10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/tables/q1_current/`: machine-readable manuscript tables.
- `00_Project/annotation/`: 21-patch boll annotation materials and provenance.

Raw GeoTIFF orthomosaics are not distributed here and are available from the corresponding author upon reasonable request. Paths to raw images in released tables are project-relative references, not included files. Private sampling manifests, personal paths, and names in operational annotation metadata were removed. The external China basemap is excluded; users must obtain external map layers under their own applicable terms. The daily weather table is also excluded because its original acquisition record and redistribution terms have not been established in the project ledger.

## Environment

From the repository root, create a Python environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Core analysis and figure commands below were tested with Python 3.14, NumPy 2.4.6, pandas 3.0.3, SciPy 1.16.3, Matplotlib 3.11.0, scikit-learn 1.9.0, GeoPandas 1.1.3, and Rasterio 1.5.0. For the ResNet-50 image-feature extraction, additionally install compatible `torch` and `torchvision` builds for the chosen platform; pretrained ImageNet weights may need downloading. Raw-image workflows also require the orthomosaics and appropriate raster paths.

## Reproduce selected results

Run from the repository root. The following were tested using the distributed processed inputs. Outputs are written under the `10_Results/.../` tree; work on a copy if the released inputs need to remain unchanged.

```bash
python reproduce.py fig3
python reproduce.py fig4
python reproduce.py fig5
python reproduce.py fig6
python reproduce.py ablation-2x2
```

`fig3`–`fig6` render the corresponding main figures to `figures/main/`; font availability can affect appearance. `ablation-2x2` reruns the mask × index comparison across spatial resolutions and writes `ablation_2x2_index_mask_multiscale.csv`. Other ablation results, including the ResNet-50 and normalization comparisons, are included as CSVs. Their full scripts are `02_Code/src/upgrade_q1/run_peer_review_ablations.py` and `run_post_review_revisions.py`; some stages require the raw orthomosaics, pretrained weights, or other original project resources and were not verified from this distribution alone. Full figure generation (`python 02_Code/src/upgrade_q1/plot_q1_upgraded_figures.py`) also requires raw imagery, the weather table, external basemaps and pre-existing project assets; it is not a self-contained command from this archive. The local manuscript/DOCX package builder depends on additional project and system resources and is included as workflow context, not as a verified standalone release command.

## Verification record provenance

The 2,808 records used by the Figure S1 plotting script are in `analysis/frozen_baseline/analysis/raw_rgb_mining/ai_reference/AI_reference_points.csv`. Their labels are AI visual interpretations, not an independent human reference; see `ai_labels.json` and the `independent_human_reference` field. The separate `human_reference_points.csv` retains its recorded `DISPLAY_CORRESPONDENCE_NOT_VERIFIED` and `usable_for_independent_accuracy=False` flags. The 21-patch boll labels were AI-assisted and accepted after user-reported review; they are not independent unassisted annotations. These distinctions matter when interpreting segmentation metrics.

## License

The authors' code is licensed under the MIT License in [LICENSE](LICENSE). The authors' processed data and annotations are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Third-party software, pretrained weights, and external geographic resources retain their respective licenses; this repository does not relicense them. Please cite the article and repository when using these materials.
