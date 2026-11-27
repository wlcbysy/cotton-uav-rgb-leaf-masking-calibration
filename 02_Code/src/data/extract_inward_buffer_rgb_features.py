#!/usr/bin/env python3
"""Batch extraction of 13 visible-light spectral vegetation indices using inward buffer polygons.

Replaces the 0.525 m² fixed central micro-ROI with the authoritative ~15.37 m² inward buffer
polygons (-0.5 m negative buffer from visible compartment boundaries) across all 33 UAV GeoTIFF
orthomosaics and 714 slots.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import warnings

# Suppress rasterio nodata shadow warning
warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")

ROOT = Path(__file__).resolve().parents[3]
GPKG_PATH = ROOT / "01_Data/processed/spatial/cotton_rgb/field_authority.gpkg"
TEMPLATE_CSV = ROOT / "01_Data/processed/model_ready/cotton_rgb_formal_714_slots.csv"
OUT_MODEL_READY = ROOT / "01_Data/processed/model_ready/cotton_rgb_formal_714_slots.csv"
OUT_DEPOSITION = ROOT / "10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/analysis/frozen_baseline/repository_deposition/complete_714_slot_analysis_input.csv"
OUT_AUTHORITY = ROOT / "10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/analysis/frozen_baseline/authority_inputs/complete_714_slot_analysis_input.csv"
OUT_OBSERVED_ONLY = ROOT / "10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/analysis/frozen_baseline/authority_inputs/observed_only_714_slots.csv"
OUT_DEPOSITION_OBSERVED = ROOT / "10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/analysis/frozen_baseline/repository_deposition/observed_only_714_slots.csv"

ROI_SET_ID = "cotton_inward_buffer_0_5m_approved_v1_20260919"
INDICES = ["ExB", "ExG", "ExR", "ExGR", "GBRI", "GLI", "IKAW", "MGRVI", "NGBDI", "NGRDI", "RGBVI", "RGRI", "VARI"]


def calc_indices(r_raw: np.ndarray, g_raw: np.ndarray, b_raw: np.ndarray):
    """Compute normalized chromaticities and all 13 vegetation indices on ExG > 0 mask."""
    valid = (r_raw > 0) | (g_raw > 0) | (b_raw > 0)
    if valid.sum() == 0:
        return None

    s = r_raw + g_raw + b_raw
    r_n = np.zeros_like(r_raw)
    g_n = np.zeros_like(g_raw)
    b_n = np.zeros_like(b_raw)
    
    # Safe division for chromaticity coordinates
    r_n[valid] = r_raw[valid] / s[valid]
    g_n[valid] = g_raw[valid] / s[valid]
    b_n[valid] = b_raw[valid] / s[valid]

    # Primary vegetation mask: ExG = 2g - r - b > 0
    exg = 2.0 * g_n - r_n - b_n
    m = valid & (exg > 0.0)
    
    if m.sum() == 0:
        return None

    # Masked chromaticities
    rm = r_n[m]
    gm = g_n[m]
    bm = b_n[m]

    # DN stats
    red_dn_mean = float(np.mean(r_raw[m]))
    green_dn_mean = float(np.mean(g_raw[m]))
    blue_dn_mean = float(np.mean(b_raw[m]))
    red_dn_median = float(np.median(r_raw[m]))
    green_dn_median = float(np.median(g_raw[m]))
    blue_dn_median = float(np.median(b_raw[m]))
    r_mean = float(np.mean(rm))
    g_mean = float(np.mean(gm))
    b_mean = float(np.mean(bm))

    # Helper for ratio indices with denominator guard
    def safe_ratio_mean(num, den):
        valid_den = np.abs(den) > 1e-9
        if not np.any(valid_den):
            return np.nan
        vals = np.full(len(num), np.nan, dtype=float)
        vals[valid_den] = num[valid_den] / den[valid_den]
        return float(np.nanmean(vals))

    exb_val = float(np.mean(1.4 * bm - gm))
    exg_val = float(np.mean(2.0 * gm - rm - bm))
    exr_val = float(np.mean(1.4 * rm - gm))
    exgr_val = float(np.mean(3.0 * gm - 2.4 * rm - bm))
    gbri_val = safe_ratio_mean(gm, bm)
    gli_val = safe_ratio_mean(2.0 * gm - rm - bm, 2.0 * gm + rm + bm)
    ikaw_val = safe_ratio_mean(rm - bm, rm + bm)
    mgrvi_val = safe_ratio_mean(gm**2 - rm**2, gm**2 + rm**2)
    ngbdi_val = safe_ratio_mean(gm - bm, gm + bm)
    ngrdi_val = safe_ratio_mean(gm - rm, gm + rm)
    rgbvi_val = safe_ratio_mean(gm**2 - rm * bm, gm**2 + rm * bm)
    rgri_val = safe_ratio_mean(rm, gm)
    vari_val = safe_ratio_mean(gm - rm, gm + rm - bm)

    return {
        "red_dn_mean": red_dn_mean,
        "green_dn_mean": green_dn_mean,
        "blue_dn_mean": blue_dn_mean,
        "red_dn_median": red_dn_median,
        "green_dn_median": green_dn_median,
        "blue_dn_median": blue_dn_median,
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "ExB": exb_val,
        "ExG": exg_val,
        "ExR": exr_val,
        "ExGR": exgr_val,
        "GBRI": gbri_val,
        "GLI": gli_val,
        "IKAW": ikaw_val,
        "MGRVI": mgrvi_val,
        "NGBDI": ngbdi_val,
        "NGRDI": ngrdi_val,
        "RGBVI": rgbvi_val,
        "RGRI": rgri_val,
        "VARI": vari_val,
        "base_valid_pixels": int(valid.sum()),
        "mask_selected_pixels": int(m.sum()),
        "roi_total_pixels": int(r_raw.size),
    }


def main():
    t0 = time.time()
    print("=" * 70)
    print("STARTING INWARD BUFFER RGB EXTRACTION PIPELINE")
    print("=" * 70)

    # 1. Load inward buffer polygons
    print(f"Loading geometry layer 'inward_buffer_plot' from: {GPKG_PATH}")
    gdf = gpd.read_file(GPKG_PATH, layer="inward_buffer_plot")
    print(f"Loaded {len(gdf)} plot geometries. Mean area: {gdf['geometry'].area.mean():.2f} m²")

    # Build geometry lookup by (year, field, plot_id)
    geom_dict = {}
    for _, row in gdf.iterrows():
        key = (int(row["year"]), str(row["field"]), int(row["plot_id"]))
        geom_dict[key] = row["geometry"]

    # 2. Load template 714 slots
    print(f"Loading template slot table: {TEMPLATE_CSV}")
    df = pd.read_csv(TEMPLATE_CSV)
    if len(df) != 714:
        raise ValueError(f"Expected 714 slots, found {len(df)}")

    # Keep a copy of original values for correlation analysis
    old_df = df.copy()

    # 3. Process direct raster observations
    observed_slots = df[df["slot_status"] == "AVAILABLE"]
    print(f"Direct observed slots to extract: {len(observed_slots)} across {observed_slots['raster_path'].nunique()} GeoTIFFs")

    # Group by raster path to open each raster only once
    grouped = observed_slots.groupby("raster_path")
    raster_idx = 0
    extracted_count = 0

    for raster_rel_path, group in grouped:
        raster_idx += 1
        # Resolve path
        raster_path = Path(raster_rel_path)
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster not found: {raster_path}")

        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            # Check if geometry reprojection is needed
            reproject = (raster_crs != gdf.crs)
            
            for idx, slot_row in group.iterrows():
                key = (int(slot_row["year"]), str(slot_row["field"]), int(slot_row["plot_id"]))
                poly = geom_dict[key]
                area_m2 = poly.area

                if reproject:
                    # Reproject polygon to raster CRS
                    poly_geom = gpd.GeoSeries([poly], crs=gdf.crs).to_crs(raster_crs).iloc[0]
                else:
                    poly_geom = poly

                out_img, _ = mask(src, [poly_geom], crop=True)
                r_raw = out_img[0].astype(float)
                g_raw = out_img[1].astype(float)
                b_raw = out_img[2].astype(float)

                res = calc_indices(r_raw, g_raw, b_raw)
                if res is None:
                    raise RuntimeError(f"Failed to extract valid vegetation pixels for {key} at {raster_path.name}")

                # Update row
                df.at[idx, "roi_set_id"] = ROI_SET_ID
                df.at[idx, "roi_area_m2"] = area_m2
                df.at[idx, "extraction_status"] = "OK"
                for k, v in res.items():
                    df.at[idx, k] = v

                extracted_count += 1

        if raster_idx % 5 == 0 or raster_idx == len(grouped):
            print(f"[{raster_idx:02d}/{len(grouped):02d}] Processed {raster_path.name} (Total extracted: {extracted_count}/{len(observed_slots)})")

    # 4. Process reconstructed slots (15 slots)
    reconstructed_slots = df[df["slot_status"] != "AVAILABLE"]
    print(f"Reconstructed slots: {len(reconstructed_slots)}")
    for idx, slot_row in reconstructed_slots.iterrows():
        key = (int(slot_row["year"]), str(slot_row["field"]), int(slot_row["plot_id"]))
        poly = geom_dict[key]
        area_m2 = poly.area
        df.at[idx, "roi_set_id"] = ROI_SET_ID
        df.at[idx, "roi_area_m2"] = area_m2

    # 5. Sanity checks
    print("-" * 50)
    print("RUNNING POST-EXTRACTION VALIDATION GATES")
    print("-" * 50)
    assert len(df) == 714, f"Row count changed: {len(df)}"
    assert df["roi_set_id"].nunique() == 1 and df["roi_set_id"].iloc[0] == ROI_SET_ID, "ROI set ID mismatch"
    assert df[INDICES].isna().sum().sum() == 0, "Found NaN in indices of full table"
    assert (df[df["slot_status"] == "AVAILABLE"]["mask_selected_pixels"] > 0).all(), "Found zero-pixel masks in observed slots"
    obs_pixels = df[df["slot_status"] == "AVAILABLE"]["mask_selected_pixels"]
    print(f"All 699 observed slots valid. Mask selected pixels mean: {obs_pixels.mean():.0f} (min: {obs_pixels.min():.0f}, max: {obs_pixels.max():.0f})")
    print(f"Average inward buffer area: {df['roi_area_m2'].mean():.2f} m²")

    # Compare correlation between new inward buffer indices and old micro-ROI indices
    print("\nScale correlation with previous micro-ROI extraction:")
    for col in ["IKAW", "ExG", "RGBVI", "VARI"]:
        r_corr = np.corrcoef(old_df[col], df[col])[0, 1]
        print(f"  {col:6s}: Pearson r = {r_corr:.4f} (mean old: {old_df[col].mean():.4f}, mean new: {df[col].mean():.4f})")

    # 6. Save model-ready table and repository depositions
    print(f"\nSaving model-ready table to: {OUT_MODEL_READY}")
    df.to_csv(OUT_MODEL_READY, index=False)
    print(f"Saving repository deposition to: {OUT_DEPOSITION}")
    df.to_csv(OUT_DEPOSITION, index=False)
    print(f"Saving authority input to: {OUT_AUTHORITY}")
    df.to_csv(OUT_AUTHORITY, index=False)

    # 7. Generate observed-only authority table
    obs_cols = [
        'year', 'field', 'flight_date', 'raster_path', 'roi_set_id', 'plot_id', 'block', 'n_code',
        'analysis_used', 'roi_area_m2', 'source_boundary_status', 'mask_mode', 'extraction_status',
        'red_dn_mean', 'green_dn_mean', 'blue_dn_mean', 'red_dn_median', 'green_dn_median', 'blue_dn_median',
        'r_mean', 'g_mean', 'b_mean', 'ExB', 'ExG', 'ExR', 'ExGR', 'GBRI', 'GLI', 'IKAW', 'MGRVI',
        'NGBDI', 'NGRDI', 'RGBVI', 'RGRI', 'VARI', 'base_valid_pixels', 'mask_selected_pixels', 'roi_total_pixels',
        'plot_id_numeric', 'plot_id_design', 'block_design', 'N_code', 'N_kg_ha', 'grid_row', 'grid_col',
        'dataset_role', 'plot', 'field_number', 'date', 'DAS', 'yield_kg', 'yield_kg_mu', 'yield_kg_ha',
        'yield_authority_source', 'yield_authority_sha256', 'analysis_available', 'slot_status'
    ]
    df_obs = df[obs_cols].copy()
    recon_mask = df_obs["slot_status"] != "AVAILABLE"
    # Nullify index columns for reconstructed slots in observed_only
    null_cols = ['red_dn_mean', 'green_dn_mean', 'blue_dn_mean', 'red_dn_median', 'green_dn_median', 'blue_dn_median',
                 'r_mean', 'g_mean', 'b_mean', 'ExB', 'ExG', 'ExR', 'ExGR', 'GBRI', 'GLI', 'IKAW', 'MGRVI',
                 'NGBDI', 'NGRDI', 'RGBVI', 'RGRI', 'VARI', 'base_valid_pixels', 'mask_selected_pixels', 'roi_total_pixels']
    df_obs.loc[recon_mask, null_cols] = np.nan

    print(f"Saving observed-only authority input to: {OUT_OBSERVED_ONLY}")
    df_obs.to_csv(OUT_OBSERVED_ONLY, index=False)
    if OUT_DEPOSITION_OBSERVED.parent.exists():
        print(f"Saving observed-only deposition to: {OUT_DEPOSITION_OBSERVED}")
        df_obs.to_csv(OUT_DEPOSITION_OBSERVED, index=False)

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"EXTRACTION COMPLETE IN {elapsed:.1f} SECONDS!")
    print("=" * 70)


if __name__ == "__main__":
    main()
