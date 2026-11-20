#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-compute pixel-exact spatial/radiometric caches and prediction tables for rich Q1 Main Figures 2-6.

Strictly read-only on 01_Data/raw and sat_full_figure_label_legend_sync_20260901.
Writes deterministic display caches to sat_q1_spatial_calibration_20261108/analysis/q1_spatial_calibration/.
"""
from __future__ import annotations

from pathlib import Path
import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.stats import pearsonr, t as student_t
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '02_Code/src/data'))
sys.path.insert(0, str(ROOT / '02_Code/src/upgrade_q1'))
from extract_raw_rgb_information import crop, arrays, rules
from run_module_b_spatial_scaling import GSD_LABELS, GSD_LEVELS_M, build_common4_plot_table

warnings.filterwarnings('ignore', category=rasterio.errors.NodataShadowWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

FORMAL_NEW = ROOT / '10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108'
FORMAL_OLD = FORMAL_NEW / 'analysis/frozen_baseline'
OLD_MINING = FORMAL_OLD / 'analysis/raw_rgb_mining'
OLD_METHOD = FORMAL_OLD / 'analysis/method_strengthening'
OLD_DM = FORMAL_OLD / 'analysis/data_mining_current'
Q1_DIR = FORMAL_NEW / 'analysis/q1_spatial_calibration'
INPUT_SLOTS = ROOT / '01_Data/processed/model_ready/cotton_rgb_formal_714_slots.csv'
GPKG = ROOT / '01_Data/processed/spatial/cotton_rgb/field_authority.gpkg'


def build_fig2_assets(df_slots: pd.DataFrame, gdf: gpd.GeoDataFrame, npz_dict: dict) -> None:
    """1) Native 3.3 mm Pure-Leaf NGRDI + Contiguous Gap (>=0.04 m2) + 1D Cross-Row Profile; 2) Quadratic 95% CI."""
    mm = pd.read_csv(Q1_DIR / 'millimeter_slot_features_699.csv')
    cand = mm[(mm.year == 2024) & (mm.DAS == 128) & (mm.N_kg_ha.isin([75, 150, 225])) & (mm.f_cover > 0.62)]
    best = cand.sort_values('gap_large_frac', ascending=False).iloc[0]
    p_id = int(best.plot_id)
    fld = str(best.field)

    s_row = df_slots[(df_slots.year == 2024) & (df_slots.field == fld) & (df_slots.plot_id == p_id) & (df_slots.DAS == 128)].iloc[0]
    with rasterio.open(s_row.raster_path) as src:
        gg = gdf.to_crs(src.crs)
        px_m = float(np.sqrt(abs(src.res[0] * src.res[1])))
        geom = gg[(gg.year == 2024) & (gg.field == fld) & (gg.plot_id == p_id)].geometry.iloc[0]
        rgb0, val0 = crop(src, geom)

    # Label contiguous gaps on full ROI first, then pick 1.8 m x 1.5 m window with clearest large gap + rows
    grn0 = rules(rgb0, val0)['green_canopeo']
    gap0 = val0 & (~grn0)
    lbl0, n_cc0 = ndimage.label(gap0)
    gap_full = np.zeros(val0.shape, dtype=np.uint8)
    if n_cc0 > 0:
        sizes0 = np.bincount(lbl0.ravel()) * (px_m * px_m)
        sizes0[0] = 0.0
        gap_full[((sizes0 > 0.0) & (sizes0 < 0.04))[lbl0]] = 1
        gap_full[(sizes0 >= 0.04)[lbl0]] = 2

    h_w = int(round(1.50 / px_m))
    w_w = int(round(1.80 / px_m))
    best_sc, best_yx = -1.0, (max(0, val0.shape[0] // 2 - h_w // 2), max(0, val0.shape[1] // 2 - w_w // 2))
    step = max(20, h_w // 6)
    for yy in range(0, max(1, val0.shape[0] - h_w), step):
        for xx in range(0, max(1, val0.shape[1] - w_w), step):
            sv = val0[yy:yy + h_w, xx:xx + w_w]
            if sv.mean() < 0.99:
                continue
            sg = gap_full[yy:yy + h_w, xx:xx + w_w]
            f_lg = (sg == 2).mean()
            f_grn = (sg == 0).mean()
            if 0.08 <= f_lg <= 0.35 and f_grn >= 0.58:
                sc = f_lg + 0.5 * f_grn
                if sc > best_sc:
                    best_sc = sc
                    best_yx = (yy, xx)

    y0, x0 = best_yx
    rgb_w = rgb0[y0:y0 + h_w, x0:x0 + w_w]
    grn = grn0[y0:y0 + h_w, x0:x0 + w_w]
    gap_cls = gap_full[y0:y0 + h_w, x0:x0 + w_w]
    r, g, _, _, _ = arrays(rgb_w)
    ngrdi = np.divide(g - r, g + r, out=np.zeros_like(g), where=(g + r) > 0)
    ngrdi_leaf = np.where(grn, ngrdi, np.nan).astype(np.float32)

    # 1D cross-row cover profile along the axis with larger oscillation SD
    prof_col = grn.mean(axis=0) * 100.0
    prof_row = grn.mean(axis=1) * 100.0
    sm_k = max(3, int(round(0.04 / px_m)))
    ker = np.ones(sm_k) / sm_k
    p_c_sm = np.convolve(prof_col, ker, mode='same')
    p_r_sm = np.convolve(prof_row, ker, mode='same')
    if np.std(p_c_sm) >= np.std(p_r_sm):
        prof_1d = p_c_sm
        prof_x_m = np.arange(len(p_c_sm)) * px_m
    else:
        # Rotate arrays 90 deg so rows run vertically and x-axis is cross-row!
        rgb_w = np.rot90(rgb_w)
        ngrdi_leaf = np.rot90(ngrdi_leaf)
        gap_cls = np.rot90(gap_cls)
        prof_1d = p_r_sm[::-1]
        prof_x_m = np.arange(len(prof_1d)) * px_m

    npz_dict['f2_anat_rgb'] = rgb_w
    npz_dict['f2_anat_ngrdi'] = ngrdi_leaf
    npz_dict['f2_anat_gap_cls'] = gap_cls
    npz_dict['f2_anat_prof_x_m'] = prof_x_m.astype(np.float32)
    npz_dict['f2_anat_prof_pct'] = prof_1d.astype(np.float32)
    npz_dict['f2_anat_meta'] = np.array([float(p_id), float(best.N_kg_ha), float(px_m)], dtype=np.float32)

    # Block-adjusted quadratic N response 95% confidence bands & R2 on common 64 support
    common = pd.read_csv(OLD_MINING / 'common_plot_features_common_DAS.csv')
    ci_rows = []
    n_grid = np.linspace(0.0, 450.0, 91)
    for dom, sub in common.groupby('domain'):
        y = sub['yield_kg_ha'].to_numpy(float) / 1000.0
        n_v = sub['N_kg_ha'].to_numpy(float)
        blk = pd.get_dummies(sub['block_design'].astype(int), drop_first=True).to_numpy(float)
        # Design with centered block effects so intercept represents mean across blocks
        blk_c = blk - blk.mean(axis=0, keepdims=True)
        X = np.column_stack([np.ones_like(n_v), n_v, n_v ** 2, blk_c])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        dof = max(1, len(y) - X.shape[1])
        s2 = float(np.sum((y - yhat) ** 2) / dof)
        cov_b = s2 * np.linalg.inv(X.T @ X)
        t_crit = float(student_t.ppf(0.975, dof))

        # Within-block R2 and unadjusted quadratic R2
        X_n = np.column_stack([np.ones_like(n_v), n_v, n_v ** 2])
        b_n, _, _, _ = np.linalg.lstsq(X_n, y, rcond=None)
        r2_quad = float(1.0 - np.sum((y - X_n @ b_n) ** 2) / np.sum((y - y.mean()) ** 2))
        n_opt = float(np.clip(-beta[1] / (2.0 * beta[2]), 0.0, 450.0)) if beta[2] < 0 else 450.0

        Xg = np.column_stack([np.ones_like(n_grid), n_grid, n_grid ** 2, np.zeros((len(n_grid), blk_c.shape[1]))])
        mu = Xg @ beta
        se = np.sqrt(np.einsum('ij,jk,ik->i', Xg, cov_b, Xg))
        for ng, m_i, s_i in zip(n_grid, mu, se):
            ci_rows.append({
                'domain': dom,
                'N': float(ng),
                'mean_mg': float(m_i),
                'low_95_mg': float(m_i - t_crit * s_i),
                'high_95_mg': float(m_i + t_crit * s_i),
                'r2_quad': r2_quad,
                'n_opt': n_opt,
            })
    pd.DataFrame(ci_rows).to_csv(Q1_DIR / 'rich_fig2_quad_ci.csv', index=False)


def build_fig3_assets(df_slots: pd.DataFrame, gdf: gpd.GeoDataFrame) -> None:
    """1) Plot-level & pixel-level Background vs Pure-Leaf chroma drift across environments; 2) Within-N anomalies."""
    # Pixel-level ExG and NGRDI density curves for Background (~M_leaf) vs Pure Leaf (M_leaf) at 141/142 DAS
    rng = np.random.default_rng(20261108)
    common = pd.read_csv(OLD_MINING / 'common_plot_features_common_DAS.csv')
    late_das = {'2023_F1': 142, '2024_F1': 141, '2024_F2': 141}

    bins = np.linspace(-0.25, 0.65, 91)
    centers = 0.5 * (bins[:-1] + bins[1:])
    dens_rows = []

    for dom, das_t in late_das.items():
        yr, fld = int(dom.split('_')[0]), dom.split('_')[1]
        p_ids = set(common[common.domain == dom]['plot_id'].astype(int))
        s_row = df_slots[(df_slots.year == yr) & (df_slots.field == fld) & (df_slots.DAS == das_t)].iloc[0]
        leaf_exg_all, bg_exg_all = [], []
        leaf_ngrdi_all, bg_ngrdi_all = [], []
        with rasterio.open(s_row.raster_path) as src:
            gg = gdf.to_crs(src.crs)
            sub_g = gg[(gg.year == yr) & (gg.field == fld) & (gg.plot_id.isin(p_ids))]
            for row_g in sub_g.itertuples():
                rgb0, val0 = crop(src, row_g.geometry)
                r, g, b, _, _ = arrays(rgb0)
                grn = rules(rgb0, val0)['green_canopeo']
                tot = r + g + b
                exg = np.divide(2.0 * g - r - b, tot, out=np.zeros_like(g), where=tot > 0)
                ngrdi = np.divide(g - r, g + r, out=np.zeros_like(g), where=(g + r) > 0)
                bg = val0 & (~grn)
                if grn.sum() > 500:
                    idx = rng.choice(int(grn.sum()), size=2500, replace=False) if grn.sum() > 2500 else np.arange(int(grn.sum()))
                    leaf_exg_all.append(exg[grn][idx])
                    leaf_ngrdi_all.append(ngrdi[grn][idx])
                if bg.sum() > 200:
                    idx_b = rng.choice(int(bg.sum()), size=2500, replace=False) if bg.sum() > 2500 else np.arange(int(bg.sum()))
                    bg_exg_all.append(exg[bg][idx_b])
                    bg_ngrdi_all.append(ngrdi[bg][idx_b])

        for cls_name, arr_exg, arr_ng in [
            ('Pure_Leaf', np.concatenate(leaf_exg_all), np.concatenate(leaf_ngrdi_all)),
            ('Mulch_Soil_BG', np.concatenate(bg_exg_all), np.concatenate(bg_ngrdi_all)),
        ]:
            h_exg, _ = np.histogram(arr_exg, bins=bins, density=True)
            h_ng, _ = np.histogram(arr_ng, bins=bins, density=True)
            for bc, de, dn in zip(centers, h_exg, h_ng):
                dens_rows.append({
                    'domain': dom,
                    'pixel_class': cls_name,
                    'bin_center': float(bc),
                    'exg_density': float(de),
                    'ngrdi_density': float(dn),
                    'exg_median': float(np.median(arr_exg)),
                    'ngrdi_median': float(np.median(arr_ng)),
                })
    pd.DataFrame(dens_rows).to_csv(Q1_DIR / 'rich_fig3_pixel_chroma_density.csv', index=False)

    # Within-N observed vs predicted yield anomalies across adjacent-field transfers
    ms_pred = pd.read_csv(Q1_DIR / 'multiscale_predictions.csv').query("prefix == '3.3 mm'")
    rn_pred = pd.read_csv(OLD_METHOD / 'resnet50_partial_ridge_predictions.csv').assign(specification='N_ResNet50')
    comb = pd.concat([
        ms_pred[ms_pred.specification.isin(['N_C_T', 'N_Leaf_T'])][['source', 'target', 'evaluation', 'specification', 'plot_id', 'N', 'observed', 'predicted']],
        rn_pred[['source', 'target', 'evaluation', 'specification', 'plot_id', 'N', 'observed', 'predicted']],
    ], ignore_index=True)

    anom_rows = []
    for (src, tgt, ev, sp), grp in comb.groupby(['source', 'target', 'evaluation', 'specification']):
        g = grp.copy()
        g['obs_within_N'] = g['observed'] - g.groupby('N')['observed'].transform('mean')
        g['pred_within_N'] = g['predicted'] - g.groupby('N')['predicted'].transform('mean')
        anom_rows.append(g)
    pd.concat(anom_rows, ignore_index=True).to_csv(Q1_DIR / 'rich_fig3_within_n_anomalies.csv', index=False)


def build_fig4_assets(df_slots: pd.DataFrame, gdf: gpd.GeoDataFrame, npz_dict: dict) -> None:
    """1) 181 DAS connected-component glare-sheet rejection visual slice; 2) 64-plot cluster size spectrum."""
    mm = pd.read_csv(Q1_DIR / 'millimeter_slot_features_699.csv')
    cand = mm[(mm.DAS == 181) & (mm.white_b140_s30_morph_frac > 0.025)].sort_values(
        'white_b140_s30_glare_frac', ascending=False
    )
    best = cand.iloc[0]
    yr, fld, p_id = int(best.year), str(best.field), int(best.plot_id)

    s_row = df_slots[(df_slots.year == yr) & (df_slots.field == fld) & (df_slots.plot_id == p_id) & (df_slots.DAS == 181)].iloc[0]
    with rasterio.open(s_row.raster_path) as src:
        gg = gdf.to_crs(src.crs)
        px_m = float(np.sqrt(abs(src.res[0] * src.res[1])))
        geom = gg[(gg.year == yr) & (gg.field == fld) & (gg.plot_id == p_id)].geometry.iloc[0]
        rgb0, val0 = crop(src, geom)

    # Find 1.8 m x 1.6 m window containing both >120 cm2 glare sheet and 0.5-120 cm2 boll clusters
    wm0 = rules(rgb0, val0)['white_b140_s30']
    lbl0, n_cc0 = ndimage.label(wm0)
    px_cm2 = (px_m * 100.0) ** 2
    cls_full = np.zeros(val0.shape, dtype=np.uint8)
    if n_cc0 > 0:
        areas_cm2 = np.bincount(lbl0.ravel()) * px_cm2
        areas_cm2[0] = 0.0
        cls_full[((areas_cm2 > 0) & (areas_cm2 < 0.5))[lbl0]] = 1
        cls_full[((areas_cm2 >= 0.5) & (areas_cm2 <= 120.0))[lbl0]] = 2
        cls_full[(areas_cm2 > 120.0)[lbl0]] = 3

    h_w = int(round(1.50 / px_m))
    w_w = int(round(1.80 / px_m))
    best_score = -1.0
    best_yx = (0, 0)
    step = max(20, h_w // 6)
    for y0 in range(0, max(1, val0.shape[0] - h_w), step):
        for x0 in range(0, max(1, val0.shape[1] - w_w), step):
            sub_v = val0[y0:y0 + h_w, x0:x0 + w_w]
            if sub_v.mean() < 0.98:
                continue
            sub_c = cls_full[y0:y0 + h_w, x0:x0 + w_w]
            f_boll = (sub_c == 2).mean()
            f_glare = (sub_c == 3).mean()
            score = min(f_boll, 0.12) * 2.0 + min(f_glare, 0.18) * 2.5
            if f_boll > 0.02 and f_glare > 0.02 and score > best_score:
                best_score = score
                best_yx = (y0, x0)

    by, bx = best_yx
    npz_dict['f4_glare_rgb'] = rgb0[by:by + h_w, bx:bx + w_w]
    npz_dict['f4_glare_cls'] = cls_full[by:by + h_w, bx:bx + w_w]
    npz_dict['f4_glare_meta'] = np.array([float(yr), float(p_id), float(best.N_kg_ha), float(px_m)], dtype=np.float32)

    # Full 64-plot connected-component area spectrum at 181 DAS (Low N <=75 vs Fertilized N >=150)
    common = pd.read_csv(OLD_MINING / 'common_plot_features_common_DAS.csv')
    log_bins = np.linspace(-1.0, 3.4, 23)  # 0.1 cm2 to 2512 cm2
    log_centers = 0.5 * (log_bins[:-1] + log_bins[1:])
    area_buckets = {'Low_N_0_75': [], 'Fert_N_150_450': []}

    for (yr_k, fld_k), grp_c in common.groupby(['year', 'field']):
        s181 = df_slots[(df_slots.year == yr_k) & (df_slots.field == fld_k) & (df_slots.DAS == 181)].iloc[0]
        with rasterio.open(s181.raster_path) as src:
            gg = gdf.to_crs(src.crs)
            px_m_i = float(np.sqrt(abs(src.res[0] * src.res[1])))
            px_cm2_i = (px_m_i * 100.0) ** 2
            for r_c in grp_c.itertuples():
                geom = gg[(gg.year == yr_k) & (gg.field == fld_k) & (gg.plot_id == r_c.plot_id)].geometry.iloc[0]
                rgb_i, val_i = crop(src, geom)
                wm_i = rules(rgb_i, val_i)['white_b140_s30']
                lbl_i, n_i = ndimage.label(wm_i)
                if n_i > 0:
                    a_cm2 = np.bincount(lbl_i.ravel())[1:] * px_cm2_i
                    grp_key = 'Low_N_0_75' if r_c.N_kg_ha <= 75 else 'Fert_N_150_450'
                    area_buckets[grp_key].append(a_cm2)

    spec_rows = []
    for grp_key, lst in area_buckets.items():
        all_a = np.concatenate(lst)
        # Weight each component by its physical area so y-axis shows share of total white-mask area!
        tot_area = float(all_a.sum())
        hist_w, _ = np.histogram(np.log10(np.clip(all_a, 0.101, 2500.0)), bins=log_bins, weights=all_a / tot_area * 100.0)
        for lc, pct_share in zip(log_centers, hist_w):
            spec_rows.append({
                'n_regime': grp_key,
                'log10_area_cm2': float(lc),
                'area_cm2': float(10.0 ** lc),
                'white_area_share_pct': float(pct_share),
            })
    pd.DataFrame(spec_rows).to_csv(Q1_DIR / 'rich_fig4_cluster_spectrum.csv', index=False)


def build_fig5_and_fig6_assets(gdf: gpd.GeoDataFrame) -> None:
    """Build Fig 5e paradox N-response curves, Fig 5g spatial active map, Fig 6a 9x6 heatmap, Fig 6c plot intervals."""
    ms_pred = pd.read_csv(Q1_DIR / 'multiscale_predictions.csv').query("prefix == '3.3 mm'")
    rn_pred = pd.read_csv(OLD_METHOD / 'resnet50_partial_ridge_predictions.csv').assign(specification='N_ResNet50')
    comb = pd.concat([
        ms_pred[ms_pred.specification.isin(['N', 'N_C_T', 'N_Leaf_T'])][['source', 'target', 'evaluation', 'specification', 'plot_id', 'block', 'N', 'observed', 'predicted']],
        rn_pred[['source', 'target', 'evaluation', 'specification', 'plot_id', 'block', 'N', 'observed', 'predicted']],
    ], ignore_index=True)

    # 1. Fig 5e: Paradox curves across all 6 transfer tasks (including N0 and fertilized plots)
    par_rows = []
    for (src, tgt, sp), grp in comb[comb.evaluation.isin(['adjacent_field', 'cross_year'])].groupby(['source', 'target', 'specification']):
        g = grp.copy()
        n0_sub = g[g.N == 0]
        shift_3n0 = float((n0_sub['observed'] - n0_sub['predicted']).mean())
        g['pred_zero'] = g['predicted']
        g['pred_plus_3N0'] = g['predicted'] + shift_3n0
        g['shift_3N0'] = shift_3n0
        par_rows.append(g)
    pd.concat(par_rows, ignore_index=True).to_csv(Q1_DIR / 'rich_fig5_paradox_predictions.csv', index=False)

    # 2. Fig 5g: Spatial polygon coordinates & UAV-stratified active selection on 2024_F2 (from 2024_F1 -> 2024_F2 N_Leaf_T)
    g_f2 = gdf[(gdf.year == 2024) & (gdf.field == 'F2')].copy()
    g_f2 = g_f2.to_crs(epsg=32644)  # metric UTM meters
    minx, miny = g_f2.bounds.minx.min(), g_f2.bounds.miny.min()
    sub_pred = comb[(comb.source == '2024_F1') & (comb.target == '2024_F2') & (comb.specification == 'N_Leaf_T')].set_index('plot_id')

    # Identify exact k=3 UAV-stratified active calibration plots (Block 1 Q25, Block 2 Q50, Block 3 Q75 among N > 0)
    fert_p = sub_pred[sub_pred.N > 0].copy()
    selected_active = {}
    for b_id, q_target, q_lbl in [(1, 0.25, 'B1: Q25'), (2, 0.50, 'B2: Q50'), (3, 0.75, 'B3: Q75')]:
        b_sub = fert_p[fert_p['block'] == b_id]
        q_val = float(np.quantile(b_sub['predicted'], q_target))
        chosen_pid = int((b_sub['predicted'] - q_val).abs().idxmin())
        selected_active[chosen_pid] = q_lbl

    poly_rows = []
    for row in g_f2.itertuples():
        pid = int(row.plot_id)
        if pid not in sub_pred.index:
            continue
        xx, yy = row.geometry.exterior.coords.xy
        pr = sub_pred.loc[pid]
        poly_rows.append({
            'plot_id': pid,
            'block': int(pr['block']),
            'N': int(pr['N']),
            'pred_mg': float(pr['predicted'] / 1000.0),
            'obs_mg': float(pr['observed'] / 1000.0),
            'active_tag': selected_active.get(pid, ''),
            'coords_x_m': ';'.join(f'{x - minx:.2f}' for x in xx),
            'coords_y_m': ';'.join(f'{y - miny:.2f}' for y in yy),
            'cx_m': float(row.geometry.centroid.x - minx),
            'cy_m': float(row.geometry.centroid.y - miny),
        })
    pd.DataFrame(poly_rows).to_csv(Q1_DIR / 'rich_fig5_active_spatial_f2.csv', index=False)

    # 3. Fig 6a: 9-task x 6-phenological-stage forward-only N+C+T gain matrix
    d_rec = pd.read_csv(OLD_DM / 'recursive_metrics.csv')
    sub_r = d_rec[(d_rec.clock == 'DAS') & (d_rec.variant == 'C+T')].copy()
    stage_bins = [
        ('Squaring\n(≤88)', 0, 89),
        ('Flowering\n(92)', 90, 95),
        ('Early boll\n(106/107)', 100, 110),
        ('Mid boll\n(128/130)', 125, 132),
        ('Late boll\n(141/142)', 139, 144),
        ('Boll open\n(181)', 179, 185),
    ]
    hm_rows = []
    for (src, tgt, ev), grp in sub_r.groupby(['source', 'target', 'evaluation']):
        for st_lbl, d_lo, d_hi in stage_bins:
            m = grp[(grp.cutoff >= d_lo) & (grp.cutoff <= d_hi)]
            if len(m) > 0:
                r_last = m.sort_values('cutoff').iloc[-1]
                val = float(r_last.gain)
            else:
                # Latest available cutoff at or before d_hi
                m_prev = grp[grp.cutoff <= d_hi]
                val = float(m_prev.sort_values('cutoff').iloc[-1].gain) if len(m_prev) > 0 else np.nan
            hm_rows.append({
                'source': src,
                'target': tgt,
                'evaluation': ev,
                'stage': st_lbl,
                'gain_kg_ha': val,
            })
    pd.DataFrame(hm_rows).to_csv(Q1_DIR / 'rich_fig6_temporal_heatmap.csv', index=False)

    # 4. Fig 6a & Fig S3: Forward-only recursive trajectories for N+Leaf+T (variant='Leaf+T') across DAS and GDD clocks
    from itertools import product as iter_product
    sys.path.insert(0, str(ROOT / '02_Code/src/modeling'))
    import mine_raw_rgb_information as legacy_mod

    mm_slots = pd.read_csv(Q1_DIR / 'millimeter_slot_features_699.csv')
    common_base = pd.read_csv(OLD_MINING / 'common_plot_features_common_DAS.csv')
    fl_time = pd.read_csv(OLD_DM / 'flight_thermal_time.csv')
    mm_slots = mm_slots.merge(fl_time[['domain', 'DAS', 'GDD']], on=['domain', 'DAS'], how='left')
    dom_list = sorted(common_base.domain.unique())

    def _history_mm(dom_k: str, clock_k: str, cut_val: float) -> pd.DataFrame | None:
        b_df = common_base[common_base.domain == dom_k].copy()
        h_rows = []
        for r_b in b_df.itertuples():
            d_s = mm_slots[
                (mm_slots.domain == dom_k) & (mm_slots.plot_id == r_b.plot_id) & (mm_slots[clock_k] <= cut_val)
            ].sort_values(clock_k)
            if d_s.empty:
                return None
            x_d = d_s[clock_k].to_numpy(float)
            c_d = d_s.f_cover.to_numpy(float)
            last_s = d_s.iloc[-1]
            span_d = x_d[-1] - x_d[0]
            h_rows.append({
                'plot_id': int(r_b.plot_id),
                'H_cover': float(c_d[-1]),
                'H_ngrdi_leaf': float(last_s.ngrdi_leaf),
                'H_auc': float(np.trapezoid(c_d, x_d) / span_d) if span_d else float(c_d[-1]),
                'H_change': float((c_d[-1] - c_d[-2]) / (x_d[-1] - x_d[-2]) * span_d) if len(d_s) > 1 else 0.0,
            })
        return b_df.merge(pd.DataFrame(h_rows), on='plot_id', validate='one_to_one')

    rec_leaf_rows = []
    for src_d in dom_list:
        for tgt_d in dom_list:
            ev_d = 'BlockCV' if src_d == tgt_d else ('adjacent_field' if src_d[:4] == tgt_d[:4] else 'cross_year')
            for clock_k in ('DAS', 'GDD'):
                for fl_r in fl_time[fl_time.domain == tgt_d].sort_values(clock_k).itertuples():
                    cut_v = float(getattr(fl_r, clock_k))
                    fa = _history_mm(src_d, clock_k, cut_v)
                    fb = _history_mm(tgt_d, clock_k, cut_v)
                    if fa is None or fb is None:
                        continue
                    ya = fa.yield_kg_ha.to_numpy(float)
                    yb = fb.yield_kg_ha.to_numpy(float)
                    grpa = fa.block_design.to_numpy(int)
                    grpb = fb.block_design.to_numpy(int)
                    iia = np.arange(len(fa))
                    folds = np.unique(grpa) if src_d == tgt_d else [0]
                    na, nb = fa.N_kg_ha.to_numpy(float), fb.N_kg_ha.to_numpy(float)

                    preds_pair = {}
                    for v_tag, f_cols in [('N', []), ('Leaf+T', ['H_cover', 'H_ngrdi_leaf', 'H_auc', 'H_change'])]:
                        xa = np.c_[na, na * na, fa[f_cols].to_numpy(float)] if f_cols else np.c_[na, na * na]
                        xb = np.c_[nb, nb * nb, fb[f_cols].to_numpy(float)] if f_cols else np.c_[nb, nb * nb]
                        p_vec = np.empty(len(fb))
                        for fld_k in folds:
                            tr_k = iia[grpa != fld_k] if fld_k else iia
                            te_k = np.flatnonzero(grpb == fld_k) if fld_k else np.arange(len(fb))
                            if f_cols:
                                sc_list = []
                                for alpha_v in legacy_mod.ALPHAS:
                                    oof_v = np.empty(len(tr_k))
                                    for blk_v in np.unique(grpa[tr_k]):
                                        kp_v = grpa[tr_k] != blk_v
                                        oof_v[~kp_v] = legacy_mod.PartialRidge(alpha_v).fit(xa[tr_k[kp_v]], ya[tr_k[kp_v]]).predict(xa[tr_k[~kp_v]])
                                    sc_list.append((float(np.sqrt(np.mean((oof_v - ya[tr_k]) ** 2))), float(np.mean(np.abs(oof_v - ya[tr_k]))), -alpha_v))
                                best_a = -min(sc_list)[2]
                            else:
                                best_a = 0.0
                            p_vec[te_k] = legacy_mod.PartialRidge(best_a).fit(xa[tr_k], ya[tr_k]).predict(xb[te_k])
                        preds_pair[v_tag] = p_vec

                    err_n = preds_pair['N'] - yb
                    err_l = preds_pair['Leaf+T'] - yb
                    rmse_n = float(np.sqrt(np.mean(err_n ** 2)))
                    rmse_l = float(np.sqrt(np.mean(err_l ** 2)))
                    members = [np.flatnonzero(grpb == v_b) for v_b in np.unique(grpb)]
                    draws_g = []
                    for picks in iter_product(members, repeat=3):
                        ix_p = np.concatenate(picks)
                        draws_g.append(float(np.sqrt(np.mean(err_n[ix_p] ** 2)) - np.sqrt(np.mean(err_l[ix_p] ** 2))))
                    rec_leaf_rows.append({
                        'source': src_d, 'target': tgt_d, 'evaluation': ev_d, 'clock': clock_k,
                        'cutoff': cut_v, 'target_date': str(fl_r.flight_date),
                        'target_DAS': int(fl_r.DAS), 'target_GDD': float(fl_r.GDD),
                        'variant': 'Leaf+T', 'n': int(len(yb)),
                        'RMSE_N': rmse_n, 'RMSE_model': rmse_l, 'gain': rmse_n - rmse_l,
                        'low': float(np.quantile(draws_g, 0.025)), 'high': float(np.quantile(draws_g, 0.975)),
                        'resamples': 27,
                    })
    pd.DataFrame(rec_leaf_rows).to_csv(Q1_DIR / 'rich_fig6_recursive_leaf.csv', index=False)

    # 5. Fig 6c: Plot-by-plot 90% prediction intervals on fertilized target holdouts
    cp = pd.read_csv(Q1_DIR / 'calibrated_interval_coverage.csv')
    int_rows = []
    for src, tgt in [('2024_F2', '2024_F1'), ('2023_F1', '2024_F2')]:
        for sp_csv, mod_lbl in [('N_Leaf_T', 'N+Leaf+T+3N0'), ('N_ResNet50', 'N+ResNet-50+3N0')]:
            cp_mod = 'N+Leaf+T+3N0' if sp_csv == 'N_Leaf_T' else 'N+ResNet-50+3N0'
            r_cp = cp[(cp.source == src) & (cp.target == tgt) & (cp.model == cp_mod)].iloc[0]
            rad = float(r_cp.radius_q90)
            off = float(r_cp.offset_kg_ha)
            p_sub = comb[(comb.source == src) & (comb.target == tgt) & (comb.specification == sp_csv) & (comb.N > 0)].sort_values(['N', 'block', 'plot_id'])
            for idx_rank, pr in enumerate(p_sub.itertuples()):
                c_pred = float(pr.predicted + off)
                lo_b = c_pred - rad
                hi_b = c_pred + rad
                covered = int(lo_b <= pr.observed <= hi_b)
                int_rows.append({
                    'task': f'{src}->{tgt}',
                    'model': mod_lbl,
                    'plot_rank': idx_rank,
                    'plot_id': int(pr.plot_id),
                    'N': int(pr.N),
                    'obs_mg': float(pr.observed / 1000.0),
                    'pred_cal_mg': float(c_pred / 1000.0),
                    'low_90_mg': float(lo_b / 1000.0),
                    'high_90_mg': float(hi_b / 1000.0),
                    'covered': covered,
                })
    pd.DataFrame(int_rows).to_csv(Q1_DIR / 'rich_fig6_plot_intervals.csv', index=False)


def build_direction_b_assets() -> None:
    """Build Direction B (Prior-Free RGB-Only Diagnosis, 2D Attribution Quadrants & Calibrated Yield Inversion) tables."""
    p64 = pd.read_csv(Q1_DIR / 'plot64_millimeter_features.csv')
    gsd_df = pd.read_csv(Q1_DIR / 'multiscale_gsd_slot_features.csv')

    # 1. Block-standardized (yield-blind, N-blind) optical features & 2D Diagnostic Attribution Plane
    rows_b = []
    for (dom, blk), g in p64.groupby(['domain', 'block_design']):
        g = g.copy()
        for col in [
            'Cearly_cover', 'C_cover', 'Cearly_exg', 'C_exg',
            'Cearly_ngrdi_leaf', 'C_ngrdi_leaf', 'T_ngrdi_leaf_auc',
            'T_auc_mean', 'T_late_change',
            'Searly_gap_large', 'S_gap_large', 'T_gap_auc',
            'Searly_prof_across', 'S_prof_across',
            'white_b140_s30_raw', 'white_b140_s30_morph_frac',
        ]:
            mu = float(g[col].mean())
            sd = float(g[col].std(ddof=0) + 1e-12)
            g[f'{col}_bz'] = (g[col] - mu) / sd
        g['I_Leaf_Phys'] = (g['C_ngrdi_leaf_bz'] + g['T_ngrdi_leaf_auc_bz'] + g['T_late_change_bz']) / 3.0
        g['I_Gap_Struct'] = (g['Searly_gap_large_bz'] + g['T_gap_auc_bz'] + g['Searly_prof_across_bz']) / 3.0
        g['yield_blk_anom'] = g['yield_kg_ha'] - float(g['yield_kg_ha'].mean())
        rows_b.append(g)

    df_b = pd.concat(rows_b, ignore_index=True).sort_values(['domain', 'plot_id']).reset_index(drop=True)
    for dom, g in df_b.groupby('domain'):
        zd = pd.get_dummies(g[['N_kg_ha', 'block_design']].astype(str), drop_first=True, dtype=float)
        Xd = np.column_stack([np.ones(len(g)), zd.to_numpy()])
        yd = g['yield_kg_ha'].to_numpy(float)
        df_b.loc[g.index, 'yield_within_NB'] = yd - Xd @ np.linalg.lstsq(Xd, yd, rcond=None)[0]
        for col in ['I_Leaf_Phys', 'I_Gap_Struct']:
            vd = g[col].to_numpy(float)
            df_b.loc[g.index, f'{col}_within_NB'] = vd - Xd @ np.linalg.lstsq(Xd, vd, rcond=None)[0]

    X_2d = np.column_stack([np.ones(len(df_b)), df_b['I_Leaf_Phys'].to_numpy(), df_b['I_Gap_Struct'].to_numpy()])
    y_2d = df_b['yield_blk_anom'].to_numpy(float)
    beta_2d = np.linalg.lstsq(X_2d, y_2d, rcond=None)[0]
    yhat_2d = X_2d @ beta_2d
    r2_2d = float(1.0 - np.sum((y_2d - yhat_2d) ** 2) / np.sum((y_2d - y_2d.mean()) ** 2))
    df_b['pred_yield_blk_anom_2d'] = yhat_2d
    df_b['r2_2d_plane'] = r2_2d
    df_b['beta_leaf_kg_ha'] = float(beta_2d[1])
    df_b['beta_gap_kg_ha'] = float(beta_2d[2])

    conds = [
        (df_b['I_Leaf_Phys'] >= 0.0) & (df_b['I_Gap_Struct'] <= 0.0),
        (df_b['I_Leaf_Phys'] >= 0.0) & (df_b['I_Gap_Struct'] > 0.0),
        (df_b['I_Leaf_Phys'] < 0.0) & (df_b['I_Gap_Struct'] <= 0.0),
        (df_b['I_Leaf_Phys'] < 0.0) & (df_b['I_Gap_Struct'] > 0.0),
    ]
    q_labels = ['Q1_Optimal_Intact', 'Q2_Stand_Gap_Limited', 'Q3_N_Senescence_Limited', 'Q4_Dual_Stressed']
    df_b['quadrant'] = np.select(conds, q_labels, default='Q1_Optimal_Intact')
    df_b.to_csv(Q1_DIR / 'direction_b_2d_attribution_quadrants.csv', index=False)

    # 2. Multi-scale In-Season Canopy N-Deficiency (N <= 75 vs N >= 150) Diagnosis across GSDs (3.3 mm to 3 m)
    # Evaluated via spatial leave-one-block-out cross-validation within each environment (zero target-block leakage).
    diag_rows = []
    roc_rows = []
    for gsd in GSD_LEVELS_M:
        sub_gsd = gsd_df[gsd_df.gsd_m == gsd].copy()
        sub_gsd['ngrdi_decoupled_gain'] = sub_gsd['ngrdi_leaf'] - sub_gsd['ngrdi_mixed']
        tbl = build_common4_plot_table(sub_gsd).reset_index(drop=True)
        p_peak = (
            sub_gsd[sub_gsd.DAS.isin([128, 130])]
            .groupby(['domain', 'plot_id'])['ngrdi_decoupled_gain']
            .mean()
            .reset_index()
        )
        tbl = tbl.merge(p_peak, on=['domain', 'plot_id'])
        for tag, cols in [
            ('Mixed_ExG', ['Cearly_exg', 'C_exg']),
            ('Mixed_CT', ['C_cover', 'C_exg', 'T_auc_mean', 'T_late_change']),
            ('Decoupled_Full', ['C_cover', 'ngrdi_decoupled_gain', 'Cearly_bg_glare', 'T_auc_mean', 'T_late_change', 'Searly_gap_large', 'Searly_prof_across', 'T_prof_auc']),
        ]:
            y_def = (tbl['N_kg_ha'] <= 75).astype(int).to_numpy()
            p_def = np.zeros(len(tbl))
            for dom, g in tbl.groupby('domain'):
                ix = g.index.to_numpy()
                X = g[cols].to_numpy(float)
                y = (g['N_kg_ha'] <= 75).astype(int).to_numpy()
                blks = g['block_design'].to_numpy(int)
                for b in np.unique(blks):
                    tr = blks != b
                    te = blks == b
                    sc = StandardScaler().fit(X[tr])
                    clf = LogisticRegression(C=0.5).fit(sc.transform(X[tr]), y[tr])
                    p_def[ix[te]] = clf.predict_proba(sc.transform(X[te]))[:, 1]
            auc = float(roc_auc_score(y_def, p_def))
            acc = float(accuracy_score(y_def, (p_def >= 0.5).astype(int)))
            diag_rows.append({
                'gsd_m': float(gsd),
                'gsd_label': GSD_LABELS[gsd],
                'model': tag,
                'roc_auc': auc,
                'accuracy_pct': acc * 100.0,
                'mean_ngrdi_gain': float(tbl['ngrdi_decoupled_gain'].mean()),
            })
            if gsd == 0.0033:
                fpr, tpr, _ = roc_curve(y_def, p_def)
                for fp, tp in zip(fpr, tpr):
                    roc_rows.append({'model': tag, 'fpr': float(fp), 'tpr': float(tp), 'auc': auc})
    pd.DataFrame(diag_rows).to_csv(Q1_DIR / 'direction_b_n_diagnosis_multiscale.csv', index=False)
    pd.DataFrame(roc_rows).to_csv(Q1_DIR / 'direction_b_n_diagnosis_roc_3p3mm.csv', index=False)

    # 3. Prior-Free (RGB-Only) vs Management-Coupled Yield Prediction across 9 Tasks
    # Evaluated on identical k=0 (full & holdout) and k=3 block-balanced median-vigor active calibration splits.
    alphas = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0]
    specs_rgb = {
        'RGB_Mixed_ExG': ['Cearly_exg_bz', 'C_exg_bz'],
        'RGB_Mixed_CT': ['C_cover_bz', 'C_exg_bz', 'T_auc_mean_bz', 'T_late_change_bz'],
        'RGB_Decoupled_InSeason': ['I_Leaf_Phys', 'I_Gap_Struct', 'C_cover_bz', 'T_auc_mean_bz'],
        'RGB_Decoupled_Bmorph': ['I_Leaf_Phys', 'I_Gap_Struct', 'C_cover_bz', 'T_auc_mean_bz', 'white_b140_s30_morph_frac_bz'],
    }
    sets = {k: d.reset_index(drop=True) for k, d in df_b.groupby('domain')}
    pred_rows = []
    sum_rows = []

    for sp, cols in specs_rgb.items():
        for src, d in sets.items():
            ii = np.arange(len(d))
            g = d.block_design.to_numpy(int)
            x = d[cols].to_numpy(float)
            y = d.yield_kg_ha.to_numpy(float)

            # BlockCV
            preds_cv = np.empty(len(d))
            for b in np.unique(g):
                tr = ii[g != b]
                te = ii[g == b]
                best_a, best_sc = 10.0, 1e18
                for a in alphas:
                    oof = np.empty(len(tr))
                    for b2 in np.unique(g[tr]):
                        m2 = g[tr] != b2
                        oof[~m2] = Ridge(alpha=a).fit(x[tr[m2]], y[tr[m2]]).predict(x[tr[~m2]])
                    rm = float(np.sqrt(np.mean((oof - y[tr]) ** 2)))
                    if rm < best_sc:
                        best_sc, best_a = rm, a
                preds_cv[te] = Ridge(alpha=best_a).fit(x[tr], y[tr]).predict(x[te])
            e_cv = preds_cv - y
            en_cv = pd.Series(e_cv).groupby(d.N_kg_ha).transform('mean').to_numpy()
            for r_i, p_i in zip(d.itertuples(), preds_cv):
                pred_rows.append({
                    'specification': sp, 'source': src, 'target': src, 'evaluation': 'BlockCV',
                    'plot_id': int(r_i.plot_id), 'block': int(r_i.block_design), 'N': float(r_i.N_kg_ha),
                    'quadrant': str(r_i.quadrant), 'observed': float(r_i.yield_kg_ha),
                    'pred_k0': float(p_i), 'pred_k3': float(p_i), 'is_cal_k3': 0,
                })
            sum_rows.append({
                'specification': sp, 'source': src, 'target': src, 'evaluation': 'BlockCV',
                'RMSE_k0_all': float(np.sqrt(np.mean(e_cv ** 2))),
                'R2_k0_all': float(r2_score(y, preds_cv)),
                'RMSE_k0': float(np.sqrt(np.mean(e_cv ** 2))), 'cRMSE_k0': float(np.std(e_cv)),
                'WithinN_RMSE': float(np.sqrt(np.mean((e_cv - en_cv) ** 2))),
                'R2_k0': float(r2_score(y, preds_cv)), 'R2_cent': float(r2_score(y, preds_cv - e_cv.mean())),
                'RMSE_k3': float(np.sqrt(np.mean(e_cv ** 2))), 'R2_k3': float(r2_score(y, preds_cv)),
            })

            # Cross-environment transfer (adjacent_field & cross_year)
            best_a, best_sc = 10.0, 1e18
            for a in alphas:
                oof = np.empty(len(ii))
                for b2 in np.unique(g):
                    m2 = g != b2
                    oof[~m2] = Ridge(alpha=a).fit(x[m2], y[m2]).predict(x[~m2])
                rm = float(np.sqrt(np.mean((oof - y) ** 2)))
                if rm < best_sc:
                    best_sc, best_a = rm, a
            m_full = Ridge(alpha=best_a).fit(x, y)
            for tar, td in sets.items():
                if tar == src:
                    continue
                ev = 'adjacent_field' if src[:4] == tar[:4] else 'cross_year'
                xt = td[cols].to_numpy(float)
                yt = td.yield_kg_ha.to_numpy(float)
                pt = m_full.predict(xt)

                # k=3 block-balanced median-vigor active calibration plots
                cal_pids = []
                for _, gb in td.groupby('block_design'):
                    v_diff = gb['T_auc_mean_bz'].abs()
                    cal_pids.append(int(gb.loc[v_diff.idxmin(), 'plot_id']))
                cal_mask = td['plot_id'].isin(cal_pids).to_numpy()
                shift = float(np.mean(yt[cal_mask] - pt[cal_mask]))
                pt_cal = pt + shift

                et_all = pt - yt
                ent_all = pd.Series(et_all).groupby(td.N_kg_ha).transform('mean').to_numpy()
                et0_ho = pt[~cal_mask] - yt[~cal_mask]
                et3_ho = pt_cal[~cal_mask] - yt[~cal_mask]

                for r_i, p0_i, p3_i, cm_i in zip(td.itertuples(), pt, pt_cal, cal_mask):
                    pred_rows.append({
                        'specification': sp, 'source': src, 'target': tar, 'evaluation': ev,
                        'plot_id': int(r_i.plot_id), 'block': int(r_i.block_design), 'N': float(r_i.N_kg_ha),
                        'quadrant': str(r_i.quadrant), 'observed': float(r_i.yield_kg_ha),
                        'pred_k0': float(p0_i), 'pred_k3': float(p3_i), 'is_cal_k3': int(cm_i),
                    })
                sum_rows.append({
                    'specification': sp, 'source': src, 'target': tar, 'evaluation': ev,
                    'RMSE_k0_all': float(np.sqrt(np.mean(et_all ** 2))),
                    'R2_k0_all': float(r2_score(yt, pt)),
                    'RMSE_k0': float(np.sqrt(np.mean(et0_ho ** 2))), 'cRMSE_k0': float(np.std(et_all)),
                    'WithinN_RMSE': float(np.sqrt(np.mean((et_all - ent_all) ** 2))),
                    'R2_k0': float(r2_score(yt[~cal_mask], pt[~cal_mask])),
                    'R2_cent': float(r2_score(yt, pt - et_all.mean())),
                    'RMSE_k3': float(np.sqrt(np.mean(et3_ho ** 2))),
                    'R2_k3': float(r2_score(yt[~cal_mask], pt_cal[~cal_mask])),
                })

    # Also evaluate Management-Coupled baselines (Agron_N_Only = 'N', Integrated_N_Decoupled = 'N_C_T_Bmorph',
    # and Integrated_N_Leaf_Bmorph = 'N_Leaf_T_Bmorph') on the exact same 9 tasks and identical k=3 block-balanced holdout plots.
    ms_3p3 = pd.read_csv(Q1_DIR / 'multiscale_predictions.csv').query("prefix == '3.3 mm'")
    # Compute N_Leaf_T_Bmorph on 3.3 mm using identical source-nested PartialRidge protocol
    sys.path.insert(0, str(ROOT / '02_Code/src/modeling'))
    import mine_raw_rgb_information as legacy_pr
    plot64_mm = pd.read_csv(Q1_DIR / 'plot64_millimeter_features.csv')
    ai_tune_mm = pd.read_csv(OLD_MINING / 'ai_reference/AI_source_tuning.csv')
    frames_mm = {dom: d.reset_index(drop=True) for dom, d in plot64_mm.groupby('domain')}
    n_leaf_bm_rows = []
    for src_m in ['2023_F1', '2024_F1', '2024_F2']:
        for tar_m in ['2023_F1', '2024_F1', '2024_F2']:
            ev_m = 'BlockCV' if src_m == tar_m else ('adjacent_field' if src_m[:4] == tar_m[:4] else 'cross_year')
            a_m, b_m = frames_mm[src_m], frames_mm[tar_m]
            ya_m = a_m.yield_kg_ha.to_numpy(float)
            ga_m = a_m.block_design.to_numpy(int)
            ii_m = np.arange(len(a_m))
            folds_m = np.unique(ga_m) if src_m == tar_m else [0]
            for fold_m in folds_m:
                tr_m = ii_m[ga_m != fold_m] if fold_m else ii_m
                te_m = np.flatnonzero(b_m.block_design.to_numpy(int) == fold_m) if fold_m else np.arange(len(b_m))
                rk_m = f'BlockCV_hold_{fold_m}' if fold_m else f'{ev_m}_hold_0'
                w_rule_m = ai_tune_mm.query("source==@src_m and role==@rk_m and specification=='N_C_T_B'").mask_rule.iloc[0]
                cols_m = ['C_cover', 'C_ngrdi_leaf', 'T_auc_mean', 'T_late_change', f'{w_rule_m}_morph_frac', f'{w_rule_m}_cluster_m2']
                na_m, nb_m = a_m.N_kg_ha.to_numpy(float), b_m.N_kg_ha.to_numpy(float)
                xa_m = np.c_[na_m, na_m * na_m, a_m[cols_m].to_numpy(float)]
                xb_m = np.c_[nb_m, nb_m * nb_m, b_m[cols_m].to_numpy(float)]
                sc_m = []
                for alpha_m in legacy_pr.ALPHAS:
                    oof_m = np.empty(len(tr_m))
                    for blk_m in np.unique(ga_m[tr_m]):
                        kp_m = ga_m[tr_m] != blk_m
                        oof_m[~kp_m] = legacy_pr.PartialRidge(alpha_m).fit(xa_m[tr_m[kp_m]], ya_m[tr_m[kp_m]]).predict(xa_m[tr_m[~kp_m]])
                    err_m = oof_m - ya_m[tr_m]
                    sc_m.append((float(np.sqrt(np.mean(err_m ** 2))), float(np.mean(np.abs(err_m))), -alpha_m, alpha_m))
                _, _, _, best_a_m = min(sc_m)
                yh_m = legacy_pr.PartialRidge(best_a_m).fit(xa_m[tr_m], ya_m[tr_m]).predict(xb_m[te_m])
                for r_te_m, p_val_m in zip(b_m.iloc[te_m].itertuples(), yh_m):
                    n_leaf_bm_rows.append({
                        'prefix': '3.3 mm', 'source': src_m, 'target': tar_m, 'evaluation': ev_m,
                        'specification': 'N_Leaf_T_Bmorph', 'plot_id': int(r_te_m.plot_id),
                        'block': int(r_te_m.block_design), 'N': float(r_te_m.N_kg_ha),
                        'observed': float(r_te_m.yield_kg_ha), 'predicted': float(p_val_m),
                    })
    ms_3p3 = pd.concat([ms_3p3, pd.DataFrame(n_leaf_bm_rows)], ignore_index=True)

    for sp_out, sp_ms in [
        ('Agron_N_Only', 'N'),
        ('Integrated_N_Decoupled', 'N_C_T_Bmorph'),
        ('Integrated_N_Leaf_Bmorph', 'N_Leaf_T_Bmorph'),
    ]:
        sub_ms = ms_3p3[ms_3p3.specification == sp_ms]
        for (src, tar, ev), grp in sub_ms.groupby(['source', 'target', 'evaluation']):
            td = sets[tar]
            g_m = grp.sort_values('plot_id').reset_index(drop=True)
            td_s = td.sort_values('plot_id').reset_index(drop=True)
            yt = g_m['observed'].to_numpy(float)
            pt = g_m['predicted'].to_numpy(float)
            if ev == 'BlockCV':
                e_cv = pt - yt
                en_cv = pd.Series(e_cv).groupby(g_m['N']).transform('mean').to_numpy()
                for r_i, p_i in zip(td_s.itertuples(), pt):
                    pred_rows.append({
                        'specification': sp_out, 'source': src, 'target': tar, 'evaluation': 'BlockCV',
                        'plot_id': int(r_i.plot_id), 'block': int(r_i.block_design), 'N': float(r_i.N_kg_ha),
                        'quadrant': str(r_i.quadrant), 'observed': float(r_i.yield_kg_ha),
                        'pred_k0': float(p_i), 'pred_k3': float(p_i), 'is_cal_k3': 0,
                    })
                sum_rows.append({
                    'specification': sp_out, 'source': src, 'target': tar, 'evaluation': 'BlockCV',
                    'RMSE_k0_all': float(np.sqrt(np.mean(e_cv ** 2))),
                    'R2_k0_all': float(r2_score(yt, pt)),
                    'RMSE_k0': float(np.sqrt(np.mean(e_cv ** 2))), 'cRMSE_k0': float(np.std(e_cv)),
                    'WithinN_RMSE': float(np.sqrt(np.mean((e_cv - en_cv) ** 2))),
                    'R2_k0': float(r2_score(yt, pt)), 'R2_cent': float(r2_score(yt, pt - e_cv.mean())),
                    'RMSE_k3': float(np.sqrt(np.mean(e_cv ** 2))), 'R2_k3': float(r2_score(yt, pt)),
                })
            else:
                cal_pids = [int(gb.loc[gb['T_auc_mean_bz'].abs().idxmin(), 'plot_id']) for _, gb in td_s.groupby('block_design')]
                cal_mask = g_m['plot_id'].isin(cal_pids).to_numpy()
                shift = float(np.mean(yt[cal_mask] - pt[cal_mask]))
                pt_cal = pt + shift
                et_all = pt - yt
                ent_all = pd.Series(et_all).groupby(g_m['N']).transform('mean').to_numpy()
                et0_ho = pt[~cal_mask] - yt[~cal_mask]
                et3_ho = pt_cal[~cal_mask] - yt[~cal_mask]
                for r_i, p0_i, p3_i, cm_i in zip(td_s.itertuples(), pt, pt_cal, cal_mask):
                    pred_rows.append({
                        'specification': sp_out, 'source': src, 'target': tar, 'evaluation': ev,
                        'plot_id': int(r_i.plot_id), 'block': int(r_i.block_design), 'N': float(r_i.N_kg_ha),
                        'quadrant': str(r_i.quadrant), 'observed': float(r_i.yield_kg_ha),
                        'pred_k0': float(p0_i), 'pred_k3': float(p3_i), 'is_cal_k3': int(cm_i),
                    })
                sum_rows.append({
                    'specification': sp_out, 'source': src, 'target': tar, 'evaluation': ev,
                    'RMSE_k0_all': float(np.sqrt(np.mean(et_all ** 2))),
                    'R2_k0_all': float(r2_score(yt, pt)),
                    'RMSE_k0': float(np.sqrt(np.mean(et0_ho ** 2))), 'cRMSE_k0': float(np.std(et_all)),
                    'WithinN_RMSE': float(np.sqrt(np.mean((et_all - ent_all) ** 2))),
                    'R2_k0': float(r2_score(yt[~cal_mask], pt[~cal_mask])),
                    'R2_cent': float(r2_score(yt, pt - et_all.mean())),
                    'RMSE_k3': float(np.sqrt(np.mean(et3_ho ** 2))),
                    'R2_k3': float(r2_score(yt[~cal_mask], pt_cal[~cal_mask])),
                })

    pd.DataFrame(pred_rows).to_csv(Q1_DIR / 'direction_b_prior_free_predictions.csv', index=False)
    pd.DataFrame(sum_rows).to_csv(Q1_DIR / 'direction_b_prior_free_summary.csv', index=False)


def main() -> None:
    df_slots = pd.read_csv(INPUT_SLOTS)
    gdf = gpd.read_file(GPKG, layer='inward_buffer_plot')
    npz_dict: dict[str, np.ndarray] = {}

    build_fig2_assets(df_slots, gdf, npz_dict)
    build_fig3_assets(df_slots, gdf)
    build_fig4_assets(df_slots, gdf, npz_dict)
    build_fig5_and_fig6_assets(gdf)
    build_direction_b_assets()

    np.savez_compressed(Q1_DIR / 'rich_figure_assets.npz', **npz_dict)
    print('Successfully generated rich_figure_assets.npz and all display CSVs for Figures 2-6 (including Direction B).')


if __name__ == '__main__':
    main()
