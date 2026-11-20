"""Module B: Millimeter-scale spatial phenotyping, pure-leaf decoupling & multi-scale GSD scaling law.

Strictly read-only on 01_Data/raw and all frozen v08 CSVs.
Writes additive outputs exclusively to FORMAL / 'analysis/q1_spatial_calibration'.
Follows identical nested leave-source-block-out PartialRidge protocol (zero target leakage).
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import sys
import time
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '02_Code/src/data'))
sys.path.insert(0, str(ROOT / '02_Code/src/modeling'))

from extract_raw_rgb_information import crop, arrays, rules, sha
import mine_raw_rgb_information as legacy

FORMAL_NEW = ROOT / '10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108'
FORMAL_OLD = FORMAL_NEW / 'analysis/frozen_baseline'
OLD_MINING = FORMAL_OLD / 'analysis/raw_rgb_mining'
OUT = FORMAL_NEW / 'analysis/q1_spatial_calibration'
INPUT_SLOTS = ROOT / '01_Data/processed/model_ready/cotton_rgb_formal_714_slots.csv'
GPKG = ROOT / '01_Data/processed/spatial/cotton_rgb/field_authority.gpkg'

warnings.filterwarnings('ignore', category=rasterio.errors.NodataShadowWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

BOLL_RULES = ('white_b120_s20', 'white_b120_s30', 'white_b140_s20', 'white_b140_s30', 'white_b160_s20')

COMMON_DATES = {
    (2023, 'F1'): [92, 107, 130, 142, 181],
    (2024, 'F1'): [92, 106, 128, 141, 181],
    (2024, 'F2'): [92, 106, 128, 141, 181],
}
GSD_LEVELS_M = [0.0033, 0.01, 0.05, 0.20, 1.00, 3.00]
GSD_LABELS = {
    0.0033: '3.3 mm',
    0.01: '1 cm',
    0.05: '5 cm',
    0.20: '20 cm',
    1.00: '1 m',
    3.00: '3 m',
}


def compute_spatial_metrics(rgb: np.ndarray, valid: np.ndarray, px_size_m: float, include_bolls: bool = False) -> dict:
    """Compute background-decoupled leaf physiology, canopy gap micro-structure, and boll-cluster morphology."""
    n_val = int(valid.sum())
    if n_val == 0:
        raise ValueError('Empty valid mask')

    r, g, b, sat, br = arrays(rgb)
    ms = rules(rgb, valid)
    green = ms['green_canopeo']
    n_grn = int(green.sum())
    px_area_m2 = px_size_m * px_size_m
    roi_valid_m2 = n_val * px_area_m2

    tot = r + g + b
    exg = np.divide(2.0 * g - r - b, tot, out=np.zeros_like(g), where=tot > 0)
    ngrdi = np.divide(g - r, g + r, out=np.zeros_like(g), where=(g + r) > 0)
    denom_vari = g + r - b
    vari = np.divide(g - r, denom_vari, out=np.zeros_like(g), where=np.abs(denom_vari) > 8.0)
    vari = np.clip(vari, -1.0, 1.0)

    f_cover = float(n_grn / n_val)
    exg_mixed = float(exg[valid].mean())
    ngrdi_mixed = float(ngrdi[valid].mean())

    if n_grn >= 5:
        exg_leaf = float(exg[green].mean())
        ngrdi_leaf = float(ngrdi[green].mean())
        ngrdi_leaf_sd = float(np.std(ngrdi[green]))
        vari_leaf = float(vari[green].mean())
    else:
        # Fallback if coarsened resolution wipes out Canopeo detections
        exg_leaf = exg_mixed
        ngrdi_leaf = ngrdi_mixed
        ngrdi_leaf_sd = float(np.std(ngrdi[valid]))
        vari_leaf = float(vari[valid].mean())

    # Specular plastic-mulch glare fraction in non-canopy background
    bg_glare = valid & (~green) & (br >= 150.0) & (sat <= 0.22)
    bg_glare_frac = float(bg_glare.sum() / n_val)
    n_non_grn = int((valid & (~green)).sum())
    bg_glare_within_gap = float(bg_glare.sum() / n_non_grn) if n_non_grn > 0 else 0.0

    # Contiguous non-canopy large gaps (>= 0.04 m2 ~ 20 cm x 20 cm missing/stunted patch)
    gap = valid & (~green)
    if px_area_m2 <= 0.04:
        lbl, n_cc = ndimage.label(gap)
        if n_cc > 0:
            sizes_m2 = np.bincount(lbl.ravel())[1:] * px_area_m2
            gap_large_frac = float(sizes_m2[sizes_m2 >= 0.04].sum() / roi_valid_m2)
            gap_p90_m2 = float(np.percentile(sizes_m2, 90))
        else:
            gap_large_frac = 0.0
            gap_p90_m2 = 0.0
    else:
        # When single pixel exceeds 0.04 m2 (e.g. >= 20 cm GSD), every non-green pixel is >= 0.04 m2
        gap_large_frac = float(gap.sum() / n_val)
        gap_p90_m2 = float(gap.sum() * px_area_m2) if gap.any() else 0.0

    # Sub-plot 0.25 m meso-scale heterogeneity & lacunarity
    cell_px = max(1, int(round(0.25 / px_size_m)))
    ny = max(1, valid.shape[0] // cell_px)
    nx = max(1, valid.shape[1] // cell_px)
    cell_vals = []
    cell_gaps = []
    if ny >= 2 and nx >= 2:
        v_crop = valid[: ny * cell_px, : nx * cell_px].reshape(ny, cell_px, nx, cell_px).swapaxes(1, 2)
        g_crop = green[: ny * cell_px, : nx * cell_px].reshape(ny, cell_px, nx, cell_px).swapaxes(1, 2)
        v_cnt = v_crop.sum(axis=(2, 3))
        g_cnt = g_crop.sum(axis=(2, 3))
        keep_cells = v_cnt >= max(1, int(0.4 * cell_px * cell_px))
        if keep_cells.sum() >= 3:
            c_frac = g_cnt[keep_cells] / v_cnt[keep_cells]
            gap_frac_cells = 1.0 - c_frac
            cover_sd_25cm = float(np.std(c_frac))
            mu_g = float(np.mean(gap_frac_cells))
            lacunarity_25cm = float(1.0 + np.var(gap_frac_cells) / (mu_g * mu_g + 1e-6))
        else:
            cover_sd_25cm = 0.0
            lacunarity_25cm = 1.0
    else:
        cover_sd_25cm = 0.0
        lacunarity_25cm = 1.0

    # Orientation-invariant cross-row & along-row canopy profile SD
    min_line = max(1, int(0.25 * min(valid.shape)))
    col_v = valid.sum(axis=0)
    row_v = valid.sum(axis=1)
    col_cov = green.sum(axis=0)[col_v >= min_line] / col_v[col_v >= min_line] if (col_v >= min_line).sum() >= 3 else np.array([f_cover])
    row_cov = green.sum(axis=1)[row_v >= min_line] / row_v[row_v >= min_line] if (row_v >= min_line).sum() >= 3 else np.array([f_cover])
    sd_a, sd_b = float(np.std(col_cov)), float(np.std(row_cov))
    profile_sd_across = max(sd_a, sd_b)
    profile_sd_along = min(sd_a, sd_b)

    res = {
        'f_cover': f_cover,
        'exg_mixed': exg_mixed,
        'ngrdi_mixed': ngrdi_mixed,
        'exg_leaf': exg_leaf,
        'ngrdi_leaf': ngrdi_leaf,
        'ngrdi_leaf_sd': ngrdi_leaf_sd,
        'vari_leaf': vari_leaf,
        'bg_glare_frac': bg_glare_frac,
        'bg_glare_within_gap': bg_glare_within_gap,
        'gap_large_frac': gap_large_frac,
        'gap_p90_m2': gap_p90_m2,
        'cover_sd_25cm': cover_sd_25cm,
        'lacunarity_25cm': lacunarity_25cm,
        'profile_sd_across': profile_sd_across,
        'profile_sd_along': profile_sd_along,
    }

    if include_bolls:
        px_cm2 = px_area_m2 * 1e4
        for w_rule in BOLL_RULES:
            w_mask = ms[w_rule]
            res[f'{w_rule}_raw'] = float(w_mask.sum() / n_val)
            b_lbl, b_n = ndimage.label(w_mask)
            if b_n > 0:
                b_cnt = np.bincount(b_lbl.ravel())[1:]
                b_cm2 = b_cnt * px_cm2
                keep_boll = (b_cm2 >= 0.5) & (b_cm2 <= 120.0)
                glare_mulch = b_cm2 > 120.0
                res[f'{w_rule}_morph_frac'] = float(b_cnt[keep_boll].sum() / n_val)
                res[f'{w_rule}_cluster_m2'] = float(keep_boll.sum() / roi_valid_m2)
                res[f'{w_rule}_glare_frac'] = float(b_cnt[glare_mulch].sum() / n_val)
            else:
                res[f'{w_rule}_morph_frac'] = 0.0
                res[f'{w_rule}_cluster_m2'] = 0.0
                res[f'{w_rule}_glare_frac'] = 0.0

    return res


def coarsen_rgb_valid(rgb: np.ndarray, valid: np.ndarray, native_px_m: float, target_gsd_m: float):
    """Physical area-weighted block averaging of RGB reflectance to coarser ground sampling distance."""
    factor = int(round(target_gsd_m / native_px_m))
    if factor <= 1:
        return rgb, valid, native_px_m
    H, W = valid.shape
    if factor >= 50:
        # Exact grid stride tiling so 0.2 m, 1 m, and 3 m grids preserve multi-pixel strips even on narrow plots
        r_starts = list(range(0, H, factor))
        c_starts = list(range(0, W, factor))
        nh, nw = len(r_starts), len(c_starts)
        out_rgb = np.zeros((nh, nw, 3), dtype=np.uint8)
        out_valid = np.zeros((nh, nw), dtype=bool)
        for i, r0 in enumerate(r_starts):
            r1 = min(H, r0 + factor)
            for j, c0 in enumerate(c_starts):
                c1 = min(W, c0 + factor)
                vb = valid[r0:r1, c0:c1]
                vc = int(vb.sum())
                area_blk = (r1 - r0) * (c1 - c0)
                if vc >= max(1, int(0.35 * area_blk)):
                    out_valid[i, j] = True
                    out_rgb[i, j] = np.clip(np.round(rgb[r0:r1, c0:c1][vb].mean(axis=0)), 0, 255).astype(np.uint8)
        if not out_valid.any():
            mean_rgb = rgb[valid].mean(axis=0).round().astype(np.uint8).reshape(1, 1, 3)
            return mean_rgb, np.array([[True]]), target_gsd_m
        return out_rgb, out_valid, native_px_m * factor

    nh, nw = H // factor, W // factor
    if nh == 0 or nw == 0:
        mean_rgb = rgb[valid].mean(axis=0).round().astype(np.uint8).reshape(1, 1, 3)
        return mean_rgb, np.array([[True]]), target_gsd_m

    v_blk = valid[: nh * factor, : nw * factor].reshape(nh, factor, nw, factor).swapaxes(1, 2)
    v_cnt = v_blk.sum(axis=(2, 3))
    out_valid = v_cnt >= max(1, int(0.5 * factor * factor))
    if not out_valid.any():
        mean_rgb = rgb[valid].mean(axis=0).round().astype(np.uint8).reshape(1, 1, 3)
        return mean_rgb, np.array([[True]]), target_gsd_m

    rgb_f = rgb[: nh * factor, : nw * factor].astype(np.float32)
    rgb_f[~valid[: nh * factor, : nw * factor]] = 0.0
    rgb_blk = rgb_f.reshape(nh, factor, nw, factor, 3).swapaxes(1, 2).sum(axis=(2, 3))
    out_rgb = np.zeros((nh, nw, 3), dtype=np.uint8)
    for c in range(3):
        ch = np.divide(rgb_blk[:, :, c], v_cnt, out=np.zeros((nh, nw), dtype=np.float32), where=out_valid)
        out_rgb[:, :, c] = np.clip(np.round(ch), 0, 255).astype(np.uint8)
    return out_rgb, out_valid, native_px_m * factor


def extract_all_slots_and_multiscale():
    f_nat = OUT / 'millimeter_slot_features_699.csv'
    f_gsd = OUT / 'multiscale_gsd_slot_features.csv'
    if f_nat.is_file() and f_gsd.is_file():
        df_native = pd.read_csv(f_nat)
        df_gsd = pd.read_csv(f_gsd)
        missing_rules = [r for r in BOLL_RULES if f'{r}_raw' not in df_native.columns]
        if not missing_rules:
            return df_native, df_gsd
        # Backfill missing boll rules strictly on the 3 DAS==181 orthomosaics
        df_slots = pd.read_csv(INPUT_SLOTS)
        gdf = gpd.read_file(GPKG, layer='inward_buffer_plot')
        common_plots = pd.read_csv(OLD_MINING / 'common_plot_features_common_DAS.csv')
        common_keys = set(zip(common_plots.domain, common_plots.plot_id.astype(int)))
        direct_181 = df_slots[(df_slots.slot_status == 'AVAILABLE') & (df_slots.DAS == 181)].copy()
        patch_nat, patch_gsd = [], []
        for rpath, sub in direct_181.groupby('raster_path'):
            yr, fld, das = int(sub.year.iloc[0]), str(sub.field.iloc[0]), int(sub.DAS.iloc[0])
            dom = f'{yr}_{fld}'
            with rasterio.open(rpath) as src:
                gg = gdf.to_crs(src.crs)
                px_m = float(np.sqrt(abs(src.res[0] * 85000.0 * src.res[1] * 111000.0))) if (src.crs and src.crs.is_geographic) else float(np.sqrt(abs(src.res[0] * src.res[1])))
                for _, row in sub.iterrows():
                    pid = int(row.plot_id)
                    geom = gg[(gg.year == yr) & (gg.field == fld) & (gg.plot_id == pid)].geometry.iloc[0]
                    rgb, valid = crop(src, geom)
                    m_nat = compute_spatial_metrics(rgb, valid, px_m, include_bolls=True)
                    keep_cols = {k: m_nat[k] for k in m_nat if any(k.startswith(r) for r in missing_rules)}
                    patch_nat.append({'domain': dom, 'plot_id': pid, 'DAS': 181, **keep_cols})
                    if (dom, pid) in common_keys:
                        for gsd in GSD_LEVELS_M:
                            if gsd == 0.0033:
                                patch_gsd.append({'gsd_m': gsd, 'domain': dom, 'plot_id': pid, 'DAS': 181, **keep_cols})
                            else:
                                c_rgb, c_val, eff_px = coarsen_rgb_valid(rgb, valid, px_m, gsd)
                                m_c = compute_spatial_metrics(c_rgb, c_val, eff_px, include_bolls=True)
                                keep_c = {k: m_c[k] for k in m_c if any(k.startswith(r) for r in missing_rules)}
                                patch_gsd.append({'gsd_m': gsd, 'domain': dom, 'plot_id': pid, 'DAS': 181, **keep_c})
        df_native = df_native.merge(pd.DataFrame(patch_nat), on=['domain', 'plot_id', 'DAS'], how='left')
        df_gsd = df_gsd.merge(pd.DataFrame(patch_gsd), on=['gsd_m', 'domain', 'plot_id', 'DAS'], how='left')
        df_native.to_csv(f_nat, index=False)
        df_gsd.to_csv(f_gsd, index=False)
        return df_native, df_gsd

    t0 = time.time()
    df_slots = pd.read_csv(INPUT_SLOTS)
    gdf = gpd.read_file(GPKG, layer='inward_buffer_plot')
    common_plots = pd.read_csv(OLD_MINING / 'common_plot_features_common_DAS.csv')
    common_keys = set(zip(common_plots.domain, common_plots.plot_id.astype(int)))

    direct = df_slots[df_slots.slot_status == 'AVAILABLE'].copy()
    slot_rows = []
    gsd_rows = []

    for j, (rpath, sub) in enumerate(direct.groupby('raster_path'), 1):
        yr = int(sub.year.iloc[0])
        fld = str(sub.field.iloc[0])
        das = int(sub.DAS.iloc[0])
        dom = f'{yr}_{fld}'
        is_common_date = das in COMMON_DATES[(yr, fld)]

        with rasterio.open(rpath) as src:
            gg = gdf.to_crs(src.crs)
            # Pixel size in meters (handle EPSG:4326 vs UTM)
            if src.crs and src.crs.is_geographic:
                px_m = float(np.sqrt(abs(src.res[0] * 85000.0 * src.res[1] * 111000.0)))
            else:
                px_m = float(np.sqrt(abs(src.res[0] * src.res[1])))

            for _, row in sub.iterrows():
                pid = int(row.plot_id)
                geom = gg[(gg.year == yr) & (gg.field == fld) & (gg.plot_id == pid)].geometry.iloc[0]
                rgb, valid = crop(src, geom)
                is_181 = (das == 181)

                base_meta = {
                    'domain': dom, 'year': yr, 'field': fld, 'plot_id': pid,
                    'block_design': int(row.block_design), 'N_kg_ha': float(row.N_kg_ha),
                    'DAS': das, 'flight_date': str(row.flight_date),
                    'yield_kg_ha': float(row.yield_kg_ha), 'native_px_m': px_m,
                }
                m_native = compute_spatial_metrics(rgb, valid, px_m, include_bolls=is_181)
                slot_rows.append({**base_meta, **m_native})

                if is_common_date and (dom, pid) in common_keys:
                    for gsd in GSD_LEVELS_M:
                        if gsd == 0.0033:
                            gsd_rows.append({**base_meta, 'gsd_m': gsd, 'gsd_label': GSD_LABELS[gsd], **m_native})
                        else:
                            c_rgb, c_val, eff_px = coarsen_rgb_valid(rgb, valid, px_m, gsd)
                            m_c = compute_spatial_metrics(c_rgb, c_val, eff_px, include_bolls=is_181)
                            gsd_rows.append({**base_meta, 'gsd_m': gsd, 'gsd_label': GSD_LABELS[gsd], **m_c})

        print(f'[{j:02d}/33] {Path(rpath).name} (DAS={das}, px={px_m*1000:.2f} mm) elapsed={time.time()-t0:.1f}s', flush=True)

    df_native = pd.DataFrame(slot_rows).sort_values(['domain', 'plot_id', 'DAS'])
    df_gsd = pd.DataFrame(gsd_rows).sort_values(['gsd_m', 'domain', 'plot_id', 'DAS'])
    assert len(df_native) == 699
    assert len(df_gsd) == 64 * 5 * len(GSD_LEVELS_M)

    df_native.to_csv(OUT / 'millimeter_slot_features_699.csv', index=False)
    df_gsd.to_csv(OUT / 'multiscale_gsd_slot_features.csv', index=False)
    return df_native, df_gsd


def build_common4_plot_table(df_slots: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 4 common in-season dates + 181 DAS into 64-plot feature table at a given GSD."""
    common_base = pd.read_csv(OLD_MINING / 'common_plot_features_common_DAS.csv')[
        ['domain', 'year', 'field', 'plot_id', 'block_design', 'N_kg_ha', 'yield_kg_ha']
    ]
    domain_rules = pd.read_csv(OUT / 'source_boll_domain_rules.csv')
    dom_rules = dict(zip(domain_rules.calibration_source, domain_rules.mask_rule))
    assert set(dom_rules) == {'2023_F1', '2024_F1', '2024_F2'}

    rows = []
    for r in common_base.itertuples():
        yr, fld, pid = int(r.year), str(r.field), int(r.plot_id)
        d = df_slots[(df_slots.domain == r.domain) & (df_slots.plot_id == pid)].sort_values('DAS')
        d_in = d[d.DAS < 180].sort_values('DAS')
        assert len(d_in) == 4, (r.domain, pid, len(d_in))
        xs = d_in.DAS.to_numpy(float)
        early_das = 130 if yr == 2023 else 128
        late_das = 142 if yr == 2023 else 141
        e = d_in[d_in.DAS == early_das].iloc[0]
        l = d_in[d_in.DAS == late_das].iloc[0]
        s6 = d[d.DAS == 181].iloc[0]
        span = float(late_das - 92.0)

        cov_y = d_in.f_cover.to_numpy(float)
        ngrdi_y = d_in.ngrdi_leaf.to_numpy(float)
        glare_y = d_in.bg_glare_frac.to_numpy(float)
        gap_y = d_in.gap_large_frac.to_numpy(float)
        prof_y = d_in.profile_sd_across.to_numpy(float)

        rec = {
            'domain': r.domain, 'year': yr, 'field': fld, 'plot_id': pid,
            'block_design': int(r.block_design), 'N_kg_ha': float(r.N_kg_ha), 'yield_kg_ha': float(r.yield_kg_ha),
            # Legacy whole-plot baselines (exact replay check)
            'Cearly_cover': float(e.f_cover), 'Cearly_exg': float(e.exg_mixed),
            'C_cover': float(l.f_cover), 'C_exg': float(l.exg_mixed),
            'T_auc_mean': float(np.trapezoid(cov_y, xs) / span),
            'T_late_change': float((l.f_cover - e.f_cover) / (late_das - early_das) * span),
            # Decoupled pure-leaf physiology & background mulch glare
            'Cearly_ngrdi_leaf': float(e.ngrdi_leaf), 'Cearly_vari_leaf': float(e.vari_leaf),
            'C_ngrdi_leaf': float(l.ngrdi_leaf), 'C_vari_leaf': float(l.vari_leaf),
            'T_ngrdi_leaf_auc': float(np.trapezoid(ngrdi_y, xs) / span),
            'Cearly_bg_glare': float(e.bg_glare_frac), 'T_bg_glare_auc': float(np.trapezoid(glare_y, xs) / span),
            # Millimeter sub-plot micro-structure (gap architecture & row closure)
            'Searly_gap_large': float(e.gap_large_frac), 'Searly_prof_across': float(e.profile_sd_across),
            'Searly_cover_sd25': float(e.cover_sd_25cm),
            'S_gap_large': float(l.gap_large_frac), 'S_prof_across': float(l.profile_sd_across),
            'S_prof_along': float(l.profile_sd_along), 'S_cover_sd25': float(l.cover_sd_25cm),
            'T_gap_auc': float(np.trapezoid(gap_y, xs) / span),
            'T_prof_auc': float(np.trapezoid(prof_y, xs) / span),
            # Boll morphology at 181 DAS (store all 4 candidate source rules so transfer uses source's rule strictly!)
        }
        for w_rule in BOLL_RULES:
            rec[f'{w_rule}_raw'] = float(s6[f'{w_rule}_raw'])
            rec[f'{w_rule}_morph_frac'] = float(s6[f'{w_rule}_morph_frac'])
            rec[f'{w_rule}_cluster_m2'] = float(s6[f'{w_rule}_cluster_m2'])
            rec[f'{w_rule}_glare_frac'] = float(s6[f'{w_rule}_glare_frac'])
        rec['B_dom_raw'] = float(s6[f"{dom_rules[r.domain]}_raw"])
        rec['B_dom_morph'] = float(s6[f"{dom_rules[r.domain]}_morph_frac"])
        rec['B_dom_cluster_m2'] = float(s6[f"{dom_rules[r.domain]}_cluster_m2"])
        rows.append(rec)

    return pd.DataFrame(rows)


def compute_confounding_associations(plot64: pd.DataFrame) -> pd.DataFrame:
    """Compare apparent pooled/unadjusted correlation vs N+block adjusted within-treatment signal."""
    features = [
        ('C_cover', 'Canopy cover (141/142 DAS)', 'Whole-plot scalar'),
        ('C_exg', 'Mixed-pixel ExG (141/142 DAS)', 'Whole-plot scalar'),
        ('T_auc_mean', 'Canopy cover integral (92–142 DAS)', 'Whole-plot scalar'),
        ('Cearly_ngrdi_leaf', 'Pure-leaf NGRDI (128/130 DAS)', 'Decoupled leaf color'),
        ('T_ngrdi_leaf_auc', 'Pure-leaf NGRDI integral', 'Decoupled leaf color'),
        ('Searly_gap_large', 'Large canopy gaps >=0.04 m2 (128/130 DAS)', 'Millimeter micro-structure'),
        ('T_gap_auc', 'Seasonal large-gap integral', 'Millimeter micro-structure'),
        ('Searly_prof_across', 'Cross-row closure heterogeneity (128/130 DAS)', 'Millimeter micro-structure'),
        ('white_b140_s30_raw', 'Raw white-pixel fraction (181 DAS)', '181-DAS boll proxy'),
        ('white_b140_s30_morph_frac', 'Filtered boll-cluster fraction (181 DAS)', '181-DAS boll proxy'),
        ('white_b140_s30_cluster_m2', 'Boll-cluster density (clusters m-2)', '181-DAS boll proxy'),
    ]
    out = []
    # Pooled across environments: unadjusted vs (domain + domain*N + domain*block) adjusted!
    z_pool = pd.get_dummies(
        plot64[['domain']].assign(
            dom_N=plot64.domain + '_' + plot64.N_kg_ha.astype(int).astype(str),
            dom_B=plot64.domain + '_B' + plot64.block_design.astype(str),
        ),
        drop_first=True, dtype=float,
    )
    X_pool = np.column_stack([np.ones(len(plot64)), z_pool.to_numpy()])
    y_all = plot64.yield_kg_ha.to_numpy(float)
    ry_all = y_all - X_pool @ np.linalg.lstsq(X_pool, y_all, rcond=None)[0]

    # Also control only environment intercept (showing N-confounded within-field correlation!)
    z_env = pd.get_dummies(plot64[['domain']], drop_first=True, dtype=float)
    X_env = np.column_stack([np.ones(len(plot64)), z_env.to_numpy()])
    ry_env = y_all - X_env @ np.linalg.lstsq(X_env, y_all, rcond=None)[0]

    for col, label, group in features:
        v = plot64[col].to_numpy(float)
        rv_env = v - X_env @ np.linalg.lstsq(X_env, v, rcond=None)[0]
        rv_adj = v - X_pool @ np.linalg.lstsq(X_pool, v, rcond=None)[0]
        r_unadj, p_unadj = pearsonr(v, y_all)
        r_env, p_env = pearsonr(rv_env, ry_env)
        r_adj, p_adj = pearsonr(rv_adj, ry_all)

        dom_adj = {}
        for dom, d_sub in plot64.groupby('domain'):
            zd = pd.get_dummies(d_sub[['N_kg_ha', 'block_design']].astype(str), drop_first=True, dtype=float)
            Xd = np.column_stack([np.ones(len(d_sub)), zd.to_numpy()])
            yd = d_sub.yield_kg_ha.to_numpy(float)
            vd = d_sub[col].to_numpy(float)
            ryd = yd - Xd @ np.linalg.lstsq(Xd, yd, rcond=None)[0]
            rvd = vd - Xd @ np.linalg.lstsq(Xd, vd, rcond=None)[0]
            dom_adj[dom] = float(pearsonr(rvd, ryd).statistic) if np.std(rvd) > 1e-12 else 0.0

        out.append({
            'feature': col, 'label': label, 'group': group,
            'pooled_unadjusted_r': float(r_unadj),
            'within_env_apparent_r': float(r_env),
            'within_N_block_partial_r': float(r_adj),
            'within_N_block_p': float(p_adj),
            'r_2023_F1': dom_adj['2023_F1'],
            'r_2024_F1': dom_adj['2024_F1'],
            'r_2024_F2': dom_adj['2024_F2'],
        })
    return pd.DataFrame(out)


def evaluate_specs_nested(plot64: pd.DataFrame, tag_prefix: str = ''):
    """Evaluate feature specifications under identical nested leave-source-block-out PartialRidge."""
    specs = {
        'N': [],
        'N_Cearly': ['Cearly_cover', 'Cearly_exg'],
        'N_C': ['C_cover', 'C_exg'],
        'N_C_T': ['C_cover', 'C_exg', 'T_auc_mean', 'T_late_change'],
        'N_Leaf_T': ['C_cover', 'C_ngrdi_leaf', 'T_auc_mean', 'T_late_change'],
        'N_Gap_early': ['Cearly_cover', 'Searly_gap_large'],
        'N_Leaf': ['Cearly_cover', 'C_cover', 'Cearly_ngrdi_leaf', 'C_ngrdi_leaf', 'T_auc_mean', 'T_late_change'],
        'N_Micro_early': ['Cearly_cover', 'Cearly_ngrdi_leaf', 'Searly_gap_large', 'Searly_prof_across'],
        'N_Micro_T': [
            'C_cover', 'C_ngrdi_leaf', 'T_auc_mean', 'T_late_change',
            'Searly_gap_large', 'T_gap_auc', 'Searly_prof_across', 'S_cover_sd25',
        ],
    }
    frames = {dom: d.reset_index(drop=True) for dom, d in plot64.groupby('domain')}
    domains = ['2023_F1', '2024_F1', '2024_F2']
    tasks = [(s, t, 'BlockCV' if s == t else 'adjacent_field' if s[:4] == t[:4] else 'cross_year')
             for s in domains for t in domains]

    pred_rows = []
    tune_rows = []

    for src, tgt, ev in tasks:
        a = frames[src]
        b = frames[tgt]
        y_a = a.yield_kg_ha.to_numpy(float)
        grp_a = a.block_design.to_numpy(int)
        ii = np.arange(len(a))
        folds = np.unique(grp_a) if src == tgt else [0]

        for spec_name, cols in specs.items():
            n_a = a.N_kg_ha.to_numpy(float)
            n_b = b.N_kg_ha.to_numpy(float)
            xa = np.c_[n_a, n_a * n_a, a[cols].to_numpy(float)] if cols else np.c_[n_a, n_a * n_a]
            xb = np.c_[n_b, n_b * n_b, b[cols].to_numpy(float)] if cols else np.c_[n_b, n_b * n_b]

            for fold in folds:
                tr = ii[grp_a != fold] if fold else ii
                te = np.flatnonzero(b.block_design.to_numpy(int) == fold) if fold else np.arange(len(b))

                if cols:
                    scores = []
                    for alpha in legacy.ALPHAS:
                        oof = np.empty(len(tr))
                        for blk in np.unique(grp_a[tr]):
                            keep = grp_a[tr] != blk
                            m_in = legacy.PartialRidge(alpha).fit(xa[tr[keep]], y_a[tr[keep]])
                            oof[~keep] = m_in.predict(xa[tr[~keep]])
                        err = oof - y_a[tr]
                        scores.append((float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err))), -alpha, alpha))
                    inner_rmse, _, _, best_alpha = min(scores)
                else:
                    best_alpha, inner_rmse = 0.0, np.nan

                m_fit = legacy.PartialRidge(best_alpha).fit(xa[tr], y_a[tr])
                yh = m_fit.predict(xb[te])
                tune_rows.append({
                    'prefix': tag_prefix, 'source': src, 'target': tgt, 'evaluation': ev,
                    'specification': spec_name, 'held_block': int(fold),
                    'alpha': float(best_alpha), 'inner_RMSE': inner_rmse,
                })
                for r_te, p_val in zip(b.iloc[te].itertuples(), yh):
                    pred_rows.append({
                        'prefix': tag_prefix, 'source': src, 'target': tgt, 'evaluation': ev,
                        'specification': spec_name, 'plot_id': int(r_te.plot_id),
                        'block': int(r_te.block_design), 'N': float(r_te.N_kg_ha),
                        'observed': float(r_te.yield_kg_ha), 'predicted': float(p_val),
                    })

    # Also evaluate source-nested 181-DAS Boll Raw vs Boll Morphology!
    ai_tune = pd.read_csv(OUT / 'source_boll_mask_rules.csv')
    for src, tgt, ev in tasks:
        a = frames[src]
        b = frames[tgt]
        y_a = a.yield_kg_ha.to_numpy(float)
        grp_a = a.block_design.to_numpy(int)
        ii = np.arange(len(a))
        folds = np.unique(grp_a) if src == tgt else [0]

        for b_spec, add_morph in [('N_C_T_Braw', False), ('N_C_T_Bmorph', True)]:
            for fold in folds:
                tr = ii[grp_a != fold] if fold else ii
                te = np.flatnonzero(b.block_design.to_numpy(int) == fold) if fold else np.arange(len(b))
                role_key = f'BlockCV_hold_{fold}' if fold else (f'{ev}_hold_0')
                w_rule = ai_tune.query(
                    "source==@src and role==@role_key and specification=='N_C_T_B'"
                ).mask_rule.iloc[0]

                base_cols = ['C_cover', 'C_exg', 'T_auc_mean', 'T_late_change']
                b_cols = [f'{w_rule}_morph_frac', f'{w_rule}_cluster_m2'] if add_morph else [f'{w_rule}_raw']
                cols = base_cols + b_cols
                n_a, n_b = a.N_kg_ha.to_numpy(float), b.N_kg_ha.to_numpy(float)
                xa = np.c_[n_a, n_a * n_a, a[cols].to_numpy(float)]
                xb = np.c_[n_b, n_b * n_b, b[cols].to_numpy(float)]

                scores = []
                for alpha in legacy.ALPHAS:
                    oof = np.empty(len(tr))
                    for blk in np.unique(grp_a[tr]):
                        keep = grp_a[tr] != blk
                        m_in = legacy.PartialRidge(alpha).fit(xa[tr[keep]], y_a[tr[keep]])
                        oof[~keep] = m_in.predict(xa[tr[~keep]])
                    err = oof - y_a[tr]
                    scores.append((float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err))), -alpha, alpha))
                inner_rmse, _, _, best_alpha = min(scores)
                m_fit = legacy.PartialRidge(best_alpha).fit(xa[tr], y_a[tr])
                yh = m_fit.predict(xb[te])
                tune_rows.append({
                    'prefix': tag_prefix, 'source': src, 'target': tgt, 'evaluation': ev,
                    'specification': b_spec, 'held_block': int(fold),
                    'alpha': float(best_alpha), 'inner_RMSE': inner_rmse,
                })
                for r_te, p_val in zip(b.iloc[te].itertuples(), yh):
                    pred_rows.append({
                        'prefix': tag_prefix, 'source': src, 'target': tgt, 'evaluation': ev,
                        'specification': b_spec, 'plot_id': int(r_te.plot_id),
                        'block': int(r_te.block_design), 'N': float(r_te.N_kg_ha),
                        'observed': float(r_te.yield_kg_ha), 'predicted': float(p_val),
                    })

    return pd.DataFrame(pred_rows), pd.DataFrame(tune_rows)


def block_intervals_vs_N(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Exact 27 whole-block conditional 95% intervals of Total & Within-N RMSE reduction vs N-only."""
    rows = []
    for (prefix, src, tgt, ev), grp in pred_df.groupby(['prefix', 'source', 'target', 'evaluation'], sort=False):
        base = grp[grp.specification == 'N'].sort_values('plot_id')
        for spec, cand in grp.groupby('specification', sort=False):
            cand = cand.sort_values('plot_id')
            assert base.plot_id.tolist() == cand.plot_id.tolist()
            e0_raw = (base.predicted - base.observed).to_numpy(copy=True)
            e1_raw = (cand.predicted - cand.observed).to_numpy(copy=True)
            n_arr = base.N.to_numpy()
            blk_arr = base.block.to_numpy()
            uniq_blks = sorted(set(blk_arr))

            for measure in ('Total', 'Within-N'):
                e0 = e0_raw.copy()
                e1 = e1_raw.copy()
                if measure == 'Within-N':
                    for lv in np.unique(n_arr):
                        tk = n_arr == lv
                        e0[tk] -= e0[tk].mean()
                        e1[tk] -= e1[tk].mean()
                rmse0 = float(np.sqrt(np.mean(e0 ** 2)))
                rmse1 = float(np.sqrt(np.mean(e1 ** 2)))
                gain = rmse0 - rmse1
                draws = []
                for bs in product(uniq_blks, repeat=3):
                    ix = np.concatenate([np.flatnonzero(blk_arr == b) for b in bs])
                    draws.append(float(np.sqrt(np.mean(e0[ix] ** 2)) - np.sqrt(np.mean(e1[ix] ** 2))))
                lo, hi = np.quantile(draws, [0.025, 0.975])
                rows.append({
                    'prefix': prefix, 'source': src, 'target': tgt, 'evaluation': ev,
                    'model': spec, 'measure': measure, 'n': len(base),
                    'RMSE_N': rmse0, 'RMSE_model': rmse1,
                    'gain_absolute': gain, 'cluster_low': float(lo), 'cluster_high': float(hi),
                })
    return pd.DataFrame(rows)


def summarize_gsd_physical_degradation(df_gsd: pd.DataFrame) -> pd.DataFrame:
    """Summarize physical mixed-pixel decay of canopy cover, gap architecture, and boll morphology across GSD."""
    ref = df_gsd[df_gsd.gsd_m == 0.0033].set_index(['domain', 'plot_id', 'DAS'])
    rows = []
    for gsd, sub in df_gsd.groupby('gsd_m'):
        sub_idx = sub.set_index(['domain', 'plot_id', 'DAS']).reindex(ref.index)
        in_mask = sub_idx.index.get_level_values('DAS') < 180
        b_mask = sub_idx.index.get_level_values('DAS') == 181

        cov_ref = ref.loc[in_mask, 'f_cover'].to_numpy()
        cov_cur = sub_idx.loc[in_mask, 'f_cover'].to_numpy()
        gap_ref = ref.loc[in_mask, 'gap_large_frac'].to_numpy()
        gap_cur = sub_idx.loc[in_mask, 'gap_large_frac'].to_numpy()
        prof_ref = ref.loc[in_mask, 'profile_sd_across'].to_numpy()
        prof_cur = sub_idx.loc[in_mask, 'profile_sd_across'].to_numpy()
        boll_ref = ref.loc[b_mask, 'white_b140_s30_morph_frac'].to_numpy()
        boll_cur = sub_idx.loc[b_mask, 'white_b140_s30_morph_frac'].to_numpy()

        rows.append({
            'gsd_m': float(gsd),
            'gsd_label': GSD_LABELS[gsd],
            'mean_canopy_cover_pct': float(100 * np.mean(cov_cur)),
            'cover_mae_vs_native_pp': float(100 * np.mean(np.abs(cov_cur - cov_ref))),
            'cover_r_vs_native': float(pearsonr(cov_cur, cov_ref).statistic) if np.std(cov_cur) > 1e-9 else float('nan'),
            'mean_large_gap_pct': float(100 * np.mean(gap_cur)),
            'gap_r_vs_native': float(pearsonr(gap_cur, gap_ref).statistic) if np.std(gap_cur) > 1e-9 else float('nan'),
            'mean_row_profile_sd_pct': float(100 * np.mean(prof_cur)),
            'row_profile_r_vs_native': float(pearsonr(prof_cur, prof_ref).statistic) if np.std(prof_cur) > 1e-9 else float('nan'),
            'mean_boll_morph_pct': float(100 * np.mean(boll_cur)),
            'boll_r_vs_native': float(pearsonr(boll_cur, boll_ref).statistic) if np.std(boll_cur) > 1e-9 else float('nan'),
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df_native, df_gsd = extract_all_slots_and_multiscale()

    # 1. Native 3.3 mm 64-plot evaluation & confounding decomposition
    plot64_native = build_common4_plot_table(df_gsd[df_gsd.gsd_m == 0.0033])
    plot64_native.to_csv(OUT / 'plot64_millimeter_features.csv', index=False)

    assoc_df = compute_confounding_associations(plot64_native)
    assoc_df.to_csv(OUT / 'management_confounding_associations.csv', index=False)

    pred_native, tune_native = evaluate_specs_nested(plot64_native, tag_prefix='3.3 mm')
    # Verify exact replay of frozen N_C_T predictions!
    frozen = pd.read_csv(OLD_MINING / 'support_predictions.csv').query(
        "variant=='common_DAS' and algorithm=='PartialRidge' and specification=='N_C_T' and evaluation in ['BlockCV','adjacent_field','cross_year']"
    )
    chk = pred_native[pred_native.specification == 'N_C_T'].merge(
        frozen, on=['source', 'target', 'evaluation', 'plot_id'], suffixes=('_new', '_old')
    )
    max_diff = float(np.max(np.abs(chk.predicted_new - chk.predicted_old)))
    print(f'Exact replay check vs frozen N_C_T (n={len(chk)}): max |diff| = {max_diff:.2e}', flush=True)
    assert len(chk) == 192 and max_diff < 1e-2

    # 2. Multi-scale GSD predictive evaluation across all 6 spatial resolutions!
    all_gsd_preds = [pred_native]
    all_gsd_tunes = [tune_native]
    for gsd in GSD_LEVELS_M[1:]:
        p_tbl = build_common4_plot_table(df_gsd[df_gsd.gsd_m == gsd])
        pr, tu = evaluate_specs_nested(p_tbl, tag_prefix=GSD_LABELS[gsd])
        all_gsd_preds.append(pr)
        all_gsd_tunes.append(tu)

    full_preds = pd.concat(all_gsd_preds, ignore_index=True)
    full_tunes = pd.concat(all_gsd_tunes, ignore_index=True)
    full_intervals = block_intervals_vs_N(full_preds)

    full_preds.to_csv(OUT / 'multiscale_predictions.csv', index=False)
    full_tunes.to_csv(OUT / 'multiscale_tuning.csv', index=False)
    full_intervals.to_csv(OUT / 'multiscale_intervals_vs_N.csv', index=False)

    phys_deg = summarize_gsd_physical_degradation(df_gsd)
    phys_deg.to_csv(OUT / 'multiscale_physical_degradation.csv', index=False)

    # Save deterministic physical multi-scale visual slices for Figure 4a (2024_F1 Plot 11 N225 at 128 & 181 DAS)
    demo_path = OUT / 'multiscale_demo_slices.npz'
    if not demo_path.is_file():
        df_slots = pd.read_csv(INPUT_SLOTS)
        gdf = gpd.read_file(GPKG, layer='inward_buffer_plot')
        demo_payload = {}
        for das_demo, tag_d in [(128, 'canopy128'), (181, 'boll181')]:
            r_row = df_slots[(df_slots.year == 2024) & (df_slots.field == 'F1') & (df_slots.plot_id == 11) & (df_slots.DAS == das_demo)].iloc[0]
            with rasterio.open(r_row.raster_path) as src:
                gg = gdf.to_crs(src.crs)
                px_m = float(np.sqrt(abs(src.res[0] * src.res[1])))
                geom = gg[(gg.year == 2024) & (gg.field == 'F1') & (gg.plot_id == 11)].geometry.iloc[0]
                rgb0, val0 = crop(src, geom)
                # Take central 2.4 m x 2.4 m square inside valid ROI for crisp cross-row visualization
                cy, cx = val0.shape[0] // 2, val0.shape[1] // 2
                half = int(round(1.20 / px_m))
                rgb_sq = rgb0[max(0, cy - half):min(val0.shape[0], cy + half), max(0, cx - half):min(val0.shape[1], cx + half)]
                val_sq = val0[max(0, cy - half):min(val0.shape[0], cy + half), max(0, cx - half):min(val0.shape[1], cx + half)]
                for gsd in GSD_LEVELS_M:
                    c_rgb, c_val, eff_px = coarsen_rgb_valid(rgb_sq, val_sq, px_m, gsd)
                    ms_c = rules(c_rgb, c_val)
                    key_g = GSD_LABELS[gsd].replace(' ', '').replace('.', 'p')
                    demo_payload[f'{tag_d}_{key_g}_rgb'] = c_rgb
                    demo_payload[f'{tag_d}_{key_g}_green'] = ms_c['green_canopeo']
                    demo_payload[f'{tag_d}_{key_g}_boll'] = ms_c['white_b140_s30']
                    demo_payload[f'{tag_d}_{key_g}_px_m'] = np.array([eff_px])
        np.savez_compressed(demo_path, **demo_payload)

    print('\n=== Confounding vs Within-N Partial Correlations ===')
    print(assoc_df[['feature', 'within_env_apparent_r', 'within_N_block_partial_r', 'within_N_block_p']].to_string(index=False))

    print('\n=== Physical Degradation Across GSD ===')
    print(phys_deg.to_string(index=False))

    print('\n=== Native 3.3 mm Mean Gains Across Tasks ===')
    nat_int = full_intervals[full_intervals.prefix == '3.3 mm']
    print(nat_int.groupby(['measure', 'model'])['gain_absolute'].mean().unstack(0).round(2).to_string())

    print('\n=== Multi-Scale GSD Mean Gains for N_Micro_T vs N_C_T ===')
    gsd_sum = full_intervals[full_intervals.model.isin(['N_C_T', 'N_Micro_T', 'N_Micro_early'])].groupby(
        ['prefix', 'evaluation', 'measure', 'model']
    )['gain_absolute'].mean().unstack('model').round(2)
    print(gsd_sum.to_string())


if __name__ == '__main__':
    main()
