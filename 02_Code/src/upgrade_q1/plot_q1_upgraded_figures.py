#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render top-journal Q1 multi-modal Main Figures 1-6 + sync Supplementary Figures S1-S8.

Visual archetypes follow Remote Sensing of Environment / ISPRS / Field Crops Research /
Artificial Intelligence in Agriculture standards:
  - Figure 2 (a-e): Millimeter RGB tiles + growth ribbons + quadratic N CI + quantitative
    sub-plot NGRDI/gap maps with colorbar + diverging bar/lollipop confounding shift.
  - Figure 3 (a-d): Pixel-level foreground/background chromaticity KDE + 1x3 1:1 observed vs.
    predicted yield parity plots (N-rate colored, +/-10% cone) + 9-task x 5-model annotated
    Total ΔRMSE benchmark matrix + paired Within-N 95% CI forest.
  - Figure 4 (a-f): 2x6 sensor GSD pyramid (3.3 mm to 3 m) + 181 DAS glare-filtering maps &
    size spectrum + Raincloud (half-violin + boxplot + spaghetti) cover inflation + 3-stage
    collapse curves + multi-scale ΔRMSE ribbons + grouped bar + 95% CI boll comparison.
  - Figure 5 (a-f): 3-component stacked orthogonal MSE variance decomposition bars (6-task mean
    & 6 individual transfers) + 1x4 geo-referenced field spatial yield choropleth maps +
    physical N-response curve paradox + commercial calibration budget curves + UAV active
    sampling field map & multi-model convergence.
  - Figure 6 (a-e): 9-task x 6-phenophase forward-only ΔRMSE heatmap (with colorbar) + 1x3
    lead-time ribbon curves + shaded 90% PI fan/ribbon charts + grouped before/after coverage
    bars + 2D Reliability-Sharpness trade-off diagram.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon, Rectangle
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '02_Code/utilities'))
import figure_style as fs

FORMAL_NEW = ROOT / '10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108'
FORMAL_OLD = FORMAL_NEW / 'analysis/frozen_baseline'
OLD_MINING = FORMAL_OLD / 'analysis/raw_rgb_mining'
OLD_METHOD = FORMAL_OLD / 'analysis/method_strengthening'
OLD_DM = FORMAL_OLD / 'analysis/data_mining_current'
Q1_DIR = FORMAL_NEW / 'analysis/q1_spatial_calibration'

FIG_MAIN = FORMAL_NEW / 'figures/main'
FIG_SUPP = FORMAL_NEW / 'figures/supplementary'
FIG_EDIT = FORMAL_NEW / 'editable'

DOMAINS = list(fs.ENV_COLORS)
ROLES = fs.ROLES
TRANSFER = ROLES[3:]
COMMON = {'2023_F1': [92, 107, 130, 142], '2024_F1': [92, 106, 128, 141], '2024_F2': [92, 106, 128, 141]}

INK = '#0F172A'
SLATE = '#334155'

# Unified paper-wide colorblind-safe visual vocabulary (Figures 1-6, S9-S10)
PALETTE = {
    'teal': '#0F766E',
    'teal_mid': '#0D9488',
    'teal_light': '#99F6E4',
    'cerulean': '#0284C7',
    'cerulean_light': '#BAE6FD',
    'ochre': '#B45309',
    'terracotta': '#C2410C',
    'amber': '#D97706',
    'amber_light': '#FDBA74',
    'crimson': '#9F1239',
    'crimson_mid': '#E11D48',
    'crimson_light': '#FDA4AF',
    'alert': '#B91C1C',
    'plum': '#882255',
    'magenta_mask': '#A31C7A',
    'slate': '#475569',
    'slate_dark': '#334155',
    'slate_mid': '#64748B',
    'slate_light': '#CBD5E1',
}

N_RATES_ALL = [0, 75, 150, 225, 300, 375, 450]
N_COLORS = fs.N_COLORS
ENV_COLORS = fs.ENV_COLORS

Q1_DIV_CMAP = LinearSegmentedColormap.from_list(
    'q1_terracotta_teal',
    ['#9A3412', '#EA580C', '#FDBA74', '#F8FAFC', '#99F6E4', '#0D9488', '#115E59'],
)
Q1_LEAF_CMAP = LinearSegmentedColormap.from_list(
    'q1_leaf_ngrdi',
    ['#F8FAFC', '#CCFBF1', '#5EEAD4', '#0D9488', '#115E59'],
)

MODEL_COLORS = {
    'N': PALETTE['slate'],
    'N_Cearly': PALETTE['ochre'],
    'N_C': PALETTE['slate_mid'],
    'N_C_T': PALETTE['cerulean'],
    'N_Leaf_T': PALETTE['teal'],
    'N_RGB52_common4': PALETTE['slate_dark'],
    'N_ResNet50_PCA52_common4': PALETTE['crimson'],
    'N_plus_3N0': PALETTE['amber'],
    'N_Leaf_T_plus_3N0': PALETTE['teal'],
    'N_ResNet50_plus_3N0': PALETTE['crimson'],
    'N_oracle': INK,
    'N_C_T_Braw': PALETTE['amber_light'],
    'N_C_T_Bmorph': PALETTE['plum'],
}

MODEL_MARKERS = {
    'N': 'o',
    'N_Cearly': '^',
    'N_C': 'o',
    'N_C_T': 's',
    'N_Leaf_T': 'v',
    'N_RGB52_common4': 'X',
    'N_ResNet50_PCA52_common4': 'P',
    'N_plus_3N0': 's',
    'N_Leaf_T_plus_3N0': 'D',
    'N_ResNet50_plus_3N0': 'X',
    'N_oracle': 'd',
    'N_C_T_Braw': 'o',
    'N_C_T_Bmorph': 'D',
}

MAIN_STEMS = (
    'Figure_1_design_and_label_budgets',
    'Figure_2_RGB_observations',
    'Figure_3_RMSE_reduction',
    'Figure_4_multiscale_spatial_scaling',
    'Figure_5_fewshot_calibration_paradox',
    'Figure_6_forecast_timing_and_coverage',
)

SUPP_STEMS = (
    'Figure_S1_segmentation_audit',
    'Figure_S2_boll_threshold_sensitivity',
    'Figure_S3_recursive_timing',
    'Figure_S4_spatial_residuals_2024F1',
)


def _safe_copy(src: Path, dst: Path) -> None:
    if dst.exists() and dst.resolve() == src.resolve():
        return
    if dst.is_symlink():
        dst.unlink()
    shutil.copyfile(src, dst)


def save_fig(fig: plt.Figure, stem: str) -> None:
    FIG_MAIN.mkdir(parents=True, exist_ok=True)
    FIG_EDIT.mkdir(parents=True, exist_ok=True)
    fs.save(fig, FIG_MAIN, stem)
    _safe_copy(FIG_MAIN / f'{stem}.svg', FIG_EDIT / f'{stem}.svg')
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis='both', which='major', length=2.4, width=0.6, pad=1.8, color=SLATE, labelsize=7.0)
    ax.tick_params(axis='both', which='minor', length=1.3, width=0.5, color='#64748B')
    ax.spines['left'].set_color(SLATE)
    ax.spines['bottom'].set_color(SLATE)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)


def series_for(df: pd.DataFrame, roles: list[tuple[str, str, str]], col: str) -> np.ndarray:
    return df.set_index(['source', 'target', 'evaluation']).reindex(roles)[col].to_numpy(dtype=float)


def is_hollow_alpha(
    tune_df: pd.DataFrame,
    spec: str,
    task: tuple[str, str, str],
    extra_tune: dict[str, pd.DataFrame] | None = None,
) -> bool:
    src, tgt, ev = task
    if extra_tune and spec in extra_tune:
        z = extra_tune[spec]
        z = z[(z.source == src) & (z.target == tgt) & (z.evaluation == ev)]
        if ev == 'BlockCV' and 'held_block' in z.columns:
            z = z[z.held_block > 0]
        elif ev != 'BlockCV' and 'held_block' in z.columns:
            z = z[z.held_block == 0]
        col = 'alpha' if 'alpha' in z.columns else 'selected_alpha'
        return len(z) > 0 and bool((z[col] >= 100000).all())
    if spec not in ('N_Cearly', 'N_C', 'N_C_T', 'N_Leaf_T', 'N_C_T_Braw', 'N_C_T_Bmorph'):
        return False
    z = tune_df[(tune_df.source == src) & (tune_df.specification == spec)]
    if ev == 'BlockCV':
        z = z[z.held_block > 0]
    else:
        z = z[(z.target == tgt) & (z.evaluation == ev) & (z.held_block == 0)]
    return len(z) > 0 and bool((z.alpha >= 100000).all())


def _gaussian_kde_1d(vals: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    diff = (grid[:, None] - vals[None, :]) / bw
    dens = np.exp(-0.5 * diff * diff).mean(axis=1) / (bw * np.sqrt(2.0 * np.pi))
    return dens


def clean_stale_supplementary_assets() -> None:
    FIG_MAIN.mkdir(parents=True, exist_ok=True)
    FIG_SUPP.mkdir(parents=True, exist_ok=True)
    FIG_EDIT.mkdir(parents=True, exist_ok=True)
    valid_supp = {f'{stem}.{ext}' for stem in SUPP_STEMS for ext in ('pdf', 'svg', 'png', 'tiff')}
    for p in list(FIG_SUPP.glob('Figure_S*')):
        if p.name not in valid_supp:
            p.unlink()
    valid_edit = {f'{stem}.svg' for stem in MAIN_STEMS + SUPP_STEMS} | {'plot_q1_upgraded_figures.py'}
    for p in list(FIG_EDIT.glob('Figure_S*')):
        if p.name not in valid_edit:
            p.unlink()


def copy_figure1_and_supplementary() -> None:
    FIG_MAIN.mkdir(parents=True, exist_ok=True)
    FIG_SUPP.mkdir(parents=True, exist_ok=True)
    FIG_EDIT.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# FIGURE 1 (a-b): 3-Stage Spatial Hierarchy + Multi-Temporal Flight/GDD10 Schedule
# ==============================================================================
def render_figure1() -> None:
    import json
    import pyproj
    from matplotlib.patches import ConnectionPatch, Polygon, Rectangle
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject
    from rasterio.windows import from_bounds as win_from_bounds
    from shapely import wkt
    from shapely.affinity import scale as shp_scale, translate
    from shapely.geometry import Polygon as ShpPolygon, shape
    from shapely.ops import transform as shp_transform, unary_union

    fs.apply()
    fig_w_mm, fig_h_mm = 180.0, 112.0
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4), facecolor='white')

    ink = '#0F172A'
    slate = '#334155'

    fig.text(0.014, 0.984, 'a', fontsize=9, fontweight='bold', color='#111111', va='top', ha='left')
    fig.text(0.014, 0.428, 'b', fontsize=9, fontweight='bold', color='#111111', va='top', ha='left')

    def _draw_geom(ax, geom, *, fc='none', ec='#64748B', lw=0.4, ls='-', alpha=1.0, zorder=2):
        if geom.is_empty:
            return
        geoms = [geom] if geom.geom_type == 'Polygon' else list(getattr(geom, 'geoms', []))
        for poly in geoms:
            if poly.geom_type != 'Polygon':
                continue
            coords = np.asarray(poly.exterior.coords)
            ax.add_patch(Polygon(coords, closed=True, facecolor=fc, edgecolor=ec,
                                 linewidth=lw, linestyle=ls, alpha=alpha, zorder=zorder))

    def _draw_unused_cell(ax, poly_disp, *, zorder=3.4):
        coords = np.asarray(poly_disp.exterior.coords)
        ax.add_patch(Polygon(coords, closed=True, facecolor='#0F172A', edgecolor='none',
                             alpha=0.50, zorder=zorder))
        bx0, by0, bx1, by1 = poly_disp.bounds
        clip_patch = Polygon(coords, closed=True, facecolor='none', edgecolor='none', transform=ax.transData)
        ax.add_patch(clip_patch)
        span = (bx1 - bx0) + (by1 - by0)
        for d in np.arange(-span, span + 1.1, 1.15):
            ln, = ax.plot([bx0 + d, bx0 + d + (by1 - by0)], [by0, by1],
                          color='#E2E8F0', lw=0.45, alpha=0.72, zorder=zorder + 0.1)
            ln.set_clip_path(clip_patch)

    # Shared top-row vertical geometry [y = 0.470 .. 0.954] aligned across Stages 1-3
    row_a_y, row_a_h = 0.470, 0.484

    # --------------------------------------------------------------------------
    # STAGE 1 (PANEL A LEFT): Xinjiang Locator + Embedded China Inset
    # --------------------------------------------------------------------------
    proj_to_km = pyproj.Transformer.from_crs(
        'EPSG:4326',
        '+proj=aea +lat_1=25 +lat_2=47 +lat_0=35 +lon_0=105 +datum=WGS84 +units=km',
        always_xy=True,
    ).transform

    cn_geo = json.loads((ROOT / '01_Data/external/geospatial/china_provinces.geojson').read_text())
    xj_geo = json.loads((ROOT / '01_Data/external/geospatial/xinjiang_prefectures_datav.geojson').read_text())

    loc_x, loc_y, loc_w, loc_h = 0.056, row_a_y, 0.218, row_a_h
    loc_aspect = (loc_w * fig_w_mm) / (loc_h * fig_h_mm)

    ax_xj = fig.add_axes([loc_x, loc_y, loc_w, loc_h], facecolor='white')
    for feat in xj_geo['features']:
        adcode = feat['properties'].get('adcode')
        g_ll = shape(feat['geometry'])
        if adcode == 659002:  # Alar City
            _draw_geom(ax_xj, g_ll, fc='#38BDF8', ec='#0369A1', lw=0.65, zorder=5)
        elif adcode == 652900:  # Aksu Prefecture
            _draw_geom(ax_xj, g_ll, fc='#E0F2FE', ec='#0284C7', lw=0.50, zorder=3)
        else:
            _draw_geom(ax_xj, g_ll, fc='#F8FAFC', ec='#94A3B8', lw=0.35, zorder=2)

    alar_lon, alar_lat = 81.31408, 40.60595
    ax_xj.plot(alar_lon, alar_lat, 'o', color='white', ms=5.0, zorder=6)
    ax_xj.plot(alar_lon, alar_lat, 'o', color=PALETTE['crimson'], ms=3.4, zorder=7)
    ax_xj.annotate(
        'Alar',
        xy=(alar_lon, alar_lat), xytext=(83.8, 37.9),
        fontsize=7.2, fontweight='bold', color=ink, ha='left', va='center',
        arrowprops=dict(arrowstyle='-', color=ink, lw=0.70),
        zorder=8,
    )

    # Exact 300 km bar at the map's 41.5 deg N reference aspect (300 / (111.32 * cos(41.5°)) = 3.60 deg lon)
    ax_xj.plot([74.4, 78.00], [34.8, 34.8], color=ink, lw=1.05, zorder=6)
    ax_xj.text(76.20, 35.25, '300 km', ha='center', va='bottom', fontsize=7.0, color=ink, zorder=6)

    xj_h_lat = 24.4
    xj_w_lon = xj_h_lat * loc_aspect / np.cos(np.radians(41.5))
    ax_xj.set_xlim(84.92 - xj_w_lon / 2, 84.92 + xj_w_lon / 2)
    ax_xj.set_ylim(33.3, 33.3 + xj_h_lat)
    ax_xj.set_xticks([76, 84, 92])
    ax_xj.set_xticklabels(['76°E', '84°E', '92°E'])
    ax_xj.set_yticks([36, 42, 48])
    ax_xj.set_yticklabels(['36°N', '42°N', '48°N'])
    ax_xj.tick_params(axis='both', which='major', length=2.0, width=0.5, pad=1.4, labelsize=7.0, color='#64748B')
    for sp in ax_xj.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor('#94A3B8')
        sp.set_linewidth(0.55)

    cn_ins_w, cn_ins_h = 0.56, 0.34
    cn_ins_aspect = (cn_ins_w * loc_w * fig_w_mm) / (cn_ins_h * loc_h * fig_h_mm)
    ax_cn = ax_xj.inset_axes([0.0, 1.0 - cn_ins_h, cn_ins_w, cn_ins_h], facecolor='white', zorder=9)
    for feat in cn_geo['features']:
        adcode = feat['properties'].get('adcode')
        g_km = shp_transform(proj_to_km, shape(feat['geometry']))
        if adcode == '100000_JD':
            _draw_geom(ax_cn, g_km, fc='none', ec='#475569', lw=0.40, zorder=3)
        elif adcode == 650000:
            _draw_geom(ax_cn, g_km, fc='#BAE6FD', ec='#0369A1', lw=0.60, zorder=4)
        else:
            _draw_geom(ax_cn, g_km, fc='#F1F5F9', ec='#94A3B8', lw=0.26, zorder=2)
    cn_h_km = 4520.0
    cn_w_km = cn_h_km * cn_ins_aspect
    ax_cn.set_xlim(-220.0 - cn_w_km / 2, -220.0 + cn_w_km / 2)
    ax_cn.set_ylim(-2020.0, -2020.0 + cn_h_km)
    ax_cn.set_xticks([])
    ax_cn.set_yticks([])
    for sp in ax_cn.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor('#94A3B8')
        sp.set_linewidth(0.50)

    ax_scs = ax_cn.inset_axes([0.74, 0.0, 0.26, 0.46], facecolor='white')
    for feat in cn_geo['features']:
        adcode = feat['properties'].get('adcode')
        g_km = shp_transform(proj_to_km, shape(feat['geometry']))
        if adcode == '100000_JD':
            _draw_geom(ax_scs, g_km, fc='none', ec='#334155', lw=0.42, zorder=4)
        else:
            _draw_geom(ax_scs, g_km, fc='#F1F5F9', ec='#94A3B8', lw=0.22, zorder=2)
    ax_scs.set_xlim(220, 1850)
    ax_scs.set_ylim(-3420, -980)
    ax_scs.set_xticks([])
    ax_scs.set_yticks([])
    for sp in ax_scs.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor('#64748B')
        sp.set_linewidth(0.45)

    # --------------------------------------------------------------------------
    # STAGE 2 (PANEL A CENTER): Unified True-Scale 1x3 RCB Orthomosaic Triptych
    #   2023 F1 (n=24) | 2024 F2 (n=21) | 2024 F1 (n=21, N rates in kg ha^-1)
    #   1:1:1 metric scale, shared B1-B3 block alignment, zero dead space
    # --------------------------------------------------------------------------
    layout = pd.read_csv(ROOT / '01_Data/processed/spatial/cotton_rgb/field_authority_full_plot.csv')
    layout = layout[layout.scope_id.isin(['2023_f1', '2024_f1', '2024_f2'])].copy()
    layout['geom_obj'] = layout.geom.map(wkt.loads)
    inward = pd.read_csv(ROOT / '01_Data/processed/spatial/cotton_rgb/field_authority_inward_buffer.csv')
    inward['geom_obj'] = inward.geom.map(wkt.loads)

    site_x, site_y, site_w, site_h = 0.285, row_a_y, 0.436, row_a_h
    site_aspect = (site_w * fig_w_mm) / (site_h * fig_h_mm)
    site = fig.add_axes([site_x, site_y, site_w, site_h], facecolor='white')

    footprints = {
        sc: unary_union(layout.loc[layout.scope_id == sc, 'geom_obj'].tolist()).convex_hull
        for sc in ('2023_f1', '2024_f2', '2024_f1')
    }
    h_max = max(footprints[sc].bounds[3] - footprints[sc].bounds[1] for sc in footprints)
    w_23 = footprints['2023_f1'].bounds[2] - footprints['2023_f1'].bounds[0]
    w_24f2 = footprints['2024_f2'].bounds[2] - footprints['2024_f2'].bounds[0]
    w_24f1 = footprints['2024_f1'].bounds[2] - footprints['2024_f1'].bounds[0]

    gap_col = 1.68
    tx_23 = 0.0
    tx_24f2 = tx_23 + w_23 + gap_col
    tx_24f1 = tx_24f2 + w_24f2 + gap_col

    def _transform_geom(g, b_orig, scale_f, tx, ty):
        g0 = translate(g, xoff=-b_orig[0], yoff=-b_orig[1])
        if scale_f != 1.0:
            g0 = shp_scale(g0, xfact=scale_f, yfact=scale_f, origin=(0.0, 0.0))
        return translate(g0, xoff=tx, yoff=ty)

    placement = [
        ('2023_f1', '2023/2023-08-23_f01_visible_orthomosaic.tif', '2023 F1', tx_23, 'extra_only'),
        ('2024_f2', '2024/2024-08-21_f02_visible_orthomosaic.tif', '2024 F2', tx_24f2, 'none'),
        ('2024_f1', '2024/2024-08-21_f01_visible_orthomosaic.tif', '2024 F1', tx_24f1, 'all_active'),
    ]

    display_hulls = {}
    plot4_disp_geom = None
    extra_2023_pids = {8, 16, 22}

    for sc, rel_tif, tag_lbl, tx, lbl_mode in placement:
        raster_path = ROOT / f'01_Data/raw/remote_sensing/orthomosaics/{rel_tif}'
        b = footprints[sc].bounds
        ty = (h_max - (b[3] - b[1])) / 2.0
        pad = 0.25
        dst_bounds = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
        dst_h, dst_w = 780, 480
        dst_transform = from_bounds(*dst_bounds, dst_w, dst_h)
        dst = np.zeros((4, dst_h, dst_w), dtype=np.uint8)
        with rasterio.open(raster_path) as src:
            reproject(source=rasterio.band(src, [1, 2, 3, 4]), destination=dst, src_transform=src.transform,
                      src_crs=src.crs, dst_transform=dst_transform, dst_crs='EPSG:32644', resampling=Resampling.average)
        inside = geometry_mask([footprints[sc].__geo_interface__], out_shape=(dst_h, dst_w),
                               transform=dst_transform, invert=True)
        dst[3, ~inside] = 0

        extent = (tx - pad, tx - pad + (dst_bounds[2] - dst_bounds[0]),
                  ty - pad, ty - pad + (dst_bounds[3] - dst_bounds[1]))
        site.imshow(np.moveaxis(dst, 0, 2), extent=extent, origin='upper', zorder=1,
                    interpolation='lanczos', clip_on=False)

        geom = _transform_geom(footprints[sc], b, 1.0, tx, ty)
        display_hulls[sc] = geom

        sub_df = layout[layout.scope_id == sc]
        for row in sub_df.itertuples():
            pid = int(row.plot_id)
            g_p = _transform_geom(row.geom_obj, b, 1.0, tx, ty)
            gx, gy = g_p.exterior.xy
            is_unused = int(row.analysis_used) == 0 or pd.isna(row.n_kg_ha)

            if is_unused:
                _draw_unused_cell(site, g_p, zorder=2.6)

            site.plot(np.asarray(gx), np.asarray(gy), color='white', lw=0.48, alpha=0.88, zorder=3.5)

            if lbl_mode == 'extra_only' and pid in extra_2023_pids:
                g_in = g_p.buffer(-0.22)
                if not g_in.is_empty and g_in.geom_type == 'Polygon':
                    site.plot(*g_in.exterior.xy, color='white', lw=0.65, ls='--', alpha=0.92, zorder=4.8)
                center = g_p.representative_point()
                site.text(
                    center.x, center.y, str(int(row.n_kg_ha)),
                    ha='center', va='center', fontsize=7.0, fontweight='bold', color='white',
                    path_effects=[pe.withStroke(linewidth=1.65, foreground='#0F172A')], zorder=7,
                )
            elif lbl_mode == 'all_active' and not is_unused:
                if pid == 4:
                    plot4_disp_geom = g_p
                    site.plot(np.asarray(gx), np.asarray(gy), color='#F59E0B', lw=1.45, zorder=6.5)
                center = g_p.representative_point()
                site.text(
                    center.x, center.y, str(int(row.n_kg_ha)),
                    ha='center', va='center', fontsize=7.0, fontweight='bold', color='white',
                    path_effects=[pe.withStroke(linewidth=1.65, foreground='#0F172A')], zorder=7,
                )

        # Crisp B1 / B2 / B3 horizontal block dividers across every field
        for b_idx in (1, 2):
            upper = unary_union([_transform_geom(g, b, 1.0, tx, ty)
                                 for g in sub_df[sub_df.grid_row.isin([2 * b_idx - 1, 2 * b_idx])].geom_obj])
            lower = unary_union([_transform_geom(g, b, 1.0, tx, ty)
                                 for g in sub_df[sub_df.grid_row.isin([2 * b_idx + 1, 2 * b_idx + 2])].geom_obj])
            y_div = (upper.bounds[1] + lower.bounds[3]) / 2.0
            site.plot([geom.bounds[0] + 0.12, geom.bounds[2] - 0.12], [y_div, y_div],
                      color='#0F172A', lw=1.15, alpha=0.90, zorder=5.2)
            site.plot([geom.bounds[0] + 0.12, geom.bounds[2] - 0.12], [y_div, y_div],
                      color='white', lw=0.65, linestyle=(0, (3.0, 1.8)), zorder=5.6)

        hull = Polygon(np.asarray(geom.exterior.coords), fill=False, edgecolor='#0F172A',
                       linewidth=0.85, zorder=6.0, clip_on=False)
        site.add_patch(hull)
        site.text((geom.bounds[0] + geom.bounds[2]) / 2.0, h_max + 0.65, tag_lbl,
                  ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=ink, zorder=7, clip_on=False)

    # Top season & design grouping rules over 2023 (n=24) vs. 2024 adjacent (n=21; N: kg ha^-1)
    y_grp = h_max + 3.85
    tick_d = 0.55
    x23_l, x23_r = display_hulls['2023_f1'].bounds[0], display_hulls['2023_f1'].bounds[2]
    site.plot([x23_l, x23_l, x23_r, x23_r], [y_grp - tick_d, y_grp, y_grp, y_grp - tick_d],
              color=PALETTE['ochre'], lw=0.85, zorder=6, clip_on=False)
    site.text((x23_l + x23_r) / 2.0, y_grp + 0.38, r'2023 ($n=24$)',
              ha='center', va='bottom', fontsize=7.0, fontweight='bold', color=PALETTE['ochre'], zorder=7, clip_on=False)

    x24_l, x24_r = display_hulls['2024_f2'].bounds[0], display_hulls['2024_f1'].bounds[2]
    site.plot([x24_l, x24_l, x24_r, x24_r], [y_grp - tick_d, y_grp, y_grp, y_grp - tick_d],
              color=PALETTE['teal'], lw=0.85, zorder=6, clip_on=False)
    site.text((x24_l + x24_r) / 2.0, y_grp + 0.38, r'2024 adjacent ($n=21$; N: $\mathrm{kg\ ha^{-1}}$)',
              ha='center', va='bottom', fontsize=7.0, fontweight='bold', color=PALETTE['teal'], zorder=7, clip_on=False)

    # Left B1, B2, B3 architectural brackets aligned across all 3 fields
    b23 = footprints['2023_f1'].bounds
    ty23 = (h_max - (b23[3] - b23[1])) / 2.0
    sub_23 = layout[layout.scope_id == '2023_f1']
    for block in (1, 2, 3):
        u_blk = unary_union([_transform_geom(g, b23, 1.0, tx_23, ty23)
                             for g in sub_23[sub_23.grid_row.isin((2 * block - 1, 2 * block))].geom_obj])
        yb0, yb1 = u_blk.bounds[1] + 0.35, u_blk.bounds[3] - 0.35
        xb = x23_l - 0.55
        site.plot([xb + 0.32, xb, xb, xb + 0.32], [yb0, yb0, yb1, yb1],
                  color=slate, lw=0.75, zorder=6, clip_on=False)
        site.text(xb - 0.28, (yb0 + yb1) / 2.0, f'B{block}',
                  ha='right', va='center', fontsize=7.0, fontweight='bold', color=ink, zorder=7, clip_on=False)

    # Bottom unified 1:1 cartographic bar: 10 m scale bar + North arrow + Unused-cell key
    bar_y = -3.35
    site.plot([0.0, 10.0], [bar_y, bar_y], color=ink, lw=1.20, zorder=6, clip_on=False)
    for xt in (0.0, 10.0):
        site.plot([xt, xt], [bar_y - 0.42, bar_y + 0.42], color=ink, lw=0.80, zorder=6, clip_on=False)
    site.text(10.65, bar_y, '10 m', ha='left', va='center', fontsize=7.0, color=ink, zorder=7, clip_on=False)

    # Crisp two-tone cartographic North needle (using non-bold 'N' so normalize keeps 7 pt)
    nx = 21.4
    ny0, ny1 = bar_y - 1.25, bar_y + 1.35
    site.add_patch(Polygon([[nx, ny1], [nx - 0.72, ny0], [nx, ny0 + 0.45]],
                           closed=True, facecolor=ink, edgecolor=ink, lw=0.45, zorder=6, clip_on=False))
    site.add_patch(Polygon([[nx, ny1], [nx + 0.72, ny0], [nx, ny0 + 0.45]],
                           closed=True, facecolor='white', edgecolor=ink, lw=0.45, zorder=6, clip_on=False))
    site.text(nx + 1.15, bar_y, 'N', ha='left', va='center', fontsize=7.0, fontweight='normal',
              color=ink, zorder=7, clip_on=False)

    # Unused-cell legend swatch on bottom-right
    sw_x, sw_w, sw_h = 30.2, 2.7, 1.85
    sw_poly = ShpPolygon([[sw_x, bar_y - sw_h / 2], [sw_x + sw_w, bar_y - sw_h / 2],
                          [sw_x + sw_w, bar_y + sw_h / 2], [sw_x, bar_y + sw_h / 2]])
    _draw_unused_cell(site, sw_poly, zorder=5.8)
    site.add_patch(Polygon(np.asarray(sw_poly.exterior.coords), closed=True,
                           facecolor='none', edgecolor=slate, lw=0.60, zorder=6.2, clip_on=False))
    site.text(sw_x + sw_w + 0.75, bar_y, 'Unused cell (2024)', ha='left', va='center',
              fontsize=7.0, color=slate, zorder=7, clip_on=False)

    # Lock exact 1:1 isotropic limits so Stage 2 fills [site_x..site_x+site_w, site_y..site_y+site_h]
    x_min_site, x_max_site = -4.15, x24_r + 0.25
    span_x_site = x_max_site - x_min_site
    span_y_site = span_x_site / site_aspect
    y_top_content = y_grp + 2.85
    y_bot_content = bar_y - 1.55
    y_mid_content = (y_top_content + y_bot_content) / 2.0
    site.set_xlim(x_min_site, x_max_site)
    site.set_ylim(y_mid_content - span_y_site / 2.0, y_mid_content + span_y_site / 2.0)
    site.axis('off')

    # --------------------------------------------------------------------------
    # STAGE 3 (PANEL A RIGHT): Single-Plot 0.5 m Buffered ROI + 3.3 mm Canopy Slices
    # Aligned flush with Stage 1/2 [y = row_a_y .. row_a_y + row_a_h] and Panel b [x_right = 0.934]
    # --------------------------------------------------------------------------
    roi_x, roi_w = 0.733, 0.201
    sq_w = 0.0636
    sq_h = sq_w * (fig_w_mm / fig_h_mm)
    sq_gap = (roi_w - 3 * sq_w) / 2.0
    sq_y = row_a_y

    roi_y = sq_y + sq_h + 0.056
    roi_h = (row_a_y + row_a_h) - roi_y
    roi_aspect = (roi_w * fig_w_mm) / (roi_h * fig_h_mm)
    ax_roi = fig.add_axes([roi_x, roi_y, roi_w, roi_h], facecolor='white')

    p4_full = layout[(layout.scope_id == '2024_f1') & (layout.plot_id == 4)].geom_obj.iloc[0]
    p4_inw = inward[(inward.scope_id == '2024_f1') & (inward.plot_id == 4)].geom_obj.iloc[0]
    p3_full = layout[(layout.scope_id == '2024_f1') & (layout.plot_id == 3)].geom_obj.iloc[0]

    bf = p4_full.bounds
    win_h = 5.55
    win_w = win_h * roi_aspect
    rx1 = bf[2] + 0.14
    rx0 = rx1 - win_w
    cy_p4 = (bf[1] + bf[3]) / 2.0
    ry0, ry1 = cy_p4 - win_h / 2.0, cy_p4 + win_h / 2.0

    tif_128 = ROOT / '01_Data/raw/remote_sensing/orthomosaics/2024/2024-08-21_f01_visible_orthomosaic.tif'
    with rasterio.open(tif_128) as src:
        win = win_from_bounds(rx0, ry0, rx1, ry1, src.transform)
        rgb_roi = np.moveaxis(src.read([1, 2, 3], window=win), 0, 2)
    ax_roi.imshow(rgb_roi, extent=(rx0, rx1, ry0, ry1), origin='upper', zorder=1, interpolation='lanczos')

    # Shade 0.5 m excluded perimeter strip between gross plot and inward-buffered ROI
    border_strip = p4_full.difference(p4_inw)
    _draw_geom(ax_roi, border_strip, fc='#0F172A', ec='none', lw=0.0, alpha=0.38, zorder=2)

    ax_roi.plot(*p3_full.exterior.xy, color='white', lw=0.60, ls=':', alpha=0.78, zorder=3)
    ax_roi.plot(*p4_full.exterior.xy, color='white', lw=1.05, ls='-', zorder=4)
    ax_roi.plot(*p4_inw.exterior.xy, color='#FBBF24', lw=1.10, ls='--', zorder=5)

    # Sub-canopy 1.5 m x 1.5 m micro-window inside Plot 4 inward ROI
    bi = p4_inw.bounds
    sub_s = 1.50
    sx0 = (bi[0] + bi[2]) / 2.0 - sub_s / 2.0
    sy0 = (bi[1] + bi[3]) / 2.0 - sub_s / 2.0
    ax_roi.add_patch(Rectangle((sx0, sy0), sub_s, sub_s, facecolor='none',
                               edgecolor='#38BDF8', linewidth=1.15, zorder=6))

    ax_roi.text(
        rx0 + 0.16, ry1 - 0.16, 'Plot 4 (N150)', ha='left', va='top',
        fontsize=7.0, fontweight='bold', color='white', zorder=7,
        bbox=dict(boxstyle='round,pad=0.10', fc='#0F172A', ec='#F59E0B', lw=0.55, alpha=0.84),
    )
    ax_roi.text(
        bi[2] - 0.16, bi[1] + 0.18, '0.5 m ROI', ha='right', va='bottom',
        fontsize=7.0, fontweight='bold', color='#FEF08A', zorder=7,
        bbox=dict(boxstyle='round,pad=0.09', fc='#0F172A', ec='#FBBF24', lw=0.50, alpha=0.82),
    )
    ax_roi.plot([rx0 + 0.22, rx0 + 2.22], [ry0 + 0.32, ry0 + 0.32], color='#0F172A', lw=2.2, zorder=6.8)
    ax_roi.plot([rx0 + 0.22, rx0 + 2.22], [ry0 + 0.32, ry0 + 0.32], color='white', lw=1.2, zorder=7)
    ax_roi.text(rx0 + 1.22, ry0 + 0.45, '2 m', ha='center', va='bottom',
                fontsize=7.0, fontweight='bold', color='white',
                path_effects=[pe.withStroke(linewidth=1.5, foreground='#0F172A')], zorder=7.5)

    ax_roi.set_xlim(rx0, rx1)
    ax_roi.set_ylim(ry0, ry1)
    ax_roi.set_xticks([])
    ax_roi.set_yticks([])
    for sp in ax_roi.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor('#D97706')
        sp.set_linewidth(0.90)

    # Zoom connector lines from Plot 4 in 2024 F1 (Stage 2) to ax_roi (Stage 3)
    if plot4_disp_geom is not None:
        pb = plot4_disp_geom.bounds
        for y_src, y_dst in [(pb[3], ry1), (pb[1], ry0)]:
            con = ConnectionPatch(
                xyA=(pb[2], y_src), coordsA=site.transData,
                xyB=(rx0, y_dst), coordsB=ax_roi.transData,
                linestyle='--', linewidth=0.65, color='#D97706', alpha=0.85, zorder=8,
            )
            fig.add_artist(con)

    stage_specs = [
        ('78 DAS', '2024-07-02_f01_visible_orthomosaic.tif'),
        ('128 DAS', '2024-08-21_f01_visible_orthomosaic.tif'),
        ('181 DAS', '2024-10-13_f01_visible_orthomosaic.tif'),
    ]
    sq_axes = []
    for idx, (st_lbl, tif_name) in enumerate(stage_specs):
        ax_sq = fig.add_axes([roi_x + idx * (sq_w + sq_gap), sq_y, sq_w, sq_h], facecolor='white')
        tif_p = ROOT / f'01_Data/raw/remote_sensing/orthomosaics/2024/{tif_name}'
        with rasterio.open(tif_p) as src:
            w_sq = win_from_bounds(sx0, sy0, sx0 + sub_s, sy0 + sub_s, src.transform)
            arr_sq = np.moveaxis(src.read([1, 2, 3], window=w_sq), 0, 2)
        ax_sq.imshow(arr_sq, extent=(0.0, sub_s, 0.0, sub_s), origin='upper', interpolation='lanczos')
        ax_sq.text(
            0.06, 0.94, st_lbl, transform=ax_sq.transAxes, ha='left', va='top',
            fontsize=7.0, fontweight='bold', color='white', zorder=6,
            bbox=dict(boxstyle='round,pad=0.08', fc='#0F172A', ec='#38BDF8' if idx == 1 else 'white',
                      lw=0.45, alpha=0.82),
        )
        if idx == 2:
            ax_sq.plot([0.74, 1.24], [0.16, 0.16], color='#0F172A', lw=2.20, zorder=4.8)
            ax_sq.plot([0.74, 1.24], [0.16, 0.16], color='white', lw=1.20, zorder=5)
            ax_sq.text(0.99, 0.24, '0.5 m', ha='center', va='bottom',
                       fontsize=7.0, fontweight='bold', color='white',
                       path_effects=[pe.withStroke(linewidth=1.5, foreground='#0F172A')], zorder=5.5)
        ax_sq.set_xlim(0.0, sub_s)
        ax_sq.set_ylim(0.0, sub_s)
        ax_sq.set_xticks([])
        ax_sq.set_yticks([])
        for sp in ax_sq.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor('#0284C7' if idx == 1 else '#64748B')
            sp.set_linewidth(0.85 if idx == 1 else 0.55)
        sq_axes.append(ax_sq)

    for x_src, x_dst in [(sx0, 0.0), (sx0 + sub_s, sub_s)]:
        con_sq = ConnectionPatch(
            xyA=(x_src, sy0), coordsA=ax_roi.transData,
            xyB=(x_dst, sub_s), coordsB=sq_axes[1].transData,
            linestyle='--', linewidth=0.65, color='#0284C7', alpha=0.85, zorder=8,
        )
        fig.add_artist(con_sq)

    # --------------------------------------------------------------------------
    # PANEL B: Full-Width Flight & GDD10 Schedule Aligned Flush [x = 0.056..0.934]
    # --------------------------------------------------------------------------
    ax_c = fig.add_axes([0.056, 0.118, 0.878, 0.258], facecolor='white')
    d_img = pd.read_csv(OLD_MINING / 'direct_image_features.csv').assign(
        domain=lambda x: x.year.astype(str) + '_' + x.field
    )
    flights = d_img[['domain', 'flight_date', 'DAS']].drop_duplicates()
    assert len(flights) == 33

    anchor_specs = [
        (92.0, '92'),
        (106.5, '106/107'),
        (129.0, '128/130'),
        (141.5, '141/142'),
        (181.0, '181'),
    ]
    for x_anc, anc_lbl in anchor_specs:
        ax_c.axvspan(x_anc - 1.0, x_anc + 1.0, fc='#F1F5F9', ec='none', zorder=0.8)
        ax_c.axvline(x_anc, color='#94A3B8', lw=0.60, ls='--', zorder=1.0)
        ax_c.text(x_anc, -0.52, anc_lbl, ha='center', va='bottom', fontsize=7.0,
                  color='#334155', clip_on=False)

    for i, dom in enumerate(DOMAINS):
        z = flights.loc[flights.domain.eq(dom)].sort_values('DAS')
        ax_c.hlines(i, 54, 184, color='#E2E8F0', lw=0.75, zorder=1.2)

        is_common = z.DAS.isin(COMMON[dom])
        is_boll = z.DAS.eq(181)
        z_other = z.loc[~(is_common | is_boll)]
        z_com = z.loc[is_common]
        z_bol = z.loc[181 == z.DAS]

        ax_c.scatter(z_other.DAS, np.repeat(i, len(z_other)), s=24,
                     facecolor='white', edgecolor='#64748B', linewidths=0.85, zorder=3)
        ax_c.scatter(z_com.DAS, np.repeat(i, len(z_com)), s=34,
                     facecolor=PALETTE['cerulean'], edgecolor='white', linewidths=0.55, zorder=5)
        ax_c.scatter(z_bol.DAS, np.repeat(i, len(z_bol)), marker='D', s=28,
                     facecolor=PALETTE['plum'], edgecolor='white', linewidths=0.55, zorder=5)

    ax_c.set_xlim(55, 183.5)
    ax_c.set_ylim(2.46, -0.46)
    ax_c.set_yticks([0, 1, 2])
    ax_c.set_yticklabels(['2023 F1', '2024 F1', '2024 F2'])
    ax_c.set_xticks([60, 80, 100, 120, 140, 160, 180])
    ax_c.xaxis.set_minor_locator(MultipleLocator(10))
    ax_c.set_xlabel('Days after sowing (DAS)', fontsize=7.8, labelpad=2.2)
    style_axis(ax_c)

    weather = pd.read_csv(ROOT / '01_Data/processed/weather/alar_2023_2024_daily_weather.csv')
    ax_gdd = ax_c.twinx()
    for yr, sow_str, col_w, ls_w, lbl_w in [
        (2023, '2023-04-15', PALETTE['ochre'], '-', r'2023 $\mathrm{GDD}_{10}$'),
        (2024, '2024-04-15', PALETTE['teal'], '--', r'2024 $\mathrm{GDD}_{10}$'),
    ]:
        sub_w = weather[weather.year == yr].copy()
        sub_w['date'] = pd.to_datetime(sub_w['date'])
        sub_w['DAS'] = (sub_w['date'] - pd.to_datetime(sow_str)).dt.days
        sub_w = sub_w[(sub_w.DAS >= 55) & (sub_w.DAS <= 183)].sort_values('DAS')
        ax_gdd.plot(sub_w.DAS, sub_w.cum_GDD, color=col_w, lw=1.15, ls=ls_w, alpha=0.88, zorder=2.2, label=lbl_w)

    ax_gdd.set_ylim(550, 3100)
    ax_gdd.set_yticks([800, 1400, 2000, 2600])
    ax_gdd.set_ylabel(r'Cumulative $\mathrm{GDD}_{10}$ (°C d)', fontsize=7.6, color='#334155', labelpad=2.8)
    ax_gdd.tick_params(axis='y', which='major', length=2.4, width=0.6, pad=1.8, color=SLATE, labelsize=7.0)
    ax_gdd.spines['right'].set_visible(True)
    ax_gdd.spines['right'].set_color(SLATE)
    ax_gdd.spines['right'].set_linewidth(0.6)
    ax_gdd.spines['top'].set_visible(False)

    leg_c_handles = [
        Line2D([], [], marker='o', ls='', ms=4.2, mfc='white', mec='#64748B', mew=0.85, label='UAV flight'),
        Line2D([], [], marker='o', ls='', ms=4.6, mfc=PALETTE['cerulean'], mec='white', mew=0.55, label='Shared stage'),
        Line2D([], [], marker='D', ls='', ms=4.0, mfc=PALETTE['plum'], mec='white', mew=0.55, label='Boll opening (181 DAS)'),
        Line2D([], [], color=PALETTE['ochre'], lw=1.15, ls='-', label=r'2023 $\mathrm{GDD}_{10}$'),
        Line2D([], [], color=PALETTE['teal'], lw=1.15, ls='--', label=r'2024 $\mathrm{GDD}_{10}$'),
    ]
    fig.legend(handles=leg_c_handles, loc='upper center', bbox_to_anchor=(0.496, 0.048),
               ncol=5, columnspacing=1.30, handletextpad=0.35, handlelength=1.35, fontsize=7.0)

    save_fig(fig, MAIN_STEMS[0])



# ==============================================================================
# FIGURE 2 (a-e): Scheme 1 (RSE / ISPRS Three-Tier Diagnostic & Decoupling Layout)
#   a: Three-environment canopy cover trajectories (92-142 DAS)
#   b: Three-environment quadratic N-yield response curves with 95% CI & R2
#   c: 3.3 mm pixel-level physical decoupling (Raw RGB, Pure-leaf NGRDI,
#      Contiguous gaps >=0.04 m2) + 1D cross-row cover oscillation profile
#   d: Clean 2D yield-limiting attribution plane (I_Leaf vs I_Gap, R2=0.67) + colorbar
#   e: Apparent r vs Within-(N, block) partial r bar comparison + Simpson reversal
# ==============================================================================
def render_figure2() -> None:
    from matplotlib.patches import Patch

    fs.apply()
    fig_w_mm, fig_h_mm = 180.0, 142.0
    # Figure setup with internal helper functions for coordinate normalization
    def _y(y_old: float) -> float:
        return y_old
    def _h(h_old: float) -> float:
        return h_old

    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4), facecolor='white')

    x_left, x_right = 0.080, 0.978
    total_w = x_right - x_left

    # --- Top shared N-rate legend ---
    n_handles = [
        Line2D([], [], color='none', label=r'N rate ($\mathrm{kg\ N\ ha^{-1}}$):')
    ] + [
        Line2D([], [], color=N_COLORS[n], marker='o', ms=3.8, lw=1.05,
               mfc=N_COLORS[n], mec='#0F172A', mew=0.35, label=f'{n}')
        for n in N_RATES_ALL
    ]
    fig.legend(handles=n_handles, loc='upper center', bbox_to_anchor=((x_left + x_right) / 2, 0.992),
               ncol=8, frameon=False, columnspacing=0.85, handlelength=1.15, handletextpad=0.25, fontsize=7.0)

    # --- Row 1 (a) & Row 2 (b): Canopy Cover Trajectories & Quadratic N-Yield Curves ---
    slots = pd.read_csv(Q1_DIR / 'millimeter_slot_features_699.csv').rename(columns={'N_kg_ha': 'N', 'f_cover': 'canopy_cover'})
    common_plots = pd.read_csv(OLD_MINING / 'common_plot_features_common_DAS.csv').rename(columns={'N_kg_ha': 'N', 'yield_kg_ha': 'yield'})
    quad_ci = pd.read_csv(Q1_DIR / 'rich_fig2_quad_ci.csv')
    keep = common_plots[['domain', 'plot_id']].drop_duplicates()
    slots = slots.merge(keep, on=['domain', 'plot_id'])

    col_gap_ab = 0.014
    panel_w_ab = (total_w - 2 * col_gap_ab) / 3
    y_bot_a, h_a = 0.788, 0.154
    y_bot_b, h_b = 0.590, 0.150

    fig.text(0.016, y_bot_a + h_a + 0.016, 'a', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')
    fig.text(0.016, y_bot_b + h_b + 0.014, 'b', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')

    for i, dom in enumerate(DOMAINS):
        xb = x_left + i * (panel_w_ab + col_gap_ab)
        ax_a = fig.add_axes([xb, y_bot_a, panel_w_ab, h_a], facecolor='white')
        sub_s = slots[(slots.domain == dom) & (slots.DAS < 180)]
        grp = sub_s.groupby(['DAS', 'N'])['canopy_cover'].agg(['mean', 'std']).reset_index()

        ax_a.axvspan(92, 142, facecolor='#F1F5F9', edgecolor='none', zorder=0.2)
        for c_das in COMMON[dom][:4]:
            ax_a.axvline(c_das, color='#CBD5E1', lw=0.45, ls=':', zorder=0.4)
        ax_a.grid(axis='both', lw=0.40, color='#F1F5F9', zorder=0)
        for n_val in N_RATES_ALL:
            g = grp[grp.N == n_val].sort_values('DAS')
            if len(g) == 0:
                continue
            xd = g['DAS'].to_numpy(float)
            ym = g['mean'].to_numpy(float) * 100.0
            ys = g['std'].fillna(0.0).to_numpy(float) * 100.0
            if n_val in (0, 300):
                ax_a.fill_between(xd, ym - ys, ym + ys, color=N_COLORS[n_val], alpha=0.14, lw=0, zorder=1)
            ax_a.plot(xd, ym, '-o', ms=2.4, lw=0.92,
                      color=N_COLORS[n_val], mfc=N_COLORS[n_val], mec='#0F172A', mew=0.25, zorder=3 + n_val / 500.0)

        if dom == '2023_F1':
            ax_a.set_xlim(83, 144)
            ax_a.set_xticks([90, 105, 120, 135])
        elif dom == '2024_F1':
            ax_a.set_xlim(55, 159)
            ax_a.set_xticks([60, 90, 120, 150])
        else:
            ax_a.set_xlim(60, 159)
            ax_a.set_xticks([60, 90, 120, 150])
        ax_a.set_ylim(12, 102)
        ax_a.set_yticks([20, 40, 60, 80, 100])
        ax_a.text(0.04, 0.10, fs.ENV_LABELS[dom], transform=ax_a.transAxes,
                  ha='left', va='bottom', fontsize=7.3, fontweight='bold', color=INK,
                  bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.88))
        style_axis(ax_a)
        if i == 0:
            ax_a.set_ylabel('Canopy cover (%)', fontsize=7.5, labelpad=2.5)
        else:
            ax_a.set_yticklabels([''] * 5)
        if i == 1:
            ax_a.set_xlabel('Days after sowing (DAS)', fontsize=7.5, labelpad=2.0)

        ax_b = fig.add_axes([xb, y_bot_b, panel_w_ab, h_b], facecolor='white')
        sub_p = common_plots[common_plots.domain == dom].copy()
        q_sub = quad_ci[quad_ci.domain == dom].sort_values('N')

        ax_b.grid(axis='both', lw=0.40, color='#F1F5F9', zorder=0)
        ax_b.fill_between(q_sub.N, q_sub.low_95_mg, q_sub.high_95_mg, color='#CBD5E1', alpha=0.42, lw=0, zorder=1)
        n_opt_v = float(q_sub.n_opt.iloc[0])
        ax_b.axvline(n_opt_v, color='#475569', lw=0.80, ls='--', zorder=2.0)
        ax_b.plot(q_sub.N, q_sub.mean_mg, color='#0F172A', lw=1.25, zorder=2.5)

        r2_v = float(q_sub.r2_quad.iloc[0])
        ax_b.text(0.05, 0.92, rf'$R^2 = {r2_v:.2f}$', transform=ax_b.transAxes,
                  ha='left', va='top', fontsize=7.1, color=INK,
                  bbox=dict(boxstyle='round,pad=0.10', facecolor='white', edgecolor='none', alpha=0.85))

        rng = np.random.default_rng(42 + i)
        xj = sub_p['N'].to_numpy(float) + rng.uniform(-6.5, 6.5, size=len(sub_p))
        yv = sub_p['yield'].to_numpy(float) / 1000.0
        for xv_i, yv_i, nv_i in zip(xj, yv, sub_p['N'].to_numpy(int)):
            ax_b.scatter([xv_i], [yv_i], s=18, facecolor=N_COLORS[nv_i], edgecolor='#0F172A', linewidth=0.40, zorder=3.5)

        ax_b.set_xlim(-15, 465)
        ax_b.set_ylim(2.7, 9.0)
        ax_b.set_xticks([0, 150, 300, 450])
        ax_b.set_yticks([3, 5, 7, 9])
        style_axis(ax_b)
        if i == 0:
            ax_b.set_ylabel(r'Yield ($\mathrm{Mg\ ha^{-1}}$)', fontsize=7.5, labelpad=2.8)
        else:
            ax_b.set_yticklabels([''] * 4)
        if i == 1:
            ax_b.set_xlabel(r'N rate ($\mathrm{kg\ N\ ha^{-1}}$)', fontsize=7.5, labelpad=2.0)

    # --- Row 3 (c): 3.3 mm Pixel-Level Physical Decoupling (3 Maps) + 1D Cross-Row Transect ---
    rich_npz = np.load(Q1_DIR / 'rich_figure_assets.npz')
    map_y, map_h = 0.352, 0.152
    fig.text(0.016, map_y + map_h + 0.018, 'c', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')

    # Aspect-preserving width for 1.80 m x 1.50 m (545 x 455 px)
    map_w = map_h * (fig_h_mm / fig_w_mm) * (545.0 / 455.0)
    map_gap = 0.010

    ax_c1 = fig.add_axes([x_left, map_y, map_w, map_h], facecolor='white')
    ax_c2 = fig.add_axes([x_left + (map_w + map_gap), map_y, map_w, map_h], facecolor='white')
    ax_c3 = fig.add_axes([x_left + 2 * (map_w + map_gap), map_y, map_w, map_h], facecolor='white')

    rgb_anat = rich_npz['f2_anat_rgb']
    ax_c1.imshow(rgb_anat, extent=(0.0, 1.80, 0.0, 1.50), origin='upper', interpolation='lanczos')
    ax_c1.text(0.50, 1.04, '3.3 mm RGB', transform=ax_c1.transAxes,
               ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=INK, clip_on=False)
    # 0.5 m scale bar in bottom-right of Raw RGB
    ax_c1.plot([1.15, 1.65], [0.14, 0.14], color='#0F172A', lw=2.3, zorder=5)
    ax_c1.plot([1.15, 1.65], [0.14, 0.14], color='white', lw=1.25, zorder=6)
    ax_c1.text(1.40, 0.21, '0.5 m', ha='center', va='bottom', fontsize=7.0, fontweight='bold', color='white',
               path_effects=[pe.withStroke(linewidth=1.5, foreground='#0F172A')], zorder=7)

    ng_img = rich_npz['f2_anat_ngrdi'].copy()
    cmap_ng = Q1_LEAF_CMAP.copy()
    cmap_ng.set_bad('#1E293B')
    im_ng = ax_c2.imshow(ng_img, cmap=cmap_ng, vmin=0.0, vmax=0.28, extent=(0.0, 1.80, 0.0, 1.50), origin='upper')
    ax_c2.text(0.50, 1.04, 'Pure-leaf NGRDI', transform=ax_c2.transAxes,
               ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=INK, clip_on=False)

    cax_ng = fig.add_axes([x_left + (map_w + map_gap) + 0.018, map_y - 0.013, map_w - 0.048, 0.0062])
    cb_ng = fig.colorbar(im_ng, cax=cax_ng, orientation='horizontal', ticks=[0.0, 0.28])
    cb_ng.ax.set_xticklabels(['0.00', '0.28'])
    cb_ng.ax.tick_params(labelsize=7.0, length=1.5, pad=1.0, color=SLATE)
    cb_ng.outline.set_linewidth(0.45)

    cls_cmap = ListedColormap([PALETTE['teal'], PALETTE['slate_mid'], PALETTE['amber']])
    ax_c3.imshow(rich_npz['f2_anat_gap_cls'], cmap=cls_cmap, vmin=0, vmax=2,
                 extent=(0.0, 1.80, 0.0, 1.50), origin='upper', interpolation='nearest')
    ax_c3.text(0.50, 1.04, 'Contiguous stand gaps', transform=ax_c3.transAxes,
               ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=INK, clip_on=False)

    gap_leg = [
        Patch(facecolor=PALETTE['teal'], edgecolor='none', label='Canopy'),
        Patch(facecolor=PALETTE['slate_mid'], edgecolor='none', label=r'$<0.04\ \mathrm{m^2}$'),
        Patch(facecolor=PALETTE['amber'], edgecolor='none', label=r'$\geq 0.04\ \mathrm{m^2}$'),
    ]
    ax_c3.legend(handles=gap_leg, loc='upper center', bbox_to_anchor=(0.585, -0.025),
                 ncol=3, frameon=False, columnspacing=0.28, handlelength=0.58, handletextpad=0.14, fontsize=7.0)

    for ax_m in (ax_c1, ax_c2, ax_c3):
        ax_m.set_xlim(0.0, 1.80)
        ax_m.set_ylim(0.0, 1.50)
        ax_m.set_xticks([])
        ax_m.set_yticks([])
        for sp in ax_m.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor('#475569')
            sp.set_linewidth(0.55)

    # Right of Row 3 (c): Comparative 1D Cross-Row Closure Transect (Matched N150: Intact Q1 Plot 19 vs. Gap-impaired Plot 4)
    import geopandas as gpd
    import rasterio
    if str(ROOT / '02_Code/src/data') not in sys.path:
        sys.path.insert(0, str(ROOT / '02_Code/src/data'))
    from extract_raw_rgb_information import crop as _crop_roi, rules as _canopeo_rules

    x_dp = 0.606
    w_dp = x_right - x_dp
    ax_dp = fig.add_axes([x_dp, map_y, w_dp, map_h], facecolor='white')
    px = rich_npz['f2_anat_prof_x_m']
    px_m_c = float(rich_npz['f2_anat_meta'][2])
    sm_k = max(3, int(round(0.05 / px_m_c)))  # 5 cm running-mean smoothing per Methods 2.3
    gap_cls_arr = rich_npz['f2_anat_gap_cls']
    py_gap = ndimage.uniform_filter1d((gap_cls_arr == 0).mean(axis=0) * 100.0, size=sm_k, mode='nearest')

    # Extract matched 1.80 m x 1.50 m center window of intact Q1 N150 reference (2024 F1, Plot 19, N150, 128 DAS)
    _slots_f2 = pd.read_csv(ROOT / '01_Data/processed/model_ready/cotton_rgb_formal_714_slots.csv')
    _s19 = _slots_f2[(_slots_f2.year == 2024) & (_slots_f2.field == 'F1') & (_slots_f2.plot_id == 19) & (_slots_f2.DAS == 128)].iloc[0]
    _gdf_inw = gpd.read_file(ROOT / '01_Data/processed/spatial/cotton_rgb/field_authority.gpkg', layer='inward_buffer_plot')
    with rasterio.open(_s19.raster_path) as _src19:
        _gg19 = _gdf_inw.to_crs(_src19.crs)
        _geom19 = _gg19[(_gg19.year == 2024) & (_gg19.field == 'F1') & (_gg19.plot_id == 19)].geometry.iloc[0]
        _rgb19, _val19 = _crop_roi(_src19, _geom19)
    _grn19 = _canopeo_rules(_rgb19, _val19)['green_canopeo']
    _h_w, _w_w = gap_cls_arr.shape[0], gap_cls_arr.shape[1]
    _cy19, _cx19 = _val19.shape[0] // 2 - _h_w // 2, _val19.shape[1] // 2 - _w_w // 2
    py_intact = ndimage.uniform_filter1d(
        _grn19[_cy19:_cy19 + _h_w, _cx19:_cx19 + _w_w].mean(axis=0) * 100.0, size=sm_k, mode='nearest'
    )

    ax_dp.grid(axis='y', lw=0.40, color='#F1F5F9', zorder=0)
    ax_dp.fill_between(px, py_gap, np.maximum(py_intact, py_gap), color=PALETTE['amber'], alpha=0.16, lw=0, zorder=1.5)
    ax_dp.axhline(float(py_intact.mean()), color=PALETTE['teal'], lw=0.65, ls='--', alpha=0.72, zorder=2.0)
    ax_dp.axhline(float(py_gap.mean()), color=PALETTE['terracotta'], lw=0.65, ls='--', alpha=0.72, zorder=2.0)
    ax_dp.plot(px, py_intact, color=PALETTE['teal'], lw=1.20, zorder=3.4)
    ax_dp.plot(px, py_gap, color=PALETTE['terracotta'], lw=1.20, zorder=3.8)
    _idx_min = int(np.argmin(py_gap))
    ax_dp.plot(px[_idx_min], py_gap[_idx_min], marker='o', ms=2.6, mfc=PALETTE['amber'], mec='white', mew=0.45, zorder=4.4)

    dp_leg = [
        Line2D([0], [0], color=PALETTE['teal'], lw=1.25,
               label=f'Intact canopy (SD = {float(np.std(py_intact)):.1f}%)'),
        Line2D([0], [0], color=PALETTE['terracotta'], lw=1.25,
               label=f'Gap-impaired (SD = {float(np.std(py_gap)):.1f}%)'),
    ]
    ax_dp.legend(handles=dp_leg, loc='lower center', bbox_to_anchor=(0.50, 1.02),
                 ncol=2, frameon=False, borderpad=0.0, columnspacing=0.65,
                 handlelength=0.95, handletextpad=0.28, fontsize=7.0)

    ax_dp.set_xlim(0.0, 1.80)
    ax_dp.set_ylim(30, 100)
    ax_dp.set_xticks([0.0, 0.6, 1.2, 1.8])
    ax_dp.set_yticks([40, 60, 80, 100])
    ax_dp.set_xlabel('Cross-row distance (m)', fontsize=7.5, labelpad=2.0)
    ax_dp.set_ylabel('Canopy cover (%)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_dp)

    # --- Row 4 (d & e): 2D Yield-Limiting Attribution Plane (d) + Confounding Correlation Shift (e) ---
    y_bot_de, h_de = 0.056, 0.210
    fig.text(0.016, y_bot_de + h_de + 0.022, 'd', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')
    fig.text(0.500, y_bot_de + h_de + 0.022, 'e', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')

    # Panel d: Clean White-Background 2D Attribution Plane + Colorbar
    q_df = pd.read_csv(Q1_DIR / 'direction_b_2d_attribution_quadrants.csv')
    w_d = 0.398
    ax_e2d = fig.add_axes([x_left, y_bot_de, w_d, h_de], facecolor='white')
    ax_e2d.grid(axis='both', lw=0.35, color='#F1F5F9', zorder=0)
    ax_e2d.axhline(0.0, color='#64748B', lw=0.75, ls='--', zorder=1.5)
    ax_e2d.axvline(0.0, color='#64748B', lw=0.75, ls='--', zorder=1.5)

    b_leaf_mg = float(q_df['beta_leaf_kg_ha'].iloc[0]) / 1000.0
    b_gap_mg = float(q_df['beta_gap_kg_ha'].iloc[0]) / 1000.0
    xg_cont = np.linspace(-1.86, 1.54, 40)
    for iso_val in [-1.0, 0.0, 1.0]:
        yg_cont = (iso_val - b_leaf_mg * xg_cont) / b_gap_mg
        ax_e2d.plot(xg_cont, yg_cont, color='#94A3B8', lw=0.60, ls=':', alpha=0.85, zorder=1.2)

    norm_anom = TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=1.5)
    sc_2d = ax_e2d.scatter(
        q_df['I_Leaf_Phys'], q_df['I_Gap_Struct'],
        c=q_df['yield_blk_anom'] / 1000.0, cmap=Q1_DIV_CMAP, norm=norm_anom,
        s=22, edgecolor='#0F172A', linewidth=0.45, alpha=0.94, zorder=3.5,
    )

    # Quadrant labels with mean block-adjusted yield anomaly (Mg ha^-1), computed dynamically from CSV
    q_means = (q_df.groupby('quadrant')['yield_blk_anom'].mean() / 1000.0).to_dict()
    def _fmt_anom(v: float) -> str:
        return f'+{v:.2f}' if v >= 0 else f'−{abs(v):.2f}'
    ax_e2d.text(1.48, -1.91, f"Q1: Intact ({_fmt_anom(q_means['Q1_Optimal_Intact'])})", ha='right', va='bottom', fontsize=7.0, fontweight='bold', color='#0F172A')
    ax_e2d.text(1.48, 2.95, f"Q2: Gap-limited ({_fmt_anom(q_means['Q2_Stand_Gap_Limited'])})", ha='right', va='top', fontsize=7.0, fontweight='bold', color='#334155')
    ax_e2d.text(-1.80, -1.91, f"Q3: Leaf-limited ({_fmt_anom(q_means['Q3_N_Senescence_Limited'])})", ha='left', va='bottom', fontsize=7.0, fontweight='bold', color='#334155')
    ax_e2d.text(-1.80, 2.95, f"Q4: Dual stress ({_fmt_anom(q_means['Q4_Dual_Stressed'])})", ha='left', va='top', fontsize=7.0, fontweight='bold', color='#0F172A')

    ax_e2d.set_xlim(-1.86, 1.54)
    ax_e2d.set_ylim(-2.00, 3.05)
    ax_e2d.set_xticks([-1.5, 0.0, 1.5])
    ax_e2d.set_yticks([-1.0, 0.0, 1.0, 2.0])
    ax_e2d.set_xlabel(r'Pure-leaf index, $I_{\mathrm{Leaf}}$ ($z$)', fontsize=7.5, labelpad=2.0)
    ax_e2d.set_ylabel(r'Stand-gap index, $I_{\mathrm{Gap}}$ ($z$)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_e2d)

    # Header & Colorbar for Panel d yield anomaly above ax_e2d (dynamic 9-fold block LOBO CV R^2)
    r2_2d = float(q_df['r2_2d_plane'].iloc[0])
    _X2d = np.column_stack([np.ones(len(q_df)), q_df['I_Leaf_Phys'].to_numpy(float), q_df['I_Gap_Struct'].to_numpy(float)])
    _y2d = q_df['yield_blk_anom'].to_numpy(float)
    _st9 = (q_df['domain'].astype(str) + '_B' + q_df['block_design'].astype(str)).to_numpy()
    _oof9 = np.zeros(len(q_df), dtype=float)
    for _sb in np.unique(_st9):
        _tr, _te = _st9 != _sb, _st9 == _sb
        _oof9[_te] = _X2d[_te] @ np.linalg.lstsq(_X2d[_tr], _y2d[_tr], rcond=None)[0]
    r2_2d_cv = float(1.0 - np.sum((_y2d - _oof9) ** 2) / np.sum((_y2d - _y2d.mean()) ** 2))
    fig.text(x_left, y_bot_de + h_de + 0.011, rf'$R^2 = {r2_2d:.2f}$ (CV $= {r2_2d_cv:.2f}$)',
             ha='left', va='center', fontsize=7.0, color=INK)
    cax_2d = fig.add_axes([x_left + w_d - 0.105, y_bot_de + h_de + 0.0075, 0.105, 0.0065])
    cb_2d = fig.colorbar(sc_2d, cax=cax_2d, orientation='horizontal', ticks=[-2.0, 0.0, 1.5])
    cb_2d.ax.xaxis.set_ticks_position('top')
    cb_2d.ax.set_xticklabels(['−2.0', '0.0', '+1.5'])
    cb_2d.ax.tick_params(labelsize=7.0, length=1.3, pad=0.7, color=SLATE)
    cb_2d.outline.set_linewidth(0.45)
    fig.text(x_left + w_d - 0.116, y_bot_de + h_de + 0.011,
             r'$\Delta Y_{\mathrm{blk}}\ (\mathrm{Mg\ ha^{-1}})$:', ha='right', va='center', fontsize=7.0, color=INK)

    # Panel e: Diverging Bar (Apparent r vs Partial r | N, block) Confounding Shift locked to [x=0.592..0.974]
    assoc = pd.read_csv(Q1_DIR / 'management_confounding_associations.csv')
    order_feats = [
        ('T_auc_mean', 'Integrated cover'),
        ('C_cover', 'Canopy cover'),
        ('C_exg', 'Mixed ExG'),
        ('Cearly_ngrdi_leaf', 'Pure-leaf NGRDI'),
        ('T_gap_auc', 'Integrated gaps'),
        ('Searly_gap_large', r'Gaps $\geq 0.04\ \mathrm{m^2}$'),
        ('Searly_prof_across', 'Row-closure SD'),
        ('white_b140_s30_raw', 'Raw white area'),
    ]
    sub_a = assoc.set_index('feature').reindex([k for k, _ in order_feats])

    x_e0 = 0.592
    w_e = x_right - x_e0
    ax_e = fig.add_axes([x_e0, y_bot_de, w_e, h_de], facecolor='white')
    ypos = np.arange(len(order_feats), dtype=float)
    ax_e.grid(axis='x', lw=0.40, color='#F1F5F9', zorder=0.5)
    ax_e.axhline(3.5, color='#CBD5E1', lw=0.60, ls=':', zorder=1)
    ax_e.axhline(6.5, color='#CBD5E1', lw=0.60, ls=':', zorder=1)
    ax_e.axvline(0.0, color='#334155', lw=0.80, ls='--', zorder=2)

    r_app = sub_a['within_env_apparent_r'].to_numpy(float)
    r_par = sub_a['within_N_block_partial_r'].to_numpy(float)

    ax_e.barh(ypos - 0.17, r_app, height=0.32, facecolor='#CBD5E1', edgecolor='#64748B', linewidth=0.55,
              alpha=0.90, label=r'Apparent $r$', zorder=2.5)
    ax_e.barh(ypos + 0.17, r_par, height=0.32, facecolor='#0F766E', edgecolor='#0F172A', linewidth=0.55,
              alpha=0.92, label=r'Partial $r$ (N, block)', zorder=3.5)

    ax_e.set_xlim(-0.77, 0.79)
    ax_e.set_ylim(len(order_feats) - 0.42, -0.55)
    ax_e.set_xticks([-0.6, -0.3, 0.0, 0.3, 0.6])
    ax_e.set_yticks(ypos)
    ax_e.set_yticklabels([lbl for _, lbl in order_feats], fontsize=7.0)
    ax_e.set_xlabel(r'Correlation with yield ($r$)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_e)
    ax_e.legend(loc='upper center', bbox_to_anchor=(0.50, 1.13), ncol=2, frameon=False,
                columnspacing=0.90, handlelength=1.0, handletextpad=0.30, fontsize=7.0)

    save_fig(fig, MAIN_STEMS[1])


# ==============================================================================
# FIGURE 3 (a-d): Scheme 1 (RSE / ISPRS Aligned Chromatic Decoupling + True 1:1
#                 Square Parity Plots + Restrained Diverging Matrix & Forest Plot)
# ==============================================================================
def render_figure3() -> None:
    fs.apply()
    fig_w_mm, fig_h_mm = 180.0, 148.0
    # Figure setup with internal helper functions for coordinate normalization
    def _y(y_old: float) -> float:
        return y_old
    def _h(h_old: float) -> float:
        return h_old

    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4), facecolor='white')

    # --- Row 1: Pixel-Level Foreground/BG KDE (a) + 1x3 True 1:1 Square Yield Parity Plots (b) ---
    xb0, xb_right = 0.436, 0.980
    gap_b = 0.011
    wb = (xb_right - xb0 - 2 * gap_b) / 3
    h1 = wb * (fig_w_mm / fig_h_mm)        # exact 1:1 geometric square
    y_bot1 = 0.706
    y_top1 = y_bot1 + h1

    fig.text(0.016, 0.980, 'a', fontsize=9, fontweight='bold', color='#111111', va='top', ha='left')
    fig.text(0.400, 0.980, 'b', fontsize=9, fontweight='bold', color='#111111', va='top', ha='left')

    # Panel a: Stacked ExG & NGRDI foreground vs background density curves with locked x-axis
    kde_df = pd.read_csv(Q1_DIR / 'rich_fig3_pixel_chroma_density.csv')
    gap_a = 0.012
    sub_h_a = (h1 - gap_a) / 2
    ax_a1 = fig.add_axes([0.074, y_bot1 + sub_h_a + gap_a, 0.312, sub_h_a], facecolor='white')
    ax_a2 = fig.add_axes([0.074, y_bot1, 0.312, sub_h_a], facecolor='white')

    dom_cols = ENV_COLORS
    for d in DOMAINS:
        c_d = dom_cols[d]
        for cls_k, ls_k, lw_k, fill_a in [('Mulch_Soil_BG', '--', 0.90, 0.0), ('Pure_Leaf', '-', 1.15, 0.14)]:
            s_sub = kde_df[(kde_df.domain == d) & (kde_df.pixel_class == cls_k)].sort_values('bin_center')
            ax_a1.plot(s_sub.bin_center, s_sub.exg_density, color=c_d, ls=ls_k, lw=lw_k)
            if fill_a > 0:
                ax_a1.fill_between(s_sub.bin_center, 0, s_sub.exg_density, color=c_d, alpha=fill_a, lw=0)

            ax_a2.plot(s_sub.bin_center, s_sub.ngrdi_density, color=c_d, ls=ls_k, lw=lw_k)
            if fill_a > 0:
                ax_a2.fill_between(s_sub.bin_center, 0, s_sub.ngrdi_density, color=c_d, alpha=fill_a, lw=0)

    # Lock both ax_a1 and ax_a2 to the exact same xlim so 0.0 reference and gridlines align 100%
    for ax_a, tag_a in [(ax_a1, 'ExG'), (ax_a2, 'NGRDI')]:
        ax_a.set_axisbelow(True)
        ax_a.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
        ax_a.axvline(0.0, color='#475569', lw=0.80, ls='--', zorder=2.8)
        ax_a.set_xlim(-0.22, 0.60)
        ax_a.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6])
        ax_a.text(0.965, 0.84, tag_a, transform=ax_a.transAxes,
                  ha='right', va='top', fontsize=7.5, fontweight='bold', color=INK)
        style_axis(ax_a)
    ax_a1.set_xticklabels([''] * 5)
    ax_a2.set_ylabel('Probability density', fontsize=7.5, labelpad=2.8)
    ax_a2.yaxis.set_label_coords(-0.110, (h1 / 2.0) / sub_h_a)
    ax_a2.set_xlabel('Chromatic index', fontsize=7.5, labelpad=2.0)

    # Column-major order for ncol=3 so Row 1 = [2023 F1, 2024 F1, 2024 F2] and Row 2 = [Pure leaf, Mulch/soil]
    a_handles = [
        Line2D([], [], color=dom_cols['2023_F1'], lw=1.15, ls='-', label='2023 F1'),
        Line2D([], [], color='#0F172A', lw=1.15, ls='-', label='Pure leaf'),
        Line2D([], [], color=dom_cols['2024_F1'], lw=1.15, ls='-', label='2024 F1'),
        Line2D([], [], color='#475569', lw=0.95, ls='--', label='Mulch/soil'),
        Line2D([], [], color=dom_cols['2024_F2'], lw=1.15, ls='-', label='2024 F2'),
    ]
    ax_a1.legend(handles=a_handles, loc='lower center', bbox_to_anchor=(0.50, 1.04),
                 ncol=3, frameon=False, fontsize=7.0, columnspacing=0.55, handlelength=1.00, handletextpad=0.22)

    # Panel b: 1 x 3 True 1:1 Square Observed vs. Predicted Yield Parity Progression (Adjacent-field k=0 true zero-label)
    pf_pred = pd.read_csv(Q1_DIR / 'direction_b_prior_free_predictions.csv').query("evaluation == 'adjacent_field'")
    pf_sum_adj = pd.read_csv(Q1_DIR / 'direction_b_prior_free_summary.csv').query("evaluation == 'adjacent_field'")

    parity_panels = [
        ('RGB_Mixed_ExG', 'RGB-only: Mixed ExG', pf_pred[pf_pred.specification == 'RGB_Mixed_ExG']),
        ('RGB_Decoupled_Bmorph', r'RGB-only: Dec.$+B_{\mathrm{morph}}$', pf_pred[pf_pred.specification == 'RGB_Decoupled_Bmorph']),
        ('Integrated_N_Leaf_Bmorph', r'$N{+}\mathrm{Leaf}{+}T{+}B_{\mathrm{morph}}$', pf_pred[pf_pred.specification == 'Integrated_N_Leaf_Bmorph']),
    ]
    grid_xy = np.linspace(2.8, 9.0, 60)
    tgt_markers = {'2024_F1': 'o', '2024_F2': '^'}
    for j, (sp_k, p_tag, sub_p) in enumerate(parity_panels):
        ax_p = fig.add_axes([xb0 + j * (wb + gap_b), y_bot1, wb, h1], facecolor='white')
        xo = sub_p['observed'].to_numpy(float) / 1000.0
        yp = sub_p['pred_k0'].to_numpy(float) / 1000.0
        nv = sub_p['N'].to_numpy(int)
        tgts = sub_p['target'].to_numpy(str)

        # Exact 2-task adjacent-field zero-label (k = 0) mean across all target plots
        s_row = pf_sum_adj[pf_sum_adj.specification == sp_k]
        r2_v = float(s_row['R2_k0_all'].mean())
        rmse_v = float(s_row['RMSE_k0_all'].mean()) / 1000.0

        ax_p.set_axisbelow(True)
        ax_p.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
        ax_p.fill_between(grid_xy, 0.90 * grid_xy, 1.10 * grid_xy, color='#94A3B8', alpha=0.15, lw=0, zorder=1.2)
        ax_p.plot(grid_xy, 0.90 * grid_xy, color='#94A3B8', lw=0.50, ls=':', zorder=1.5)
        ax_p.plot(grid_xy, 1.10 * grid_xy, color='#94A3B8', lw=0.50, ls=':', zorder=1.5)
        ax_p.plot([2.8, 9.0], [2.8, 9.0], color='#334155', lw=0.85, ls='--', zorder=2.2)

        for xo_i, yp_i, nv_i, tg_i in zip(xo, yp, nv, tgts):
            ax_p.scatter([xo_i], [yp_i], s=19, marker=tgt_markers.get(tg_i, 'o'),
                         facecolor=N_COLORS.get(int(nv_i), '#0284C7'),
                         edgecolor='#0F172A', linewidth=0.40, alpha=0.92, zorder=3.5)

        ax_p.set_xlim(3.15, 8.68)
        ax_p.set_ylim(3.15, 8.68)
        ax_p.set_xticks([4, 6, 8])
        ax_p.set_yticks([4, 6, 8])
        style_axis(ax_p)
        if j == 0:
            ax_p.set_ylabel(r'Predicted yield ($\mathrm{Mg\ ha^{-1}}$)', fontsize=7.5, labelpad=2.5)
            sub_leg_b = [
                Line2D([], [], marker='^', ls='', ms=3.8, mfc='white', mec='#0F172A', mew=0.60, label='2024 F1→2024 F2'),
                Line2D([], [], marker='o', ls='', ms=3.8, mfc='white', mec='#0F172A', mew=0.60, label='2024 F2→2024 F1'),
                Patch(facecolor='#94A3B8', edgecolor='none', alpha=0.25, label='±10%'),
            ]
            ax_p.legend(handles=sub_leg_b, loc='lower right', frameon=True, facecolor='white', edgecolor='none',
                        framealpha=0.88, fontsize=7.0, handlelength=0.85, handletextpad=0.20, borderpad=0.06, labelspacing=0.12)
        else:
            ax_p.set_yticklabels([''] * 3)
        if j == 1:
            ax_p.set_xlabel(r'Observed yield ($\mathrm{Mg\ ha^{-1}}$)', fontsize=7.5, labelpad=2.0)

        r2_str = f'−{abs(r2_v):.2f}' if r2_v < 0 else f'{r2_v:.2f}'
        ax_p.text(0.042, 0.968, p_tag, transform=ax_p.transAxes, ha='left', va='top',
                  fontsize=7.0, fontweight='bold', color=INK, zorder=4.6,
                  bbox=dict(boxstyle='round,pad=0.04', facecolor='white', edgecolor='none', alpha=0.88))
        ax_p.text(0.042, 0.848, rf'$R^2 = {r2_str}$' + '\n' + f'RMSE = {rmse_v:.2f}',
                  transform=ax_p.transAxes, ha='left', va='top', fontsize=7.0, linespacing=1.02, color=INK, zorder=4.6,
                  bbox=dict(boxstyle='round,pad=0.04', facecolor='white', edgecolor='none', alpha=0.86))

    # Shared discrete N-rate legend above 1x3 parity plots
    b_n_handles = [
        Line2D([], [], color='none', label=r'N rate ($\mathrm{kg\ N\ ha^{-1}}$):')
    ] + [
        Line2D([], [], marker='o', ls='', ms=3.8, mfc=N_COLORS[n], mec='#0F172A', mew=0.35, label=f'{n}')
        for n in N_RATES_ALL
    ]
    fig.legend(handles=b_n_handles, loc='upper center', bbox_to_anchor=((xb0 + xb_right) / 2, 0.982),
               ncol=8, frameon=False, columnspacing=0.58, handlelength=0.85, handletextpad=0.20, fontsize=7.0)

    # --- Row 2: 9-Task x 5-Model Restrained Total ΔRMSE Matrix (c) + Paired Within-N 95% CI Forest (d) ---
    ms_int = pd.read_csv(Q1_DIR / 'multiscale_intervals_vs_N.csv')
    ms_tune = pd.read_csv(Q1_DIR / 'multiscale_tuning.csv')
    nat_int = ms_int[ms_int.prefix == '3.3 mm']
    nat_tune = ms_tune[ms_tune.prefix == '3.3 mm']
    blocked = pd.read_csv(OLD_METHOD / 'blocked_inference.csv')
    step1 = pd.read_csv(OLD_METHOD / 'REFRAME_STEP1_INTERVALS.csv').query("family=='representation_vs_N'")

    specs = [
        ('N_Cearly', r'$N{+}C_{\mathrm{early}}$', r'$N{+}C_{\mathrm{early}}$', nat_int[nat_int.model == 'N_Cearly']),
        ('N_C_T', r'$N{+}C{+}T$', r'$N{+}C{+}T$', nat_int[nat_int.model == 'N_C_T']),
        ('N_Leaf_T', r'$N{+}\mathrm{Leaf}{+}T$', r'$N{+}\mathrm{Leaf}{+}T$', nat_int[nat_int.model == 'N_Leaf_T']),
        ('N_RGB52_common4', r'$N{+}\mathrm{RGB52}$', r'$N{+}\mathrm{RGB52}$', pd.concat([
            blocked.query("model=='N_RGB52_common4' and base=='N'"),
            step1.query("candidate=='N_RGB52_common4'"),
        ])),
        ('N_ResNet50_PCA52_common4', r'$N{+}\mathrm{ResNet\text{-}50}$', r'$N{+}\mathrm{ResNet\text{-}50}$', step1.query("candidate=='N_ResNet50_PCA52_common4'")),
    ]

    y_bot2, y_top2 = 0.064, 0.566
    h2 = y_top2 - y_bot2
    fig.text(0.016, y_top2 + 0.048, 'c', fontsize=9, fontweight='bold', color='#111111', va='top', ha='left')
    fig.text(0.512, y_top2 + 0.048, 'd', fontsize=9, fontweight='bold', color='#111111', va='top', ha='left')

    # Panel c: 10-row (9 tasks + 9-task mean) x 5-model Total ΔRMSE Benchmark Matrix
    mat_tot = np.zeros((10, len(specs)), dtype=float)
    for c_i, (_, _, _, blk) in enumerate(specs):
        vals = series_for(blk[blk.measure == 'Total'], ROLES, 'gain_absolute')
        mat_tot[:9, c_i] = vals
        mat_tot[9, c_i] = float(np.mean(vals))

    ax_c = fig.add_axes([0.144, y_bot2, 0.368, h2], facecolor='white')
    norm_c = TwoSlopeNorm(vmin=-180.0, vcenter=0.0, vmax=240.0)
    im_c = ax_c.imshow(mat_tot, cmap=Q1_DIV_CMAP, norm=norm_c, aspect='auto')
    for r_i in range(10):
        for c_i in range(len(specs)):
            v = mat_tot[r_i, c_i]
            rgba_c = Q1_DIV_CMAP(norm_c(np.clip(v, -180.0, 240.0)))
            lum_c = 0.299 * rgba_c[0] + 0.587 * rgba_c[1] + 0.114 * rgba_c[2]
            txt_c = 'white' if lum_c < 0.46 else '#0F172A'
            fw = 'bold' if r_i == 9 else 'normal'
            val_int = int(round(v))
            val_str = f'−{abs(val_int)}' if val_int < 0 else f'{val_int}'
            ax_c.text(c_i, r_i, val_str, ha='center', va='center', fontsize=7.0, fontweight=fw, color=txt_c)

    for h_div in (2.5, 4.5, 8.5):
        ax_c.axhline(h_div, color='#0F172A', lw=0.85 if h_div > 8 else 0.60)
    ax_c.set_xticks([-0.05, 0.96, 1.94, 2.90, 4.12])
    ax_c.set_xticklabels([s[1] for s in specs], fontsize=7.0)
    lbls_y = [fs.TASK_LABELS[t] for t in ROLES] + ['9-task mean']
    ax_c.set_yticks(np.arange(10))
    ax_c.set_yticklabels(lbls_y, fontsize=7.0)
    ax_c.get_yticklabels()[-1].set_fontweight('bold')
    ax_c.tick_params(axis='both', which='major', length=2.0, width=0.5, pad=1.8, color=SLATE)
    for sp in ax_c.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(SLATE)
        sp.set_linewidth(0.55)

    # Horizontal colorbar with explicit unit label above Panel c
    fig.text(0.258, y_top2 + 0.015, r'Total $\Delta\mathrm{RMSE}$ ($\mathrm{kg\ ha^{-1}}$):',
             ha='right', va='center', fontsize=7.0, color=INK)
    cax_c = fig.add_axes([0.266, y_top2 + 0.011, 0.200, 0.0080])
    cb_c = fig.colorbar(im_c, cax=cax_c, orientation='horizontal', ticks=[-150, 0, 120, 240])
    cb_c.ax.set_xticklabels(['−150', '0', '120', '240'])
    cb_c.ax.xaxis.set_ticks_position('top')
    cb_c.ax.tick_params(labelsize=7.0, length=1.5, pad=0.8, color=SLATE)
    cb_c.outline.set_linewidth(0.45)

    # Panel d: Paired Within-N 95% CI Forest Plot (Clean White/Slate Background)
    ax_d = fig.add_axes([0.526, y_bot2, 0.454, h2], facecolor='white')
    pos = np.arange(10, dtype=float)
    for r_i in range(9):
        if r_i % 2 == 1:
            ax_d.axhspan(r_i - 0.48, r_i + 0.48, fc='#F8FAFC', ec='none', zorder=0.2)
    ax_d.axhspan(8.52, 9.50, fc='#F1F5F9', ec='none', zorder=0.3)
    for h_div in (2.5, 4.5, 8.5):
        ax_d.axhline(h_div, color='#64748B' if h_div > 8 else '#CBD5E1', lw=0.85 if h_div > 8 else 0.65, zorder=1)

    ax_d.set_axisbelow(True)
    ax_d.grid(axis='x', lw=0.40, color='#F1F5F9', zorder=0.5)
    ax_d.axvline(0, color='#334155', lw=0.85, ls='--', zorder=2.5)

    forest_cols = {k: MODEL_COLORS[k] for k, _, _, _ in specs}

    extra_tune_3d = {
        'N_RGB52_common4': pd.read_csv(OLD_MINING / 'methodology_current/matched_RGB_tuning.csv').query(
            "specification == 'N_RGB52_common4'"
        ),
        'N_ResNet50_PCA52_common4': pd.read_csv(Q1_DIR / 'resnet_mask_pca_predictions.csv').query(
            "specification == 'N_ResNet50_Unmasked_p52'"
        ),
    }

    n_specs = len(specs)
    step_y = 0.155
    for m_i, (key, _, _, block) in enumerate(specs):
        sub = block.loc[block.measure.eq('Within-N')]
        x_val = series_for(sub, ROLES, 'gain_absolute')
        lo_val = series_for(sub, ROLES, 'cluster_low')
        hi_val = series_for(sub, ROLES, 'cluster_high')
        c = forest_cols[key]
        mk = MODEL_MARKERS[key]
        y_val = pos[:-1] + (m_i - (n_specs - 1) / 2.0) * step_y
        ax_d.hlines(y_val, lo_val, hi_val, color=c, lw=0.95, alpha=0.88, zorder=3)
        for xx, yy, task in zip(x_val, y_val, ROLES):
            is_open = is_hollow_alpha(nat_tune, key, task, extra_tune=extra_tune_3d)
            ax_d.plot(xx, yy, ls='', marker=mk, ms=3.8, color=c,
                      mfc='white' if is_open else c,
                      mec=c if is_open else '#0F172A',
                      mew=0.80 if is_open else 0.35, zorder=4)
        mean_v = float(np.mean(x_val))
        se_v = float(np.std(x_val, ddof=1) / np.sqrt(len(x_val)))
        ym = pos[-1] + (m_i - (n_specs - 1) / 2.0) * step_y
        ax_d.hlines(ym, mean_v - se_v, mean_v + se_v, color=c, lw=1.25, zorder=4.2)
        ax_d.plot(mean_v, ym, ls='', marker=mk, ms=4.4, color=c, mfc=c, mec='#0F172A', mew=0.50, zorder=4.5)

    ax_d.set_xlim(-130, 194)
    ax_d.set_ylim(9.50, -0.50)
    ax_d.set_xticks([-100, -50, 0, 50, 100, 150])
    ax_d.set_yticks(pos)
    ax_d.set_yticklabels([''] * 10)
    ax_d.set_xlabel(r'Within-$N$ $\Delta\mathrm{RMSE}$ ($\mathrm{kg\ ha^{-1}}$)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_d)

    # Column-major order for ncol=3 so Row 1 = [N+C_early, N+C+T, N+Leaf+T], Row 2 = [N+RGB52, N+ResNet-50, Open gate]
    spec_handles = [
        Line2D([], [], color=forest_cols[k], marker=MODEL_MARKERS[k], ls='-', lw=1.0, ms=3.9,
               mfc=forest_cols[k], mec='#0F172A', mew=0.35, label=full_lbl)
        for k, _, full_lbl, _ in specs
    ]
    open_handle = Line2D([], [], color='none', marker='s', ls='', ms=3.8,
                         mfc='white', mec='#0F766E', mew=0.80, label=r'Open: $\alpha = 10^5$ bound')
    d_handles = [spec_handles[0], spec_handles[3], spec_handles[1], spec_handles[4], spec_handles[2], open_handle]
    ax_d.legend(handles=d_handles, loc='lower center', bbox_to_anchor=(0.50, 1.010),
                ncol=3, frameon=False, columnspacing=0.65, handlelength=1.05, handletextpad=0.22, labelspacing=0.14, fontsize=7.0)

    save_fig(fig, MAIN_STEMS[2])


# ==============================================================================
# FIGURE 4 (a-f): Scheme 1 — Multi-Scale Sensor GSD Pyramid & Cross-Row Waveforms (a),
#                 3.3 mm -> 3 m Scaling Law (b-d), 181 DAS Glare Rejection & Gains (e-f)
# ==============================================================================
def render_figure4() -> None:
    fs.apply()
    fig_w_mm, fig_h_mm = 180.0, 170.0
    # Figure setup with internal helper functions for coordinate normalization
    def _y(y_old: float) -> float:
        return y_old
    def _h(h_old: float) -> float:
        return h_old

    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4), facecolor='white')

    demo = np.load(Q1_DIR / 'multiscale_demo_slices.npz')
    rich_npz = np.load(Q1_DIR / 'rich_figure_assets.npz')

    def _morph_boll_mask(raw_mask: np.ndarray, px_m_val: float) -> np.ndarray:
        lbl, n_cc = ndimage.label(raw_mask.astype(bool))
        if n_cc == 0:
            return np.zeros_like(raw_mask, dtype=bool)
        areas_cm2 = np.bincount(lbl.ravel())[1:] * ((px_m_val * 100.0) ** 2)
        keep_ids = np.where((areas_cm2 >= 0.5) & (areas_cm2 <= 120.0))[0] + 1
        return np.isin(lbl, keep_ids)

    # --- Row 1 (a): 2 x 4 Exact 1:1 Sub-Canopy Pyramid (3.3 mm RGB, 3.3 mm mask, 1 cm, 5 cm)
    #                + 2 x 1 Cross-Row Spatial Waveforms (0.0-2.4 m across 3 planted rows) ---
    fig.text(0.012, 0.986, 'a', fontsize=9, fontweight='bold', color='#111111', va='top', ha='left')
    x_left, x_img_right = 0.066, 0.580
    col_gap = 0.005
    cell_w = (x_img_right - x_left - 3 * col_gap) / 4
    cell_h = cell_w * (fig_w_mm / fig_h_mm)  # Exact 1:1 geometric square
    row_gap = 0.007
    y_top_a = 0.964

    vis_cols = [
        ('3p3mm', '3.3 mm RGB', False),
        ('3p3mm', '3.3 mm mask', True),
        ('1cm', '1 cm', True),
        ('5cm', '5 cm', True),
    ]
    row_specs = [('canopy128', 'green', '128 DAS'), ('boll181', 'boll', '181 DAS')]
    for r_i, (r_key, m_sfx, r_label) in enumerate(row_specs):
        yc = y_top_a - (r_i + 1) * cell_h - r_i * row_gap
        fig.text(x_left - 0.008, yc + cell_h / 2, r_label, ha='right', va='center', fontsize=7.2, fontweight='bold', color=INK)
        for c_i, (tag, g_lbl, apply_tint) in enumerate(vis_cols):
            xc = x_left + c_i * (cell_w + col_gap)
            if r_i == 0:
                fig.text(xc + cell_w / 2, y_top_a + 0.005, g_lbl, ha='center', va='bottom', fontsize=7.0, fontweight='bold', color=INK)

            ax = fig.add_axes([xc, yc, cell_w, cell_h])
            rgb = demo[f'{r_key}_{tag}_rgb'].astype(np.float32) / 255.0
            raw_m = demo[f'{r_key}_{tag}_{m_sfx}'].astype(bool)
            px_m_tile = float(demo[f'{r_key}_{tag}_px_m'][0])
            vis = rgb.copy()
            if apply_tint:
                if r_key == 'canopy128':
                    gap = ~raw_m
                    tint = np.array([0.85, 0.47, 0.02], dtype=np.float32)
                    vis[gap] = 0.46 * vis[gap] + 0.54 * tint
                else:
                    msk_b = _morph_boll_mask(raw_m, px_m_tile)
                    tint = np.array([0.64, 0.11, 0.48], dtype=np.float32)
                    vis[msk_b] = 0.35 * vis[msk_b] + 0.65 * tint
            # Strictly nearest-neighbor interpolation to preserve authentic physical pixel blocks
            ax.imshow(np.clip(vis, 0, 1), interpolation='nearest', aspect='equal')
            if r_i == 0 and c_i == 0:
                # 0.5 m physical scale bar on 2.4 m x 2.4 m tile
                sb_frac = 0.50 / 2.40
                ax.plot([0.07, 0.07 + sb_frac], [0.09, 0.09], transform=ax.transAxes, color='white', lw=2.2, solid_capstyle='butt', zorder=5)
                ax.text(0.07 + sb_frac / 2, 0.14, '0.5 m', transform=ax.transAxes, ha='center', va='bottom',
                        fontsize=7.0, fontweight='bold', color='white',
                        path_effects=[pe.withStroke(linewidth=1.5, foreground='#0F172A')], zorder=6)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor('#475569')
                sp.set_linewidth(0.50)

        # Paired Cross-Row Spatial Waveform on the right (x = 0.0 to 2.4 m across the 3 cotton rows)
        ax_prof = fig.add_axes([0.642, yc, 0.340, cell_h], facecolor='white')
        ax_prof.set_axisbelow(True)
        ax_prof.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
        prof_specs = [
            ('3p3mm', '3.3 mm', PALETTE['teal'], '-', 1.15),
            ('1cm', '1 cm', PALETTE['cerulean'], '--', 0.95),
            ('5cm', '5 cm', PALETTE['amber'], '-', 1.05),
            ('20cm', '≥20 cm', PALETTE['crimson'], ':', 1.20),
        ]
        if r_i == 0:
            for p_tag, p_lbl, p_col, p_ls, p_lw in prof_specs:
                arr_m = demo[f'canopy128_{p_tag}_green'].astype(float) * 100.0
                prof_1d = ndimage.uniform_filter1d(arr_m.mean(axis=0), size=max(1, len(arr_m) // 28))
                xm = np.linspace(0.0, 2.4, len(prof_1d))
                ax_prof.plot(xm, prof_1d, ls=p_ls, lw=p_lw, color=p_col, label=p_lbl, zorder=3)
            ax_prof.set_ylim(58, 103)
            ax_prof.set_yticks([60, 80, 100])
            ax_prof.set_ylabel('Canopy cover (%)', fontsize=7.5, labelpad=2.0)
            ax_prof.set_xticklabels([])
            ax_prof.legend(loc='lower center', bbox_to_anchor=(0.50, 1.03), ncol=4, frameon=False,
                           fontsize=7.0, columnspacing=0.68, handlelength=1.05, handletextpad=0.22, borderpad=0.0)
        else:
            for p_tag, p_lbl, p_col, p_ls, p_lw in prof_specs:
                raw_b = demo[f'boll181_{p_tag}_boll'].astype(bool)
                px_b = float(demo[f'boll181_{p_tag}_px_m'][0])
                arr_b = _morph_boll_mask(raw_b, px_b).astype(float) * 100.0
                prof_1d = ndimage.uniform_filter1d(arr_b.mean(axis=0), size=max(1, len(arr_b) // 28))
                xm = np.linspace(0.0, 2.4, len(prof_1d))
                ax_prof.plot(xm, prof_1d, ls=p_ls, lw=p_lw, color=p_col, label=p_lbl, zorder=3)
            ax_prof.set_ylim(-1.5, 33)
            ax_prof.set_yticks([0, 15, 30])
            ax_prof.set_ylabel('Open-boll area (%)', fontsize=7.5, labelpad=2.0)
            ax_prof.set_xticks([0.0, 0.8, 1.6, 2.4])
            ax_prof.set_xlabel('Cross-row distance (m)', fontsize=7.5, labelpad=2.0)
        ax_prof.set_xlim(0.0, 2.40)
        style_axis(ax_prof)

    # Compact mask tint legend under the 2x4 tile pyramid
    a_tint_handles = [
        Patch(facecolor=PALETTE['amber'], edgecolor='#334155', lw=0.4, label='Inter-row gap'),
        Patch(facecolor=PALETTE['magenta_mask'], edgecolor='#334155', lw=0.4, label=r'Boll cluster (0.5–120 $\mathrm{cm^2}$)'),
    ]
    fig.legend(handles=a_tint_handles, loc='upper center',
               bbox_to_anchor=((x_left + x_img_right) / 2, y_top_a - 2 * cell_h - row_gap - 0.004),
               ncol=2, frameon=False, fontsize=7.0, columnspacing=0.90, handlelength=0.95, handletextpad=0.25)

    # --- Row 2 (b, c, d): Simulated Multi-Scale Spatial Coarsening Sensitivity Analysis ---
    y_bot2, y_top2 = 0.366, 0.594
    h2 = y_top2 - y_bot2
    fig.text(0.012, y_top2 + 0.036, 'b', fontsize=9, fontweight='bold', color='#111111', va='top', ha='left')
    fig.text(0.346, y_top2 + 0.036, 'c', fontsize=9, fontweight='bold', color='#111111', va='top', ha='left')
    fig.text(0.668, y_top2 + 0.036, 'd', fontsize=9, fontweight='bold', color='#111111', va='top', ha='left')

    gsd_slots = pd.read_csv(Q1_DIR / 'multiscale_gsd_slot_features.csv')
    common64 = pd.read_csv(OLD_MINING / 'common_plot_features_common_DAS.csv')[['domain', 'plot_id']].drop_duplicates()
    all_das = gsd_slots[gsd_slots.DAS != 181].merge(common64, on=['domain', 'plot_id'])
    gsd_order = ['3.3 mm', '1 cm', '5 cm', '20 cm', '1 m', '3 m']
    x_idx = np.arange(len(gsd_order), dtype=float)

    # Panel b: Raincloud Plot (Right Half-Violin KDE + Boxplot + 64-plot Spaghetti + Mean Curve)
    ax_b = fig.add_axes([0.066, y_bot2, 0.268, h2], facecolor='white')
    ax_b.set_axisbelow(True)
    ax_b.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
    piv_cov = (
        all_das.groupby(['domain', 'plot_id', 'gsd_label'])['f_cover']
        .mean()
        .unstack('gsd_label')[gsd_order]
        * 100.0
    )

    for row_vals in piv_cov.to_numpy():
        ax_b.plot(x_idx - 0.10, row_vals, color='#94A3B8', lw=0.32, alpha=0.25, zorder=1)

    for gi, g in enumerate(gsd_order):
        vals = piv_cov[g].to_numpy(float)
        if np.std(vals) > 0.25:
            yg = np.linspace(max(55.0, vals.min() - 2.0), min(100.2, vals.max() + 1.0), 80)
            bw = max(0.8, 1.06 * float(np.std(vals)) * (len(vals) ** (-0.2)))
            dens = _gaussian_kde_1d(vals, yg, bw)
            dens_scaled = dens / max(1e-6, dens.max()) * 0.34
            ax_b.fill_betweenx(yg, gi + 0.04, gi + 0.04 + dens_scaled, facecolor='#38BDF8', edgecolor='#0284C7',
                               alpha=0.45, linewidth=0.55, zorder=2.5)

    bp = ax_b.boxplot([piv_cov[g].to_numpy() for g in gsd_order], positions=x_idx - 0.10, widths=0.18,
                      patch_artist=True, showfliers=False, zorder=3.5)
    for box in bp['boxes']:
        box.set(facecolor='#E0F2FE', edgecolor='#0369A1', linewidth=0.65, alpha=0.92)
    for med in bp['medians']:
        med.set(color='#0F172A', linewidth=1.1)
    for w_el in bp['whiskers'] + bp['caps']:
        w_el.set(color='#0369A1', linewidth=0.55)

    ax_b.plot(x_idx - 0.10, piv_cov.mean(axis=0).to_numpy(), 'o-', color=PALETTE['crimson'], lw=1.15, ms=3.5,
              mfc=PALETTE['crimson_mid'], mec='#1E293B', mew=0.4, zorder=4.5)
    b_leg_handles = [
        Line2D([], [], color=PALETTE['crimson'], marker='o', lw=1.15, ms=3.5, mfc=PALETTE['crimson_mid'], mec='#1E293B', mew=0.4, label='Mean'),
        Line2D([], [], color='#94A3B8', lw=0.65, label='Individual plots'),
    ]
    ax_b.legend(handles=b_leg_handles, loc='lower right', frameon=False, fontsize=7.0,
                handlelength=1.0, handletextpad=0.22, labelspacing=0.18)
    ax_b.set_xlim(-0.30, 5.43)
    ax_b.set_ylim(54, 102.5)
    ax_b.set_xticks(x_idx)
    ax_b.set_xticklabels(gsd_order, fontsize=7.0)
    ax_b.set_yticks([60, 70, 80, 90, 100])
    ax_b.set_xlabel('Spatial resolution (GSD)', fontsize=7.5, labelpad=2.0)
    ax_b.set_ylabel('Apparent canopy cover (%)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_b)

    # Panel c: Split into two sub-panels: Top = Trait correlation (r); Bottom = Diagnostic AUC (synced with Table S1)
    phys = pd.read_csv(Q1_DIR / 'multiscale_physical_degradation.csv').sort_values('gsd_m')
    diag_ms = pd.read_csv(Q1_DIR / 'n_limitation_same_window_multiscale.csv')
    diag_ms = diag_ms[(diag_ms.window == 'Full_92_142') & (diag_ms.model == 'Decoupled_p8')].sort_values('gsd_m')
    gap_c = 0.014
    h_c_bot = 0.080
    h_c_top = h2 - h_c_bot - gap_c
    ax_c_top = fig.add_axes([0.392, y_bot2 + h_c_bot + gap_c, 0.262, h_c_top], facecolor='white')
    ax_c_bot = fig.add_axes([0.392, y_bot2, 0.262, h_c_bot], facecolor='white')

    ax_c_top.set_axisbelow(True)
    ax_c_top.axvspan(2.0, 5.3, fc='#F8FAFC', ec='none', zorder=0.1)
    ax_c_top.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
    ax_c_top.axhline(0.0, color='#64748B', lw=0.75, ls='--', zorder=1.5)
    ax_c_top.axvline(2.0, color='#475569', lw=0.85, ls=':', zorder=1.8)

    for col, lbl, col_hex, mk in [
        ('cover_r_vs_native', 'Canopy cover', PALETTE['cerulean'], 'o'),
        ('gap_r_vs_native', r'Gaps $\geq$0.04 $\mathrm{m^2}$', PALETTE['teal'], 's'),
        ('row_profile_r_vs_native', 'Row-closure SD', PALETTE['amber'], '^'),
        ('boll_r_vs_native', 'Boll clusters', PALETTE['plum'], 'D'),
    ]:
        y_vals = phys[col].to_numpy(float).copy()
        ax_c_top.plot(x_idx, y_vals, marker=mk, ms=3.4, lw=1.00,
                      color=col_hex, mfc=col_hex, mec='#0F172A', mew=0.35, label=lbl, zorder=3)

    # Mark mathematically undefined r (zero feature variance: Row-closure SD at 3 m; Boll clusters at >=20 cm)
    y_row_1m = float(phys['row_profile_r_vs_native'].to_numpy(float)[4])
    y_boll_5cm = float(phys['boll_r_vs_native'].to_numpy(float)[2])
    ax_c_top.plot([4.0, 4.30], [y_row_1m, 0.08], color=PALETTE['amber'], lw=0.65, ls=':', zorder=3.8)
    ax_c_top.text(4.34, 0.08, 'NA', fontsize=7.0, fontweight='bold', color=PALETTE['ochre'], ha='left', va='center',
                  bbox=dict(boxstyle='round,pad=0.04', facecolor='white', edgecolor=PALETTE['amber'], lw=0.45, alpha=0.94), zorder=4.2)
    ax_c_top.plot([2.0, 2.32], [y_boll_5cm, 0.50], color=PALETTE['plum'], lw=0.65, ls=':', zorder=3.8)
    ax_c_top.text(2.36, 0.50, 'NA', fontsize=7.0, fontweight='bold', color=PALETTE['plum'], ha='left', va='center',
                  bbox=dict(boxstyle='round,pad=0.04', facecolor='white', edgecolor=PALETTE['plum'], lw=0.45, alpha=0.94), zorder=4.2)

    ax_c_top.set_xlim(-0.20, 5.20)
    ax_c_top.set_ylim(-0.08, 1.08)
    ax_c_top.set_xticks(x_idx)
    ax_c_top.set_xticklabels([''] * len(gsd_order))
    ax_c_top.set_yticks([0.0, 0.5, 1.0])
    ax_c_top.set_ylabel(r'Correlation ($r$)', fontsize=7.5, labelpad=1.5)
    style_axis(ax_c_top)
    ax_c_top.legend(loc='lower center', bbox_to_anchor=(0.50, 1.02), ncol=2, frameon=False,
                    fontsize=7.0, columnspacing=0.60, handlelength=0.85, handletextpad=0.16, borderpad=0.02, labelspacing=0.10)

    ax_c_bot.set_axisbelow(True)
    ax_c_bot.axvspan(2.0, 5.25, fc='#F8FAFC', ec='none', zorder=0.1)
    ax_c_bot.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
    ax_c_bot.axhline(0.5, color='#64748B', lw=0.70, ls='--', zorder=1.5)
    ax_c_bot.axvline(2.0, color='#475569', lw=0.85, ls=':', zorder=1.8)
    auc_dec = diag_ms['LOBO_ROC_AUC'].to_numpy(float)
    ax_c_bot.plot(x_idx, auc_dec, marker='P', ms=3.6, lw=1.05, ls='-',
                  color=PALETTE['teal'], mfc=PALETTE['teal'], mec='#0F172A', mew=0.35, zorder=3.5)
    ax_c_bot.set_xlim(-0.20, 5.20)
    ax_c_bot.set_ylim(0.46, 0.94)
    ax_c_bot.set_xticks(x_idx)
    ax_c_bot.set_xticklabels(gsd_order, fontsize=7.0)
    ax_c_bot.set_yticks([0.5, 0.7, 0.9])
    ax_c_bot.set_xlabel('Spatial resolution (GSD)', fontsize=7.5, labelpad=2.0)
    ax_c_bot.set_ylabel('ROC AUC', fontsize=7.5, labelpad=2.0)
    style_axis(ax_c_bot)

    # Panel d: Multi-Scale Mean ΔRMSE Trajectories (Clean White + Standard +/-1 SE Band)
    ms_int = pd.read_csv(Q1_DIR / 'multiscale_intervals_vs_N.csv')
    ax_d = fig.add_axes([0.720, y_bot2, 0.262, h2], facecolor='white')
    ax_d.set_axisbelow(True)
    ax_d.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
    ax_d.axhline(0.0, color='#334155', lw=0.85, ls='--', zorder=2.0)
    ax_d.axvline(2.0, color='#475569', lw=0.85, ls=':', zorder=1.8)

    for mod_k, lbl, col_hex, mk in [
        ('N_C_T_Bmorph', r'$N{+}C{+}T{+}B_{\mathrm{morph}}$', PALETTE['plum'], 'D'),
        ('N_Leaf_T', r'$N{+}\mathrm{Leaf}{+}T$', PALETTE['teal'], 'v'),
        ('N_C_T', r'$N{+}C{+}T$', PALETTE['cerulean'], 's'),
    ]:
        means, ses = [], []
        for g in gsd_order:
            arr = ms_int[(ms_int.prefix == g) & (ms_int.model == mod_k) & (ms_int.measure == 'Total')]['gain_absolute'].to_numpy(float)
            means.append(float(arr.mean()))
            ses.append(float(arr.std(ddof=1) / np.sqrt(len(arr))))
        m_arr, s_arr = np.array(means), np.array(ses)
        ax_d.fill_between(x_idx, m_arr - s_arr, m_arr + s_arr, color=col_hex, alpha=0.13, lw=0, zorder=2)
        ax_d.plot(x_idx, m_arr, marker=mk, ms=3.8, lw=1.10, color=col_hex,
                  mfc=col_hex, mec='#0F172A', mew=0.35, label=lbl, zorder=3)

    ax_d.set_xlim(-0.20, 5.20)
    ax_d.set_ylim(-195, 340)
    ax_d.set_xticks(x_idx)
    ax_d.set_xticklabels(gsd_order, fontsize=7.0)
    ax_d.set_yticks([-150, 0, 150, 300])
    ax_d.set_xlabel('Spatial resolution (GSD)', fontsize=7.5, labelpad=2.0)
    ax_d.set_ylabel(r'Total $\Delta\mathrm{RMSE}$ ($\mathrm{kg\ ha^{-1}}$)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_d)
    ax_d.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='none', framealpha=0.88,
                fontsize=7.0, handlelength=0.95, handletextpad=0.20, borderpad=0.06, labelspacing=0.14)

    # --- Row 3 (e + f): 181 DAS Plastic-Mulch Glare Rejection & Size Spectrum (e) + 6-Transfer Grouped Bars (f) ---
    y_bot3, y_top3 = 0.056, 0.292
    h3 = y_top3 - y_bot3
    fig.text(0.012, y_top3 + 0.024, 'e', fontsize=9, fontweight='bold', color='#111111', va='top', ha='left')
    fig.text(0.628, y_top3 + 0.024, 'f', fontsize=9, fontweight='bold', color='#111111', va='top', ha='left')

    # Exact 1:1 square 1.50 m x 1.50 m (417 x 417 px) tiles aligned with left margin x_left = 0.066
    im_w = 0.132
    im_h = im_w * (fig_w_mm / fig_h_mm)
    im_y = y_bot3 + 0.050
    ax_e1 = fig.add_axes([x_left, im_y, im_w, im_h])
    ax_e2 = fig.add_axes([x_left + im_w + 0.007, im_y, im_w, im_h])

    glare_rgb = rich_npz['f4_glare_rgb'][:, :417]
    glare_cls = rich_npz['f4_glare_cls'][:, :417]
    ax_e1.imshow(glare_rgb, interpolation='nearest', aspect='equal')
    ax_e1.text(0.50, 1.04, '181 DAS RGB', transform=ax_e1.transAxes,
               ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=INK, clip_on=False)
    # 0.5 m physical scale bar on 1.50 m x 1.50 m window
    sb_e = 0.50 / 1.50
    ax_e1.plot([0.07, 0.07 + sb_e], [0.09, 0.09], transform=ax_e1.transAxes, color='white', lw=2.2, solid_capstyle='butt', zorder=5)
    ax_e1.text(0.07 + sb_e / 2, 0.14, '0.5 m', transform=ax_e1.transAxes, ha='center', va='bottom',
               fontsize=7.0, fontweight='bold', color='white',
               path_effects=[pe.withStroke(linewidth=1.5, foreground='#0F172A')], zorder=6)

    # Correct connected-component class indices: 2 = True Bolls (0.5-120 cm2), 3 = Mulch Glare (>120 cm2)
    vis_g = (glare_rgb.astype(np.float32) / 255.0) * 0.38
    vis_g[glare_cls == 2] = np.array([0.64, 0.11, 0.48], dtype=np.float32)
    vis_g[glare_cls == 3] = np.array([0.85, 0.47, 0.02], dtype=np.float32)
    ax_e2.imshow(np.clip(vis_g, 0, 1), interpolation='nearest', aspect='equal')
    ax_e2.text(0.50, 1.04, 'Morphological filter', transform=ax_e2.transAxes,
               ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=INK, clip_on=False)

    for ax_m in (ax_e1, ax_e2):
        ax_m.set_xticks([])
        ax_m.set_yticks([])
        for sp in ax_m.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor('#334155')
            sp.set_linewidth(0.55)

    b_patch_handles = [
        Patch(facecolor=PALETTE['magenta_mask'], edgecolor='#0F172A', lw=0.4, label=r'0.5–120 $\mathrm{cm^2}$ bolls'),
        Patch(facecolor=PALETTE['amber'], edgecolor='#0F172A', lw=0.4, label=r'>120 $\mathrm{cm^2}$ glare'),
    ]
    fig.legend(handles=b_patch_handles, loc='upper center', bbox_to_anchor=(x_left + im_w + 0.0035, im_y - 0.004),
               ncol=1, frameon=False, fontsize=7.0, handlelength=0.85, handletextpad=0.22, labelspacing=0.14)

    spec_df = pd.read_csv(Q1_DIR / 'rich_fig4_cluster_spectrum.csv')
    ax_e3 = fig.add_axes([0.398, y_bot3, 0.222, h3], facecolor='white')
    ax_e3.set_axisbelow(True)
    ax_e3.axvspan(np.log10(0.5), np.log10(120.0), fc='#FDF2F8', ec='none', alpha=0.65, zorder=0.1)
    ax_e3.axvspan(np.log10(120.0), 3.30, fc='#FFF7ED', ec='none', alpha=0.75, zorder=0.1)
    ax_e3.axvline(np.log10(0.5), color=PALETTE['plum'], lw=0.75, ls='--', zorder=1.8)
    ax_e3.axvline(np.log10(120.0), color=PALETTE['amber'], lw=0.85, ls='--', zorder=1.8)
    ax_e3.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)

    for reg_k, reg_lbl, c_r in [
        ('Low_N_0_75', 'N0–N75', PALETTE['amber']),
        ('Fert_N_150_450', 'N150–N450', PALETTE['plum']),
    ]:
        sr = spec_df[spec_df.n_regime == reg_k].sort_values('log10_area_cm2')
        ax_e3.fill_between(sr.log10_area_cm2, 0, sr.white_area_share_pct, color=c_r, alpha=0.14, lw=0)
        ax_e3.plot(sr.log10_area_cm2, sr.white_area_share_pct, color=c_r, lw=1.15, label=reg_lbl)

    ax_e3.text(np.log10(0.5) + 0.08, 20.3, '0.5', fontsize=7.0, fontweight='bold', color=PALETTE['plum'], ha='left', va='top')
    ax_e3.text(np.log10(120.0) + 0.08, 20.3, '120', fontsize=7.0, fontweight='bold', color=PALETTE['ochre'], ha='left', va='top')
    ax_e3.set_xlim(-0.72, 3.30)
    ax_e3.set_ylim(0, 21.5)
    ax_e3.set_xticks([0, 1, 2, 3])
    ax_e3.set_xticklabels(['1', '10', '100', '1000'])
    ax_e3.set_xlabel(r'Component area ($\mathrm{cm^2}$)', fontsize=7.5, labelpad=2.0)
    ax_e3.set_ylabel('White-pixel share (%)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_e3)
    ax_e3.legend(loc='upper left', bbox_to_anchor=(0.18, 0.84), frameon=True, facecolor='white', edgecolor='none',
                 framealpha=0.88, fontsize=7.0, handlelength=0.95, handletextpad=0.22, borderpad=0.05, labelspacing=0.16)

    # Panel f: Horizontal Grouped Bar + 95% CI Whiskers across 6 Transfers + 6-Transfer Mean (+/-1 SE)
    ax_f = fig.add_axes([0.734, y_bot3, 0.248, h3], facecolor='white')
    ax_f.set_axisbelow(True)
    pos_t = np.arange(len(TRANSFER) + 1, dtype=float)
    ax_f.axhspan(pos_t[-1] - 0.50, pos_t[-1] + 0.50, fc='#F1F5F9', ec='none', zorder=0.2)
    ax_f.axhline(1.5, color='#CBD5E1', lw=0.65, zorder=1)
    ax_f.axhline(5.5, color='#64748B', lw=0.80, zorder=1)
    ax_f.grid(axis='x', lw=0.38, color='#F1F5F9', zorder=0.5)
    ax_f.axvline(0.0, color='#334155', lw=0.85, ls='--', zorder=2.0)

    nat_tot = ms_int[(ms_int.prefix == '3.3 mm') & (ms_int.measure == 'Total')]
    for m_i, (mod_k, lbl, fc_hex, ec_hex) in enumerate([
        ('N_C_T_Braw', r'$B_{\mathrm{raw}}$', '#FDBA74', PALETTE['ochre']),
        ('N_C_T_Bmorph', r'$B_{\mathrm{morph}}$', PALETTE['plum'], '#4A044E'),
    ]):
        sub = nat_tot[nat_tot.model == mod_k]
        xv = series_for(sub, TRANSFER, 'gain_absolute')
        lo = series_for(sub, TRANSFER, 'cluster_low')
        hi = series_for(sub, TRANSFER, 'cluster_high')
        yv = pos_t[:-1] + (m_i - 0.5) * 0.34
        ax_f.barh(yv, xv, height=0.31, facecolor=fc_hex, edgecolor=ec_hex, linewidth=0.55, alpha=0.88, label=lbl, zorder=3)
        ax_f.hlines(yv, lo, hi, color='#0F172A', lw=0.70, zorder=4)

        mv = float(np.mean(xv))
        se_mv = float(np.std(xv, ddof=1) / np.sqrt(len(xv)))
        ym = pos_t[-1] + (m_i - 0.5) * 0.34
        ax_f.barh(ym, mv, height=0.31, facecolor=fc_hex, edgecolor='#0F172A', linewidth=0.75, alpha=0.96, zorder=3.5)
        ax_f.hlines(ym, mv - se_mv, mv + se_mv, color='#0F172A', lw=0.85, zorder=4.2)

    ax_f.set_xlim(-108, 872)
    ax_f.set_ylim(pos_t[-1] + 0.52, -0.52)
    ax_f.set_xticks([0, 400, 800])
    ax_f.set_yticks(pos_t)
    ax_f.set_yticklabels(['2024 F1 → 2024 F2', '2024 F2 → 2024 F1', '2023 F1 → 2024 F1', '2023 F1 → 2024 F2', '2024 F1 → 2023 F1', '2024 F2 → 2023 F1', '6-task mean'], fontsize=7.0)
    ax_f.get_yticklabels()[-1].set_fontweight('bold')
    ax_f.set_xlabel(r'Total $\Delta\mathrm{RMSE}$ ($\mathrm{kg\ ha^{-1}}$)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_f)
    ax_f.legend(loc='upper right', bbox_to_anchor=(0.995, 0.995), frameon=True, facecolor='white', edgecolor='none',
                framealpha=0.90, fontsize=7.0, handlelength=0.85, handletextpad=0.20, borderpad=0.05, labelspacing=0.12)

    save_fig(fig, MAIN_STEMS[3])


# ==============================================================================
# FIGURE 5 (a-f): Clean 3x2 Quantitative Grid (Zero Schematic Box Maps):
#                 Row 1: 6-Task Mean Orthogonal MSE Stack (a) + 6-Direction MSE Stack (b)
#                 Row 2: Physical N-Response Paradox Curves (c) + Plot-Level Residual Shift Distribution (d)
#                 Row 3: Commercial Calibration Strategy Curves (e) + Multi-Model Active Convergence Curves (f)
# ==============================================================================
def render_figure5() -> None:
    fs.apply()
    fig_w_mm, fig_h_mm = 180.0, 158.0
    # Figure setup with internal helper functions for coordinate normalization
    def _y(y_old: float) -> float:
        return y_old
    def _h(h_old: float) -> float:
        return h_old

    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4), facecolor='white')

    dec_base = pd.read_csv(Q1_DIR / 'hybrid_transfer_decomposition.csv')
    dec_mask = pd.read_csv(Q1_DIR / 'resnet_masked_p12_table3_decomposition.csv')
    dec = pd.concat([dec_base, dec_mask], ignore_index=True)
    p_para = pd.read_csv(Q1_DIR / 'rich_fig5_paradox_predictions.csv')
    sub_f2 = p_para[(p_para.source == '2023_F1') & (p_para.target == '2024_F2')]

    # --- Row 1 (a + b): 3-Component Orthogonal MSE Variance Decomposition (Within-N + Between-N + Bias^2) ---
    y_bot1, y_top1 = 0.708, 0.934
    h1 = y_top1 - y_bot1
    fig.text(0.012, y_top1 + 0.038, 'a', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')
    fig.text(0.466, y_top1 + 0.038, 'b', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')

    models_a = [
        ('N__zero_label', r'$N$-only ($k{=}0$)'),
        ('N_Leaf_T__zero_label', r'$N{+}\mathrm{Leaf}{+}T$ ($k{=}0$)'),
        ('N_ResNet50_Masked_p12__zero_label', r'Masked RN50-12 ($k{=}0$)'),
        ('N_ResNet50_PCA52_common4__zero_label', r'Unmasked RN50-52 ($k{=}0$)'),
        ('N__plus_3N0', r'$N{+}3\mathrm{N0}$'),
        ('N_Leaf_T__plus_3N0', r'$N{+}\mathrm{Leaf}{+}T{+}3\mathrm{N0}$'),
        ('N_ResNet50_Masked_p12__plus_3N0', r'Masked RN50-12${+}3\mathrm{N0}$'),
        ('N_ResNet50_Unmasked_p12__plus_3N0', r'Unmasked RN50-12${+}3\mathrm{N0}$'),
        ('N_ResNet50_PCA52_common4__plus_3N0', r'Unmasked RN50-52${+}3\mathrm{N0}$'),
        ('N__oracle', r'$N$-only shift ref.'),
    ]
    ax_a = fig.add_axes([0.184, y_bot1, 0.260, h1], facecolor='white')
    ax_a.set_axisbelow(True)
    ax_a.grid(axis='x', lw=0.38, color='#F1F5F9', zorder=0.5)
    ya = np.arange(len(models_a), dtype=float)

    w_mse_list, b_mse_list, bias_mse_list = [], [], []
    for mid, _ in models_a:
        sub_m = dec[dec.model_id == mid].pivot(index=['source', 'target', 'evaluation'], columns='measure', values='estimate').reindex(TRANSFER)
        wm = float(np.mean((sub_m['Within-N'].to_numpy(float) / 1000.0) ** 2))
        bm = float(np.mean((sub_m['Between-N'].to_numpy(float) / 1000.0) ** 2))
        bim = float(np.mean((sub_m['Bias'].to_numpy(float) / 1000.0) ** 2))
        w_mse_list.append(wm)
        b_mse_list.append(bm)
        bias_mse_list.append(bim)

    w_arr = np.array(w_mse_list)
    b_arr = np.array(b_mse_list)
    bi_arr = np.array(bias_mse_list)

    c_wn, c_bn, c_bi = PALETTE['teal'], PALETTE['amber'], PALETTE['slate_mid']
    ax_a.barh(ya, w_arr, height=0.62, facecolor=c_wn, edgecolor='#0F172A', lw=0.45, zorder=3)
    ax_a.barh(ya, b_arr, left=w_arr, height=0.62, facecolor=c_bn, edgecolor='#0F172A', lw=0.45, zorder=3)
    ax_a.barh(ya, bi_arr, left=w_arr + b_arr, height=0.62, facecolor=c_bi, edgecolor='#0F172A', lw=0.45, zorder=3)
    ax_a.axhline(3.5, color='#64748B', lw=0.75, ls='--', zorder=2)
    ax_a.axhline(8.5, color='#94A3B8', lw=0.65, ls=':', zorder=2)

    ax_a.set_xlim(0, 1.23)
    ax_a.set_ylim(len(models_a) - 0.42, -0.52)
    ax_a.set_xticks([0.0, 0.4, 0.8, 1.2])
    ax_a.set_yticks(ya)
    ax_a.set_yticklabels([lbl for _, lbl in models_a], fontsize=7.0)
    ax_a.get_yticklabels()[5].set_fontweight('bold')
    ax_a.get_yticklabels()[6].set_fontweight('bold')
    ax_a.set_xlabel(r'Transfer MSE ($\mathrm{Mg^2\ ha^{-2}}$)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_a)

    # Panel b: 1x3 Side-by-Side Small-Multiples Stacked MSE Comparison (One Row per Task, Solid Opacity)
    yt = np.arange(len(TRANSFER), dtype=float)
    task_full_labels = [
        '2024 F1 → 2024 F2',
        '2024 F2 → 2024 F1',
        '2023 F1 → 2024 F1',
        '2023 F1 → 2024 F2',
        '2024 F1 → 2023 F1',
        '2024 F2 → 2023 F1',
    ]
    comp_models = [
        ('N_Leaf_T__plus_3N0', r'$N{+}\mathrm{Leaf}{+}T$' + '\n' + r'($+3\mathrm{N0}$)'),
        ('N_ResNet50_Masked_p12__plus_3N0', r'$N{+}\mathrm{Masked\ RN50}$' + '\n' + r'($+3\mathrm{N0}$)'),
        ('N_ResNet50_PCA52_common4__plus_3N0', r'$N{+}\mathrm{Unmasked\ RN50}$' + '\n' + r'($+3\mathrm{N0}$)'),
    ]
    xb0_b, xb1_b = 0.588, 0.982
    gap_sub_b = 0.010
    w_sub_b = (xb1_b - xb0_b - 2 * gap_sub_b) / 3.0
    for j_b, (mid, m_title) in enumerate(comp_models):
        ax_b_sub = fig.add_axes([xb0_b + j_b * (w_sub_b + gap_sub_b), y_bot1, w_sub_b, h1], facecolor='white')
        ax_b_sub.set_axisbelow(True)
        for r_i in range(len(TRANSFER)):
            if r_i % 2 == 1:
                ax_b_sub.axhspan(r_i - 0.46, r_i + 0.46, fc='#F8FAFC', ec='none', zorder=0.1)
        ax_b_sub.axhline(1.5, color='#CBD5E1', lw=0.70, zorder=1)
        ax_b_sub.grid(axis='x', lw=0.38, color='#F1F5F9', zorder=0.5)

        sub_m = dec[dec.model_id == mid].pivot(index=['source', 'target', 'evaluation'], columns='measure', values='estimate').reindex(TRANSFER)
        wm = (sub_m['Within-N'].to_numpy(float) / 1000.0) ** 2
        bm = (sub_m['Between-N'].to_numpy(float) / 1000.0) ** 2
        bim = (sub_m['Bias'].to_numpy(float) / 1000.0) ** 2
        ax_b_sub.barh(yt, wm, height=0.56, facecolor=c_wn, edgecolor='#0F172A', lw=0.38, zorder=3)
        ax_b_sub.barh(yt, bm, left=wm, height=0.56, facecolor=c_bn, edgecolor='#0F172A', lw=0.38, zorder=3)
        ax_b_sub.barh(yt, bim, left=wm + bm, height=0.56, facecolor=c_bi, edgecolor='#0F172A', lw=0.38, zorder=3)

        ax_b_sub.set_xlim(0, 3.08)
        ax_b_sub.set_ylim(len(TRANSFER) - 0.45, -0.52)
        ax_b_sub.set_xticks([0, 1, 2, 3])
        ax_b_sub.set_yticks(yt)
        if j_b == 0:
            ax_b_sub.set_yticklabels(task_full_labels, fontsize=7.0)
        else:
            ax_b_sub.set_yticklabels([''] * len(TRANSFER))
        if j_b == 1:
            ax_b_sub.set_xlabel(r'Transfer MSE ($\mathrm{Mg^2\ ha^{-2}}$)', fontsize=7.5, labelpad=2.0)
        ax_b_sub.text(0.50, 1.03, m_title, transform=ax_b_sub.transAxes,
                      ha='center', va='bottom', fontsize=7.0, fontweight='bold', color=INK, clip_on=False)
        style_axis(ax_b_sub)

    mse_handles = [
        Patch(facecolor=c_wn, edgecolor='#0F172A', lw=0.45, label=r'Within-$N$ MSE'),
        Patch(facecolor=c_bn, edgecolor='#0F172A', lw=0.45, label=r'Between-$N$ MSE'),
        Patch(facecolor=c_bi, edgecolor='#0F172A', lw=0.45, label=r'$\mathrm{Bias^2}$'),
    ]
    fig.legend(handles=mse_handles, loc='upper center', bbox_to_anchor=(0.308, 0.990),
               ncol=3, frameon=False, columnspacing=0.80, handlelength=0.95, handletextpad=0.22, fontsize=7.0)

    # --- Row 2 (c + d): Physical N-Response Curves (c) + Plot-Level Residual Shift Distribution (d) ---
    y_bot2, y_top2 = 0.382, 0.624
    h2 = y_top2 - y_bot2
    fig.text(0.012, y_top2 + 0.024, 'c', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')
    fig.text(0.506, y_top2 + 0.024, 'd', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')

    # Panel c: Physical N-Response Curves & Slope-Bias Mismatch under Scalar Shift (2023 F1 -> 2024 F2)
    ax_c = fig.add_axes([0.068, y_bot2, 0.418, h2], facecolor='white')
    ax_c.set_axisbelow(True)
    ax_c.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)

    obs_sub = sub_f2[sub_f2.specification == 'N_Leaf_T']
    ax_c.scatter(obs_sub['N'], obs_sub['observed'] / 1000.0, s=13, facecolor='#E2E8F0', edgecolor='#64748B',
                 linewidth=0.40, alpha=0.75, zorder=2.0)

    obs_n = obs_sub.groupby('N')['observed'].mean() / 1000.0
    lf_n = sub_f2[sub_f2.specification == 'N_Leaf_T'].groupby('N')[['pred_zero', 'pred_plus_3N0']].mean() / 1000.0
    rn_n = sub_f2[sub_f2.specification == 'N_ResNet50'].groupby('N')[['pred_zero', 'pred_plus_3N0']].mean() / 1000.0

    nx = obs_n.index.to_numpy(float)
    ax_c.plot(nx, obs_n.to_numpy(), 'o-', color='#0F172A', lw=1.25, ms=3.6, label='Observed mean', zorder=5)
    ax_c.plot(nx, lf_n['pred_zero'].to_numpy(), '--', color=PALETTE['teal_mid'], lw=0.95, label=r'$N{+}\mathrm{Leaf}{+}T$ ($k{=}0$)', zorder=3)
    ax_c.plot(nx, lf_n['pred_plus_3N0'].to_numpy(), '-', color=PALETTE['teal'], lw=1.25, label=r'$N{+}\mathrm{Leaf}{+}T{+}3\mathrm{N0}$', zorder=4.2)
    ax_c.plot(nx, rn_n['pred_zero'].to_numpy(), '--', color=PALETTE['crimson_mid'], lw=0.95, label=r'$N{+}\mathrm{Unmasked\ RN50}$ ($k{=}0$)', zorder=3)
    ax_c.plot(nx, rn_n['pred_plus_3N0'].to_numpy(), '-', color=PALETTE['crimson'], lw=1.25, label=r'$N{+}\mathrm{Unmasked\ RN50}{+}3\mathrm{N0}$', zorder=4.5)

    # Vertical shift arrows at N = 300 illustrating additive shift
    y_lf0, y_lf3 = float(lf_n.loc[300, 'pred_zero']), float(lf_n.loc[300, 'pred_plus_3N0'])
    y_rn0, y_rn3 = float(rn_n.loc[300, 'pred_zero']), float(rn_n.loc[300, 'pred_plus_3N0'])
    ax_c.annotate('', xy=(295, y_lf3 - 0.04), xytext=(295, y_lf0 + 0.04),
                  arrowprops=dict(arrowstyle='->', color=PALETTE['teal'], lw=1.05, mutation_scale=7.5), zorder=4.6)
    ax_c.annotate('', xy=(305, y_rn3 + 0.04), xytext=(305, y_rn0 - 0.04),
                  arrowprops=dict(arrowstyle='->', color=PALETTE['crimson'], lw=1.05, mutation_scale=7.5), zorder=4.6)

    ax_c.set_xlim(-12, 462)
    ax_c.set_ylim(2.50, 8.60)
    ax_c.set_xticks([0, 150, 300, 450])
    ax_c.set_yticks([3, 5, 7])
    ax_c.set_xlabel(r'N rate ($\mathrm{kg\ N\ ha^{-1}}$)', fontsize=7.5, labelpad=2.0)
    ax_c.set_ylabel(r'Yield ($\mathrm{Mg\ ha^{-1}}$)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_c)
    ax_c.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.88,
                fontsize=7.0, handlelength=1.05, handletextpad=0.20, borderpad=0.06, labelspacing=0.12)

    # Panel d: Quantitative Plot-Level Residual Shift Distribution (Pred - Obs, Mg ha^-1) in 2023 F1 -> 2024 F2 (n = 21)
    ax_d = fig.add_axes([0.562, y_bot2, 0.420, h2], facecolor='white')
    ax_d.set_axisbelow(True)
    ax_d.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
    ax_d.axhline(0.0, color='#334155', lw=0.90, ls='--', zorder=2.0)
    ax_d.axvline(1.5, color='#CBD5E1', lw=0.70, zorder=1.0)

    lf_f2 = sub_f2[sub_f2.specification == 'N_Leaf_T'].sort_values(['N', 'block'])
    rn_f2 = sub_f2[sub_f2.specification == 'N_ResNet50'].sort_values(['N', 'block'])

    res_groups = [
        (r'$N{+}\mathrm{Leaf}{+}T$' + '\n' + r'($k{=}0$)', (lf_f2['pred_zero'] - lf_f2['observed']) / 1000.0, lf_f2['N'].to_numpy(int), PALETTE['teal_mid']),
        (r'$N{+}\mathrm{Leaf}{+}T$' + '\n' + r'($+3\mathrm{N0}$)', (lf_f2['pred_plus_3N0'] - lf_f2['observed']) / 1000.0, lf_f2['N'].to_numpy(int), PALETTE['teal']),
        (r'$N{+}\mathrm{Unmasked}$' + '\n' + r'RN50 ($k{=}0$)', (rn_f2['pred_zero'] - rn_f2['observed']) / 1000.0, rn_f2['N'].to_numpy(int), PALETTE['crimson_mid']),
        (r'$N{+}\mathrm{Unmasked}$' + '\n' + r'RN50 ($+3\mathrm{N0}$)', (rn_f2['pred_plus_3N0'] - rn_f2['observed']) / 1000.0, rn_f2['N'].to_numpy(int), PALETTE['crimson']),
    ]

    rng_d = np.random.default_rng(42)
    for gi, (g_lbl, res_s, nv_arr, c_box) in enumerate(res_groups):
        res_v = res_s.to_numpy(float)
        fert_mask = nv_arr > 0
        n0_mask = nv_arr == 0

        # Boxplot for fertilized holdout plots (n = 18)
        bp_d = ax_d.boxplot([res_v[fert_mask]], positions=[gi - 0.12], widths=0.22,
                            patch_artist=True, showfliers=False, zorder=3.0)
        for b_el in bp_d['boxes']:
            b_el.set(facecolor=c_box, edgecolor='#0F172A', alpha=0.24, linewidth=0.65)
        for m_el in bp_d['medians']:
            m_el.set(color='#0F172A', linewidth=1.15)
        for w_el in bp_d['whiskers'] + bp_d['caps']:
            w_el.set(color='#334155', linewidth=0.55)

        # Jittered plot-level residuals: Fertilized holdout (circles colored by N rate) + N0 checks (amber diamonds)
        jit_f = rng_d.uniform(-0.05, 0.05, size=int(fert_mask.sum()))
        for r_i_v, n_i_v, jf_i in zip(res_v[fert_mask], nv_arr[fert_mask], jit_f):
            ax_d.scatter([gi + 0.14 + jf_i], [r_i_v], s=17, marker='o',
                         facecolor=N_COLORS.get(int(n_i_v), PALETTE['cerulean']),
                         edgecolor='#0F172A', linewidth=0.38, alpha=0.90, zorder=3.8)
        jit_0 = np.linspace(-0.04, 0.04, int(n0_mask.sum()))
        ax_d.scatter(gi + 0.14 + jit_0, res_v[n0_mask], s=24, marker='D',
                     facecolor=PALETTE['amber'], edgecolor='#0F172A', linewidth=0.55, zorder=4.5)

    ax_d.set_xlim(-0.36, 3.34)
    ax_d.set_ylim(-2.78, 1.32)
    ax_d.set_xticks([0, 1, 2, 3])
    ax_d.set_xticklabels([g[0] for g in res_groups], fontsize=7.0)
    ax_d.set_yticks([-2, -1, 0, 1])
    ax_d.set_ylabel(r'Yield residual ($\mathrm{Mg\ ha^{-1}}$)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_d)

    d_leg_handles = [
        Line2D([], [], marker='D', ls='', ms=4.2, mfc=PALETTE['amber'], mec='#0F172A', mew=0.55, label='N0 check'),
        Line2D([], [], marker='o', ls='', ms=3.8, mfc=N_COLORS[75], mec='#0F172A', mew=0.38, label='N75'),
        Line2D([], [], marker='o', ls='', ms=3.8, mfc=N_COLORS[225], mec='#0F172A', mew=0.38, label='N225'),
        Line2D([], [], marker='o', ls='', ms=3.8, mfc=N_COLORS[450], mec='#0F172A', mew=0.38, label='N450'),
    ]
    ax_d.legend(handles=d_leg_handles, loc='upper right', ncol=2, frameon=True, facecolor='white', edgecolor='none',
                framealpha=0.90, fontsize=7.0, columnspacing=0.55, handlelength=0.80, handletextpad=0.18, borderpad=0.06, labelspacing=0.12)

    # --- Row 3 (e + f): Shared-Y 1x2 Commercial Budget Curves (e: strategies; f: multi-model active convergence) ---
    y_bot3, y_top3 = 0.060, 0.302
    h3 = y_top3 - y_bot3
    fig.text(0.014, y_top3 + 0.024, 'e', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')
    fig.text(0.526, y_top3 + 0.024, 'f', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')

    curves = pd.read_csv(Q1_DIR / 'fewshot_calibration_curves.csv')
    c_fert = curves[curves.support == 'unsampled_fertilized']
    n_ora_rmse = float(dec[(dec.model_id == 'N__oracle') & (dec.measure == 'Total')]['estimate'].mean())

    ax_e = fig.add_axes([0.082, y_bot3, 0.432, h3], facecolor='white')
    ax_e.set_axisbelow(True)
    ax_e.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
    ax_e.axhline(n_ora_rmse, color='#334155', lw=0.85, ls=':', zorder=1.8)
    ax_e.axvline(3.0, color='#64748B', lw=0.75, ls='--', zorder=1.5)
    ax_e.text(0.08, n_ora_rmse - 16.0, r'$N$-only shift ref.', fontsize=7.0, color='#334155', ha='left', va='top')

    leaf_c = c_fert[c_fert.specification == 'N_Leaf_T']
    rnd = leaf_c[leaf_c.strategy == 'Random_all'].groupby('k')[['mean_RMSE', 'q10_RMSE', 'q90_RMSE']].mean().reset_index()
    ax_e.fill_between(rnd.k, rnd.q10_RMSE, rnd.q90_RMSE, color='#94A3B8', alpha=0.20, lw=0, label='Random (10–90%)', zorder=1)

    for st_k, lbl, col_hex, mk, ls_s in [
        ('Random_all', 'Random mean', PALETTE['slate_mid'], 'o', '--'),
        ('N0_check', 'N0 check', PALETTE['amber'], 's', '-'),
        ('Farmer_N225', 'Standard N225', PALETTE['cerulean'], '^', '-'),
        ('UAV_Stratified_Pred', 'Pred.-stratified', PALETTE['teal'], 'D', '-'),
    ]:
        g_s = leaf_c[leaf_c.strategy == st_k].groupby('k')['mean_RMSE'].mean().reset_index()
        ax_e.plot(g_s.k, g_s.mean_RMSE, ls=ls_s, lw=1.10, marker=mk, ms=3.7,
                  color=col_hex, mfc=col_hex, mec='#0F172A', mew=0.35, label=lbl, zorder=3.5)

    ax_e.set_xlim(-0.16, 6.16)
    ax_e.set_ylim(490, 1120)
    ax_e.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax_e.set_yticks([600, 750, 900, 1050])
    ax_e.set_xlabel(r'Calibration plots ($k$)', fontsize=7.5, labelpad=2.0)
    ax_e.set_ylabel(r'Transfer RMSE ($\mathrm{kg\ ha^{-1}}$)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_e)
    ax_e.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.88,
                fontsize=7.0, handlelength=1.0, handletextpad=0.20, borderpad=0.06, labelspacing=0.12)

    # Panel f: Shared-Y Multi-Model Quantile-Stratified Calibration Convergence (k = 0..6)
    ax_f = fig.add_axes([0.548, y_bot3, 0.432, h3], facecolor='white')
    ax_f.set_axisbelow(True)
    ax_f.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
    ax_f.axhline(n_ora_rmse, color='#334155', lw=0.85, ls=':', zorder=1.8)
    ax_f.axvline(3.0, color='#64748B', lw=0.75, ls='--', zorder=1.5)

    rn_masked_curves = pd.read_csv(Q1_DIR / 'resnet_mask_pca_fewshot_curves.csv')
    c_fert_ext = pd.concat([
        c_fert[['specification', 'strategy', 'k', 'mean_RMSE']],
        rn_masked_curves[['specification', 'strategy', 'k', 'mean_RMSE']],
    ], ignore_index=True)

    for sp_k, lbl, col_hex, mk, ls_m in [
        ('N', r'$N$-only', PALETTE['slate'], 'o', '--'),
        ('N_C_T', r'$N{+}C{+}T$', PALETTE['cerulean'], 's', '-'),
        ('N_Leaf_T', r'$N{+}\mathrm{Leaf}{+}T$', PALETTE['teal'], 'v', '-'),
        ('N_ResNet50_Masked_p12', r'$N{+}\mathrm{Masked\ ResNet\text{-}50}$', '#7C3AED', 'd', '-.'),
        ('N_ResNet50_PCA52_common4', r'$N{+}\mathrm{Unmasked\ ResNet\text{-}50}$', PALETTE['crimson'], 'P', '-'),
    ]:
        fg = c_fert_ext[(c_fert_ext.specification == sp_k) & (c_fert_ext.strategy == 'UAV_Stratified_Pred')].groupby('k')['mean_RMSE'].mean().reset_index()
        ax_f.plot(fg.k, fg.mean_RMSE, ls=ls_m, lw=1.10, marker=mk, ms=3.7,
                  color=col_hex, mfc=col_hex, mec='#0F172A', mew=0.35, label=lbl, zorder=3.5)

    ax_f.set_xlim(-0.16, 6.16)
    ax_f.set_ylim(490, 1120)
    ax_f.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax_f.set_yticks([600, 750, 900, 1050])
    ax_f.set_yticklabels([''] * 4)
    ax_f.set_xlabel(r'Calibration plots ($k$)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_f)
    ax_f.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.88,
                fontsize=7.0, handlelength=1.0, handletextpad=0.20, borderpad=0.06, labelspacing=0.12)

    save_fig(fig, MAIN_STEMS[4])


# ==============================================================================
# FIGURE 6 (a-d): Full-Width 1x3 Forward-Only Forecast Horizon Trajectories (a),
#                 Shaded 90% PI Fan Ribbons (b), Grouped Before/After Coverage Bars (c),
#                 2D Calibration Accuracy-Reliability Phase Diagram (d)
# ==============================================================================
def render_figure6() -> None:
    fs.apply()
    fig_w_mm, fig_h_mm = 180.0, 140.0
    # Figure setup with internal helper functions for coordinate normalization
    def _y(y_old: float) -> float:
        return y_old
    def _h(h_old: float) -> float:
        return h_old

    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4), facecolor='white')

    # --- Row 1 (a): Full-Width 1x3 Continuous Forward-Only Recursive Forecast Trajectories ---
    y_bot1, y_top1 = 0.620, 0.930
    h1 = y_top1 - y_bot1
    fig.text(0.012, 0.978, 'a', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')

    d_rec = pd.concat([
        pd.read_csv(OLD_DM / 'recursive_metrics.csv'),
        pd.read_csv(Q1_DIR / 'rich_fig6_recursive_leaf.csv'),
    ], ignore_index=True)
    selected = [ROLES[0], ROLES[3], ROLES[8]]
    x_a0, x_right = 0.072, 0.982
    gap_a = 0.015
    wa = (x_right - x_a0 - 2 * gap_a) / 3.0
    top_keys = [
        ('C', r'$N{+}C$', PALETTE['slate_mid'], 'o', ':'),
        ('C+T', r'$N{+}C{+}T$', PALETTE['cerulean'], 's', '--'),
        ('Leaf+T', r'$N{+}\mathrm{Leaf}{+}T$', PALETTE['teal'], 'D', '-'),
    ]

    for j, task in enumerate(selected):
        ax = fig.add_axes([x_a0 + j * (wa + gap_a), y_bot1, wa, h1], facecolor='white')
        ax.set_axisbelow(True)
        ax.axvspan(120, 142, fc='#F1F5F9', ec='none', zorder=0.2)
        ax.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
        ax.axhline(0.0, color='#334155', lw=0.85, ls='--', zorder=2.0)

        for variant, lbl_k, col_c, mk_s, ls_s in top_keys:
            z = d_rec.loc[
                d_rec.source.eq(task[0]) & d_rec.target.eq(task[1]) & d_rec.clock.eq('DAS') & d_rec.variant.eq(variant)
            ].sort_values('cutoff')
            if variant in ('C+T', 'Leaf+T'):
                ax.fill_between(z.cutoff, z.low, z.high, color=col_c, alpha=0.12 if variant == 'C+T' else 0.16, lw=0, zorder=2.0)
            ax.plot(z.cutoff, z.gain, color=col_c, ls=ls_s, marker=mk_s,
                    ms=3.1 if variant != 'Leaf+T' else 3.4,
                    lw=1.00 if variant != 'Leaf+T' else 1.18,
                    mfc=col_c, mec='#0F172A', mew=0.35, label=lbl_k,
                    zorder=3.0 if variant == 'C' else (3.3 if variant == 'C+T' else 3.8))

        if task == ROLES[3]:
            ax.set_xlim(60, 184)
            ax.set_xticks([60, 90, 120, 150, 180])
        else:
            ax.set_xlim(82, 184)
            ax.set_xticks([90, 120, 150, 180])
        ax.set_ylim(-385, 510)
        ax.set_yticks([-300, -150, 0, 150, 300, 450])
        style_axis(ax)
        if j == 0:
            ax.set_ylabel(r'Total $\Delta\mathrm{RMSE}$ ($\mathrm{kg\ ha^{-1}}$)', fontsize=7.5, labelpad=2.0)
        else:
            ax.set_yticklabels([''] * 6)
        if j == 1:
            ax.set_xlabel('Days after sowing (DAS)', fontsize=7.5, labelpad=2.2)

        short_tag = {
            ROLES[0]: '2023 F1 CV',
            ROLES[3]: '2024 F1 → 2024 F2',
            ROLES[8]: '2024 F2 → 2023 F1',
        }[task]
        ax.text(0.95, 0.94, short_tag, transform=ax.transAxes, ha='right', va='top',
                fontsize=7.0, fontweight='bold', color='#0F172A',
                bbox=dict(boxstyle='round,pad=0.14', facecolor='white', edgecolor='none', alpha=0.86))

    a_leg_handles = [
        Line2D([], [], color=PALETTE['slate_mid'], ls=':', lw=1.00, marker='o', ms=3.5, mfc=PALETTE['slate_mid'], mec='#0F172A', mew=0.35, label=r'$N{+}C$'),
        Line2D([], [], color=PALETTE['cerulean'], ls='--', lw=1.05, marker='s', ms=3.5, mfc=PALETTE['cerulean'], mec='#0F172A', mew=0.35, label=r'$N{+}C{+}T$'),
        Line2D([], [], color=PALETTE['teal'], ls='-', lw=1.18, marker='D', ms=3.6, mfc=PALETTE['teal'], mec='#0F172A', mew=0.35, label=r'$N{+}\mathrm{Leaf}{+}T$'),
        Patch(facecolor='#F1F5F9', edgecolor='#CBD5E1', lw=0.45, label='Boll filling (120–142 DAS)'),
    ]
    fig.legend(handles=a_leg_handles, loc='upper center', bbox_to_anchor=(0.528, 0.988),
               ncol=4, frameon=False, columnspacing=1.05, handlelength=1.20, handletextpad=0.28, fontsize=7.0)

    # --- Row 2 (b, c, d): Shaded 90% PI Fan Ribbons (b) + Grouped Coverage Bars (c) + Reliability-Sharpness Diagram (d) ---
    y_bot2, y_top2 = 0.068, 0.490
    h2 = y_top2 - y_bot2
    fig.text(0.012, y_top2 + 0.036, 'b', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')
    fig.text(0.320, y_top2 + 0.036, 'c', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')
    fig.text(0.674, y_top2 + 0.036, 'd', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')

    # Panel b: Shaded 90% Prediction Interval Fan / Ribbon Charts (Tight Per-Task Plot Rank X-Axis)
    p_int = pd.read_csv(Q1_DIR / 'rich_fig6_plot_intervals.csv')
    sub_h_b = (h2 - 0.034) / 2.0
    ax_b1 = fig.add_axes([0.060, y_bot2 + sub_h_b + 0.034, 0.244, sub_h_b], facecolor='white')
    ax_b2 = fig.add_axes([0.060, y_bot2, 0.244, sub_h_b], facecolor='white')

    for ax_bi, t_key, t_tag in [
        (ax_b1, '2024_F2->2024_F1', '2024 F2 → 2024 F1'),
        (ax_b2, '2023_F1->2024_F2', '2023 F1 → 2024 F2'),
    ]:
        ax_bi.set_axisbelow(True)
        ax_bi.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
        s_lf = p_int[(p_int.task == t_key) & (p_int.model == 'N+Leaf+T+3N0')].sort_values('plot_rank')
        s_rn = p_int[(p_int.task == t_key) & (p_int.model == 'N+ResNet-50+3N0')].sort_values('plot_rank')
        xr = s_lf['plot_rank'].to_numpy(float) + 1.0

        ax_bi.fill_between(xr, s_lf.low_90_mg, s_lf.high_90_mg, facecolor=PALETTE['teal_mid'], edgecolor=PALETTE['teal'],
                           alpha=0.22, lw=0.60, zorder=2.0)
        ax_bi.plot(xr, s_lf.pred_cal_mg, color=PALETTE['teal'], lw=1.05, zorder=2.8)

        ax_bi.fill_between(xr, s_rn.low_90_mg, s_rn.high_90_mg, facecolor=PALETTE['crimson_mid'], edgecolor=PALETTE['crimson'],
                           alpha=0.18, lw=0.60, ls='--', zorder=2.2)
        ax_bi.plot(xr, s_rn.pred_cal_mg, color=PALETTE['crimson'], lw=1.00, ls='--', zorder=2.9)

        ax_bi.scatter(xr, s_lf.obs_mg, s=12, facecolor='#0F172A', edgecolor='white', linewidth=0.30, zorder=4.0)
        missed_rn = s_rn[s_rn.covered == 0]
        if len(missed_rn) > 0:
            ax_bi.scatter(missed_rn.plot_rank.to_numpy(float) + 1.0, missed_rn.obs_mg, s=28,
                          facecolor='none', edgecolor=PALETTE['alert'], linewidth=0.95, zorder=5.0)
        missed_lf = s_lf[s_lf.covered == 0]
        if len(missed_lf) > 0:
            ax_bi.scatter(missed_lf.plot_rank.to_numpy(float) + 1.0, missed_lf.obs_mg, s=46, marker='D',
                          facecolor='none', edgecolor=PALETTE['teal'], linewidth=0.90, zorder=5.2)

        n_max = int(xr.max())
        ax_bi.set_xlim(0.6, n_max + 0.4)
        ax_bi.set_ylim(2.8, 9.5)
        ax_bi.set_xticks([1, 6, 11, 16] if n_max == 16 else [1, 6, 12, 18])
        ax_bi.set_yticks([3, 6, 9])
        style_axis(ax_bi)
        ax_bi.text(0.04, 0.92, t_tag, transform=ax_bi.transAxes, ha='left', va='top',
                   fontsize=7.0, fontweight='bold', color='#0F172A',
                   bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.86))

    ax_b2.set_xlabel('Ordered plots (N75–N450)', fontsize=7.5, labelpad=2.0)
    fig.text(0.018, y_bot2 + h2 / 2.0, r'Yield ($\mathrm{Mg\ ha^{-1}}$)', rotation=90,
             va='center', ha='center', fontsize=7.5, color='#0F172A')

    b_leg_handles = [
        Patch(facecolor=PALETTE['teal_mid'], edgecolor=PALETTE['teal'], alpha=0.30, lw=0.6, label=r'$N{+}\mathrm{Leaf}{+}T{+}3\mathrm{N0}$'),
        Patch(facecolor=PALETTE['crimson_mid'], edgecolor=PALETTE['crimson'], alpha=0.24, lw=0.6, ls='--', label=r'$N{+}\mathrm{ResNet\text{-}50}{+}3\mathrm{N0}$'),
        Line2D([], [], marker='o', ls='', ms=3.6, mfc='#0F172A', mec='white', mew=0.3, label='Observed'),
        Line2D([], [], marker='D', ls='', ms=4.8, mfc='none', mec=PALETTE['teal'], mew=0.9, label='Missed (Leaf)'),
        Line2D([], [], marker='o', ls='', ms=4.8, mfc='none', mec=PALETTE['alert'], mew=0.9, label='Missed (RN)'),
    ]
    ax_b1.legend(handles=b_leg_handles, loc='lower right', ncol=2, frameon=True, facecolor='white', edgecolor='none',
                 framealpha=0.88, fontsize=7.0, columnspacing=0.42, handlelength=0.80, handletextpad=0.18, borderpad=0.04, labelspacing=0.08)

    # Panel c: Grouped Horizontal Before (k=0) vs After (+3N0) Empirical Coverage Bars across 6 Transfers + Mean
    cp_base = pd.read_csv(Q1_DIR / 'calibrated_interval_coverage.csv')
    cp_mask = pd.read_csv(Q1_DIR / 'resnet_masked_p12_coverage.csv').rename(columns={'model': 'model_raw'})
    cp_mask['model'] = cp_mask['model_raw'].map({
        'N_ResNet50_Masked_p12__zero_label': 'N+Masked ResNet-50',
        'N_ResNet50_Masked_p12__plus_3N0': 'N+Masked ResNet-50+3N0',
    })
    cp = pd.concat([
        cp_base[['source', 'target', 'evaluation', 'model', 'coverage']],
        cp_mask.dropna(subset=['model'])[['source', 'target', 'evaluation', 'model', 'coverage']],
    ], ignore_index=True)

    ax_c = fig.add_axes([0.444, y_bot2, 0.220, h2], facecolor='white')
    ax_c.set_axisbelow(True)
    yd = np.arange(len(TRANSFER) + 1, dtype=float)
    ax_c.axhspan(yd[-1] - 0.50, yd[-1] + 0.50, fc='#F8FAFC', ec='none', zorder=0.1)
    ax_c.axhline(1.5, color='#CBD5E1', lw=0.65, zorder=1.0)
    ax_c.axhline(5.5, color='#94A3B8', lw=0.80, zorder=1.0)
    ax_c.grid(axis='x', lw=0.38, color='#F1F5F9', zorder=0.5)
    ax_c.axvline(90.0, color=PALETTE['teal'], ls='--', lw=0.90, zorder=2.5)

    bar_specs = [
        ('N+Leaf+T', r'Leaf+$T$ ($k{=}0$)', PALETTE['teal_light'], PALETTE['teal'], -0.30),
        ('N+Leaf+T+3N0', r'Leaf+$T$+3N0', PALETTE['teal'], '#0F172A', -0.18),
        ('N+Masked ResNet-50', r'Masked ($k{=}0$)', '#DDD6FE', '#7C3AED', -0.06),
        ('N+Masked ResNet-50+3N0', r'Masked+3N0', '#7C3AED', '#0F172A', +0.06),
        ('N+ResNet-50', r'Unmasked ($k{=}0$)', PALETTE['crimson_light'], PALETTE['crimson'], +0.18),
        ('N+ResNet-50+3N0', r'Unmasked+3N0', PALETTE['crimson'], '#0F172A', +0.30),
    ]
    for csv_m, b_lbl, fc_b, ec_b, dy in bar_specs:
        z = cp[cp.model == csv_m].set_index(['source', 'target', 'evaluation']).reindex(TRANSFER)
        t_cov = z.coverage.to_numpy(float) * 100.0
        vals = np.append(t_cov, float(t_cov.mean()))
        se_cov = float(t_cov.std(ddof=1) / np.sqrt(len(t_cov)))
        ax_c.barh(yd + dy, vals, height=0.115, facecolor=fc_b, edgecolor=ec_b, linewidth=0.40, label=b_lbl, zorder=3.0)
        ax_c.errorbar(vals[-1], yd[-1] + dy, xerr=se_cov, fmt='none', ecolor='#0F172A', elinewidth=0.60, capsize=1.2, zorder=4.0)

    ax_c.set_xlim(0, 102.5)
    ax_c.set_ylim(yd[-1] + 0.52, -0.52)
    ax_c.set_xticks([0, 25, 50, 75, 100])
    ax_c.set_yticks(yd)
    ax_c.set_yticklabels(['2024 F1 → 2024 F2', '2024 F2 → 2024 F1', '2023 F1 → 2024 F1', '2023 F1 → 2024 F2', '2024 F1 → 2023 F1', '2024 F2 → 2023 F1', '6-task mean'], fontsize=7.0)
    ax_c.get_yticklabels()[-1].set_fontweight('bold')
    ax_c.set_xlabel('90% PI coverage (%)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_c)
    ax_c.legend(loc='lower center', bbox_to_anchor=(0.26, 1.008), ncol=3, frameon=False,
                columnspacing=0.42, handlelength=0.72, handletextpad=0.16, labelspacing=0.10, fontsize=7.0)

    # Panel d: 2D Calibration Accuracy vs. Reliability Phase Diagram (Mean RMSE vs. Mean 90% Coverage ± 1 SE)
    dec6 = pd.concat([
        pd.read_csv(Q1_DIR / 'hybrid_transfer_decomposition.csv'),
        pd.read_csv(Q1_DIR / 'resnet_masked_p12_table3_decomposition.csv'),
    ], ignore_index=True)
    ax_d = fig.add_axes([0.728, y_bot2, 0.254, h2], facecolor='white')
    ax_d.set_axisbelow(True)
    ax_d.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
    ax_d.axhline(90.0, color='#334155', lw=0.85, ls='--', zorder=2.0)
    ax_d.text(1.098, 90.8, 'Nominal 90%', fontsize=7.0, color='#334155', ha='right', va='bottom')

    pareto_families = [
        ('N__zero_label', 'N__plus_3N0', 'N-only', 'N+3N0', r'$N$-only', PALETTE['slate'], 'o', 0.0, 3.8, 3.8),
        ('N_Leaf_T__zero_label', 'N_Leaf_T__plus_3N0', 'N+Leaf+T', 'N+Leaf+T+3N0', r'$N{+}\mathrm{Leaf}{+}T$', PALETTE['teal'], 'D', 0.0, 3.8, 3.8),
        ('N_ResNet50_Masked_p12__zero_label', 'N_ResNet50_Masked_p12__plus_3N0', 'N+Masked ResNet-50', 'N+Masked ResNet-50+3N0', r'$N{+}\mathrm{Masked\ RN50}$', '#7C3AED', 'd', 0.0, 3.8, 3.8),
        ('N_ResNet50_PCA52_common4__zero_label', 'N_ResNet50_PCA52_common4__plus_3N0', 'N+ResNet-50', 'N+ResNet-50+3N0', r'$N{+}\mathrm{Unmasked\ RN50}$', PALETTE['crimson'], 'P', 0.0, 3.8, 3.8),
    ]
    for id0, id3, m0, m3, f_lbl, col_p, mk_p, arc_rad, sh_a, sh_b in pareto_families:
        r0_arr = dec6[(dec6.model_id == id0) & (dec6.measure == 'Total')]['estimate'].to_numpy(float) / 1000.0
        r3_arr = dec6[(dec6.model_id == id3) & (dec6.measure == 'Total')]['estimate'].to_numpy(float) / 1000.0
        c0_arr = cp[cp.model == m0]['coverage'].to_numpy(float) * 100.0
        c3_arr = cp[cp.model == m3]['coverage'].to_numpy(float) * 100.0

        r0, r3 = float(r0_arr.mean()), float(r3_arr.mean())
        c0, c3 = float(c0_arr.mean()), float(c3_arr.mean())
        r3_se = float(r3_arr.std(ddof=1) / np.sqrt(len(r3_arr)))
        c3_se = float(c3_arr.std(ddof=1) / np.sqrt(len(c3_arr)))

        ax_d.annotate('', xy=(r3, c3), xytext=(r0, c0),
                      arrowprops=dict(arrowstyle='->', color=col_p, lw=1.05, shrinkA=sh_a, shrinkB=sh_b,
                                      mutation_scale=6.5, connectionstyle=f'arc3,rad={arc_rad}'), zorder=3.0)
        ax_d.errorbar(r3, c3, xerr=r3_se, yerr=c3_se, fmt='none', ecolor=col_p, elinewidth=0.70, capsize=1.8, alpha=0.80, zorder=3.8)
        ax_d.plot(r0, c0, marker=mk_p, ms=4.8, mfc='white', mec=col_p, mew=1.0, ls='', zorder=4.0)
        ax_d.plot(r3, c3, marker=mk_p, ms=5.2, mfc=col_p, mec='#0F172A', mew=0.5, ls='', label=f_lbl, zorder=4.5)

    ax_d.set_xlim(0.55, 1.11)
    ax_d.set_ylim(56, 102)
    ax_d.set_xticks([0.60, 0.75, 0.90, 1.05])
    ax_d.set_yticks([60, 70, 80, 90, 100])
    ax_d.set_xlabel(r'Transfer RMSE ($\mathrm{Mg\ ha^{-1}}$)', fontsize=7.5, labelpad=2.0)
    ax_d.set_ylabel('90% PI coverage (%)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_d)

    d_leg_handles = [
        Line2D([], [], marker='o', ls='', ms=4.6, mfc=PALETTE['slate'], mec='#0F172A', mew=0.45, label=r'$N$-only'),
        Line2D([], [], marker='D', ls='', ms=4.6, mfc=PALETTE['teal'], mec='#0F172A', mew=0.45, label=r'$N{+}\mathrm{Leaf}{+}T$'),
        Line2D([], [], marker='d', ls='', ms=5.0, mfc='#7C3AED', mec='#0F172A', mew=0.45, label=r'$N{+}\mathrm{Masked\ RN50}$'),
        Line2D([], [], marker='P', ls='', ms=5.0, mfc=PALETTE['crimson'], mec='#0F172A', mew=0.45, label=r'$N{+}\mathrm{Unmasked\ RN50}$'),
        Line2D([], [], marker='o', ls='', ms=4.4, mfc='white', mec=PALETTE['slate'], mew=0.95, label=r'Open: $k{=}0$'),
    ]
    ax_d.legend(handles=d_leg_handles, loc='lower left', frameon=True, facecolor='white', edgecolor='none', framealpha=0.88,
                fontsize=7.0, handlelength=0.85, handletextpad=0.20, borderpad=0.06, labelspacing=0.11)

    save_fig(fig, MAIN_STEMS[5])


def save_supp_fig(fig: plt.Figure, stem: str) -> None:
    FIG_SUPP.mkdir(parents=True, exist_ok=True)
    FIG_EDIT.mkdir(parents=True, exist_ok=True)
    fs.save(fig, FIG_SUPP, stem)
    _safe_copy(FIG_SUPP / f'{stem}.svg', FIG_EDIT / f'{stem}.svg')
    plt.close(fig)


# ==============================================================================
# SUPPLEMENTARY FIGURE S9: 21-Crop Boll Morphological Upper-Threshold Sensitivity
#                          Diagnostic (Precision, Recall, F1, CDF & Yield r)
# ==============================================================================
def render_figure_s9() -> None:
    fs.apply()
    fig_w_mm, fig_h_mm = 180.0, 68.0
    # Figure setup with internal helper functions for coordinate normalization
    def _y(y_old: float) -> float:
        return y_old
    def _h(h_old: float) -> float:
        return h_old

    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4), facecolor='white')

    sens = pd.read_csv(Q1_DIR / 'boll_cc_threshold_sensitivity_21crops.csv')
    sens = sens[sens['rule'] == 'white_b140_s30'].reset_index(drop=True)
    comp = pd.read_csv(Q1_DIR / 'boll_cc_components_21crops.csv')

    y0, h0 = 0.18, 0.70
    fig.text(0.012, 0.95, 'a', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')
    fig.text(0.352, 0.95, 'b', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')
    fig.text(0.682, 0.95, 'c', fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')

    # Panel a: Precision, Recall, F1 across upper area cutoffs on 21 reviewed crops
    ax_a = fig.add_axes([0.068, y0, 0.258, h0], facecolor='white')
    ax_a.set_axisbelow(True)
    ax_a.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
    xk = np.arange(len(sens), dtype=float)
    xlbls = ['30', '60', '90', '120', '150', '200', '300', '400', '∞']
    ax_a.axvline(3.0, color=PALETTE['amber'], lw=0.85, ls='--', zorder=1.8)

    ax_a.plot(xk, sens['macro_precision'], 'o-', color=PALETTE['cerulean'], lw=1.05, ms=3.4, label='Precision (known)', zorder=3.5)
    ax_a.plot(xk, sens['macro_recall'], 's-', color=PALETTE['teal'], lw=1.05, ms=3.4, label='Recall', zorder=3.6)
    ax_a.plot(xk, sens['macro_F1'], 'D-', color=PALETTE['plum'], lw=1.15, ms=3.6, label=r'$F_1$ (known)', zorder=3.8)
    ax_a.plot(xk, sens['macro_precision_art_bg'], '^--', color=PALETTE['ochre'], lw=0.95, ms=3.4, label='Precision (+glare)', zorder=3.4)
    ax_a.plot(xk, sens['macro_F1_art_bg'], 'v--', color=PALETTE['crimson'], lw=0.95, ms=3.4, label=r'$F_1$ (+glare)', zorder=3.4)

    ax_a.set_xlim(-0.3, len(sens) - 0.7)
    ax_a.set_ylim(0.38, 1.05)
    ax_a.set_xticks(xk)
    ax_a.set_xticklabels(xlbls, fontsize=7.0)
    ax_a.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax_a.set_xlabel(r'Upper area threshold ($\mathrm{cm^2}$)', fontsize=7.5, labelpad=2.0)
    ax_a.set_ylabel('Segmentation metric', fontsize=7.5, labelpad=2.0)
    style_axis(ax_a)
    ax_a.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.90,
                fontsize=7.0, handlelength=0.95, handletextpad=0.20, borderpad=0.05, labelspacing=0.10)

    # Panel b: Empirical Cumulative Pixel Share vs Connected-Component Area (21 reviewed crops)
    ax_b = fig.add_axes([0.404, y0, 0.252, h0], facecolor='white')
    ax_b.set_axisbelow(True)
    ax_b.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
    ax_b.axvline(120.0, color=PALETTE['amber'], lw=0.85, ls='--', zorder=2.0)

    comp_ge05 = comp[comp['area_cm2'] >= 0.5]
    c_boll = comp_ge05[comp_ge05['fiber_px'] > 0].sort_values('area_cm2')
    xb_area = c_boll['area_cm2'].to_numpy(float)
    yb_cdf = np.cumsum(c_boll['fiber_px'].to_numpy(float)) / max(1.0, float(c_boll['fiber_px'].sum())) * 100.0
    ax_b.plot(xb_area, yb_cdf, '-', color=PALETTE['plum'], lw=1.20, label='Confirmed boll pixels', zorder=3.6)

    c_art = comp_ge05[comp_ge05['in_artifact_rect_px'] > 0].sort_values('area_cm2')
    xa_area = c_art['area_cm2'].to_numpy(float)
    ya_cdf = np.cumsum(c_art['in_artifact_rect_px'].to_numpy(float)) / max(1.0, float(c_art['in_artifact_rect_px'].sum())) * 100.0
    ax_b.plot(xa_area, ya_cdf, '--', color=PALETTE['amber'], lw=1.10, label='Glare/artifact pixels', zorder=3.4)

    pct_boll_120 = float(c_boll[c_boll['area_cm2'] <= 120.0]['fiber_px'].sum() / max(1.0, float(c_boll['fiber_px'].sum())) * 100.0)
    ax_b.text(124.0, 26.0, rf'$120\ \mathrm{{cm^2}}$' + f'\n({pct_boll_120:.1f}% bolls)', fontsize=7.0, color=PALETTE['ochre'], ha='left', va='bottom')
    ax_b.set_xscale('log')
    ax_b.set_xlim(0.45, 900.0)
    ax_b.set_ylim(0, 105)
    ax_b.set_xticks([1, 10, 100, 500])
    ax_b.set_xticklabels(['1', '10', '100', '500'])
    ax_b.set_yticks([0, 25, 50, 75, 100])
    ax_b.set_xlabel(r'Component area ($\mathrm{cm^2}$)', fontsize=7.5, labelpad=2.0)
    ax_b.set_ylabel('Cumulative pixel share (%)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_b)
    ax_b.legend(loc='upper left', frameon=True, facecolor='white', edgecolor='none', framealpha=0.90,
                fontsize=7.0, handlelength=0.95, handletextpad=0.20, borderpad=0.05, labelspacing=0.12)

    # Panel c: 64-Plot Yield Correlation Sensitivity across Upper Thresholds (181 DAS)
    ax_c = fig.add_axes([0.736, y0, 0.244, h0], facecolor='white')
    ax_c.set_axisbelow(True)
    ax_c.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
    ax_c.axhline(0.0, color='#64748B', lw=0.75, ls=':', zorder=1.5)
    ax_c.axvline(3.0, color=PALETTE['amber'], lw=0.85, ls='--', zorder=1.8)

    ax_c.plot(xk, sens['plot64_within_env_r'], 'o-', color=PALETTE['slate'], lw=1.05, ms=3.5, label=r'Apparent $r$', zorder=3.4)
    ax_c.plot(xk, sens['plot64_within_N_block_r'], 'D-', color=PALETTE['teal'], lw=1.15, ms=3.7, label=r'Partial $r$ (N, block)', zorder=3.8)

    ax_c.set_xlim(-0.3, len(sens) - 0.7)
    ax_c.set_ylim(-0.68, 0.54)
    ax_c.set_xticks(xk)
    ax_c.set_xticklabels(xlbls, fontsize=7.0)
    ax_c.set_yticks([-0.6, -0.3, 0.0, 0.3])
    ax_c.set_xlabel(r'Upper area threshold ($\mathrm{cm^2}$)', fontsize=7.5, labelpad=2.0)
    ax_c.set_ylabel(r'Correlation with yield ($r$)', fontsize=7.5, labelpad=2.0)
    style_axis(ax_c)
    ax_c.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.90,
                fontsize=7.0, handlelength=0.95, handletextpad=0.20, borderpad=0.05, labelspacing=0.12)

    save_supp_fig(fig, 'Figure_S2_boll_threshold_sensitivity')


# ==============================================================================
# SUPPLEMENTARY FIGURE S10: 2024 F1 Plot-Level Spatial Residual Maps & Moran's I
#                           (N+Leaf+T+3N0 vs. N+ResNet-50+3N0 Overlaid on Orthomosaic)
# ==============================================================================
def render_figure_s10() -> None:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject
    from shapely import wkt
    from shapely.ops import unary_union

    fs.apply()
    fig_w_mm, fig_h_mm = 180.0, 108.0
    # Figure setup with internal helper functions for coordinate normalization
    def _y(y_old: float) -> float:
        return y_old
    def _h(h_old: float) -> float:
        return h_old

    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4), facecolor='white')

    res_df = pd.read_csv(Q1_DIR / 'spatial_residuals_2024F1_plot_level.csv')
    mor_df = pd.read_csv(Q1_DIR / 'spatial_residuals_2024F1_morans_i.csv')

    layout = pd.read_csv(ROOT / '01_Data/processed/spatial/cotton_rgb/field_authority_full_plot.csv')
    sub_full = layout[layout.scope_id == '2024_f1'].copy()
    sub_full['geom_obj'] = sub_full.geom.map(wkt.loads)

    inward = pd.read_csv(ROOT / '01_Data/processed/spatial/cotton_rgb/field_authority_inward_buffer.csv')
    sub_inw = inward[inward.scope_id == '2024_f1'].copy()
    sub_inw['geom_obj'] = sub_inw.geom.map(wkt.loads)
    inw_map = dict(zip(sub_inw['plot_id'].astype(int), sub_inw['geom_obj']))

    hull_geom = unary_union(sub_full['geom_obj'].tolist()).convex_hull
    b = hull_geom.bounds
    pad = 0.45
    dst_bounds = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
    dst_h, dst_w = 720, 440
    dst_transform = from_bounds(*dst_bounds, dst_w, dst_h)
    dst = np.zeros((4, dst_h, dst_w), dtype=np.uint8)
    tif_path = ROOT / '01_Data/raw/remote_sensing/orthomosaics/2024/2024-08-21_f01_visible_orthomosaic.tif'
    with rasterio.open(tif_path) as src:
        reproject(
            source=rasterio.band(src, [1, 2, 3, 4]),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs='EPSG:32644',
            resampling=Resampling.average,
        )
    inside = geometry_mask([hull_geom.__geo_interface__], out_shape=(dst_h, dst_w), transform=dst_transform, invert=True)
    dst[3, ~inside] = 0
    ortho_rgba = np.moveaxis(dst, 0, 2)

    panel_specs = [
        ('a', '2024_F2', 'N+Leaf+T', r'2024 F2 $\rightarrow$ 2024 F1' + '\n' + r'$N{+}\mathrm{Leaf}{+}T{+}3\mathrm{N0}$'),
        ('b', '2024_F2', 'N+ResNet-50', r'2024 F2 $\rightarrow$ 2024 F1' + '\n' + r'$N{+}\mathrm{ResNet\text{-}50}{+}3\mathrm{N0}$'),
        ('c', '2023_F1', 'N+Leaf+T', r'2023 F1 $\rightarrow$ 2024 F1' + '\n' + r'$N{+}\mathrm{Leaf}{+}T{+}3\mathrm{N0}$'),
        ('d', '2023_F1', 'N+ResNet-50', r'2023 F1 $\rightarrow$ 2024 F1' + '\n' + r'$N{+}\mathrm{ResNet\text{-}50}{+}3\mathrm{N0}$'),
    ]

    cmap_res = Q1_DIV_CMAP
    norm_res = TwoSlopeNorm(vmin=-1.8, vcenter=0.0, vmax=1.8)

    x0_map, gap_map = 0.048, 0.022
    w_map = (0.976 - x0_map - 3 * gap_map) / 4.0
    h_map = 0.710
    y0_map = 0.145

    map_aspect = (w_map * fig_w_mm) / (h_map * fig_h_mm)
    span_w = max(dst_bounds[2] - dst_bounds[0], (dst_bounds[3] - dst_bounds[1]) * map_aspect)
    span_h = span_w / map_aspect
    cx_mid = (dst_bounds[0] + dst_bounds[2]) / 2.0
    cy_mid = (dst_bounds[1] + dst_bounds[3]) / 2.0

    for idx, (p_let, src_key, m_key, p_title) in enumerate(panel_specs):
        xm = x0_map + idx * (w_map + gap_map)
        fig.text(xm - 0.008, y0_map + h_map + 0.115, p_let, fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')
        ax = fig.add_axes([xm, y0_map, w_map, h_map], facecolor='white')
        ax.imshow(ortho_rgba, extent=(dst_bounds[0], dst_bounds[2], dst_bounds[1], dst_bounds[3]),
                  origin='upper', interpolation='lanczos', zorder=1)

        # Full 24-plot boundaries in white
        for row_f in sub_full.itertuples():
            gx, gy = row_f.geom_obj.exterior.xy
            ax.plot(np.asarray(gx), np.asarray(gy), color='white', lw=0.45, alpha=0.80, zorder=2.5)

        # Block dividers B1, B2, B3
        for b_idx in (1, 2):
            upper = unary_union(sub_full[sub_full.grid_row.isin([2 * b_idx - 1, 2 * b_idx])]['geom_obj'].tolist())
            lower = unary_union(sub_full[sub_full.grid_row.isin([2 * b_idx + 1, 2 * b_idx + 2])]['geom_obj'].tolist())
            y_div = (upper.bounds[1] + lower.bounds[3]) / 2.0
            ax.plot([b[0] + 0.1, b[2] - 0.1], [y_div, y_div], color='#0F172A', lw=1.05, zorder=4.5)
            ax.plot([b[0] + 0.1, b[2] - 0.1], [y_div, y_div], color='white', lw=0.60, ls='--', zorder=4.6)

        if idx == 0:
            for block in (1, 2, 3):
                u_b = unary_union(sub_full[sub_full.grid_row.isin([2 * block - 1, 2 * block])]['geom_obj'].tolist())
                ax.text(b[0] - 0.35, (u_b.bounds[1] + u_b.bounds[3]) / 2.0, f'B{block}',
                        ha='right', va='center', fontsize=7.0, fontweight='bold', color=INK, zorder=6)

        # Overlay residual choropleth inside the 19 common plots
        sub_r = res_df[(res_df.source == src_key) & (res_df.model == m_key)]
        for r_row in sub_r.itertuples():
            pid = int(r_row.plot_id)
            g_inw = inw_map[pid]
            res_mg = float(r_row.resid_n0_cal) / 1000.0
            is_n0 = int(r_row.N) == 0
            fc_rgba = cmap_res(norm_res(np.clip(res_mg, -1.8, 1.8)))
            poly_patch = MplPolygon(
                np.asarray(g_inw.exterior.coords),
                closed=True,
                facecolor=fc_rgba,
                edgecolor=PALETTE['amber'] if is_n0 else '#0F172A',
                linewidth=1.10 if is_n0 else 0.55,
                alpha=0.88,
                zorder=3.5,
            )
            ax.add_patch(poly_patch)
            pt = g_inw.representative_point()
            val_str = f'−{abs(res_mg):.2f}' if res_mg < 0 else f'+{res_mg:.2f}'
            tag_txt = f'N0\n{val_str}' if is_n0 else val_str
            lum_r = 0.2126 * fc_rgba[0] + 0.7152 * fc_rgba[1] + 0.0722 * fc_rgba[2]
            txt_col = 'white' if lum_r < 0.46 else '#0F172A'
            ax.text(pt.x, pt.y, tag_txt, ha='center', va='center', fontsize=7.0,
                    fontweight='bold' if is_n0 else 'normal', color=txt_col, zorder=5.0)

        # Summary header & Moran's I statistics
        m_sub = mor_df[
            (mor_df.source == src_key) & (mor_df.model == m_key) & (mor_df.calibration == '+3N0 calibrated')
            & (mor_df.subset == 'Non-N0 holdout (n=16)')
        ].iloc[0]
        rmse_mg = float(m_sub['RMSE_kg_ha']) / 1000.0
        mi_val = float(m_sub['morans_I_queen_12m'])
        p_val = float(m_sub['p_value_queen_12m'])
        mi_str = f'−{abs(mi_val):.2f}' if mi_val < 0 else f'{mi_val:.2f}'
        ax.text(0.50, 1.052, p_title, transform=ax.transAxes, ha='center', va='bottom',
                fontsize=7.0, fontweight='bold', color=INK, linespacing=1.22, clip_on=False)
        ax.text(0.50, 1.004, rf'RMSE = {rmse_mg:.2f}, $I = {mi_str}$ ($P = {p_val:.2f}$)',
                transform=ax.transAxes, ha='center', va='bottom', fontsize=7.0, color=SLATE, clip_on=False)

        if idx == 0:
            bar_y = b[1] - 0.95
            ax.plot([b[0], b[0] + 10.0], [bar_y, bar_y], color=INK, lw=1.15, clip_on=False)
            ax.text(b[0] + 5.0, bar_y - 0.35, '10 m', ha='center', va='top', fontsize=7.0, color=INK, clip_on=False)

        ax.set_xlim(cx_mid - span_w / 2.0, cx_mid + span_w / 2.0)
        ax.set_ylim(cy_mid - span_h / 2.0, cy_mid + span_h / 2.0)
        ax.axis('off')

    # Shared colorbar at the bottom
    cax = fig.add_axes([0.242, 0.068, 0.260, 0.018])
    sm = ScalarMappable(norm=norm_res, cmap=cmap_res)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=[-1.8, -0.9, 0.0, 0.9, 1.8])
    cb.ax.set_xticklabels(['−1.8', '−0.9', '0.0', '+0.9', '+1.8'])
    cb.ax.tick_params(labelsize=7.0, length=1.8, pad=1.2, color=SLATE)
    cb.outline.set_linewidth(0.50)
    fig.text(0.232, 0.077, r'Yield residual, $\hat{Y} - Y$ ($\mathrm{Mg\ ha^{-1}}$):',
             ha='right', va='center', fontsize=7.0, fontweight='bold', color=INK)

    leg_handles = [
        Patch(facecolor='white', edgecolor=PALETTE['amber'], lw=1.15, label='N0 check ($k = 3$)'),
        Patch(facecolor='white', edgecolor='#0F172A', lw=0.60, label='Fertilized holdout ($n = 16$)'),
        Patch(facecolor='none', edgecolor='#64748B', lw=0.60, label='Excluded ($n = 5$)'),
    ]
    fig.legend(handles=leg_handles, loc='center left', bbox_to_anchor=(0.526, 0.077),
               ncol=3, frameon=False, fontsize=7.0, columnspacing=0.65, handlelength=0.85, handletextpad=0.20)

    save_supp_fig(fig, 'Figure_S4_spatial_residuals_2024F1')


# ==============================================================================
# SUPPLEMENTARY FIGURE S1: Cross-Environment Canopy & Boll Segmentation Audit
#   a-b: 2x3 Spatial-Zoom Audit Tiles (Full 0.5 m ROI & 6x6 strata -> sRGB -> Mask)
#   c-d: 3x3 Cross-Environment Precision & F1-Score Matrices (%)
#   e-f: 30-ROI Full-Area Predicted Fraction vs. 36-Point Reference Bounds
# ==============================================================================
def render_figure_s1() -> None:
    import geopandas as gpd
    from matplotlib.patches import ConnectionPatch
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds as win_from_bounds, transform as win_transform

    sys.path.insert(0, str(ROOT / '02_Code/src/data'))
    from extract_raw_rgb_information import GPKG, crop as crop_roi, rules as seg_rules

    fs.apply()
    fig_w_mm, fig_h_mm = 180.0, 160.0
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4), facecolor='white')

    pts_ref = pd.read_csv(OLD_MINING / 'ai_reference/AI_reference_points.csv')
    gdf_roi = gpd.read_file(GPKG, layer='inward_buffer_plot')

    tile_mm = 43.0
    tile_w = tile_mm / fig_w_mm
    tile_h = tile_mm / fig_h_mm
    col_lefts = [0.082, 0.398, 0.714]
    row_ys = [0.668, 0.376]

    col_titles = [
        r'Full 0.5 m ROI & $6{\times}6$ strata',
        '3.3 mm sRGB close-up',
        'Segmented binary mask',
    ]
    for c_i, c_ttl in enumerate(col_titles):
        fig.text(col_lefts[c_i] + tile_w / 2.0, row_ys[0] + tile_h + 0.012, c_ttl,
                 ha='center', va='bottom', fontsize=7.2, fontweight='bold', color=INK)

    phase_specs = [
        ('a', 'canopy', '2024-08-21', 'green_canopeo', row_ys[0],
         '128 DAS: Canopy foliage', PALETTE['teal'], np.array([13, 148, 136]), '#10B981'),
        ('b', 'cotton', '2024-10-13', 'white_b140_s30', row_ys[1],
         '181 DAS: Exposed fiber', PALETTE['cerulean'], np.array([56, 189, 248]), '#38BDF8'),
    ]

    for p_let, phase, day_str, rule_key, ypos, side_lbl, accent_col, tint_rgb, pt_fg_col in phase_specs:
        fig.text(0.014, ypos + tile_h, p_let, fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')
        fig.text(0.048, ypos + tile_h / 2.0, side_lbl, rotation=90,
                 ha='center', va='center', fontsize=7.4, fontweight='bold', color=accent_col)

        rpath = ROOT / f'01_Data/raw/remote_sensing/orthomosaics/2024/{day_str}_f02_visible_orthomosaic.tif'
        with rasterio.open(rpath) as src:
            pix = abs(src.transform.a)
            geom_roi = gdf_roi[
                (gdf_roi.year == 2024) & (gdf_roi.field == 'F2') & (gdf_roi.plot_id == 6)
            ].to_crs(src.crs).geometry.iloc[0]
            full_rgb, full_valid = crop_roi(src, geom_roi)
            w_roi = win_from_bounds(*geom_roi.bounds, src.transform).round_offsets().round_lengths()
            fh, fw = full_rgb.shape[:2]
            side = max(fh, fw)
            pad_y = (side - fh) // 2
            pad_x = (side - fw) // 2
            w_sq = Window(w_roi.col_off - pad_x, w_roi.row_off - pad_y, side, side)
            sq_rgb = np.moveaxis(src.read([1, 2, 3], window=w_sq, boundless=True, fill_value=0), 0, 2)
            sq_inside = geometry_mask([geom_roi], out_shape=(side, side),
                                      transform=win_transform(w_sq, src.transform), invert=True)
            sq_t = win_transform(w_sq, src.transform)

        full_mask = seg_rules(full_rgb, full_valid)[rule_key]
        full_pct_str = f'ROI = {100.0 * full_mask.sum() / full_valid.sum():.1f}%'
        cy, cx = fh // 2, fw // 2

        # Tile 0: Full 0.5 m inward-buffered ROI + 6x6 strata + 36 stratified points
        ax_r = fig.add_axes([col_lefts[0], ypos, tile_w, tile_h], facecolor='white')
        canvas_roi = np.clip(sq_rgb, 0, 255).astype(np.float32)
        canvas_roi[~sq_inside] = canvas_roi[~sq_inside] * 0.42
        ax_r.imshow(np.clip(canvas_roi, 0, 255).astype(np.uint8), interpolation='lanczos')
        ax_r.set_xlim(0, side)
        ax_r.set_ylim(side, 0)
        ax_r.set_xticks([])
        ax_r.set_yticks([])

        rx, ry = geom_roi.exterior.xy
        px_roi, py_roi = (~sq_t) * (np.asarray(rx), np.asarray(ry))
        ax_r.plot(px_roi, py_roi, color='#FBBF24', lw=0.95, ls='--', zorder=4)

        for s_idx in range(1, 6):
            ax_r.plot([pad_x, pad_x + fw], [pad_y + fh * s_idx / 6.0, pad_y + fh * s_idx / 6.0],
                      color='white', lw=0.45, ls=':', alpha=0.72, zorder=3)
            ax_r.plot([pad_x + fw * s_idx / 6.0, pad_x + fw * s_idx / 6.0], [pad_y, pad_y + fh],
                      color='white', lw=0.45, ls=':', alpha=0.72, zorder=3)

        sub_pts = pts_ref[(pts_ref.domain == '2024_F2') & (pts_ref.plot_id == 6) & (pts_ref.phase == phase)]
        px_true = pad_x + sub_pts.x.to_numpy(float)
        py_true = pad_y + sub_pts.y.to_numpy(float)
        target_lbl = 'g' if phase == 'canopy' else 'c'
        is_fg = (sub_pts.label == target_lbl).to_numpy()
        is_unc = (sub_pts.label == 'u').to_numpy()
        is_bg = ~(is_fg | is_unc)
        if is_fg.any():
            ax_r.scatter(px_true[is_fg], py_true[is_fg], s=18,
                         facecolors=pt_fg_col, edgecolors='white', linewidths=0.55, zorder=5.2)
        if is_bg.any():
            ax_r.scatter(px_true[is_bg], py_true[is_bg], s=15,
                         facecolors='white', edgecolors='#0F172A', linewidths=0.55, zorder=5.0)
        if is_unc.any():
            ax_r.scatter(px_true[is_unc], py_true[is_unc], s=16,
                         facecolors='none', edgecolors='#FBBF24', linewidths=0.80, zorder=5.1)

        zx0, zy0, zs = pad_x + cx - 192, pad_y + cy - 192, 384
        ax_r.add_patch(Rectangle((zx0, zy0), zs, zs, fill=False,
                                 edgecolor=PALETTE['terracotta'], lw=1.15, zorder=6))

        bar_1m = 1.0 / pix
        ax_r.plot([side - 38 - bar_1m, side - 38], [side - 52, side - 52], color='#0F172A', lw=2.3, zorder=5.8)
        ax_r.plot([side - 38 - bar_1m, side - 38], [side - 52, side - 52], color='white', lw=1.25, zorder=6.0)
        ax_r.text(side - 38 - bar_1m / 2.0, side - 74, '1 m', ha='center', va='bottom',
                  fontsize=7.0, fontweight='bold', color='white',
                  path_effects=[pe.withStroke(linewidth=1.5, foreground='#0F172A')], zorder=6.2)
        for sp in ax_r.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor('#64748B')
            sp.set_linewidth(0.65)

        # Central 384x384 close-up patch directly from full_rgb / full_valid
        raw_sub = np.clip(full_rgb[cy - 192:cy + 192, cx - 192:cx + 192], 0, 255).astype(np.uint8)
        val_sub = full_valid[cy - 192:cy + 192, cx - 192:cx + 192]
        mask_sub = seg_rules(raw_sub, val_sub)[rule_key]
        overlay = raw_sub.astype(np.float32)
        overlay[mask_sub] = 0.34 * overlay[mask_sub] + 0.66 * tint_rgb
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        hh, ww = raw_sub.shape[:2]

        tile_axes = [ax_r]
        for k_idx, img_arr in enumerate((raw_sub, overlay)):
            ax_sub = fig.add_axes([col_lefts[k_idx + 1], ypos, tile_w, tile_h], facecolor='white')
            ax_sub.imshow(img_arr, interpolation='lanczos')
            ax_sub.set_xlim(0, ww)
            ax_sub.set_ylim(hh, 0)
            ax_sub.set_xticks([])
            ax_sub.set_yticks([])
            if k_idx == 0:
                bar_05 = 0.5 / pix
                ax_sub.plot([ww - 20 - bar_05, ww - 20], [hh - 20, hh - 20], color='#0F172A', lw=2.3, zorder=4.8)
                ax_sub.plot([ww - 20 - bar_05, ww - 20], [hh - 20, hh - 20], color='white', lw=1.25, zorder=5.0)
                ax_sub.text(ww - 20 - bar_05 / 2.0, hh - 28, '0.5 m', ha='center', va='bottom',
                            fontsize=7.0, fontweight='bold', color='white',
                            path_effects=[pe.withStroke(linewidth=1.5, foreground='#0F172A')], zorder=5.2)
            else:
                ax_sub.text(
                    0.05, 0.95, full_pct_str, transform=ax_sub.transAxes, ha='left', va='top',
                    fontsize=7.0, fontweight='bold', color='white', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.10', fc='#0F172A', ec=pt_fg_col, lw=0.55, alpha=0.84),
                )
            for sp in ax_sub.spines.values():
                sp.set_visible(True)
                sp.set_edgecolor(PALETTE['terracotta'] if k_idx == 0 else accent_col)
                sp.set_linewidth(0.85 if k_idx == 0 else 0.75)
            tile_axes.append(ax_sub)

        # Zoom connector lines from 384x384 box in Tile 0 to Tile 1
        for y_src, y_dst in [(zy0, 0), (zy0 + zs, hh)]:
            con = ConnectionPatch(
                xyA=(zx0 + zs, y_src), coordsA=ax_r.transData,
                xyB=(0, y_dst), coordsB=tile_axes[1].transData,
                linestyle='--', linewidth=0.65, color=PALETTE['terracotta'], alpha=0.82, zorder=8,
            )
            fig.add_artist(con)

    # Shared stratified-point legend centered cleanly below Row 2
    s1_leg = [
        Line2D([], [], ls='', marker='o', ms=4.5, mfc='#10B981', mec='white', mew=0.55, label='Foliage target (128 DAS)'),
        Line2D([], [], ls='', marker='o', ms=4.5, mfc='#38BDF8', mec='white', mew=0.55, label='Fiber target (181 DAS)'),
        Line2D([], [], ls='', marker='o', ms=4.2, mfc='white', mec='#0F172A', mew=0.60, label='Background'),
        Line2D([], [], ls='', marker='o', ms=4.3, mfc='none', mec='#FBBF24', mew=0.85, label='Uncertain boundary'),
    ]
    fig.legend(handles=s1_leg, loc='upper center', bbox_to_anchor=(0.520, 0.356),
               ncol=4, frameon=False, columnspacing=1.15, handletextpad=0.28, fontsize=7.0)

    # Bottom Row (c, d, e, f): 3x3 Cross-Environment Matrices + 30-ROI Full-Area Validation
    y_bot, h_bot = 0.068, 0.198
    m_val = pd.read_csv(OLD_MINING / 'ai_reference/AI_mask_validation_metrics.csv').query("phase == 'cotton'")
    env_disp = [fs.ENV_LABELS[d] for d in DOMAINS]

    mat_xs = [0.092, 0.312]
    mat_w = 0.184
    for idx_m, (m_let, m_col, m_title) in enumerate([
        ('c', 'precision', 'Boll precision (%)'),
        ('d', 'F1', r'Boll $F_1$-score (%)'),
    ]):
        fig.text(mat_xs[idx_m] - 0.022, y_bot + h_bot + 0.036, m_let,
                 fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')
        ax_m = fig.add_axes([mat_xs[idx_m], y_bot, mat_w, h_bot], facecolor='white')
        mat_arr = (
            m_val.pivot(index='calibration_source', columns='validation_domain', values=m_col)
            .reindex(index=DOMAINS, columns=DOMAINS)
            .to_numpy(float) * 100.0
        )
        ax_m.imshow(mat_arr, aspect='auto', vmin=42.0, vmax=96.0, cmap=Q1_LEAF_CMAP)
        for edge_v in (0.5, 1.5):
            ax_m.axhline(edge_v, color='white', lw=1.0)
            ax_m.axvline(edge_v, color='white', lw=1.0)
        for ri, ci in np.ndindex(mat_arr.shape):
            val_ij = mat_arr[ri, ci]
            ax_m.text(ci, ri, f'{val_ij:.0f}', ha='center', va='center', fontsize=7.2,
                      fontweight='bold', color='white' if val_ij >= 78.0 else INK)
        ax_m.set_xticks([0, 1, 2])
        ax_m.set_xticklabels(env_disp, fontsize=7.0)
        ax_m.set_yticks([0, 1, 2])
        ax_m.set_yticklabels(env_disp if idx_m == 0 else ['', '', ''], fontsize=7.0)
        ax_m.tick_params(axis='both', length=0, pad=2.2)
        ax_m.set_xlabel('Validation target', fontsize=7.5, labelpad=2.4)
        if idx_m == 0:
            ax_m.set_ylabel('Calibration source', fontsize=7.5, labelpad=2.5)
        ax_m.set_title(m_title, fontsize=7.5, fontweight='bold', color=INK, pad=4.5)
        for sp in ax_m.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor('#CBD5E1')
            sp.set_linewidth(0.55)

    area_val = pd.read_csv(OLD_MINING / 'ai_reference/AI_ROI_area_validation.csv').query(
        'calibration_source == validation_domain'
    )
    roi_xs = [0.566, 0.794]
    roi_w = 0.186
    for idx_a, (a_let, ph_key, a_title, a_col, ylims, yticks) in enumerate([
        ('e', 'canopy', 'Canopy ROI (128/130 DAS)', PALETTE['teal'], (26, 102), [30, 50, 70, 90]),
        ('f', 'cotton', 'Boll ROI (181 DAS)', PALETTE['plum'], (0, 50), [0, 15, 30, 45]),
    ]):
        fig.text(roi_xs[idx_a] - 0.022, y_bot + h_bot + 0.036, a_let,
                 fontweight='bold', fontsize=9, color='#111111', va='top', ha='left')
        ax_a = fig.add_axes([roi_xs[idx_a], y_bot, roi_w, h_bot], facecolor='white')
        ax_a.set_axisbelow(True)
        ax_a.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
        z_a = area_val[area_val.phase == ph_key].sort_values('predicted_full_ROI_fraction')
        xv = np.arange(1, len(z_a) + 1, dtype=float)
        ax_a.vlines(xv, 100.0 * z_a.reference_fraction_lower, 100.0 * z_a.reference_fraction_upper,
                    color='#94A3B8', lw=0.80, zorder=2.0)
        ax_a.scatter(xv, 100.0 * z_a.predicted_full_ROI_fraction, s=11,
                     facecolor=a_col, edgecolor='#0F172A', linewidth=0.30, zorder=3.5)
        ax_a.set_xlim(0.0, 31.0)
        ax_a.set_ylim(*ylims)
        ax_a.set_xticks([1, 10, 20, 30])
        ax_a.set_yticks(yticks)
        ax_a.set_xlabel('Validation ROI rank', fontsize=7.5, labelpad=2.2)
        if idx_a == 0:
            ax_a.set_ylabel('Area fraction (%)', fontsize=7.5, labelpad=2.0)
        ax_a.set_title(a_title, fontsize=7.5, fontweight='bold', color=INK, pad=4.5)
        style_axis(ax_a)

    save_supp_fig(fig, 'Figure_S1_segmentation_audit')


# ==============================================================================
# SUPPLEMENTARY FIGURE S3: Forward-Only Recursive Forecast Trajectories Across
#                          Calendar (DAS) and Thermal-Time (GDD10) Clocks
#                          (6 Remaining Tasks Complementing Fig. 6a, Including N+Leaf+T)
# ==============================================================================
def render_figure_s3() -> None:
    fs.apply()
    fig_w_mm, fig_h_mm = 180.0, 168.0
    fig = plt.figure(figsize=(fig_w_mm / 25.4, fig_h_mm / 25.4), facecolor='white')

    d_rec = pd.concat([
        pd.read_csv(OLD_DM / 'recursive_metrics.csv'),
        pd.read_csv(Q1_DIR / 'rich_fig6_recursive_leaf.csv'),
    ], ignore_index=True)

    rest_specs = [
        ('a', ROLES[1], 'Within-field CV', (-1950, 320), [-1600, -800, 0]),
        ('b', ROLES[2], None, (-215, 175), [-150, 0, 150]),
        ('c', ROLES[4], 'Adjacent-field', (-420, 480), [-300, 0, 300]),
        ('d', ROLES[5], 'Cross-year', (-2450, 1080), [-2000, -1000, 0, 1000]),
        ('e', ROLES[6], None, (-2500, 1080), [-2000, -1000, 0, 1000]),
        ('f', ROLES[7], None, (-940, 260), [-800, -400, 0]),
    ]

    model_keys = [
        ('C', r'$N{+}C$', PALETTE['slate_mid'], 'o', ':'),
        ('C+T', r'$N{+}C{+}T$', PALETTE['cerulean'], 's', '--'),
        ('Leaf+T', r'$N{+}\mathrm{Leaf}{+}T$', PALETTE['teal'], 'D', '-'),
    ]

    leg_handles = [
        Line2D([], [], color=PALETTE['slate_mid'], ls=':', lw=1.00, marker='o', ms=3.5,
               mfc=PALETTE['slate_mid'], mec='#0F172A', mew=0.35, label=r'$N{+}C$'),
        Line2D([], [], color=PALETTE['cerulean'], ls='--', lw=1.05, marker='s', ms=3.5,
               mfc=PALETTE['cerulean'], mec='#0F172A', mew=0.35, label=r'$N{+}C{+}T$'),
        Line2D([], [], color=PALETTE['teal'], ls='-', lw=1.18, marker='D', ms=3.6,
               mfc=PALETTE['teal'], mec='#0F172A', mew=0.35, label=r'$N{+}\mathrm{Leaf}{+}T$'),
        Patch(facecolor='#F1F5F9', edgecolor='#CBD5E1', lw=0.45, label='Boll filling window'),
    ]
    fig.legend(handles=leg_handles, loc='upper center', bbox_to_anchor=(0.560, 0.992),
               ncol=4, frameon=False, columnspacing=1.15, handlelength=1.25, handletextpad=0.28, fontsize=7.0)

    x_das, x_gdd, col_w = 0.262, 0.636, 0.346
    fig.text(x_das + col_w / 2.0, 0.944, 'Calendar clock (DAS)',
             ha='center', va='bottom', fontsize=7.6, fontweight='bold', color=INK)
    fig.text(x_gdd + col_w / 2.0, 0.944, r'Thermal-time clock ($\mathrm{GDD}_{10}$)',
             ha='center', va='bottom', fontsize=7.6, fontweight='bold', color=INK)

    # Row vertical positions (with slightly larger visual breaks between evaluation regimes)
    h_row = 0.116
    row_bots = [0.816, 0.680, 0.526, 0.372, 0.236, 0.100]

    for r_idx, (p_let, task, grp_hdr, ylims, yticks) in enumerate(rest_specs):
        yb = row_bots[r_idx]
        if grp_hdr is not None and r_idx > 0:
            y_sep = (row_bots[r_idx - 1] + yb + h_row) / 2.0
            fig.add_artist(Line2D([0.012, 0.982], [y_sep, y_sep], color='#E2E8F0', lw=0.70, ls='-'))

        fig.text(0.012, yb + h_row - 0.004, p_let, fontweight='bold', fontsize=9,
                 color='#111111', va='top', ha='left')
        if grp_hdr is not None:
            fig.text(0.030, yb + h_row - 0.008, grp_hdr, fontsize=7.0, fontweight='bold',
                     color=PALETTE['teal'] if r_idx < 2 else (PALETTE['cerulean'] if r_idx == 2 else PALETTE['ochre']),
                     va='top', ha='left')
            fig.text(0.030, yb + h_row * 0.42, fs.TASK_LABELS[task], fontsize=7.2,
                     fontweight='bold', color=INK, va='center', ha='left')
        else:
            fig.text(0.030, yb + h_row * 0.52, fs.TASK_LABELS[task], fontsize=7.2,
                     fontweight='bold', color=INK, va='center', ha='left')

        for c_idx, (clock_key, x0, xlims, xticks, win_span, x_lbl) in enumerate([
            ('DAS', x_das, (54, 185), [60, 90, 120, 150, 180], (120, 142), 'Days after sowing (DAS)'),
            ('GDD', x_gdd, (740, 2930), [1000, 1600, 2200, 2800], (1960, 2360), r'Cumulative $\mathrm{GDD}_{10}$ (°C d)'),
        ]):
            ax = fig.add_axes([x0, yb, col_w, h_row], facecolor='white')
            ax.set_axisbelow(True)
            ax.axvspan(win_span[0], win_span[1], fc='#F1F5F9', ec='none', zorder=0.2)
            ax.grid(axis='both', lw=0.38, color='#F1F5F9', zorder=0.5)
            ax.axhline(0.0, color='#334155', lw=0.80, ls='--', zorder=2.0)

            for variant, lbl_k, col_c, mk_s, ls_s in model_keys:
                z = d_rec.loc[
                    d_rec.source.eq(task[0]) & d_rec.target.eq(task[1])
                    & d_rec.clock.eq(clock_key) & d_rec.variant.eq(variant)
                ].sort_values('cutoff')
                if variant in ('C+T', 'Leaf+T'):
                    ax.fill_between(z.cutoff, z.low, z.high, color=col_c,
                                    alpha=0.12 if variant == 'C+T' else 0.16, lw=0, zorder=1.8)
                ax.plot(z.cutoff, z.gain, color=col_c, ls=ls_s, marker=mk_s,
                        ms=2.9 if variant != 'Leaf+T' else 3.2,
                        lw=0.95 if variant != 'Leaf+T' else 1.15,
                        mfc=col_c, mec='#0F172A', mew=0.32,
                        zorder=3.0 if variant == 'C' else (3.3 if variant == 'C+T' else 3.8))

            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            if r_idx < len(rest_specs) - 1:
                ax.set_xticklabels([''] * len(xticks))
            else:
                ax.set_xlabel(x_lbl, fontsize=7.5, labelpad=2.4)
            if c_idx == 1:
                ax.set_yticklabels([''] * len(yticks))
            style_axis(ax)

    fig.text(0.192, 0.515, r'Total $\Delta\mathrm{RMSE}$ vs. $N$-only ($\mathrm{kg\ ha^{-1}}$; $\leftarrow$ worse | better $\rightarrow$)',
             rotation=90, ha='center', va='center', fontsize=8.0, color=INK)

    save_supp_fig(fig, 'Figure_S3_recursive_timing')


def main() -> None:
    fs.apply()
    copy_figure1_and_supplementary()
    render_figure1()
    render_figure2()
    render_figure3()
    render_figure4()
    render_figure5()
    render_figure6()
    render_figure_s1()
    render_figure_s9()
    render_figure_s3()
    render_figure_s10()
    clean_stale_supplementary_assets()
    print('Successfully rendered Q1 flagship Main Figures 1-6 and Supplementary Figures S1-S4.')


if __name__ == '__main__':
    main()
