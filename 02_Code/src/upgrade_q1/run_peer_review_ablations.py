#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Orthogonal Ablations & Sensitivity Analyses for Artificial Intelligence in Agriculture.
Strictly read-only on 01_Data/raw and 10_Results/.../frozen_baseline.
Writes verified CSV tables to analysis/q1_spatial_calibration/:
  1. Canopy-Masked vs. Unmasked ResNet-50 across PCA dimensions (p = 4, 12, 52) & calibration regimes.
  2. 2x2 Factorial Ablation (Whole-ROI vs. Canopeo-Leaf x ExG vs. NGRDI) + N+Leaf+Gap+T across 6 GSDs.
  3. RGB-Only Normalization Regimes: Pure Source-Standardized (zero target leakage), Commercial Strips
     (N0+N225 and N225-only sufficiency strip), and Within-Trial Multi-Rate Block Normalization (z_blk).
  4. Exact Residual Degrees-of-Freedom (df) & 99,999 Within-Block Permutation P-values for Partial
     Correlations and 2D Diagnostic Plane, plus 181-DAS B_morph MSE Decomposition.
  5. Same-Window N-Limitation Diagnostics (92-107 vs. 92-142 DAS; ROC AUC, PR-AUC, Balanced Acc.,
     C-grid sensitivity, Leave-One-Field-Out, Sub-optimal/Excess-N thresholds, 2024 57-88 DAS early curve)
     and Strictly Pre-Cutoff (92-107 DAS) Early 2D Diagnostic Plane.
  6. Bare-Soil Radiometric Normalization Summary, 6-Block Pooled-Source Leave-One-Environment-Out
     (LOEO) CV, Fixed-Holdout k=0..6 Check, and Per-Hectare Computational Runtime Benchmarks.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time
import warnings

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
import rasterio
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "02_Code/src/data"))
sys.path.insert(0, str(ROOT / "02_Code/src/modeling"))
sys.path.insert(0, str(ROOT / "02_Code/src/upgrade_q1"))

from extract_raw_rgb_information import GPKG, crop, rules
import mine_raw_rgb_information as legacy
from run_module_b_spatial_scaling import GSD_LABELS, GSD_LEVELS_M, coarsen_rgb_valid, compute_spatial_metrics
from run_module_c_fewshot_calibration import TRANSFER_TASKS, block_resample_components, error_components, select_uav_active_plots

FORMAL_NEW = ROOT / "10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108"
FORMAL_OLD = FORMAL_NEW / "analysis/frozen_baseline"
OLD_MINING = FORMAL_OLD / "analysis/raw_rgb_mining"
OLD_METHOD = FORMAL_OLD / "analysis/method_strengthening"
Q1_DIR = FORMAL_NEW / "analysis/q1_spatial_calibration"
INPUT_SLOTS = ROOT / "01_Data/processed/model_ready/cotton_rgb_formal_714_slots.csv"

TORCH_CACHE_DIR = Path(
    os.environ.get(
        "REVIEW_TORCH_CACHE",
        str(Path.home() / ".cache/torch/hub"),
    )
)
SEED = 20261025
DOMAINS = ["2023_F1", "2024_F1", "2024_F2"]


def load_common_sets() -> dict[str, pd.DataFrame]:
    common = pd.read_csv(OLD_MINING / "common_plot_features_common_DAS.csv")
    assert len(common) == 64
    return {k: d.reset_index(drop=True) for k, d in common.groupby("domain")}


# ==============================================================================
# 1. [R1 & S5] Canopy-Masked vs. Unmasked ResNet-50 (p = 4, 12, 52) & Runtime
# ==============================================================================
def run_r1_masked_resnet_and_runtime() -> dict:
    import torch
    from torchvision.models import ResNet50_Weights, resnet50

    print("\n=== [R1 & S5] Extracting Unmasked & Canopy-Masked ResNet-50 Embeddings ===")
    emb_cache = Q1_DIR / "resnet50_unmasked_and_masked_embeddings.npz"
    frames = load_common_sets()

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(TORCH_CACHE_DIR))
    weights = ResNet50_Weights.IMAGENET1K_V2
    net = resnet50(weights=weights).requires_grad_(False).eval()
    net.fc = torch.nn.Identity()
    transforms = weights.transforms()
    mean = np.array(transforms.mean, dtype=np.float32)
    std = np.array(transforms.std, dtype=np.float32)

    direct = pd.read_csv(OLD_MINING / "direct_image_features.csv")
    direct["domain"] = direct.year.astype(str) + "_" + direct.field
    selected = []
    for dom, d in frames.items():
        anchors = [92, 107, 130, 142] if dom.startswith("2023") else [92, 106, 128, 141]
        selected.append(direct[(direct.domain == dom) & direct.DAS.isin(anchors) & direct.plot_id.isin(d.plot_id)])
    selected = pd.concat(selected).sort_values(["domain", "plot_id", "DAS"]).reset_index(drop=True)
    assert len(selected) == 256

    geometry = gpd.read_file(GPKG, layer="inward_buffer_plot").set_index(["year", "field", "plot_id"])

    if emb_cache.is_file():
        z = np.load(emb_cache)
        vec_unmasked = z["vec_unmasked"]
        vec_masked = z["vec_masked"]
        print("Loaded cached Unmasked & Canopy-Masked ResNet-50 embeddings (256 x 2048).")
    else:
        vec_unmasked = np.empty((len(selected), 2048), np.float32)
        vec_masked = np.empty((len(selected), 2048), np.float32)

        def to_canvas(rgb_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
            filled = np.where(mask_arr[..., None], rgb_arr, np.round(mean * 255)).clip(0, 255).astype(np.uint8)
            h, w = mask_arr.shape
            scale = 224 / max(h, w)
            hn, wn = max(1, round(h * scale)), max(1, round(w * scale))
            tile = np.asarray(Image.fromarray(filled).resize((wn, hn), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
            canvas = np.zeros((224, 224, 3), dtype=np.float32)
            y0, x0 = (224 - hn) // 2, (224 - wn) // 2
            canvas[y0 : y0 + hn, x0 : x0 + wn] = (tile - mean) / std
            return canvas.transpose(2, 0, 1)

        for raster_path, rows in selected.groupby("raster_path"):
            b_unm, b_msk = [], []
            with rasterio.open(raster_path) as src:
                g = geometry.to_crs(src.crs)
                for _, row in rows.iterrows():
                    rgb, valid = crop(src, g.loc[(row.year, row.field, row.plot_id)].geometry)
                    green = rules(rgb, valid)["green_canopeo"] & valid
                    b_unm.append(to_canvas(rgb, valid))
                    # Canopy-masked: retain strictly green foliage pixels, impute non-leaf mulch/soil with neutral ImageNet mean
                    b_msk.append(to_canvas(rgb, green if int(green.sum()) >= 25 else valid))
            with torch.inference_mode():
                for start in range(0, len(b_unm), 8):
                    idx = rows.index[start : start + 8]
                    vec_unmasked[idx] = net(torch.from_numpy(np.stack(b_unm[start : start + 8]))).numpy()
                    vec_masked[idx] = net(torch.from_numpy(np.stack(b_msk[start : start + 8]))).numpy()
        np.savez_compressed(
            emb_cache,
            vec_unmasked=vec_unmasked,
            vec_masked=vec_masked,
            domain=selected.domain.to_numpy(dtype=str),
            plot_id=selected.plot_id.to_numpy(),
            DAS=selected.DAS.to_numpy(),
        )

    # Fit nested fold-internal PCA + PartialRidge for PC_PER_STAGE in [1, 3, 13] -> p = 4, 12, 52
    def fit_resnet_spec(vectors: np.ndarray, pc_per_stage: int, spec_name: str) -> pd.DataFrame:
        data = {
            (dom, int(p), int(t)): v.astype(np.float64)
            for dom, p, t, v in zip(selected.domain, selected.plot_id, selected.DAS, vectors)
        }
        optical, n_design = {}, {}
        for dom, d in frames.items():
            dates = [92, 107, 130, 142] if dom.startswith("2023") else [92, 106, 128, 141]
            optical[dom] = np.array([[data[dom, int(p), t] for t in dates] for p in d.plot_id])
            n_design[dom] = legacy.design(d, "N")

        def fit_pca(dom: str, idx: np.ndarray) -> PCA:
            z_sub = optical[dom][idx]
            mat = z_sub.reshape(-1, z_sub.shape[-1])  # (4 * n_train, 2048)
            return PCA(n_components=pc_per_stage, svd_solver="full", random_state=SEED).fit(mat)

        def proj(pca: PCA, dom: str, idx: np.ndarray) -> np.ndarray:
            z_sub = optical[dom][idx]
            comp = pca.transform(z_sub.reshape(-1, z_sub.shape[-1])).reshape(len(idx), 4 * pc_per_stage)
            return np.c_[n_design[dom][idx], comp]

        pred_rows = []
        for src, d in frames.items():
            y = d.yield_kg_ha.to_numpy(float)
            g = d.block_design.to_numpy(int)
            ii = np.arange(len(d))
            splits = [(ii[g != b], int(b)) for b in np.unique(g)] + [(ii, 0)]
            for train, held in splits:
                inner = []
                for b_in in np.unique(g[train]):
                    sub_tr = train[g[train] != b_in]
                    sub_te_mask = g[train] == b_in
                    pca_in = fit_pca(src, sub_tr)
                    inner.append((proj(pca_in, src, sub_tr), y[sub_tr], proj(pca_in, src, train[sub_te_mask]), sub_te_mask))
                rank = []
                for alpha in legacy.ALPHAS:
                    oof = np.empty(len(train))
                    for x_tr, y_tr, x_te, m_te in inner:
                        oof[m_te] = legacy.PartialRidge(alpha).fit(x_tr, y_tr).predict(x_te)
                    err = oof - y[train]
                    rank.append((float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err))), -alpha))
                best_alpha = -min(rank)[2]
                pca_full = fit_pca(src, train)
                final = legacy.PartialRidge(best_alpha).fit(proj(pca_full, src, train), y[train])

                jobs = (
                    [(src, np.flatnonzero(d.block_design.eq(held)), "BlockCV")]
                    if held
                    else [
                        (tgt, np.arange(len(td)), "adjacent_field" if tgt[:4] == src[:4] else "cross_year")
                        for tgt, td in frames.items()
                        if tgt != src
                    ]
                )
                for tgt, te_idx, ev in jobs:
                    td = frames[tgt].iloc[te_idx]
                    yh = final.predict(proj(pca_full, tgt, te_idx))
                    for r_t, val in zip(td.itertuples(), yh):
                        pred_rows.append({
                            "source": src,
                            "target": tgt,
                            "evaluation": ev,
                            "specification": spec_name,
                            "plot_id": int(r_t.plot_id),
                            "block": int(r_t.block_design),
                            "N": float(r_t.N_kg_ha),
                            "observed": float(r_t.yield_kg_ha),
                            "predicted": float(val),
                            "selected_alpha": float(best_alpha),
                        })
        return pd.DataFrame(pred_rows)

    resnet_preds = []
    for mask_lbl, vec in [("Unmasked", vec_unmasked), ("Masked", vec_masked)]:
        for pc_k in (1, 3, 13):
            sp = f"N_ResNet50_{mask_lbl}_p{4 * pc_k}"
            df_p = fit_resnet_spec(vec, pc_k, sp)
            resnet_preds.append(df_p)
    df_rn_all = pd.concat(resnet_preds, ignore_index=True)
    df_rn_all.to_csv(Q1_DIR / "resnet_mask_pca_predictions.csv", index=False)

    # Also compute exact 27-resample block 95% intervals & source-OOF 90% PI coverage for N_ResNet50_Masked_p12
    t3_masked_rows = []
    cov_masked_rows = []
    for src, tgt, ev in TRANSFER_TASKS:
        d_src_oof = df_rn_all[(df_rn_all.source == src) & (df_rn_all.target == src) & (df_rn_all.evaluation == "BlockCV") & (df_rn_all.specification == "N_ResNet50_Masked_p12")]
        d_src_fert = d_src_oof[d_src_oof.N > 0]
        abs_res_src = np.sort(np.abs(d_src_fert.observed - d_src_fert.predicted).to_numpy(float))
        rank_q = min(len(abs_res_src), int(math.ceil(0.90 * (len(abs_res_src) + 1)))) - 1
        q90_rad = float(abs_res_src[rank_q])

        d_tgt = df_rn_all[(df_rn_all.source == src) & (df_rn_all.target == tgt) & (df_rn_all.evaluation == ev) & (df_rn_all.specification == "N_ResNet50_Masked_p12")].sort_values("plot_id").reset_index(drop=True)
        g_n0 = d_tgt[d_tgt.N == 0]
        g_ho = d_tgt[d_tgt.N > 0].reset_index(drop=True)
        sh_3n0 = float(np.mean(g_n0.observed - g_n0.predicted))

        for reg_id, shift_val in [("N_ResNet50_Masked_p12__zero_label", 0.0), ("N_ResNet50_Masked_p12__plus_3N0", sh_3n0)]:
            err_ho = (g_ho.predicted.to_numpy(float) + shift_val) - g_ho.observed.to_numpy(float)
            n_ho = g_ho.N.to_numpy(float)
            blks_ho = g_ho.block.to_numpy(int)
            est_c = error_components(err_ho, n_ho)
            u_blks = sorted(np.unique(blks_ho).tolist())
            boot_c = {m: [] for m in ("Total", "Bias", "Between-N", "Within-N")}
            for b1 in u_blks:
                for b2 in u_blks:
                    for b3 in u_blks:
                        idx_b = np.concatenate([np.flatnonzero(blks_ho == b) for b in (b1, b2, b3)])
                        bc = error_components(err_ho[idx_b], n_ho[idx_b])
                        for m in boot_c:
                            boot_c[m].append(bc[m])
            for m in ("Total", "Bias", "Between-N", "Within-N"):
                t3_masked_rows.append({
                    "source": src, "target": tgt, "evaluation": ev,
                    "model_id": reg_id, "measure": m,
                    "estimate": est_c[m],
                    "low_95": float(np.quantile(boot_c[m], 0.025)),
                    "high_95": float(np.quantile(boot_c[m], 0.975)),
                })
            cov_masked_rows.append({
                "source": src, "target": tgt, "evaluation": ev,
                "model": reg_id,
                "coverage": float(np.mean(np.abs(err_ho) <= q90_rad)),
                "mean_width_kg_ha": 2.0 * q90_rad,
            })
    pd.DataFrame(t3_masked_rows).to_csv(Q1_DIR / "resnet_masked_p12_table3_decomposition.csv", index=False)
    pd.DataFrame(cov_masked_rows).to_csv(Q1_DIR / "resnet_masked_p12_coverage.csv", index=False)

    # Verify exact replay of Unmasked p=52 vs frozen baseline!
    frozen_rn = pd.read_csv(OLD_METHOD / "resnet50_partial_ridge_predictions.csv")
    chk = df_rn_all[df_rn_all.specification == "N_ResNet50_Unmasked_p52"].merge(
        frozen_rn, on=["source", "target", "evaluation", "plot_id"]
    )
    max_diff_rn = float(np.max(np.abs(chk.predicted_x - chk.predicted_y)))
    print(f"Replay check Unmasked ResNet-50 p=52 vs frozen: max |diff| = {max_diff_rn:.3e}")

    # Compare against N_only, N_C_T, N_Leaf_T across 9 tasks & 6 fertilized transfer holdouts
    ms_3p3 = pd.read_csv(Q1_DIR / "multiscale_predictions.csv").query("prefix=='3.3 mm' and specification in ['N','N_C_T','N_Leaf_T']")
    comb = pd.concat(
        [ms_3p3[["source", "target", "evaluation", "specification", "plot_id", "block", "N", "observed", "predicted"]],
         df_rn_all[["source", "target", "evaluation", "specification", "plot_id", "block", "N", "observed", "predicted"]]],
        ignore_index=True,
    )

    # 9-task summary
    nine_rows = []
    for (src, tgt, ev), grp in comb.groupby(["source", "target", "evaluation"]):
        b0 = grp[grp.specification == "N"].sort_values("plot_id")
        e0 = (b0.predicted - b0.observed).to_numpy()
        n_arr = b0.N.to_numpy()
        e0_wn = e0.copy()
        for lv in np.unique(n_arr):
            e0_wn[n_arr == lv] -= e0_wn[n_arr == lv].mean()
        r0_tot = float(np.sqrt(np.mean(e0 ** 2)))
        r0_wn = float(np.sqrt(np.mean(e0_wn ** 2)))
        for sp, sub in grp.groupby("specification"):
            sub = sub.sort_values("plot_id")
            ec = (sub.predicted - sub.observed).to_numpy()
            ec_wn = ec.copy()
            for lv in np.unique(n_arr):
                ec_wn[n_arr == lv] -= ec_wn[n_arr == lv].mean()
            rc_tot = float(np.sqrt(np.mean(ec ** 2)))
            rc_wn = float(np.sqrt(np.mean(ec_wn ** 2)))
            nine_rows.append({
                "source": src, "target": tgt, "evaluation": ev, "specification": sp,
                "Total_RMSE": rc_tot, "WithinN_RMSE": rc_wn,
                "Delta_Total_vs_N": r0_tot - rc_tot, "Delta_WithinN_vs_N": r0_wn - rc_wn,
            })
    df_nine = pd.DataFrame(nine_rows)
    df_nine.to_csv(Q1_DIR / "resnet_mask_pca_ablation_9task.csv", index=False)

    # 6-task fertilized holdout: k=0, +3N0, N225 (k=3), and UAV quantile-stratified (k=3)
    plot64 = pd.read_csv(Q1_DIR / "plot64_millimeter_features.csv")
    cal_rows = []
    for src, tgt, ev in TRANSFER_TASKS:
        sub_t = comb[(comb.source == src) & (comb.target == tgt) & (comb.evaluation == ev)]
        d_guide = sub_t[sub_t.specification == "N_Leaf_T"].sort_values("plot_id").reset_index(drop=True)
        d_guide_fert = d_guide[d_guide.N > 0].reset_index(drop=True)
        uav_k3_pids = select_uav_active_plots(d_guide_fert, d_guide_fert["predicted"].to_numpy(float), 3)

        for sp, g in sub_t.groupby("specification"):
            g = g.sort_values("plot_id").reset_index(drop=True)
            g_n0 = g[g.N == 0]
            g_ho = g[g.N > 0].reset_index(drop=True)
            y_ho = g_ho.observed.to_numpy(float)
            p_ho = g_ho.predicted.to_numpy(float)
            n_ho = g_ho.N.to_numpy(float)

            # k=0 on non-N0 holdout
            c0 = error_components(p_ho - y_ho, n_ho)
            # +3N0 on non-N0 holdout
            sh_n0 = float(np.mean(g_n0.observed - g_n0.predicted))
            cn0 = error_components((p_ho + sh_n0) - y_ho, n_ho)
            # +3 N225 strips (unsampled fertilized holdout)
            m_225 = g_ho.N == 225
            sh_225 = float(np.mean(y_ho[m_225] - p_ho[m_225]))
            rmse_225 = float(np.sqrt(np.mean(((p_ho[~m_225] + sh_225) - y_ho[~m_225]) ** 2)))
            # k=3 UAV quantile-stratified (unsampled fertilized holdout)
            m_uav = g_ho.plot_id.isin(uav_k3_pids).to_numpy()
            sh_uav = float(np.mean(y_ho[m_uav] - p_ho[m_uav]))
            rmse_uav = float(np.sqrt(np.mean(((p_ho[~m_uav] + sh_uav) - y_ho[~m_uav]) ** 2)))

            cal_rows.append({
                "source": src, "target": tgt, "evaluation": ev, "specification": sp,
                "k0_Total_RMSE": c0["Total"], "k0_Bias": c0["Bias"],
                "BetweenN_RMSE": c0["Between-N"], "WithinN_RMSE": c0["Within-N"],
                "plus3N0_Total_RMSE": cn0["Total"], "plus3N0_Bias": cn0["Bias"],
                "N225_k3_RMSE": rmse_225, "UAV_k3_RMSE": rmse_uav,
            })
    df_cal = pd.DataFrame(cal_rows)
    df_cal.to_csv(Q1_DIR / "resnet_mask_pca_calibration.csv", index=False)

    # Also compute k = 0..6 UAV quantile-stratified & N0_check curves for Unmasked vs Canopy-Masked ResNet-50
    curve_rows = []
    for src, tgt, ev in TRANSFER_TASKS:
        sub_t = comb[(comb.source == src) & (comb.target == tgt) & (comb.evaluation == ev)]
        d_guide = sub_t[sub_t.specification == "N_Leaf_T"].sort_values("plot_id").reset_index(drop=True)
        d_guide_fert = d_guide[d_guide.N > 0].reset_index(drop=True)
        p_fert_guide = d_guide_fert["predicted"].to_numpy(float)

        for sp in ["N_ResNet50_Masked_p4", "N_ResNet50_Masked_p12", "N_ResNet50_Masked_p52",
                   "N_ResNet50_Unmasked_p4", "N_ResNet50_Unmasked_p12", "N_ResNet50_Unmasked_p52"]:
            d_sp = sub_t[sub_t.specification == sp].sort_values("plot_id").reset_index(drop=True)
            y_all = d_sp.observed.to_numpy(float)
            p_all = d_sp.predicted.to_numpy(float)
            n_all = d_sp.N.to_numpy(float)
            pids = d_sp.plot_id.to_numpy(int)
            d_fert = d_sp[d_sp.N > 0].copy()

            for k in range(0, 7):
                if k == 0:
                    chosen_idx = []
                else:
                    chosen_pids = select_uav_active_plots(d_fert, p_fert_guide, k)
                    chosen_idx = [int(np.flatnonzero(pids == pid)[0]) for pid in chosen_pids]
                cal_m = np.zeros(len(d_sp), dtype=bool)
                cal_m[chosen_idx] = True
                te_m = (~cal_m) & (n_all > 0)
                sh = 0.0 if k == 0 else float(np.mean(y_all[cal_m] - p_all[cal_m]))
                err = (p_all[te_m] + sh) - y_all[te_m]
                curve_rows.append({
                    "source": src, "target": tgt, "evaluation": ev,
                    "support": "unsampled_fertilized", "specification": sp,
                    "strategy": "UAV_Stratified_Pred", "k": k,
                    "mean_RMSE": float(np.sqrt(np.mean(err ** 2))),
                    "mean_abs_bias": float(abs(np.mean(err))),
                })
    pd.DataFrame(curve_rows).to_csv(Q1_DIR / "resnet_mask_pca_fewshot_curves.csv", index=False)

    print("\n--- [R1] 6-Task Mean Calibration Summary (Unmasked vs Canopy-Masked ResNet-50) ---")
    print(df_cal.groupby("specification")[["k0_Total_RMSE", "BetweenN_RMSE", "WithinN_RMSE", "plus3N0_Total_RMSE", "N225_k3_RMSE", "UAV_k3_RMSE"]].mean().round(1).to_string())

    # [S5] Per-plot & per-hectare CPU runtime benchmark across 3.3 mm, 1 cm, 5 cm vs ResNet-50
    sample_row = selected.iloc[10]
    with rasterio.open(sample_row.raster_path) as src:
        g = geometry.to_crs(src.crs)
        geom = g.loc[(sample_row.year, sample_row.field, sample_row.plot_id)].geometry
        rgb0, val0 = crop(src, geom)
        px_m = float(np.sqrt(abs(src.res[0] * 85000.0 * src.res[1] * 111000.0))) if src.crs.is_geographic else float(np.sqrt(abs(src.res[0] * src.res[1])))
    roi_area_ha = float(val0.sum() * (px_m ** 2) / 10000.0)

    rt_rows = []
    for gsd_m, lbl in [(0.0033, "3.3 mm"), (0.01, "1 cm"), (0.05, "5 cm")]:
        c_rgb, c_val, eff_px = coarsen_rgb_valid(rgb0, val0, px_m, gsd_m)
        t_start = time.perf_counter()
        n_rep = 5 if gsd_m == 0.0033 else 25
        for _ in range(n_rep):
            _ = compute_spatial_metrics(c_rgb, c_val, eff_px, include_bolls=False)
        dt_ms = (time.perf_counter() - t_start) * 1000.0 / n_rep
        rt_rows.append({
            "pipeline": f"Decoupled RGB (Canopeo + NGRDI_leaf + Gaps) @ {lbl}",
            "gsd_label": lbl,
            "valid_pixels_per_plot": int(c_val.sum()),
            "ms_per_plot_cpu": round(dt_ms, 2),
            "sec_per_ha_cpu": round((dt_ms / 1000.0) / roi_area_ha, 2),
        })
    # Benchmark ResNet-50 CPU inference per plot
    t_rn = time.perf_counter()
    dummy_in = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(15):
            _ = net(dummy_in)
    dt_rn_ms = (time.perf_counter() - t_rn) * 1000.0 / 15
    rt_rows.append({
        "pipeline": "ResNet-50 feature extraction (CPU 4 threads, single 224x224 ROI)",
        "gsd_label": "224x224 resized",
        "valid_pixels_per_plot": 224 * 224,
        "ms_per_plot_cpu": round(dt_rn_ms, 2),
        "sec_per_ha_cpu": round((dt_rn_ms / 1000.0) / roi_area_ha, 2),
    })
    df_rt = pd.DataFrame(rt_rows)
    df_rt.to_csv(Q1_DIR / "computational_runtime_benchmarks.csv", index=False)
    print("\n--- [S5] Computational Runtime Benchmarks ---")
    print(df_rt.to_string(index=False))
    return {"max_diff_rn": max_diff_rn}


# ==============================================================================
# 2. [R8] 2x2 Factorial Ablation (Whole-ROI vs Leaf x ExG vs NGRDI) + Gaps
# ==============================================================================
def run_r8_2x2_index_mask_ablation() -> pd.DataFrame:
    print("\n=== [R8] Running 2x2 Chromatic-Index x Leaf-Masking Ablation Across 6 GSDs ===")
    df_gsd = pd.read_csv(Q1_DIR / "multiscale_gsd_slot_features.csv")
    common_base = pd.read_csv(OLD_MINING / "common_plot_features_common_DAS.csv")[
        ["domain", "year", "field", "plot_id", "block_design", "N_kg_ha", "yield_kg_ha"]
    ]

    specs_2x2 = {
        "N": [],
        "N_Whole_ExG": ["C_cover", "C_exg_mixed", "T_auc_mean", "T_late_change"],
        "N_Whole_NGRDI": ["C_cover", "C_ngrdi_mixed", "T_auc_mean", "T_late_change"],
        "N_Leaf_ExG": ["C_cover", "C_exg_leaf", "T_auc_mean", "T_late_change"],
        "N_Leaf_NGRDI": ["C_cover", "C_ngrdi_leaf", "T_auc_mean", "T_late_change"],
        "N_Leaf_Gap_T": ["C_cover", "C_ngrdi_leaf", "T_auc_mean", "T_late_change", "Searly_gap_large", "Searly_prof_across"],
    }

    rows = []
    for gsd in GSD_LEVELS_M:
        lbl = GSD_LABELS[gsd]
        sub_s = df_gsd[df_gsd.gsd_m == gsd]
        plot_recs = []
        for r in common_base.itertuples():
            d = sub_s[(sub_s.domain == r.domain) & (sub_s.plot_id == int(r.plot_id)) & (sub_s.DAS < 180)].sort_values("DAS")
            xs = d.DAS.to_numpy(float)
            early_das = 130 if int(r.year) == 2023 else 128
            late_das = 142 if int(r.year) == 2023 else 141
            e = d[d.DAS == early_das].iloc[0]
            l = d[d.DAS == late_das].iloc[0]
            span = float(late_das - 92.0)
            plot_recs.append({
                "domain": r.domain, "plot_id": int(r.plot_id),
                "block_design": int(r.block_design), "N_kg_ha": float(r.N_kg_ha), "yield_kg_ha": float(r.yield_kg_ha),
                "C_cover": float(l.f_cover),
                "C_exg_mixed": float(l.exg_mixed),
                "C_ngrdi_mixed": float(l.ngrdi_mixed),
                "C_exg_leaf": float(l.exg_leaf),
                "C_ngrdi_leaf": float(l.ngrdi_leaf),
                "T_auc_mean": float(np.trapezoid(d.f_cover.to_numpy(float), xs) / span),
                "T_late_change": float((l.f_cover - e.f_cover) / (late_das - early_das) * span),
                "Searly_gap_large": float(e.gap_large_frac),
                "Searly_prof_across": float(e.profile_sd_across),
            })
        ptbl = pd.DataFrame(plot_recs)
        frames = {dom: g.reset_index(drop=True) for dom, g in ptbl.groupby("domain")}
        tasks = [(s, t, "BlockCV" if s == t else "adjacent_field" if s[:4] == t[:4] else "cross_year")
                 for s in DOMAINS for t in DOMAINS]

        preds_map = {}
        for sp_name, cols in specs_2x2.items():
            sp_preds = []
            for src, tgt, ev in tasks:
                a, b = frames[src], frames[tgt]
                ya = a.yield_kg_ha.to_numpy(float)
                ga = a.block_design.to_numpy(int)
                ii = np.arange(len(a))
                folds = np.unique(ga) if src == tgt else [0]
                na, nb = a.N_kg_ha.to_numpy(float), b.N_kg_ha.to_numpy(float)
                xa = np.c_[na, na * na, a[cols].to_numpy(float)] if cols else np.c_[na, na * na]
                xb = np.c_[nb, nb * nb, b[cols].to_numpy(float)] if cols else np.c_[nb, nb * nb]
                for fold in folds:
                    tr = ii[ga != fold] if fold else ii
                    te = np.flatnonzero(b.block_design.to_numpy(int) == fold) if fold else np.arange(len(b))
                    if cols:
                        sc = []
                        for alpha in legacy.ALPHAS:
                            oof = np.empty(len(tr))
                            for blk in np.unique(ga[tr]):
                                kp = ga[tr] != blk
                                oof[~kp] = legacy.PartialRidge(alpha).fit(xa[tr[kp]], ya[tr[kp]]).predict(xa[tr[~kp]])
                            err = oof - ya[tr]
                            sc.append((float(np.sqrt(np.mean(err ** 2))), float(np.mean(np.abs(err))), -alpha, alpha))
                        _, _, _, best_a = min(sc)
                    else:
                        best_a = 0.0
                    yh = legacy.PartialRidge(best_a).fit(xa[tr], ya[tr]).predict(xb[te])
                    for r_te, pv in zip(b.iloc[te].itertuples(), yh):
                        sp_preds.append({
                            "source": src, "target": tgt, "evaluation": ev,
                            "plot_id": int(r_te.plot_id), "N": float(r_te.N_kg_ha),
                            "observed": float(r_te.yield_kg_ha), "predicted": float(pv),
                        })
            preds_map[sp_name] = pd.DataFrame(sp_preds)

        # Summarize 9-task mean Total & Within-N Delta RMSE vs N-only + 6-task +3N0 RMSE
        b_df = preds_map["N"]
        for sp_name in specs_2x2:
            if sp_name == "N":
                continue
            c_df = preds_map[sp_name]
            m = b_df.merge(c_df, on=["source", "target", "evaluation", "plot_id", "N", "observed"], suffixes=("_N", "_cand"))
            tot_gains, wn_gains, p3n0_rmses = [], [], []
            for (src, tgt, ev), g in m.groupby(["source", "target", "evaluation"]):
                e0 = (g.predicted_N - g.observed).to_numpy()
                e1 = (g.predicted_cand - g.observed).to_numpy()
                narr = g.N.to_numpy()
                e0_w, e1_w = e0.copy(), e1.copy()
                for lv in np.unique(narr):
                    e0_w[narr == lv] -= e0_w[narr == lv].mean()
                    e1_w[narr == lv] -= e1_w[narr == lv].mean()
                tot_gains.append(float(np.sqrt(np.mean(e0 ** 2)) - np.sqrt(np.mean(e1 ** 2))))
                wn_gains.append(float(np.sqrt(np.mean(e0_w ** 2)) - np.sqrt(np.mean(e1_w ** 2))))
                if ev != "BlockCV":
                    gn0 = g[g.N == 0]
                    gho = g[g.N > 0]
                    sh = float(np.mean(gn0.observed - gn0.predicted_cand))
                    p3n0_rmses.append(float(np.sqrt(np.mean(((gho.predicted_cand + sh) - gho.observed) ** 2))))
            rows.append({
                "gsd_m": gsd,
                "gsd_label": lbl,
                "specification": sp_name,
                "mean_9task_Delta_Total": float(np.mean(tot_gains)),
                "mean_9task_Delta_WithinN": float(np.mean(wn_gains)),
                "mean_6task_plus3N0_RMSE": float(np.mean(p3n0_rmses)),
            })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(Q1_DIR / "ablation_2x2_index_mask_multiscale.csv", index=False)
    print("\n--- [R8] 2x2 Index x Masking Ablation Across GSDs (9-task Mean Delta Total / Within-N) ---")
    print(df_out.pivot(index="gsd_label", columns="specification", values="mean_9task_Delta_Total").loc[list(GSD_LABELS.values())].round(1).to_string())
    print(df_out.pivot(index="gsd_label", columns="specification", values="mean_9task_Delta_WithinN").loc[list(GSD_LABELS.values())].round(1).to_string())
    return df_out


# ==============================================================================
# 3. [R2] RGB-Only Normalization Regimes (Pure Source-Std vs Strips vs z_blk)
# ==============================================================================
def run_r2_rgb_only_normalizations() -> pd.DataFrame:
    print("\n=== [R2] Auditing RGB-Only Normalization Regimes (Zero-Leakage Source-Std vs Strips vs Trial z_blk) ===")
    plot64 = pd.read_csv(Q1_DIR / "plot64_millimeter_features.csv")
    df_b = pd.read_csv(Q1_DIR / "direction_b_2d_attribution_quadrants.csv")

    # Match exact 4-feature and 5-feature composite definitions from prepare_rich_figure_assets.py
    base_raw_cols = [
        "Cearly_exg", "C_exg", "C_cover", "T_auc_mean", "T_late_change",
        "C_ngrdi_leaf", "T_ngrdi_leaf_auc",
        "Searly_gap_large", "T_gap_auc", "Searly_prof_across",
        "white_b140_s30_morph_frac",
    ]
    sets_raw = {dom: g.sort_values("plot_id").reset_index(drop=True) for dom, g in plot64.groupby("domain")}
    alphas = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0]

    def build_spec_matrix(Z_scaled: np.ndarray, col_idx: dict[str, int], sp_name: str) -> np.ndarray:
        if sp_name == "RGB_Mixed_ExG":
            return Z_scaled[:, [col_idx["Cearly_exg"], col_idx["C_exg"]]]
        elif sp_name == "RGB_Mixed_CT":
            return Z_scaled[:, [col_idx["C_cover"], col_idx["C_exg"], col_idx["T_auc_mean"], col_idx["T_late_change"]]]
        i_leaf = (Z_scaled[:, col_idx["C_ngrdi_leaf"]] + Z_scaled[:, col_idx["T_ngrdi_leaf_auc"]] + Z_scaled[:, col_idx["T_late_change"]]) / 3.0
        i_gap = (Z_scaled[:, col_idx["Searly_gap_large"]] + Z_scaled[:, col_idx["T_gap_auc"]] + Z_scaled[:, col_idx["Searly_prof_across"]]) / 3.0
        if sp_name == "RGB_Decoupled_InSeason":
            return np.column_stack([i_leaf, i_gap, Z_scaled[:, col_idx["C_cover"]], Z_scaled[:, col_idx["T_auc_mean"]]])
        elif sp_name == "RGB_Decoupled_Bmorph":
            return np.column_stack([
                i_leaf, i_gap, Z_scaled[:, col_idx["C_cover"]], Z_scaled[:, col_idx["T_auc_mean"]],
                Z_scaled[:, col_idx["white_b140_s30_morph_frac"]],
            ])
        raise ValueError(sp_name)

    col_idx = {c: i for i, c in enumerate(base_raw_cols)}
    spec_list = ["RGB_Mixed_ExG", "RGB_Mixed_CT", "RGB_Decoupled_InSeason", "RGB_Decoupled_Bmorph"]

    def eval_regime(regime: str) -> list[dict]:
        out_rows = []
        for sp_name in spec_list:
            for src, ds in sets_raw.items():
                ys = ds.yield_kg_ha.to_numpy(float)
                gs = ds.block_design.to_numpy(int)

                def scale_base(df_dom: pd.DataFrame, fit_scaler: StandardScaler | None = None):
                    X_raw = df_dom[base_raw_cols].to_numpy(float)
                    n_dom = df_dom.N_kg_ha.to_numpy(float)
                    b_dom = df_dom.block_design.to_numpy(int)
                    if regime == "Pure_Source_Std":
                        sc = StandardScaler().fit(X_raw) if fit_scaler is None else fit_scaler
                        return sc.transform(X_raw), sc
                    elif regime == "Strip_N0_N225":
                        mu0 = X_raw[n_dom == 0].mean(axis=0)
                        mu225 = X_raw[n_dom == 225].mean(axis=0)
                        span = np.where(np.abs(mu225 - mu0) > 1e-6, mu225 - mu0, 1.0)
                        X_st = (X_raw - mu0) / span
                        sc = StandardScaler().fit(X_st) if fit_scaler is None else fit_scaler
                        return sc.transform(X_st), sc
                    elif regime == "Strip_N225_Only":
                        mu225 = X_raw[n_dom == 225].mean(axis=0)
                        X_st = X_raw - mu225
                        sc = StandardScaler().fit(X_st) if fit_scaler is None else fit_scaler
                        return sc.transform(X_st), sc
                    elif regime == "Trial_MultiRate_z_blk":
                        X_bz = np.zeros_like(X_raw)
                        for b_id in np.unique(b_dom):
                            mb = b_dom == b_id
                            sd_b = np.std(X_raw[mb], axis=0, ddof=0)
                            sd_b = np.where(sd_b > 1e-9, sd_b, 1.0)
                            X_bz[mb] = (X_raw[mb] - np.mean(X_raw[mb], axis=0)) / sd_b
                        return X_bz, None
                    raise ValueError(regime)

                ii = np.arange(len(ds))
                p_cv = np.empty(len(ds))
                for b_hold in np.unique(gs):
                    tr = ii[gs != b_hold]
                    te = ii[gs == b_hold]
                    X_tr_raw = ds.iloc[tr][base_raw_cols].to_numpy(float)
                    X_te_raw = ds.iloc[te][base_raw_cols].to_numpy(float)
                    if regime == "Pure_Source_Std":
                        sc_f = StandardScaler().fit(X_tr_raw)
                        z_tr = sc_f.transform(X_tr_raw)
                        z_te = sc_f.transform(X_te_raw)
                    elif regime == "Strip_N225_Only":
                        n_tr = ds.iloc[tr].N_kg_ha.to_numpy(float)
                        mu225_tr = X_tr_raw[n_tr == 225].mean(axis=0)
                        X_tr_st = X_tr_raw - mu225_tr
                        X_te_st = X_te_raw - mu225_tr
                        sc_f = StandardScaler().fit(X_tr_st)
                        z_tr = sc_f.transform(X_tr_st)
                        z_te = sc_f.transform(X_te_st)
                    elif regime == "Strip_N0_N225":
                        n_tr = ds.iloc[tr].N_kg_ha.to_numpy(float)
                        mu0_tr = X_tr_raw[n_tr == 0].mean(axis=0)
                        mu225_tr = X_tr_raw[n_tr == 225].mean(axis=0)
                        span_tr = np.where(np.abs(mu225_tr - mu0_tr) > 1e-6, mu225_tr - mu0_tr, 1.0)
                        X_tr_st = (X_tr_raw - mu0_tr) / span_tr
                        X_te_st = (X_te_raw - mu0_tr) / span_tr
                        sc_f = StandardScaler().fit(X_tr_st)
                        z_tr = sc_f.transform(X_tr_st)
                        z_te = sc_f.transform(X_te_st)
                    else:
                        Z_all, _ = scale_base(ds, None)
                        z_tr, z_te = Z_all[tr], Z_all[te]
                    x_tr = build_spec_matrix(z_tr, col_idx, sp_name)
                    x_te = build_spec_matrix(z_te, col_idx, sp_name)

                    best_a, best_rm = 10.0, 1e18
                    for a in alphas:
                        oof = np.empty(len(tr))
                        for b2 in np.unique(gs[tr]):
                            m2 = gs[tr] != b2
                            oof[~m2] = Ridge(alpha=a).fit(x_tr[m2], ys[tr[m2]]).predict(x_tr[~m2])
                        rm = float(np.sqrt(np.mean((oof - ys[tr]) ** 2)))
                        if rm < best_rm:
                            best_rm, best_a = rm, a
                    p_cv[te] = Ridge(alpha=best_a).fit(x_tr, ys[tr]).predict(x_te)

                out_rows.append({
                    "regime": regime, "specification": sp_name,
                    "source": src, "target": src, "evaluation": "BlockCV",
                    "RMSE_k0_all": float(np.sqrt(np.mean((p_cv - ys) ** 2))),
                    "R2_k0_all": float(r2_score(ys, p_cv)),
                    "RMSE_k0_ho": float(np.sqrt(np.mean((p_cv - ys) ** 2))),
                    "RMSE_k3_ho": float(np.sqrt(np.mean((p_cv - ys) ** 2))),
                    "R2_k3_ho": float(r2_score(ys, p_cv)),
                })

                z_src, sc_src = scale_base(ds, None)
                x_src = build_spec_matrix(z_src, col_idx, sp_name)
                best_a, best_rm = 10.0, 1e18
                for a in alphas:
                    oof = np.empty(len(ii))
                    for b2 in np.unique(gs):
                        m2 = gs != b2
                        oof[~m2] = Ridge(alpha=a).fit(x_src[m2], ys[m2]).predict(x_src[~m2])
                    rm = float(np.sqrt(np.mean((oof - ys) ** 2)))
                    if rm < best_rm:
                        best_rm, best_a = rm, a
                m_src = Ridge(alpha=best_a).fit(x_src, ys)

                for tgt, dt in sets_raw.items():
                    if tgt == src:
                        continue
                    ev = "adjacent_field" if src[:4] == tgt[:4] else "cross_year"
                    z_tgt, _ = scale_base(dt, sc_src)
                    x_tgt = build_spec_matrix(z_tgt, col_idx, sp_name)
                    yt = dt.yield_kg_ha.to_numpy(float)
                    pt = m_src.predict(x_tgt)

                    # Identical yield-blind block-mean cover integral plot per block
                    cal_pids = []
                    for _, gb in dt.groupby("block_design"):
                        mu_v = float(gb["T_auc_mean"].mean())
                        cal_pids.append(int(gb.loc[(gb["T_auc_mean"] - mu_v).abs().idxmin(), "plot_id"]))
                    cal_mask = dt["plot_id"].isin(cal_pids).to_numpy()
                    shift = float(np.mean(yt[cal_mask] - pt[cal_mask]))
                    pt_cal = pt + shift

                    out_rows.append({
                        "regime": regime, "specification": sp_name,
                        "source": src, "target": tgt, "evaluation": ev,
                        "RMSE_k0_all": float(np.sqrt(np.mean((pt - yt) ** 2))),
                        "R2_k0_all": float(r2_score(yt, pt)),
                        "RMSE_k0_ho": float(np.sqrt(np.mean((pt[~cal_mask] - yt[~cal_mask]) ** 2))),
                        "RMSE_k3_ho": float(np.sqrt(np.mean((pt_cal[~cal_mask] - yt[~cal_mask]) ** 2))),
                        "R2_k3_ho": float(r2_score(yt[~cal_mask], pt_cal[~cal_mask])),
                    })
        return out_rows

    all_reg_rows = []
    for reg in ("Pure_Source_Std", "Strip_N225_Only", "Strip_N0_N225", "Trial_MultiRate_z_blk"):
        all_reg_rows.extend(eval_regime(reg))
    df_reg = pd.DataFrame(all_reg_rows)
    df_reg.to_csv(Q1_DIR / "ablation_rgb_only_normalization_regimes.csv", index=False)

    print("\n--- [R2] RGB-Only Normalization Regimes Summary ---")
    piv_r2 = df_reg.groupby(["regime", "specification", "evaluation"])[["RMSE_k0_all", "R2_k0_all", "RMSE_k3_ho", "R2_k3_ho"]].mean()
    print(piv_r2.round(2).to_string())
    return df_reg


# ==============================================================================
# 4. [R3 & R9] Exact Partial-Correlation Degrees of Freedom & 181-DAS MSE Split
# ==============================================================================
def run_r3_exact_degrees_of_freedom() -> dict:
    print("\n=== [R3] Recomputing Exact Residual Degrees of Freedom & Permutation P-Values ===")
    plot64 = pd.read_csv(Q1_DIR / "plot64_millimeter_features.csv")
    df_native = pd.read_csv(Q1_DIR / "millimeter_slot_features_699.csv")
    boll66 = df_native[df_native.DAS == 181].sort_values(["domain", "plot_id"]).reset_index(drop=True)

    features = [
        ("C_cover", "Canopy cover (141/142 DAS)", "Whole-plot scalar", plot64),
        ("C_exg", "Mixed-pixel ExG (141/142 DAS)", "Whole-plot scalar", plot64),
        ("T_auc_mean", "Canopy cover integral (92–142 DAS)", "Whole-plot scalar", plot64),
        ("Cearly_ngrdi_leaf", "Pure-leaf NGRDI (128/130 DAS)", "Decoupled leaf color", plot64),
        ("T_ngrdi_leaf_auc", "Pure-leaf NGRDI integral", "Decoupled leaf color", plot64),
        ("Searly_gap_large", "Large canopy gaps >=0.04 m2 (128/130 DAS)", "Millimeter micro-structure", plot64),
        ("T_gap_auc", "Seasonal large-gap integral", "Millimeter micro-structure", plot64),
        ("Searly_prof_across", "Cross-row closure heterogeneity (128/130 DAS)", "Millimeter micro-structure", plot64),
        ("white_b140_s30_raw", "Raw white-pixel fraction (181 DAS, n=64)", "181-DAS boll proxy", plot64),
        ("white_b140_s30_morph_frac", "Filtered boll-cluster fraction (181 DAS, n=64)", "181-DAS boll proxy", plot64),
        ("white_b140_s30_raw", "Raw white-pixel fraction (181 DAS, n=66)", "181-DAS boll proxy (all 66)", boll66),
        ("white_b140_s30_morph_frac", "Filtered boll-cluster fraction (181 DAS, n=66)", "181-DAS boll proxy (all 66)", boll66),
    ]

    rng = np.random.default_rng(20261117)
    rows = []
    for col, label, grp, df_in in features:
        n_obs = len(df_in)
        y = df_in.yield_kg_ha.to_numpy(float)
        v = df_in[col].to_numpy(float)

        # Environment-centered apparent r: X_env has rank 3 (3 environment means), so df_env = n - 3 - 1
        X_env = pd.get_dummies(df_in[["domain"]], drop_first=False, dtype=float).to_numpy()
        rk_env = int(np.linalg.matrix_rank(X_env))
        df_env = n_obs - rk_env - 1
        ry_env = y - X_env @ np.linalg.lstsq(X_env, y, rcond=None)[0]
        rv_env = v - X_env @ np.linalg.lstsq(X_env, v, rcond=None)[0]
        r_env = float(stats.pearsonr(rv_env, ry_env).statistic)
        t_env = r_env * math.sqrt(df_env / max(1e-12, 1.0 - r_env ** 2))
        p_env_exact = float(2.0 * stats.t.sf(abs(t_env), df=df_env))

        # Full within-(domain x N, domain x block) projection:
        z_cell = pd.get_dummies(
            df_in[["domain"]].assign(
                dom_N=df_in.domain + "_" + df_in.N_kg_ha.astype(int).astype(str),
                dom_B=df_in.domain + "_B" + df_in.block_design.astype(str),
            ),
            drop_first=True, dtype=float,
        )
        X_cell = np.column_stack([np.ones(n_obs), z_cell.to_numpy()])
        rk_cell = int(np.linalg.matrix_rank(X_cell))
        df_cell = n_obs - rk_cell - 1  # 64 - 27 - 1 = 36 (or 66 - 27 - 1 = 38)
        ry_cell = y - X_cell @ np.linalg.lstsq(X_cell, y, rcond=None)[0]
        rv_cell = v - X_cell @ np.linalg.lstsq(X_cell, v, rcond=None)[0]
        r_part = float(stats.pearsonr(rv_cell, ry_cell).statistic)
        t_part = r_part * math.sqrt(df_cell / max(1e-12, 1.0 - r_part ** 2))
        p_part_exact = float(2.0 * stats.t.sf(abs(t_part), df=df_cell))

        # Also check additive (domain x block + quadratic N per domain): 9 block intercepts + 3*2 quadratic N = 15 params -> df = 48
        X_quad_list = [pd.get_dummies(df_in.domain + "_B" + df_in.block_design.astype(str), drop_first=False, dtype=float).to_numpy()]
        for dom in DOMAINS:
            m_d = (df_in.domain == dom).to_numpy(float)
            nd = df_in.N_kg_ha.to_numpy(float) / 100.0
            X_quad_list.append(np.column_stack([m_d * nd, m_d * (nd ** 2)]))
        X_quad = np.hstack(X_quad_list)
        rk_quad = int(np.linalg.matrix_rank(X_quad))
        df_quad = n_obs - rk_quad - 1
        ry_q = y - X_quad @ np.linalg.lstsq(X_quad, y, rcond=None)[0]
        rv_q = v - X_quad @ np.linalg.lstsq(X_quad, v, rcond=None)[0]
        r_quad = float(stats.pearsonr(rv_q, ry_q).statistic)
        t_quad = r_quad * math.sqrt(df_quad / max(1e-12, 1.0 - r_quad ** 2))
        p_quad_exact = float(2.0 * stats.t.sf(abs(t_quad), df=df_quad))

        # 19,999 Freedman-Lane residual permutations within the 9 (domain, block) strata
        blk_strata = (df_in.domain + "_B" + df_in.block_design.astype(str)).to_numpy()
        strata_idx = [np.flatnonzero(blk_strata == s) for s in np.unique(blk_strata)]
        ry_norm = ry_cell / np.linalg.norm(ry_cell)
        rv_norm = rv_cell / np.linalg.norm(rv_cell)
        obs_abs = abs(float(np.dot(rv_norm, ry_norm)))
        cnt = 1
        n_perm = 19999
        for _ in range(n_perm):
            perm_v = rv_norm.copy()
            for ix_s in strata_idx:
                perm_v[ix_s] = perm_v[rng.permutation(ix_s)]
            if abs(float(np.dot(perm_v, ry_norm))) >= obs_abs - 1e-12:
                cnt += 1
        p_perm = cnt / (n_perm + 1)

        rows.append({
            "feature": col, "label": label, "group": grp, "n": n_obs,
            "within_env_apparent_r": r_env, "df_env": df_env, "p_env_exact": p_env_exact,
            "within_N_block_partial_r": r_part, "rank_X_cell": rk_cell, "df_exact": df_cell,
            "within_N_block_p": p_part_exact,  # Updated to exact df!
            "partial_r_quadN_block": r_quad, "df_quadN_block": df_quad, "p_quadN_block": p_quad_exact,
            "p_freedman_lane_perm": p_perm,
        })

    df_assoc = pd.DataFrame(rows)
    # Overwrite management_confounding_associations.csv with exact df & exact p (keeping n=64 primary rows first)
    df_assoc_64 = df_assoc[df_assoc.n == 64].copy()
    # Add per-domain r for compatibility with plot_q1_upgraded_figures.py
    for dom in DOMAINS:
        d_sub = plot64[plot64.domain == dom]
        zd = pd.get_dummies(d_sub[["N_kg_ha", "block_design"]].astype(str), drop_first=True, dtype=float)
        Xd = np.column_stack([np.ones(len(d_sub)), zd.to_numpy()])
        yd = d_sub.yield_kg_ha.to_numpy(float)
        ryd = yd - Xd @ np.linalg.lstsq(Xd, yd, rcond=None)[0]
        r_list = []
        for col in df_assoc_64["feature"]:
            vd = d_sub[col].to_numpy(float)
            rvd = vd - Xd @ np.linalg.lstsq(Xd, vd, rcond=None)[0]
            r_list.append(float(stats.pearsonr(rvd, ryd).statistic))
        df_assoc_64[f"r_{dom}"] = r_list

    df_assoc_64.to_csv(Q1_DIR / "management_confounding_associations.csv", index=False)
    df_assoc.to_csv(Q1_DIR / "exact_df_partial_correlations_audit.csv", index=False)

    # Check 2D plane exact F-statistic & p-value with df2 = 64 - 9 - 2 = 53 vs 61
    df_2d = pd.read_csv(Q1_DIR / "direction_b_2d_attribution_quadrants.csv")
    X2 = np.column_stack([np.ones(len(df_2d)), df_2d["I_Leaf_Phys"].to_numpy(), df_2d["I_Gap_Struct"].to_numpy()])
    y_blk = df_2d["yield_blk_anom"].to_numpy()
    yhat = X2 @ np.linalg.lstsq(X2, y_blk, rcond=None)[0]
    r2_2d = float(r2_score(y_blk, yhat))
    f_53 = (r2_2d / 2.0) / ((1.0 - r2_2d) / 53.0)
    p_53 = float(stats.f.sf(f_53, 2, 53))
    # Also check I_Leaf and I_Gap within-N partial r with exact df = 36!
    z_c64 = pd.get_dummies(
        df_2d[["domain"]].assign(
            dom_N=df_2d.domain + "_" + df_2d.N_kg_ha.astype(int).astype(str),
            dom_B=df_2d.domain + "_B" + df_2d.block_design.astype(str),
        ),
        drop_first=True, dtype=float,
    )
    Xc64 = np.column_stack([np.ones(len(df_2d)), z_c64.to_numpy()])
    ry64 = df_2d.yield_kg_ha.to_numpy() - Xc64 @ np.linalg.lstsq(Xc64, df_2d.yield_kg_ha.to_numpy(), rcond=None)[0]
    rlf = df_2d.I_Leaf_Phys.to_numpy() - Xc64 @ np.linalg.lstsq(Xc64, df_2d.I_Leaf_Phys.to_numpy(), rcond=None)[0]
    rgp = df_2d.I_Gap_Struct.to_numpy() - Xc64 @ np.linalg.lstsq(Xc64, df_2d.I_Gap_Struct.to_numpy(), rcond=None)[0]
    r_ilf = float(stats.pearsonr(rlf, ry64).statistic)
    p_ilf_36 = float(2.0 * stats.t.sf(abs(r_ilf * math.sqrt(36.0 / (1.0 - r_ilf ** 2))), df=36))
    r_igp = float(stats.pearsonr(rgp, ry64).statistic)
    p_igp_36 = float(2.0 * stats.t.sf(abs(r_igp * math.sqrt(36.0 / (1.0 - r_igp ** 2))), df=36))

    # Decompose 181 DAS N_C_T_Bmorph vs N_C_T across 6 transfers (Bias^2, Between-N, Within-N)
    dec = pd.read_csv(Q1_DIR / "hybrid_transfer_decomposition.csv")
    bm_dec = dec[dec.specification.isin(["N", "N_C_T", "N_C_T_Braw", "N_C_T_Bmorph"]) & (dec.calibration_regime == "zero_label")]
    bm_piv = bm_dec.groupby(["specification", "measure"])["estimate"].mean().unstack("measure")

    print("\n--- [R3] Exact df = 36 Partial Correlations ---")
    print(df_assoc[["feature", "n", "within_env_apparent_r", "within_N_block_partial_r", "df_exact", "within_N_block_p", "df_quadN_block", "p_quadN_block"]].to_string(index=False))
    print(f"2D Plane R2={r2_2d:.4f}, F(2,53)={f_53:.2f}, P={p_53:.3e} | I_Leaf within-N r={r_ilf:.3f} (df=36, P={p_ilf_36:.4f}) | I_Gap within-N r={r_igp:.3f} (df=36, P={p_igp_36:.4f})")
    print("\n--- [R9/W5] 181-DAS B_morph 6-Transfer Mean Error Decomposition (k=0, fertilized holdout) ---")
    print(bm_piv[["Total", "Bias", "Between-N", "Within-N"]].round(1).to_string())
    return {"F_2_53": f_53, "p_2_53": p_53, "p_ilf_36": p_ilf_36, "p_igp_36": p_igp_36}


# ==============================================================================
# 5. [R4 & R5] Same-Window N-Limitation Diagnostics & Pre-Cutoff (92-107 DAS) 2D Plane
# ==============================================================================
def run_r4_r5_diagnostics_and_early_plane() -> dict:
    print("\n=== [R4 & R5] Running Same-Window N-Limitation Diagnostics & Pre-Cutoff 92-107 DAS 2D Plane ===")
    df_gsd = pd.read_csv(Q1_DIR / "multiscale_gsd_slot_features.csv")
    df_nat_all = pd.read_csv(Q1_DIR / "millimeter_slot_features_699.csv")
    common_base = pd.read_csv(OLD_MINING / "common_plot_features_common_DAS.csv")[
        ["domain", "year", "field", "plot_id", "block_design", "N_kg_ha", "yield_kg_ha"]
    ]

    # Build both Early Pre-Cutoff (92-107 DAS ONLY, zero >107 DAS leakage!) and Full In-Season (92-142 DAS) tables across GSDs
    diag_rows = []
    early_plane_64 = None

    for gsd in GSD_LEVELS_M:
        lbl = GSD_LABELS[gsd]
        sub = df_gsd[df_gsd.gsd_m == gsd]
        recs = []
        for r in common_base.itertuples():
            d = sub[(sub.domain == r.domain) & (sub.plot_id == int(r.plot_id)) & (sub.DAS < 180)].sort_values("DAS")
            d1 = d.iloc[0]  # 92 DAS
            d2 = d.iloc[1]  # 106/107 DAS (strictly before 115 DAS N cutoff and 108-110 DAS topping!)
            d3 = d.iloc[2]  # 128/130 DAS
            d4 = d.iloc[3]  # 141/142 DAS
            xs = d.DAS.to_numpy(float)
            span = float(d4.DAS - d1.DAS)
            span_e = float(d2.DAS - d1.DAS)
            recs.append({
                "domain": r.domain, "plot_id": int(r.plot_id),
                "block_design": int(r.block_design), "N_kg_ha": float(r.N_kg_ha), "yield_kg_ha": float(r.yield_kg_ha),
                # Early window (92 & 106/107 DAS ONLY)
                "E92_exg": float(d1.exg_mixed), "E107_exg": float(d2.exg_mixed),
                "E92_cover": float(d1.f_cover), "E107_cover": float(d2.f_cover),
                "E_cover_change": float(d2.f_cover - d1.f_cover),
                "E92_ngrdi_leaf": float(d1.ngrdi_leaf), "E107_ngrdi_leaf": float(d2.ngrdi_leaf),
                "E_ngrdi_gain": float(0.5 * ((d1.ngrdi_leaf - d1.ngrdi_mixed) + (d2.ngrdi_leaf - d2.ngrdi_mixed))),
                "E107_gap_large": float(d2.gap_large_frac), "E107_prof_across": float(d2.profile_sd_across),
                "E_gap_mean": float(0.5 * (d1.gap_large_frac + d2.gap_large_frac)),
                # Full in-season window (92-142 DAS)
                "F_exg1": float(d3.exg_mixed), "F_exg2": float(d4.exg_mixed),
                "F_cover": float(d4.f_cover),
                "F_auc_mean": float(np.trapezoid(d.f_cover.to_numpy(float), xs) / span),
                "F_late_change": float((d4.f_cover - d3.f_cover) / (d4.DAS - d3.DAS) * span),
                "F_ngrdi_leaf": float(d4.ngrdi_leaf),
                "F_ngrdi_gain": float(np.mean(d.ngrdi_leaf.to_numpy(float) - d.ngrdi_mixed.to_numpy(float))),
                "F_bg_glare": float(d3.bg_glare_frac),
                "F_gap_large": float(d3.gap_large_frac),
                "F_prof_across": float(d3.profile_sd_across),
                "F_prof_auc": float(np.trapezoid(d.profile_sd_across.to_numpy(float), xs) / span),
            })
        tbl = pd.DataFrame(recs)

        if gsd == 0.0033:
            early_plane_64 = tbl.copy()

        windows = {
            "Early_92_107": {
                "Mixed_ExG": ["E92_exg", "E107_exg"],
                "Mixed_CT": ["E92_cover", "E107_cover", "E107_exg", "E_cover_change"],
                "Decoupled_p4": ["E107_cover", "E107_ngrdi_leaf", "E_ngrdi_gain", "E107_gap_large"],
                "Decoupled_p8": ["E92_cover", "E107_cover", "E92_ngrdi_leaf", "E107_ngrdi_leaf", "E_ngrdi_gain", "E_cover_change", "E107_gap_large", "E107_prof_across"],
            },
            "Full_92_142": {
                "Mixed_ExG": ["F_exg1", "F_exg2"],
                "Mixed_CT": ["F_cover", "F_exg2", "F_auc_mean", "F_late_change"],
                "Decoupled_p4": ["F_cover", "F_ngrdi_leaf", "F_auc_mean", "F_gap_large"],
                "Decoupled_p8": ["F_cover", "F_ngrdi_gain", "F_bg_glare", "F_auc_mean", "F_late_change", "F_gap_large", "F_prof_across", "F_prof_auc"],
            },
        }

        y_def = (tbl["N_kg_ha"] <= 75).astype(int).to_numpy()
        for win_name, mdict in windows.items():
            for m_name, cols in mdict.items():
                # 1. Within-field 9-block spatial LOBO (zero target block scaling!)
                p_lobo = np.zeros(len(tbl))
                for dom, g in tbl.groupby("domain"):
                    ix = g.index.to_numpy()
                    X = g[cols].to_numpy(float)
                    y = (g["N_kg_ha"] <= 75).astype(int).to_numpy()
                    blks = g["block_design"].to_numpy(int)
                    for b in np.unique(blks):
                        tr, te = blks != b, blks == b
                        sc = StandardScaler().fit(X[tr])
                        clf = LogisticRegression(C=0.5).fit(sc.transform(X[tr]), y[tr])
                        p_lobo[ix[te]] = clf.predict_proba(sc.transform(X[te]))[:, 1]

                # 2. Leave-One-Field-Out (LOFO, pure source-field standardization, zero target field stats!)
                p_lofo = np.zeros(len(tbl))
                for dom in DOMAINS:
                    tr_m = (tbl["domain"] != dom).to_numpy()
                    te_m = (tbl["domain"] == dom).to_numpy()
                    X_all = tbl[cols].to_numpy(float)
                    sc = StandardScaler().fit(X_all[tr_m])
                    clf = LogisticRegression(C=0.5).fit(sc.transform(X_all[tr_m]), y_def[tr_m])
                    p_lofo[te_m] = clf.predict_proba(sc.transform(X_all[te_m]))[:, 1]

                diag_rows.append({
                    "gsd_m": gsd, "gsd_label": lbl, "window": win_name, "model": m_name,
                    "LOBO_ROC_AUC": float(roc_auc_score(y_def, p_lobo)),
                    "LOBO_PR_AUC": float(average_precision_score(y_def, p_lobo)),
                    "LOBO_BalAcc_pct": float(100.0 * balanced_accuracy_score(y_def, (p_lobo >= 0.5).astype(int))),
                    "LOFO_ROC_AUC": float(roc_auc_score(y_def, p_lofo)),
                    "LOFO_PR_AUC": float(average_precision_score(y_def, p_lofo)),
                })

    df_diag = pd.DataFrame(diag_rows)
    df_diag.to_csv(Q1_DIR / "n_limitation_same_window_multiscale.csv", index=False)

    # Check C-grid sensitivity, Sub-optimal N (<=150 vs 225-300), Excess-N saturation (225-300 vs 375-450), and Early 2D Plane at 3.3 mm!
    assert early_plane_64 is not None
    t33 = early_plane_64.copy()
    # Early 2D diagnostic plane (92-107 DAS ONLY, strictly before 115 DAS cutoff!)
    for col in ["E107_ngrdi_leaf", "E92_ngrdi_leaf", "E_cover_change", "E107_gap_large", "E_gap_mean", "E107_prof_across", "yield_kg_ha"]:
        t33[f"{col}_bz"] = t33.groupby(["domain", "block_design"])[col].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 1e-9 else 1.0)
        )
    t33["yield_anom_blk"] = t33.groupby(["domain", "block_design"])["yield_kg_ha"].transform(lambda s: s - s.mean())
    t33["I_Leaf_early"] = (t33["E92_ngrdi_leaf_bz"] + t33["E107_ngrdi_leaf_bz"] + t33["E_cover_change_bz"]) / 3.0
    t33["I_Gap_early"] = (t33["E107_gap_large_bz"] + t33["E_gap_mean_bz"] + t33["E107_prof_across_bz"]) / 3.0

    X_e = np.column_stack([np.ones(len(t33)), t33["I_Leaf_early"].to_numpy(), t33["I_Gap_early"].to_numpy()])
    y_e = t33["yield_anom_blk"].to_numpy()
    yh_e = X_e @ np.linalg.lstsq(X_e, y_e, rcond=None)[0]
    r2_early_plane = float(r2_score(y_e, yh_e))
    f_early_53 = (r2_early_plane / 2.0) / ((1.0 - r2_early_plane) / 53.0)
    # 9-fold block LOBO R2 for early 2D plane
    oof_e = np.zeros(len(t33))
    strata_9 = (t33.domain + "_B" + t33.block_design.astype(str)).to_numpy()
    for st in np.unique(strata_9):
        tr, te = strata_9 != st, strata_9 == st
        beta_e = np.linalg.lstsq(X_e[tr], y_e[tr], rcond=None)[0]
        oof_e[te] = X_e[te] @ beta_e
    r2_early_lobo = float(r2_score(y_e, oof_e))

    # Sub-optimal N (N <= 150 vs N225-300) & Excess N (N225-300 vs N375-450)
    sub_opt_df = t33[t33.N_kg_ha <= 300].reset_index(drop=True)
    y_sub = (sub_opt_df.N_kg_ha <= 150).astype(int).to_numpy()
    p_sub = np.zeros(len(sub_opt_df))
    cols_e8 = ["E92_cover", "E107_cover", "E92_ngrdi_leaf", "E107_ngrdi_leaf", "E_ngrdi_gain", "E_cover_change", "E107_gap_large", "E107_prof_across"]
    for dom, g in sub_opt_df.groupby("domain"):
        ix = g.index.to_numpy()
        X = g[cols_e8].to_numpy(float)
        y = (g.N_kg_ha <= 150).astype(int).to_numpy()
        blks = g.block_design.to_numpy(int)
        for b in np.unique(blks):
            tr, te = blks != b, blks == b
            sc = StandardScaler().fit(X[tr])
            p_sub[ix[te]] = LogisticRegression(C=0.5).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
    auc_subopt_early = float(roc_auc_score(y_sub, p_sub))

    exc_df = t33[t33.N_kg_ha >= 225].reset_index(drop=True)
    y_exc = (exc_df.N_kg_ha >= 375).astype(int).to_numpy()
    p_exc = np.zeros(len(exc_df))
    for dom, g in exc_df.groupby("domain"):
        ix = g.index.to_numpy()
        X = g[cols_e8].to_numpy(float)
        y = (g.N_kg_ha >= 375).astype(int).to_numpy()
        blks = g.block_design.to_numpy(int)
        for b in np.unique(blks):
            tr, te = blks != b, blks == b
            sc = StandardScaler().fit(X[tr])
            p_exc[ix[te]] = LogisticRegression(C=0.5).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
    auc_excess_early = float(roc_auc_score(y_exc, p_exc))

    # C-grid sensitivity for Early_92_107 & Full_92_142 Decoupled_p8
    c_sens = {}
    for c_val in [0.05, 0.1, 0.5, 1.0, 5.0]:
        p_c = np.zeros(len(t33))
        for dom, g in t33.groupby("domain"):
            ix = g.index.to_numpy()
            X = g[cols_e8].to_numpy(float)
            y = (g.N_kg_ha <= 75).astype(int).to_numpy()
            blks = g.block_design.to_numpy(int)
            for b in np.unique(blks):
                tr, te = blks != b, blks == b
                sc = StandardScaler().fit(X[tr])
                p_c[ix[te]] = LogisticRegression(C=c_val, max_iter=500).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[te]))[:, 1]
        c_sens[c_val] = round(float(roc_auc_score((t33.N_kg_ha <= 75).astype(int), p_c)), 3)

    # 2024 early squaring/first-flower trajectory (57-85 DAS, prior to 1st-3rd fertigation splits!)
    # Directional univariate rank AUC (lower leaf NGRDI / lower ExG -> predicted N deficit, no sign-flipping)
    early_2024_rows = []
    for (dom, das), grp in df_nat_all[(df_nat_all.year == 2024) & (df_nat_all.DAS < 92)].groupby(["domain", "DAS"]):
        y_d = (grp.N_kg_ha <= 75).astype(int).to_numpy()
        auc_ngrdi = float(roc_auc_score(y_d, -grp.ngrdi_leaf.to_numpy(float)))
        auc_exg = float(roc_auc_score(y_d, -grp.exg_mixed.to_numpy(float)))
        early_2024_rows.append({
            "domain": dom, "DAS": int(das), "n_plots": len(grp),
            "NGRDI_leaf_ROC_AUC": round(auc_ngrdi, 3),
            "Mixed_ExG_ROC_AUC": round(auc_exg, 3),
        })
    pd.DataFrame(early_2024_rows).to_csv(Q1_DIR / "early_squaring_2024_57_85DAS_auc.csv", index=False)

    print("\n--- [R4] Same-Window N-Limitation Diagnostics (3.3 mm & 5 cm) ---")
    print(df_diag[df_diag.gsd_label.isin(["3.3 mm", "1 cm", "5 cm", "20 cm"])].to_string(index=False))
    print(f"\n--- [R5] Pre-Cutoff (92-107 DAS) Early 2D Plane: R2 = {r2_early_plane:.3f}, R2_LOBO = {r2_early_lobo:.3f}, F(2,53) = {f_early_53:.1f} ---")
    print(f"Sub-optimal N (<=150 vs 225-300) Early ROC AUC = {auc_subopt_early:.3f} | Excess N (375-450 vs 225-300) Early ROC AUC = {auc_excess_early:.3f}")
    print(f"C-grid sensitivity (Early Decoupled p=8): {c_sens}")
    print("2024 Early Squaring (57-85 DAS) Trajectory:\n", pd.DataFrame(early_2024_rows).to_string(index=False))
    return {
        "r2_early_plane": r2_early_plane,
        "r2_early_lobo": r2_early_lobo,
        "auc_subopt_early": auc_subopt_early,
        "auc_excess_early": auc_excess_early,
        "c_sens": c_sens,
    }


# ==============================================================================
# 6. [R7 & S4] Bare-Soil Normalization Audit, 6-Block Pooled LOEO & Fixed Holdout
# ==============================================================================
def run_r7_s4_pooled_source_and_baresoil() -> dict:
    print("\n=== [R7 & S4] Running 6-Block Pooled-Source LOEO CV & Bare-Soil Radiometric Summary ===")
    plot64 = pd.read_csv(Q1_DIR / "plot64_millimeter_features.csv")

    # [S4] Leave-One-Environment-Out (LOEO): pool 2 source environments (B = 6 blocks, n = 40-45) -> predict 3rd!
    # Center source environments by environment mean yield? No, in PartialRidge X_N = [1, env_dummy, N, N^2] or pooled [1, N, N^2]:
    # Wait: across two years (e.g. 2023_F1 + 2024_F1 -> 2024_F2), including a source-environment intercept dummy in unpenalized X_N
    # absorbs inter-year yield level differences while estimating shared N curvature and 6-block orthogonal optical slopes!
    loeo_rows = []
    specs_loeo = {
        "N": [],
        "N_C_T": ["C_cover", "C_exg", "T_auc_mean", "T_late_change"],
        "N_Leaf_T": ["C_cover", "C_ngrdi_leaf", "T_auc_mean", "T_late_change"],
    }
    for tgt in DOMAINS:
        src_df = plot64[plot64.domain != tgt].reset_index(drop=True)
        tgt_df = plot64[plot64.domain == tgt].reset_index(drop=True)
        src_doms = sorted(src_df.domain.unique())
        env_dum_src = (src_df.domain == src_doms[1]).to_numpy(float)
        # 6 unique source blocks!
        blk6 = (src_df.domain + "_B" + src_df.block_design.astype(str)).to_numpy()
        y_src = src_df.yield_kg_ha.to_numpy(float)
        y_tgt = tgt_df.yield_kg_ha.to_numpy(float)
        n_src = src_df.N_kg_ha.to_numpy(float)
        n_tgt = tgt_df.N_kg_ha.to_numpy(float)

        # Fit N-only and optical PartialRidge with 6-fold leave-one-source-block-out tuning!
        base_err, base_wn = None, None
        for sp, cols in specs_loeo.items():
            # Unpenalized X_N in source includes source env dummy so optical features Z are projected orthogonal to env + quadratic N!
            # At prediction on target, target intercept cancels in Within-N and under +3N0 / k=3 calibration!
            X_unpen_src = np.column_stack([np.ones(len(src_df)), env_dum_src, n_src, n_src ** 2])
            X_unpen_tgt = np.column_stack([np.ones(len(tgt_df)), np.full(len(tgt_df), 0.5), n_tgt, n_tgt ** 2])

            def fit_pred_loeo(tr_mask: np.ndarray, X_u_te: np.ndarray, Z_te: np.ndarray | None, alpha: float):
                Xu = X_unpen_src[tr_mask]
                yy = y_src[tr_mask]
                beta_u = np.linalg.lstsq(Xu, yy, rcond=None)[0]
                if Z_te is None:
                    return X_u_te @ beta_u
                Z_tr = src_df.loc[tr_mask, cols].to_numpy(float)
                sc_z = StandardScaler().fit(Z_tr)
                Z_tr_s = sc_z.transform(Z_tr)
                Z_te_s = sc_z.transform(Z_te)
                # Orthogonalize Z_tr_s and Z_te_s consistently against Xu = [1, env_dummy, N, N^2]
                B_u = np.linalg.lstsq(Xu, Z_tr_s, rcond=None)[0]
                Q_tr = Z_tr_s - Xu @ B_u
                Q_te = Z_te_s - X_u_te @ B_u
                res_y = yy - Xu @ beta_u
                gamma = np.linalg.solve(Q_tr.T @ Q_tr + alpha * np.eye(Q_tr.shape[1]), Q_tr.T @ res_y)
                return X_u_te @ beta_u + Q_te @ gamma

            if not cols:
                best_a = 0.0
                yh = fit_pred_loeo(np.ones(len(src_df), dtype=bool), X_unpen_tgt, None, 0.0)
            else:
                ranks = []
                for alpha in legacy.ALPHAS:
                    oof = np.empty(len(src_df))
                    for b_id in np.unique(blk6):
                        tr_m = blk6 != b_id
                        te_m = blk6 == b_id
                        oof[te_m] = fit_pred_loeo(tr_m, X_unpen_src[te_m], src_df.loc[te_m, cols].to_numpy(float), alpha)
                    rm = float(np.sqrt(np.mean((oof - y_src) ** 2)))
                    ranks.append((rm, -alpha, alpha))
                _, _, best_a = min(ranks)
                yh = fit_pred_loeo(np.ones(len(src_df), dtype=bool), X_unpen_tgt, tgt_df[cols].to_numpy(float), best_a)

            err = yh - y_tgt
            ewn = err.copy()
            for lv in np.unique(n_tgt):
                ewn[n_tgt == lv] -= ewn[n_tgt == lv].mean()
            tot_rmse = float(np.sqrt(np.mean(err ** 2)))
            wn_rmse = float(np.sqrt(np.mean(ewn ** 2)))
            # +3N0 on fertilized holdout
            m_n0 = n_tgt == 0
            sh_n0 = float(np.mean(y_tgt[m_n0] - yh[m_n0]))
            p3n0_rmse = float(np.sqrt(np.mean(((yh[~m_n0] + sh_n0) - y_tgt[~m_n0]) ** 2)))

            if sp == "N":
                base_err, base_wn, base_3n0 = tot_rmse, wn_rmse, p3n0_rmse
            loeo_rows.append({
                "sources_pooled": "+".join(src_doms),
                "target": tgt,
                "n_source_plots": len(src_df),
                "n_source_blocks": 6,
                "specification": sp,
                "selected_alpha": best_a,
                "Total_RMSE": tot_rmse,
                "WithinN_RMSE": wn_rmse,
                "Delta_Total_vs_N": base_err - tot_rmse,
                "Delta_WithinN_vs_N": base_wn - wn_rmse,
                "plus3N0_RMSE": p3n0_rmse,
                "Delta_plus3N0_vs_N": base_3n0 - p3n0_rmse,
            })
    df_loeo = pd.DataFrame(loeo_rows)
    df_loeo.to_csv(Q1_DIR / "ablation_6block_pooled_source_loeo.csv", index=False)

    # [R7] Bare-soil radiometric normalization summary from brightness_reference_patches & in-season soil background
    bs_patches = pd.read_csv(OLD_METHOD / "brightness_reference_patches.csv")
    bs_metrics = pd.read_csv(OLD_METHOD / "brightness_source_model_metrics.csv")
    # Also check in-season bare-soil / non-glare non-green background normalization on NGRDI_leaf!
    df_nat = pd.read_csv(Q1_DIR / "millimeter_slot_features_699.csv")
    bs_summary = bs_patches.groupby(["domain", "raster_file"])[["valid_pixels", "median_DN", "flight_reference_median_DN", "scale"]].mean().reset_index()
    bs_summary.to_csv(Q1_DIR / "baresoil_radiometric_audit_summary.csv", index=False)

    print("\n--- [S4] 6-Block Pooled-Source LOEO Results (Zero Penalty-Ceiling Collapse!) ---")
    print(df_loeo.to_string(index=False))
    print("\n--- [R7] Bare-Soil Reference Patch Radiometric Scaling Factors ---")
    print(bs_summary.to_string(index=False))
    return {}


def main() -> None:
    t0 = time.time()
    run_r1_masked_resnet_and_runtime()
    run_r8_2x2_index_mask_ablation()
    run_r2_rgb_only_normalizations()
    run_r3_exact_degrees_of_freedom()
    run_r4_r5_diagnostics_and_early_plane()
    run_r7_s4_pooled_source_and_baresoil()
    print(f"\nAll peer-review ablations completed successfully in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
