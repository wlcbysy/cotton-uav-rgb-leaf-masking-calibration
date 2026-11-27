#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Quantitative Audits & Robustness Checks.
Strictly READ-ONLY on 01_Data/raw and analysis/frozen_baseline.
Writes verified CSVs to analysis/q1_spatial_calibration/:
  1. Table 3 27-resample fixed-center intervals for N_ResNet50_Masked_p12, N_ResNet50_Unmasked_p12,
     and N_ResNet50_Masked_p52 -> resnet_masked_p12_table3_decomposition.csv and resnet_masked_p12_coverage.csv.
  2. Time-window-matched RGB_Mixed_CT_Bmorph (92-181 DAS, p=5) across all 4 normalization regimes ->
     ablation_rgb_only_normalization_regimes.csv.
  3. Exact 90% PI empirical coverage & width under UAV quantile-stratified k=3 (n-3 holdout) ->
     post_review_quantstrat_k3_coverage.csv.
  4. I_Leaf-I_Gap correlation, Q1-Q4 N-rate breakdown & within-(N,block) residuals, and zero-leakage
     (training-standardized, no target z_blk) 2D plane LOBO/LOFO R^2 -> post_review_2d_plane_audit.csv.
  5. Exact same-subset (n-k) post-harvest shift reference for k=0..6 AND no-UAV N-rate-stratified
     baseline (k=3, 6) -> post_review_calibration_audit.csv.
  6. BH-FDR adjusted P-values for Fig. 2e partial correlations AND exact 2^9=512 whole-block sign-flip
     permutation P-values across the 9 unique physical blocks -> post_review_permutation_fdr_audit.csv.
  7. Offline photometric stress test (exposure x0.70-x1.30 + overcast blue shift) on Canopeo
     2G - R - B > 20 across all 64 real complete-case plots -> post_review_photometric_stress_test.csv.
"""
from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "02_Code/src/data"))
sys.path.insert(0, str(ROOT / "02_Code/src/modeling"))
sys.path.insert(0, str(ROOT / "02_Code/src/upgrade_q1"))

from extract_raw_rgb_information import GPKG, crop
from run_module_c_fewshot_calibration import (
    TRANSFER_TASKS,
    block_resample_components,
    error_components,
    select_uav_active_plots,
)

FORMAL_NEW = ROOT / "10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108"
FORMAL_OLD = FORMAL_NEW / "analysis/frozen_baseline"
OLD_METHOD = FORMAL_OLD / "analysis/method_strengthening"
Q1_DIR = FORMAL_NEW / "analysis/q1_spatial_calibration"


def task1_fix_table3_and_add_unmasked_p12() -> None:
    print("\n=== Recomputing ResNet-50 Table 3 & Coverage (Fixed-Center + Matched p=12) ===")
    df_rn_all = pd.read_csv(Q1_DIR / "resnet_mask_pca_predictions.csv")
    specs = ["N_ResNet50_Masked_p12", "N_ResNet50_Unmasked_p12", "N_ResNet50_Masked_p52"]

    t3_rows = []
    cov_rows = []
    for sp in specs:
        for src, tgt, ev in TRANSFER_TASKS:
            d_src_oof = df_rn_all[
                (df_rn_all.source == src)
                & (df_rn_all.target == src)
                & (df_rn_all.evaluation == "BlockCV")
                & (df_rn_all.specification == sp)
            ]
            d_src_fert = d_src_oof[d_src_oof.N > 0]
            abs_res_src = np.sort(np.abs(d_src_fert.observed - d_src_fert.predicted).to_numpy(float))
            rank_q = min(len(abs_res_src), int(math.ceil(0.90 * (len(abs_res_src) + 1)))) - 1
            q90_rad = float(abs_res_src[rank_q])

            d_tgt = (
                df_rn_all[
                    (df_rn_all.source == src)
                    & (df_rn_all.target == tgt)
                    & (df_rn_all.evaluation == ev)
                    & (df_rn_all.specification == sp)
                ]
                .sort_values("plot_id")
                .reset_index(drop=True)
            )
            g_n0 = d_tgt[d_tgt.N == 0]
            g_ho = d_tgt[d_tgt.N > 0].reset_index(drop=True)
            sh_3n0 = float(np.mean(g_n0.observed - g_n0.predicted))

            for reg_id, shift_val in [(f"{sp}__zero_label", 0.0), (f"{sp}__plus_3N0", sh_3n0)]:
                err_ho = (g_ho.predicted.to_numpy(float) + shift_val) - g_ho.observed.to_numpy(float)
                n_ho = g_ho.N.to_numpy(float)
                blks_ho = g_ho.block.to_numpy(int)
                # Unified fixed-center 27-resample protocol matching hybrid_transfer_decomposition.csv!
                res = block_resample_components(err_ho, n_ho, blks_ho)
                for m in ("Total", "Bias", "Between-N", "Within-N"):
                    est, lo, hi = res[m]
                    t3_rows.append(
                        {
                            "source": src,
                            "target": tgt,
                            "evaluation": ev,
                            "model_id": reg_id,
                            "measure": m,
                            "estimate": est,
                            "low_95": lo,
                            "high_95": hi,
                        }
                    )
                cov_rows.append(
                    {
                        "source": src,
                        "target": tgt,
                        "evaluation": ev,
                        "model": reg_id,
                        "coverage": float(np.mean(np.abs(err_ho) <= q90_rad)),
                        "mean_width_kg_ha": 2.0 * q90_rad,
                    }
                )

    df_t3 = pd.DataFrame(t3_rows)
    df_cov = pd.DataFrame(cov_rows)
    df_t3.to_csv(Q1_DIR / "resnet_masked_p12_table3_decomposition.csv", index=False)
    df_cov.to_csv(Q1_DIR / "resnet_masked_p12_coverage.csv", index=False)

    # Verify zero 0.0 lower bounds in Within-N!
    wn_min = df_t3[df_t3.measure == "Within-N"]["low_95"].min()
    print(f"Verified min Within-N low_95 across all specs: {wn_min:.2f} (previously 0.00!)")
    piv = df_t3.pivot_table(index="model_id", columns="measure", values="estimate", aggfunc="mean")
    print("\n6-task mean Table 3 decomposition (Masked vs Unmasked matched p=12 & Masked p=52):")
    print(piv[["Total", "Bias", "Between-N", "Within-N"]].round(1).to_string())
    print("\n6-task mean 90% PI coverage & width:")
    print(df_cov.groupby("model")[["coverage", "mean_width_kg_ha"]].mean().round(4).to_string())


def task2_add_rgb_mixed_ct_bmorph() -> None:
    print("\n=== Adding RGB_Mixed_CT_Bmorph (92-181 DAS, p=5) to Table S2 CSV ===")
    plot64 = pd.read_csv(Q1_DIR / "plot64_millimeter_features.csv")
    df_existing = pd.read_csv(Q1_DIR / "ablation_rgb_only_normalization_regimes.csv")
    df_existing = df_existing[df_existing.specification != "RGB_Mixed_CT_Bmorph"].copy()

    base_raw_cols = [
        "Cearly_exg", "C_exg", "C_cover", "T_auc_mean", "T_late_change",
        "C_ngrdi_leaf", "T_ngrdi_leaf_auc",
        "Searly_gap_large", "T_gap_auc", "Searly_prof_across",
        "white_b140_s30_morph_frac",
    ]
    col_idx = {c: i for i, c in enumerate(base_raw_cols)}
    sets_raw = {dom: g.sort_values("plot_id").reset_index(drop=True) for dom, g in plot64.groupby("domain")}
    alphas = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0]

    def build_ct_bmorph(Z_scaled: np.ndarray) -> np.ndarray:
        return Z_scaled[
            :,
            [
                col_idx["C_cover"],
                col_idx["C_exg"],
                col_idx["T_auc_mean"],
                col_idx["T_late_change"],
                col_idx["white_b140_s30_morph_frac"],
            ],
        ]

    def scale_base(df_dom: pd.DataFrame, regime: str, fit_scaler: StandardScaler | None = None):
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

    new_rows = []
    for regime in ["Pure_Source_Std", "Strip_N225_Only", "Strip_N0_N225", "Trial_MultiRate_z_blk"]:
        for src, ds in sets_raw.items():
            ys = ds.yield_kg_ha.to_numpy(float)
            gs = ds.block_design.to_numpy(int)

            # BlockCV (exact match to run_r2_rgb_only_normalizations)
            ii = np.arange(len(ds))
            oof = np.empty(len(ds))
            for b_id in np.unique(gs):
                tr_m = ii[gs != b_id]
                te_m = ii[gs == b_id]
                X_tr_raw = ds.iloc[tr_m][base_raw_cols].to_numpy(float)
                X_te_raw = ds.iloc[te_m][base_raw_cols].to_numpy(float)
                if regime == "Pure_Source_Std":
                    sc_f = StandardScaler().fit(X_tr_raw)
                    Z_tr = sc_f.transform(X_tr_raw)
                    Z_te = sc_f.transform(X_te_raw)
                elif regime == "Strip_N225_Only":
                    n_tr = ds.iloc[tr_m].N_kg_ha.to_numpy(float)
                    mu225_tr = X_tr_raw[n_tr == 225].mean(axis=0)
                    X_tr_st = X_tr_raw - mu225_tr
                    X_te_st = X_te_raw - mu225_tr
                    sc_f = StandardScaler().fit(X_tr_st)
                    Z_tr = sc_f.transform(X_tr_st)
                    Z_te = sc_f.transform(X_te_st)
                elif regime == "Strip_N0_N225":
                    n_tr = ds.iloc[tr_m].N_kg_ha.to_numpy(float)
                    mu0_tr = X_tr_raw[n_tr == 0].mean(axis=0)
                    mu225_tr = X_tr_raw[n_tr == 225].mean(axis=0)
                    span_tr = np.where(np.abs(mu225_tr - mu0_tr) > 1e-6, mu225_tr - mu0_tr, 1.0)
                    X_tr_st = (X_tr_raw - mu0_tr) / span_tr
                    X_te_st = (X_te_raw - mu0_tr) / span_tr
                    sc_f = StandardScaler().fit(X_tr_st)
                    Z_tr = sc_f.transform(X_tr_st)
                    Z_te = sc_f.transform(X_te_st)
                else:
                    Z_all, _ = scale_base(ds, regime, None)
                    Z_tr, Z_te = Z_all[tr_m], Z_all[te_m]
                X_tr, X_te = build_ct_bmorph(Z_tr), build_ct_bmorph(Z_te)
                y_tr = ys[tr_m]
                g_tr = gs[tr_m]
                best_a, best_rm = 10.0, float("inf")
                for a in alphas:
                    in_oof = np.empty(len(y_tr))
                    for ib in np.unique(g_tr):
                        itr, ite = g_tr != ib, g_tr == ib
                        m = Ridge(alpha=a).fit(X_tr[itr], y_tr[itr])
                        in_oof[ite] = m.predict(X_tr[ite])
                    rm = float(np.sqrt(np.mean((in_oof - y_tr) ** 2)))
                    if rm < best_rm:
                        best_rm, best_a = rm, a
                oof[te_m] = Ridge(alpha=best_a).fit(X_tr, y_tr).predict(X_te)
            rmse_cv = float(np.sqrt(np.mean((oof - ys) ** 2)))
            r2_cv = float(r2_score(ys, oof))
            new_rows.append(
                {
                    "regime": regime,
                    "specification": "RGB_Mixed_CT_Bmorph",
                    "source": src,
                    "target": src,
                    "evaluation": "BlockCV",
                    "RMSE_k0_all": rmse_cv,
                    "R2_k0_all": r2_cv,
                    "RMSE_k0_ho": rmse_cv,
                    "RMSE_k3_ho": rmse_cv,
                    "R2_k3_ho": r2_cv,
                }
            )

            # Full source fit -> transfers
            Z_src_full, sc_full = scale_base(ds, regime)
            X_src_full = build_ct_bmorph(Z_src_full)
            best_a, best_rm = alphas[0], float("inf")
            for a in alphas:
                in_oof = np.empty(len(ys))
                for ib in np.unique(gs):
                    itr, ite = gs != ib, gs == ib
                    in_oof[ite] = Ridge(alpha=a).fit(X_src_full[itr], ys[itr]).predict(X_src_full[ite])
                rm = float(np.sqrt(np.mean((in_oof - ys) ** 2)))
                if rm < best_rm:
                    best_rm, best_a = rm, a
            mdl = Ridge(alpha=best_a).fit(X_src_full, ys)

            for tgt, dt in sets_raw.items():
                if tgt == src:
                    continue
                ev = "adjacent_field" if src[:4] == tgt[:4] else "cross_year"
                yt = dt.yield_kg_ha.to_numpy(float)
                bt = dt.block_design.to_numpy(int)
                Z_tgt, _ = scale_base(dt, regime, sc_full)
                yh = mdl.predict(build_ct_bmorph(Z_tgt))
                rmse_all = float(np.sqrt(np.mean((yh - yt) ** 2)))
                r2_all = float(r2_score(yt, yh))

                # Identical yield-blind block-mean canopy-cover integral plot per block (matching run_r2_rgb_only_normalizations)
                cal_pids = []
                for _, gb in dt.groupby("block_design"):
                    mu_v = float(gb["T_auc_mean"].mean())
                    cal_pids.append(int(gb.loc[(gb["T_auc_mean"] - mu_v).abs().idxmin(), "plot_id"]))
                cal_mask = dt["plot_id"].isin(cal_pids).to_numpy()
                ho_mask = ~cal_mask
                sh = float(np.mean(yt[cal_mask] - yh[cal_mask]))
                rmse_k0_ho = float(np.sqrt(np.mean((yh[ho_mask] - yt[ho_mask]) ** 2)))
                rmse_k3_ho = float(np.sqrt(np.mean(((yh[ho_mask] + sh) - yt[ho_mask]) ** 2)))
                r2_k3_ho = float(r2_score(yt[ho_mask], yh[ho_mask] + sh))
                new_rows.append(
                    {
                        "regime": regime,
                        "specification": "RGB_Mixed_CT_Bmorph",
                        "source": src,
                        "target": tgt,
                        "evaluation": ev,
                        "RMSE_k0_all": rmse_all,
                        "R2_k0_all": r2_all,
                        "RMSE_k0_ho": rmse_k0_ho,
                        "RMSE_k3_ho": rmse_k3_ho,
                        "R2_k3_ho": r2_k3_ho,
                    }
                )

    df_out = pd.concat([df_existing, pd.DataFrame(new_rows)], ignore_index=True)
    df_out.to_csv(Q1_DIR / "ablation_rgb_only_normalization_regimes.csv", index=False)

    sub = df_out[df_out.specification == "RGB_Mixed_CT_Bmorph"]
    print("RGB_Mixed_CT_Bmorph (92-181 DAS, p=5) summary by regime & evaluation:")
    print(sub.groupby(["regime", "evaluation"])[["RMSE_k0_all", "R2_k0_all", "RMSE_k3_ho", "R2_k3_ho"]].mean().round(2).to_string())


def task3_quantstrat_k3_coverage_and_task5_calibration_audit() -> None:
    print("\n=== Quantile-Stratified k=3 PI Coverage + Same-Subset Shift Ref & No-UAV N-Stratified Baseline ===")
    ms = pd.read_csv(Q1_DIR / "multiscale_predictions.csv").query("prefix == '3.3 mm'")
    rn = pd.read_csv(Q1_DIR / "resnet_mask_pca_predictions.csv")

    models = [
        ("N-only", ms[ms.specification == "N"]),
        ("N+C+T", ms[ms.specification == "N_C_T"]),
        ("N+Leaf+T", ms[ms.specification == "N_Leaf_T"]),
        ("N+Masked ResNet-50 (p=12)", rn[rn.specification == "N_ResNet50_Masked_p12"]),
        ("N+Unmasked ResNet-50 (p=12)", rn[rn.specification == "N_ResNet50_Unmasked_p12"]),
        ("N+Unmasked ResNet-50 (p=52)", rn[rn.specification == "N_ResNet50_Unmasked_p52"]),
    ]
    df_leaf = ms[ms.specification == "N_Leaf_T"]

    # 90% PI coverage under UAV quantile-stratified k=3 (using N+Leaf+T predicted quantiles, matching Table S4)
    cov_q3_rows = []
    for m_lbl, df_m in models:
        for src, tgt, ev in TRANSFER_TASKS:
            d_src = df_m[(df_m.source == src) & (df_m.target == src) & (df_m.evaluation == "BlockCV") & (df_m.N > 0)]
            abs_res = np.sort(np.abs(d_src.observed - d_src.predicted).to_numpy(float))
            rank_q = min(len(abs_res), int(math.ceil(0.90 * (len(abs_res) + 1)))) - 1
            q90 = float(abs_res[rank_q])

            d_tgt = df_m[(df_m.source == src) & (df_m.target == tgt) & (df_m.evaluation == ev) & (df_m.N > 0)].sort_values("plot_id").reset_index(drop=True)
            d_lf_tgt = df_leaf[(df_leaf.source == src) & (df_leaf.target == tgt) & (df_leaf.evaluation == ev) & (df_leaf.N > 0)].sort_values("plot_id").reset_index(drop=True)

            cal_pids = set(select_uav_active_plots(d_lf_tgt, d_lf_tgt.predicted.to_numpy(float), 3))
            cal_mask = d_tgt["plot_id"].astype(int).isin(cal_pids).to_numpy()
            ho_mask = ~cal_mask

            sh_q3 = float(np.mean(d_tgt.observed.to_numpy(float)[cal_mask] - d_tgt.predicted.to_numpy(float)[cal_mask]))
            err_ho = (d_tgt.predicted.to_numpy(float)[ho_mask] + sh_q3) - d_tgt.observed.to_numpy(float)[ho_mask]
            cov_q3_rows.append(
                {
                    "source": src,
                    "target": tgt,
                    "evaluation": ev,
                    "model": m_lbl,
                    "n_holdout": int(ho_mask.sum()),
                    "rmse_q3": float(np.sqrt(np.mean(err_ho ** 2))),
                    "coverage_q3": float(np.mean(np.abs(err_ho) <= q90)),
                    "mean_width_kg_ha": 2.0 * q90,
                }
            )
    df_cov_q3 = pd.DataFrame(cov_q3_rows)
    df_cov_q3.to_csv(Q1_DIR / "post_review_quantstrat_k3_coverage.csv", index=False)
    print("UAV quantile-stratified k=3 6-task mean RMSE & 90% PI coverage:")
    print(df_cov_q3.groupby("model")[["rmse_q3", "coverage_q3", "mean_width_kg_ha"]].mean().round(4).to_string())

    # Same-subset (n-k) post-harvest shift reference for k=0..6 AND no-UAV N-rate-stratified baseline (k=3, 6)
    cal_audit_rows = []
    df_n = ms[ms.specification == "N"]
    for k in range(0, 7):
        rm_oracle_sub = []
        rm_n_qstrat = []
        rm_lf_qstrat = []
        rm_n_nstrat = []
        rm_lf_nstrat = []
        for src, tgt, ev in TRANSFER_TASKS:
            dn = df_n[(df_n.source == src) & (df_n.target == tgt) & (df_n.evaluation == ev) & (df_n.N > 0)].sort_values("plot_id").reset_index(drop=True)
            dl = df_leaf[(df_leaf.source == src) & (df_leaf.target == tgt) & (df_leaf.evaluation == ev) & (df_leaf.N > 0)].sort_values("plot_id").reset_index(drop=True)
            y = dn.observed.to_numpy(float)
            pn = dn.predicted.to_numpy(float)
            pl = dl.predicted.to_numpy(float)
            blks = dn.block.to_numpy(int)
            ns = dn.N.to_numpy(float)

            if k == 0:
                ho = np.ones(len(dn), dtype=bool)
                sh_n = sh_l = 0.0
            else:
                cal_pids = set(select_uav_active_plots(dl, pl, k))
                cal = dn["plot_id"].astype(int).isin(cal_pids).to_numpy()
                ho = ~cal
                sh_n = float(np.mean(y[cal] - pn[cal]))
                sh_l = float(np.mean(y[cal] - pl[cal]))

            # Exact oracle shift on THIS SAME n-k holdout!
            sh_ora_sub = float(np.mean(y[ho] - pn[ho]))
            rm_oracle_sub.append(float(np.sqrt(np.mean(((pn[ho] + sh_ora_sub) - y[ho]) ** 2))))
            rm_n_qstrat.append(float(np.sqrt(np.mean(((pn[ho] + sh_n) - y[ho]) ** 2))))
            rm_lf_qstrat.append(float(np.sqrt(np.mean(((pl[ho] + sh_l) - y[ho]) ** 2))))

            # No-UAV N-rate-stratified baseline (deterministic quantile of N rate per block)
            if k in (3, 6):
                cal_ns_pids = set(select_uav_active_plots(dn, ns, k))
                cal_ns = dn["plot_id"].astype(int).isin(cal_ns_pids).to_numpy()
                ho_ns = ~cal_ns
                sh_n_ns = float(np.mean(y[cal_ns] - pn[cal_ns]))
                sh_l_ns = float(np.mean(y[cal_ns] - pl[cal_ns]))
                rm_n_nstrat.append(float(np.sqrt(np.mean(((pn[ho_ns] + sh_n_ns) - y[ho_ns]) ** 2))))
                rm_lf_nstrat.append(float(np.sqrt(np.mean(((pl[ho_ns] + sh_l_ns) - y[ho_ns]) ** 2))))

        cal_audit_rows.append(
            {
                "k": k,
                "N_only_qstrat_RMSE": float(np.mean(rm_n_qstrat)),
                "N_Leaf_T_qstrat_RMSE": float(np.mean(rm_lf_qstrat)),
                "N_only_same_subset_oracle_RMSE": float(np.mean(rm_oracle_sub)),
                "N_only_noUAV_Nstrat_RMSE": float(np.mean(rm_n_nstrat)) if rm_n_nstrat else np.nan,
                "N_Leaf_T_noUAV_Nstrat_RMSE": float(np.mean(rm_lf_nstrat)) if rm_lf_nstrat else np.nan,
            }
        )
    df_cal_audit = pd.DataFrame(cal_audit_rows)
    df_cal_audit.to_csv(Q1_DIR / "post_review_calibration_audit.csv", index=False)
    print("\nSame-subset (n-k) oracle shift & No-UAV N-stratified baseline:")
    print(df_cal_audit.round(1).to_string(index=False))


def task4_2d_plane_audit() -> None:
    print("\n=== 2D Plane Correlation, Q1-Q4 Breakdown & Zero-Leakage LOBO/LOFO R^2 ===")
    q2d = pd.read_csv(Q1_DIR / "direction_b_2d_attribution_quadrants.csv")
    r_lg, p_lg = stats.pearsonr(q2d["I_Leaf_Phys"], q2d["I_Gap_Struct"])
    print(f"r(I_Leaf, I_Gap) = {r_lg:.3f}, P = {p_lg:.4e}")

    q_rows = []
    for q, g in q2d.groupby("quadrant"):
        n_low = int((g.N_kg_ha <= 75).sum())
        n_mid = int(((g.N_kg_ha >= 150) & (g.N_kg_ha <= 300)).sum())
        n_hi = int((g.N_kg_ha >= 375).sum())
        q_rows.append(
            {
                "quadrant": q,
                "n": len(g),
                "n_N0_75": n_low,
                "n_N150_300": n_mid,
                "n_N375_450": n_hi,
                "mean_yield_kg_ha": float(g.yield_kg_ha.mean()),
                "mean_blk_anom_kg_ha": float(g.yield_blk_anom.mean()),
                "mean_within_NB_resid_kg_ha": float(g.yield_within_NB.mean()),
            }
        )
    df_q = pd.DataFrame(q_rows)
    print("\nQ1-Q4 N-rate composition & within-(N,block) residuals:")
    print(df_q.round(1).to_string(index=False))

    # Zero-leakage (training-standardized raw features -> I_Leaf_raw, I_Gap_raw, ZERO target-block z_blk!)
    leaf_cols = ["C_ngrdi_leaf", "T_ngrdi_leaf_auc", "T_late_change"]
    gap_cols = ["Searly_gap_large", "T_gap_auc", "Searly_prof_across"]
    blk_key = q2d["domain"].astype(str) + "_B" + q2d["block_design"].astype(str)
    y_anom = q2d["yield_blk_anom"].to_numpy(float)
    y_raw = q2d["yield_kg_ha"].to_numpy(float)

    # Zero-leakage LOBO (9 env-blocks): fit StandardScaler strictly on training 8 blocks, build I_Leaf/I_Gap, fit OLS
    pred_lobo_anom = np.empty(len(q2d))
    pred_lobo_raw = np.empty(len(q2d))
    for b in blk_key.unique():
        tr = (blk_key != b).to_numpy()
        te = (blk_key == b).to_numpy()
        sc_l = StandardScaler().fit(q2d.loc[tr, leaf_cols])
        sc_g = StandardScaler().fit(q2d.loc[tr, gap_cols])
        X_tr = np.column_stack([sc_l.transform(q2d.loc[tr, leaf_cols]).mean(axis=1), sc_g.transform(q2d.loc[tr, gap_cols]).mean(axis=1)])
        X_te = np.column_stack([sc_l.transform(q2d.loc[te, leaf_cols]).mean(axis=1), sc_g.transform(q2d.loc[te, gap_cols]).mean(axis=1)])
        X_tr_i = np.column_stack([np.ones(tr.sum()), X_tr])
        X_te_i = np.column_stack([np.ones(te.sum()), X_te])
        pred_lobo_anom[te] = X_te_i @ np.linalg.lstsq(X_tr_i, y_anom[tr], rcond=None)[0]
        pred_lobo_raw[te] = X_te_i @ np.linalg.lstsq(X_tr_i, y_raw[tr], rcond=None)[0]

    # Zero-leakage LOFO (3 environments)
    pred_lofo_anom = np.empty(len(q2d))
    for dom in q2d["domain"].unique():
        tr = (q2d["domain"] != dom).to_numpy()
        te = (q2d["domain"] == dom).to_numpy()
        sc_l = StandardScaler().fit(q2d.loc[tr, leaf_cols])
        sc_g = StandardScaler().fit(q2d.loc[tr, gap_cols])
        X_tr = np.column_stack([sc_l.transform(q2d.loc[tr, leaf_cols]).mean(axis=1), sc_g.transform(q2d.loc[tr, gap_cols]).mean(axis=1)])
        X_te = np.column_stack([sc_l.transform(q2d.loc[te, leaf_cols]).mean(axis=1), sc_g.transform(q2d.loc[te, gap_cols]).mean(axis=1)])
        X_tr_i = np.column_stack([np.ones(tr.sum()), X_tr])
        X_te_i = np.column_stack([np.ones(te.sum()), X_te])
        pred_lofo_anom[te] = X_te_i @ np.linalg.lstsq(X_tr_i, y_anom[tr], rcond=None)[0]

    r2_lobo_noleak_anom = float(r2_score(y_anom, pred_lobo_anom))
    r2_lobo_noleak_raw = float(r2_score(y_raw, pred_lobo_raw))
    r2_lofo_noleak_anom = float(r2_score(y_anom, pred_lofo_anom))
    print(f"\nZero-leakage (training-standardized, no target z_blk) 2D plane (92-142 DAS):")
    print(f"  LOBO R^2 (on yield_blk_anom) = {r2_lobo_noleak_anom:.3f}, LOBO R^2 (on raw yield) = {r2_lobo_noleak_raw:.3f}")
    print(f"  LOFO R^2 (on yield_blk_anom) = {r2_lofo_noleak_anom:.3f}")

    # Save summary
    df_q["r_I_Leaf_I_Gap"] = r_lg
    df_q["p_I_Leaf_I_Gap"] = p_lg
    df_q["zero_leak_LOBO_R2_anom"] = r2_lobo_noleak_anom
    df_q["zero_leak_LOBO_R2_raw"] = r2_lobo_noleak_raw
    df_q["zero_leak_LOFO_R2_anom"] = r2_lofo_noleak_anom
    df_q.to_csv(Q1_DIR / "post_review_2d_plane_audit.csv", index=False)


def task6_fdr_and_exact_whole_block_perm() -> None:
    print("\n=== BH-FDR Partial Correlations & Exact 2^9=512 Deduplicated Whole-Block Sign-Flip Permutations ===")
    pc = pd.read_csv(Q1_DIR / "exact_df_partial_correlations_audit.csv")
    pvals = pc["within_N_block_p"].to_numpy(float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)
    qvals = np.empty(m)
    qvals[order] = bh
    pc["within_N_block_q_BH"] = qvals
    pc.to_csv(Q1_DIR / "exact_df_partial_correlations_audit.csv", index=False)
    print("BH-FDR adjusted P-values for Fig. 2e partial correlations (df=36/38):")
    print(pc[["feature", "within_N_block_partial_r", "within_N_block_p", "within_N_block_q_BH"]].round(4).to_string(index=False))

    # Exact 2^9 = 512 whole-block sign-flip permutation across the 9 UNIQUE physical target blocks!
    ms = pd.read_csv(Q1_DIR / "multiscale_predictions.csv").query("prefix == '3.3 mm'")
    # For each unique physical block (target, block), average paired MSE difference (N_only - N_Leaf_T) across its incoming transfers
    # Separately compute for (a) BlockCV (9 physical blocks), (b) All 6 transfers aggregated to 9 unique physical target blocks!
    perm_rows = []
    for mode_lbl, eval_filter in [("Within_field_BlockCV_9blocks", ["BlockCV"]), ("Cross_env_transfers_9_unique_target_blocks", ["adjacent_field", "cross_year"])]:
        dn = ms[(ms.specification == "N") & (ms.evaluation.isin(eval_filter))]
        dl = ms[(ms.specification == "N_Leaf_T") & (ms.evaluation.isin(eval_filter))]
        mg = dn.merge(dl, on=["source", "target", "evaluation", "plot_id", "block", "N", "observed"], suffixes=("_n", "_l"))
        mg["d_sq_err"] = (mg["predicted_n"] - mg["observed"]) ** 2 - (mg["predicted_l"] - mg["observed"]) ** 2
        # Deduplicate to 9 unique physical target blocks!
        blk_d = mg.groupby(["target", "block"])["d_sq_err"].mean().to_numpy(float)
        obs_mean = float(np.mean(blk_d))
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(blk_d))))
        perm_means = (signs @ blk_d) / len(blk_d)
        p_exact = float(np.mean(np.abs(perm_means) >= abs(obs_mean) - 1e-12))
        perm_rows.append(
            {
                "mode": mode_lbl,
                "n_unique_physical_blocks": len(blk_d),
                "n_positive_blocks": int((blk_d > 0).sum()),
                "mean_paired_dMSE": obs_mean,
                "p_exact_whole_block_signflip": p_exact,
            }
        )
    df_perm = pd.DataFrame(perm_rows)
    df_perm.to_csv(Q1_DIR / "post_review_permutation_fdr_audit.csv", index=False)
    print("\nExact 2^9 = 512 deduplicated whole-block sign-flip permutation tests (N+Leaf+T vs N-only):")
    print(df_perm.to_string(index=False))


def task7_photometric_stress_test() -> None:
    print("\n=== Bare-Soil Audit & Offline Photometric Stress Test on Canopeo (All 64 Complete-Case Plots) ===")
    bs = pd.read_csv(OLD_METHOD / "brightness_reference_patches.csv")
    print("brightness_reference_patches cols:", list(bs.columns))

    plot64 = pd.read_csv(Q1_DIR / "plot64_millimeter_features.csv")
    slots = pd.read_csv(ROOT / "01_Data/processed/model_ready/cotton_rgb_formal_714_slots.csv")
    slots["domain"] = slots["year"].astype(str) + "_" + slots["field"].astype(str)
    gdf = gpd.read_file(GPKG, layer="inward_buffer_plot")

    # Mid-boll-filling anchor flights: 2023_F1 (130 DAS), 2024_F1 (128 DAS), 2024_F2 (128 DAS)
    anchor_das = {"2023_F1": 130, "2024_F1": 128, "2024_F2": 128}
    factors = [
        ("exp_0.70", 0.70, 0.70, 0.70),
        ("exp_0.85", 0.85, 0.85, 0.85),
        ("baseline_1.00", 1.00, 1.00, 1.00),
        ("exp_1.15", 1.15, 1.15, 1.15),
        ("exp_1.30", 1.30, 1.30, 1.30),
        ("overcast_blue_shift", 0.90, 0.95, 1.12),
    ]

    stress_rows = []
    for dom, das in anchor_das.items():
        pids = set(plot64.loc[plot64.domain == dom, "plot_id"].astype(int))
        sub_slots = slots[
            (slots.domain == dom)
            & (slots.DAS == das)
            & (slots.slot_status == "AVAILABLE")
            & (slots.plot_id.astype(int).isin(pids))
        ].sort_values("plot_id")
        assert len(sub_slots) == len(pids), f"Expected {len(pids)} plots for {dom} DAS {das}, got {len(sub_slots)}"
        rpath_raw = sub_slots["raster_path"].iloc[0]
        rpath = Path(rpath_raw) if Path(rpath_raw).is_absolute() else (ROOT / rpath_raw)
        if not rpath.exists():
            # Fallback to workspace-relative raw orthomosaics path
            yr_str = dom[:4]
            rpath = ROOT / "01_Data/raw/remote_sensing/orthomosaics" / yr_str / Path(rpath_raw).name
        assert rpath.exists(), f"Missing raw orthomosaic: {rpath}"

        with rasterio.open(rpath) as src:
            gg = gdf.to_crs(src.crs)
            for _, srow in sub_slots.iterrows():
                pid = int(srow.plot_id)
                yr = int(srow.year)
                fld = str(srow.field)
                geom = gg[(gg.year == yr) & (gg.field == fld) & (gg.plot_id == pid)].geometry.iloc[0]
                rgb, valid = crop(src, geom)
                R0 = rgb[:, :, 0].astype(np.float32)
                G0 = rgb[:, :, 1].astype(np.float32)
                B0 = rgb[:, :, 2].astype(np.float32)
                n_val = int(valid.sum())
                m0 = valid & (R0 < 0.95 * G0) & (B0 < 0.95 * G0) & ((2.0 * G0 - R0 - B0) > 20.0)
                f0 = float(m0.sum() / max(1, n_val))
                ng0 = float(np.mean((G0[m0] - R0[m0]) / np.maximum(G0[m0] + R0[m0], 1e-6))) if m0.any() else np.nan
                for tag, kr, kg, kb in factors:
                    R = np.clip(R0 * kr, 0.0, 255.0)
                    G = np.clip(G0 * kg, 0.0, 255.0)
                    B = np.clip(B0 * kb, 0.0, 255.0)
                    m = valid & (R < 0.95 * G) & (B < 0.95 * G) & ((2.0 * G - R - B) > 20.0)
                    inter = float((m & m0).sum())
                    union = float((m | m0).sum())
                    iou = inter / max(1.0, union)
                    f = float(m.sum() / max(1, n_val))
                    ng = float(np.mean((G[m] - R[m]) / np.maximum(G[m] + R[m], 1e-6))) if m.any() else np.nan
                    d_ng = ng - ng0
                    stress_rows.append(
                        {
                            "domain": dom,
                            "DAS": das,
                            "plot_id": pid,
                            "N_kg_ha": float(srow.N_kg_ha),
                            "file": rpath.name,
                            "scenario": tag,
                            "iou_vs_base": iou,
                            "df_cover_pp": (f - f0) * 100.0,
                            "d_ngrdi_leaf": d_ng,
                            "abs_d_ngrdi_leaf": abs(d_ng),
                        }
                    )

    dfs = pd.DataFrame(stress_rows)
    assert len(dfs) == 64 * len(factors), len(dfs)
    dfs.to_csv(Q1_DIR / "post_review_photometric_stress_test.csv", index=False)
    print(f"Offline Photometric Stress Test on Canopeo (2G-R-B > 20) across n = {dfs[['domain', 'plot_id']].drop_duplicates().shape[0]} real complete-case plots:")
    summary = dfs.groupby("scenario").agg(
        iou_mean=("iou_vs_base", "mean"),
        iou_min=("iou_vs_base", "min"),
        df_cover_mean=("df_cover_pp", "mean"),
        df_cover_max_abs=("df_cover_pp", lambda s: float(np.max(np.abs(s)))),
        d_ngrdi_mean=("d_ngrdi_leaf", "mean"),
        d_ngrdi_abs_mean=("abs_d_ngrdi_leaf", "mean"),
        d_ngrdi_min=("d_ngrdi_leaf", "min"),
        d_ngrdi_max=("d_ngrdi_leaf", "max"),
        d_ngrdi_max_abs=("abs_d_ngrdi_leaf", "max"),
    )
    print(summary.round(4).to_string())


if __name__ == "__main__":
    task1_fix_table3_and_add_unmasked_p12()
    task2_add_rgb_mixed_ct_bmorph()
    task3_quantstrat_k3_coverage_and_task5_calibration_audit()
    task4_2d_plane_audit()
    task6_fdr_and_exact_whole_block_perm()
    task7_photometric_stress_test()
