"""
Diagnostic script for the 2026-11-10 Main Figure Review (sat_q1_spatial_calibration_20261108).
Strictly diagnostic: DOES NOT re-fit or alter any frozen predictive yield models.
Computes:
  1. Item 1 & 2: Multi-scale aggregation factors, effective pixel sizes, and zero-variance / NA audit.
  2. Item 3: 21 reviewed boll masks (00_Project/annotation) connected-component upper threshold
     sensitivity (60, 90, 120, 200, 400, inf cm^2) for precision, recall, F1, IoU, and artifact rejection.
  3. Item 5: Figure 3b "RGB-only: 3.3 mm" audit (z_blk vs k=3 target yield shift; k=0 vs k=3 R^2 and RMSE).
  4. Item 9: Figure 2d quadrant audit and Leave-One-Block-Out (LOBO, 9 env-block folds) validation.
  5. Item 13b: 2024 F1 plot-level spatial residuals (N+Leaf+T+3N0 vs N+ResNet-50+3N0) and Moran's I.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import pearsonr
import geopandas as gpd
import rasterio

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "02_Code/src/data"))
sys.path.insert(0, str(ROOT / "02_Code/utilities"))
from extract_raw_rgb_information import rules, GPKG
from prepare_boll_ai_drafts import ARTIFACTS

REL = ROOT / "10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108"
Q1_DIR = REL / "analysis/q1_spatial_calibration"
MOD_B = Q1_DIR
ANN_DIR = ROOT / "00_Project/annotation"


def run_item1_and_item2_audit():
    print("=== ITEM 1 & 2: Multi-scale GSD Aggregation & Zero-Variance Audit ===")
    df_gsd = pd.read_csv(MOD_B / "multiscale_gsd_slot_features.csv")
    df_deg = pd.read_csv(MOD_B / "multiscale_physical_degradation.csv")
    plot64_nat = pd.read_csv(MOD_B / "plot64_millimeter_features.csv")

    # Summarize native pixel size by domain and coarsening factors
    gsd_levels = [0.0033, 0.01, 0.05, 0.20, 1.00, 3.00]
    gsd_labels = {
        0.0033: "3.3 mm",
        0.01: "1 cm (sim.)",
        0.05: "5 cm (sim.)",
        0.20: "20 cm (sim.)",
        1.00: "1 m (sim.)",
        3.00: "3 m (sim.)",
    }
    agg_rows = []
    for dom, grp in df_gsd[df_gsd.gsd_m == 0.0033].groupby("domain"):
        px_m = float(grp.native_px_m.median())
        for g in gsd_levels:
            factor = 1 if g == 0.0033 else max(1, int(round(g / px_m)))
            eff_px_m = px_m * factor
            method = (
                "Native orthomosaic (no aggregation)"
                if factor == 1
                else (
                    f"Non-overlapping {factor}x{factor} block average (>=50% valid pixel threshold)"
                    if factor < 50
                    else f"Stride-tiled {factor}x{factor} block average (>=35% valid pixel threshold; plot-mean fallback)"
                )
            )
            agg_rows.append({
                "domain": dom,
                "target_gsd_m": g,
                "target_gsd_label": gsd_labels[g],
                "native_px_mm": round(px_m * 1000.0, 3),
                "integer_factor": factor,
                "effective_px_cm": round(eff_px_m * 100.0, 2),
                "aggregation_method": method,
            })
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(Q1_DIR / "diagnostic_item1_multiscale_aggregation.csv", index=False)

    # Check standard deviation and undefined correlations across GSDs
    ref = df_gsd[df_gsd.gsd_m == 0.0033].set_index(["domain", "plot_id", "DAS"])
    deg_clean_rows = []
    for gsd, sub in df_gsd.groupby("gsd_m"):
        sub_idx = sub.set_index(["domain", "plot_id", "DAS"]).reindex(ref.index)
        in_mask = sub_idx.index.get_level_values("DAS") < 180
        b_mask = sub_idx.index.get_level_values("DAS") == 181

        cov_ref = ref.loc[in_mask, "f_cover"].to_numpy()
        cov_cur = sub_idx.loc[in_mask, "f_cover"].to_numpy()
        gap_ref = ref.loc[in_mask, "gap_large_frac"].to_numpy()
        gap_cur = sub_idx.loc[in_mask, "gap_large_frac"].to_numpy()
        prof_ref = ref.loc[in_mask, "profile_sd_across"].to_numpy()
        prof_cur = sub_idx.loc[in_mask, "profile_sd_across"].to_numpy()
        boll_ref = ref.loc[b_mask, "white_b140_s30_morph_frac"].to_numpy()
        boll_cur = sub_idx.loc[b_mask, "white_b140_s30_morph_frac"].to_numpy()
        braw_cur = sub_idx.loc[b_mask, "white_b140_s30_raw"].to_numpy()

        deg_clean_rows.append({
            "gsd_m": float(gsd),
            "gsd_label": gsd_labels[float(gsd)],
            "mean_canopy_cover_pct": float(100 * np.mean(cov_cur)),
            "sd_canopy_cover_pct": float(100 * np.std(cov_cur)),
            "cover_r_vs_native": float(pearsonr(cov_cur, cov_ref).statistic) if np.std(cov_cur) > 1e-9 else np.nan,
            "mean_large_gap_pct": float(100 * np.mean(gap_cur)),
            "sd_large_gap_pct": float(100 * np.std(gap_cur)),
            "gap_r_vs_native": float(pearsonr(gap_cur, gap_ref).statistic) if np.std(gap_cur) > 1e-9 else np.nan,
            "mean_row_profile_sd_pct": float(100 * np.mean(prof_cur)),
            "sd_row_profile_sd_pct": float(100 * np.std(prof_cur)),
            "row_profile_r_vs_native": float(pearsonr(prof_cur, prof_ref).statistic) if np.std(prof_cur) > 1e-9 else np.nan,
            "mean_boll_morph_pct": float(100 * np.mean(boll_cur)),
            "sd_boll_morph_pct": float(100 * np.std(boll_cur)),
            "mean_boll_raw_pct": float(100 * np.mean(braw_cur)),
            "sd_boll_raw_pct": float(100 * np.std(braw_cur)),
            "boll_r_vs_native": float(pearsonr(boll_cur, boll_ref).statistic) if np.std(boll_cur) > 1e-9 else np.nan,
        })
    df_deg_clean = pd.DataFrame(deg_clean_rows)
    df_deg_clean.to_csv(Q1_DIR / "diagnostic_item2_degradation_na_audit.csv", index=False)
    print(df_deg_clean.to_string(index=False))
    return df_agg, df_deg_clean


def run_item3_boll_threshold_diagnostic():
    print("\n=== ITEM 3: 21-Plot Reviewed Boll Mask Upper Threshold Sensitivity ===")
    manifest = pd.read_csv(ANN_DIR / "sampling_manifest.csv")
    df_native = pd.read_csv(MOD_B / "millimeter_slot_features_699.csv")
    px_map = (
        df_native[df_native.DAS == 181]
        .set_index(["domain", "plot_id"])["native_px_m"]
        .to_dict()
    )

    upper_thresholds_cm2 = [30.0, 60.0, 90.0, 120.0, 150.0, 200.0, 300.0, 400.0, np.inf]
    lower_cm2 = 0.5
    candidate_rules = ["white_b120_s20", "white_b120_s30", "white_b140_s20", "white_b140_s30", "white_b160_s20"]

    rows_per_plot = []
    comp_areas_records = []

    for r in manifest.itertuples():
        name = r.image_id
        dom = r.domain
        pid = int(r.plot_id)
        px_m = float(px_map.get((dom, pid), 0.0033))
        px_cm2 = (px_m * 100.0) ** 2

        rgb = np.asarray(Image.open(ANN_DIR / "images" / f"{name}.png").convert("RGB"))
        valid = np.asarray(Image.open(ANN_DIR / "valid_masks" / f"{name}.png")) > 0
        lab = np.asarray(Image.open(ANN_DIR / "completed_masks" / f"{name}.png"))

        keep_known = valid & (lab != 255)
        y_true = lab[keep_known] == 1

        # Also construct explicit artifact mask from ARTIFACTS rectangles
        art_mask = np.zeros(valid.shape, dtype=bool)
        for x0, y0, x1, y1 in ARTIFACTS.get(name, []):
            art_mask[y0:y1, x0:x1] = True
        art_valid = art_mask & valid

        rule_preds = rules(rgb, valid)

        # Record connected-component sizes for primary rule white_b140_s30
        w0 = rule_preds["white_b140_s30"] & valid
        lbl0, n0 = ndi.label(w0)
        if n0 > 0:
            areas_px = np.bincount(lbl0.ravel())[1:]
            areas_cm2 = areas_px * px_cm2
            # Check how much of each component is confirmed fiber (lab==1), confirmed bg (lab==0), or artifact rect
            for cid in range(1, n0 + 1):
                cmask = lbl0 == cid
                n_px = int(areas_px[cid - 1])
                a_cm2 = float(areas_cm2[cid - 1])
                n_f1 = int(((lab == 1) & cmask).sum())
                n_bg = int(((lab == 0) & cmask).sum())
                n_art = int((art_valid & cmask).sum())
                n_unk = int(((lab == 255) & cmask).sum())
                comp_areas_records.append({
                    "image_id": name,
                    "domain": dom,
                    "plot_id": pid,
                    "N": int(r.N),
                    "comp_id": cid,
                    "area_px": n_px,
                    "area_cm2": a_cm2,
                    "fiber_px": n_f1,
                    "bg_px": n_bg,
                    "unknown_px": n_unk,
                    "in_artifact_rect_px": n_art,
                    "majority_type": (
                        "Artifact_rect" if n_art > 0.5 * n_px
                        else ("Confirmed_fiber" if n_f1 >= n_bg else "Confirmed_bg")
                    ),
                })

        for rname in candidate_rules:
            w_raw = rule_preds[rname] & valid
            lbl, n_cc = ndi.label(w_raw)
            if n_cc > 0:
                cc_sizes = np.bincount(lbl.ravel())
                cc_cm2 = cc_sizes * px_cm2
            else:
                cc_cm2 = np.zeros(1, dtype=float)

            for u_cm2 in upper_thresholds_cm2:
                if n_cc > 0:
                    valid_cc = (cc_cm2 >= lower_cm2) & (cc_cm2 <= u_cm2)
                    valid_cc[0] = False
                    pred_filt = valid_cc[lbl] & valid
                else:
                    pred_filt = np.zeros_like(valid, dtype=bool)

                p_known = pred_filt[keep_known]
                tp = int((y_true & p_known).sum())
                fp = int((~y_true & p_known).sum())
                fn = int((y_true & ~p_known).sum())
                tn = int((~y_true & ~p_known).sum())

                # Also evaluate treating manually withheld ARTIFACTS rectangles as background glare (false positives if predicted positive)
                keep_plus_art = keep_known | art_valid
                y_plus_art = (lab[keep_plus_art] == 1) & (~art_valid[keep_plus_art])
                p_plus_art = pred_filt[keep_plus_art]
                tp_a = int((y_plus_art & p_plus_art).sum())
                fp_a = int((~y_plus_art & p_plus_art).sum())
                fn_a = int((y_plus_art & ~p_plus_art).sum())

                ratio = lambda a, b: float(a / b) if b > 0 else np.nan
                rows_per_plot.append({
                    "image_id": name,
                    "domain": dom,
                    "plot_id": pid,
                    "N": int(r.N),
                    "rule": rname,
                    "upper_cm2": "inf" if np.isinf(u_cm2) else int(u_cm2),
                    "upper_cm2_num": 9999.0 if np.isinf(u_cm2) else float(u_cm2),
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "precision": ratio(tp, tp + fp),
                    "recall": ratio(tp, tp + fn),
                    "F1": ratio(2 * tp, 2 * tp + fp + fn),
                    "IoU": ratio(tp, tp + fp + fn),
                    "TP_with_art_bg": tp_a,
                    "FP_with_art_bg": fp_a,
                    "FN_with_art_bg": fn_a,
                    "precision_with_art_bg": ratio(tp_a, tp_a + fp_a),
                    "recall_with_art_bg": ratio(tp_a, tp_a + fn_a),
                    "F1_with_art_bg": ratio(2 * tp_a, 2 * tp_a + fp_a + fn_a),
                    "artifact_positive_px": int((pred_filt & art_valid).sum()),
                    "unknown_positive_px": int((pred_filt & valid & (lab == 255)).sum()),
                })

    df_plot_thresh = pd.DataFrame(rows_per_plot)
    df_comp = pd.DataFrame(comp_areas_records)
    df_plot_thresh.to_csv(Q1_DIR / "boll_cc_threshold_per_plot_21crops.csv", index=False)
    df_comp.to_csv(Q1_DIR / "boll_cc_components_21crops.csv", index=False)

    # Summarize across the 21 plots for each rule & upper threshold
    sum_rows = []
    for (rname, u_str, u_num), grp in df_plot_thresh.groupby(["rule", "upper_cm2", "upper_cm2_num"], sort=False):
        tp_s, fp_s, fn_s = int(grp.TP.sum()), int(grp.FP.sum()), int(grp.FN.sum())
        tpa_s, fpa_s, fna_s = int(grp.TP_with_art_bg.sum()), int(grp.FP_with_art_bg.sum()), int(grp.FN_with_art_bg.sum())
        sum_rows.append({
            "rule": rname,
            "upper_cm2": str(u_str),
            "upper_cm2_num": float(u_num),
            "macro_precision": float(grp.precision.mean()),
            "macro_recall": float(grp.recall.mean()),
            "macro_F1": float(grp.F1.mean()),
            "macro_IoU": float(grp.IoU.mean()),
            "pooled_precision": float(tp_s / (tp_s + fp_s)) if (tp_s + fp_s) else np.nan,
            "pooled_recall": float(tp_s / (tp_s + fn_s)) if (tp_s + fn_s) else np.nan,
            "pooled_F1": float(2 * tp_s / (2 * tp_s + fp_s + fn_s)) if (2 * tp_s + fp_s + fn_s) else np.nan,
            "macro_precision_art_bg": float(grp.precision_with_art_bg.mean()),
            "macro_recall_art_bg": float(grp.recall_with_art_bg.mean()),
            "macro_F1_art_bg": float(grp.F1_with_art_bg.mean()),
            "pooled_precision_art_bg": float(tpa_s / (tpa_s + fpa_s)) if (tpa_s + fpa_s) else np.nan,
            "pooled_recall_art_bg": float(tpa_s / (tpa_s + fna_s)) if (tpa_s + fna_s) else np.nan,
            "pooled_F1_art_bg": float(2 * tpa_s / (2 * tpa_s + fpa_s + fna_s)) if (2 * tpa_s + fpa_s + fna_s) else np.nan,
            "total_TP": tp_s,
            "total_FP": fp_s,
            "total_FN": fn_s,
            "total_artifact_positive_px": int(grp.artifact_positive_px.sum()),
            "total_unknown_positive_px": int(grp.unknown_positive_px.sum()),
        })
    df_sum = pd.DataFrame(sum_rows).sort_values(["rule", "upper_cm2_num"])
    # Also compute 64-plot full-ROI B_morph yield correlation sensitivity across upper thresholds
    plot64_nat = pd.read_csv(MOD_B / "plot64_millimeter_features.csv")
    slots_df = pd.read_csv(ROOT / "01_Data/processed/model_ready/cotton_rgb_formal_714_slots.csv")
    direct_181 = slots_df[(slots_df.slot_status == "AVAILABLE") & (slots_df.DAS == 181)].copy()
    gdf_in = gpd.read_file(GPKG, layer="inward_buffer_plot")
    common_keys = set(zip(plot64_nat.domain, plot64_nat.plot_id.astype(int)))
    from extract_raw_rgb_information import crop as crop_roi

    roi_thresh_rows = []
    for rpath, sub in direct_181.groupby("raster_path"):
        yr, fld = int(sub.year.iloc[0]), str(sub.field.iloc[0])
        dom = f"{yr}_{fld}"
        with rasterio.open(rpath) as src:
            gg = gdf_in.to_crs(src.crs)
            px_m = (
                float(np.sqrt(abs(src.res[0] * 85000.0 * src.res[1] * 111000.0)))
                if (src.crs and src.crs.is_geographic)
                else float(np.sqrt(abs(src.res[0] * src.res[1])))
            )
            px_cm2 = (px_m * 100.0) ** 2
            for _, row in sub.iterrows():
                pid = int(row.plot_id)
                if (dom, pid) not in common_keys:
                    continue
                geom = gg[(gg.year == yr) & (gg.field == fld) & (gg.plot_id == pid)].geometry.iloc[0]
                rgb_p, val_p = crop_roi(src, geom)
                n_val = max(int(val_p.sum()), 1)
                w_p = rules(rgb_p, val_p)["white_b140_s30"] & val_p
                lbl_p, n_cc_p = ndi.label(w_p)
                cc_cm2_p = np.bincount(lbl_p.ravel()) * px_cm2 if n_cc_p > 0 else np.zeros(1, dtype=float)
                rec = {"domain": dom, "plot_id": pid}
                for u_cm2 in upper_thresholds_cm2:
                    key = "inf" if np.isinf(u_cm2) else str(int(u_cm2))
                    if n_cc_p > 0:
                        keep_c = (cc_cm2_p >= lower_cm2) & (cc_cm2_p <= u_cm2)
                        keep_c[0] = False
                        rec[f"morph_{key}"] = float(cc_cm2_p[keep_c].sum() / (n_val * px_cm2))
                    else:
                        rec[f"morph_{key}"] = 0.0
                roi_thresh_rows.append(rec)

    df_roi64 = plot64_nat[["domain", "plot_id", "block_design", "N_kg_ha", "yield_kg_ha"]].merge(
        pd.DataFrame(roi_thresh_rows), on=["domain", "plot_id"]
    )
    z_pool = pd.get_dummies(
        df_roi64[["domain"]].assign(
            dom_N=df_roi64.domain + "_" + df_roi64.N_kg_ha.astype(int).astype(str),
            dom_B=df_roi64.domain + "_B" + df_roi64.block_design.astype(str),
        ),
        drop_first=True,
        dtype=float,
    )
    X_pool = np.column_stack([np.ones(len(df_roi64)), z_pool.to_numpy()])
    y64 = df_roi64.yield_kg_ha.to_numpy(float)
    ry_nb = y64 - X_pool @ np.linalg.lstsq(X_pool, y64, rcond=None)[0]
    z_env = pd.get_dummies(df_roi64[["domain"]], drop_first=True, dtype=float)
    X_env = np.column_stack([np.ones(len(df_roi64)), z_env.to_numpy()])
    ry_env = y64 - X_env @ np.linalg.lstsq(X_env, y64, rcond=None)[0]

    r_env_map, r_nb_map, p_nb_map, mean_frac_map = {}, {}, {}, {}
    for u_cm2 in upper_thresholds_cm2:
        key = "inf" if np.isinf(u_cm2) else str(int(u_cm2))
        v = df_roi64[f"morph_{key}"].to_numpy(float)
        rv_env = v - X_env @ np.linalg.lstsq(X_env, v, rcond=None)[0]
        rv_nb = v - X_pool @ np.linalg.lstsq(X_pool, v, rcond=None)[0]
        r_env_map[key] = round(float(pearsonr(rv_env, ry_env).statistic), 4)
        st, pv = pearsonr(rv_nb, ry_nb)
        r_nb_map[key] = round(float(st), 4)
        p_nb_map[key] = round(float(pv), 4)
        mean_frac_map[key] = round(float(100.0 * v.mean()), 2)

    df_sum["plot64_mean_area_pct"] = df_sum["upper_cm2"].map(mean_frac_map)
    df_sum["plot64_within_env_r"] = df_sum["upper_cm2"].map(r_env_map)
    df_sum["plot64_within_N_block_r"] = df_sum["upper_cm2"].map(r_nb_map)
    df_sum["plot64_within_N_block_p"] = df_sum["upper_cm2"].map(p_nb_map)
    df_sum.to_csv(Q1_DIR / "boll_cc_threshold_sensitivity_21crops.csv", index=False)
    print(df_sum[df_sum.rule == "white_b140_s30"].to_string(index=False))
    return df_sum, df_comp


def run_item5_rgb_only_audit():
    print("\n=== ITEM 5: Figure 3b 'RGB-only: 3.3 mm' Zero-Label (k=0) vs 3-Plot Calibrated (k=3) Audit ===")
    pred_df = pd.read_csv(Q1_DIR / "direction_b_prior_free_predictions.csv")
    sum_df = pd.read_csv(Q1_DIR / "direction_b_prior_free_summary.csv")

    rows = []
    for (spec, ev), grp in pred_df.groupby(["specification", "evaluation"]):
        # Full target field (all plots, k=0)
        y_all = grp.observed.to_numpy(float)
        p0_all = grp.pred_k0.to_numpy(float)
        rmse_k0_all = float(np.sqrt(np.mean((p0_all - y_all) ** 2)))
        r2_k0_all = float(1.0 - np.sum((y_all - p0_all) ** 2) / np.sum((y_all - y_all.mean()) ** 2))
        r_pear_all = float(pearsonr(y_all, p0_all).statistic)

        # Holdout target plots (is_cal_k3 == 0, n-3 per task)
        sub = grp[grp.is_cal_k3 == 0]
        yh = sub.observed.to_numpy(float)
        p0_h = sub.pred_k0.to_numpy(float)
        p3_h = sub.pred_k3.to_numpy(float)
        rmse_k0_h = float(np.sqrt(np.mean((p0_h - yh) ** 2)))
        r2_k0_h = float(1.0 - np.sum((yh - p0_h) ** 2) / np.sum((yh - yh.mean()) ** 2))
        rmse_k3_h = float(np.sqrt(np.mean((p3_h - yh) ** 2)))
        r2_k3_h = float(1.0 - np.sum((yh - p3_h) ** 2) / np.sum((yh - yh.mean()) ** 2))
        r_pear_h = float(pearsonr(yh, p3_h).statistic)

        rows.append({
            "specification": spec,
            "evaluation": ev,
            "n_total": len(grp),
            "n_holdout": len(sub),
            "RMSE_k0_all": round(rmse_k0_all, 1),
            "R2_k0_all": round(r2_k0_all, 3),
            "pearson_r2_all": round(r_pear_all ** 2, 3),
            "RMSE_k0_holdout": round(rmse_k0_h, 1),
            "R2_k0_holdout": round(r2_k0_h, 3),
            "RMSE_k3_holdout": round(rmse_k3_h, 1),
            "R2_k3_holdout": round(r2_k3_h, 3),
            "pearson_r2_holdout": round(r_pear_h ** 2, 3),
        })
    df_item5 = pd.DataFrame(rows)
    df_item5.to_csv(Q1_DIR / "diagnostic_item5_rgb_prior_free_audit.csv", index=False)
    print(df_item5.to_string(index=False))
    return df_item5


def run_item9_quadrant_lobo_audit():
    print("\n=== ITEM 9: Figure 2d Quadrant & Leave-One-Block-Out (LOBO) Validation ===")
    q_df = pd.read_csv(Q1_DIR / "direction_b_2d_attribution_quadrants.csv")
    x = q_df["I_Leaf_Phys"].to_numpy(float)
    y = q_df["I_Gap_Struct"].to_numpy(float)
    z = q_df["yield_blk_anom"].to_numpy(float) / 1000.0
    z_nb = q_df["yield_within_NB"].to_numpy(float) / 1000.0

    # In-sample plane fit
    A = np.c_[np.ones_like(x), x, y]
    coef_in, *_ = np.linalg.lstsq(A, z, rcond=None)
    z_hat_in = A @ coef_in
    r_in, p_in = pearsonr(z, z_hat_in)
    r2_in = float(1.0 - np.sum((z - z_hat_in) ** 2) / np.sum((z - z.mean()) ** 2))
    rmse_in = float(np.sqrt(np.mean((z - z_hat_in) ** 2)))

    # Leave-One-Environment-Block-Out (9 folds: 3 domains x 3 blocks)
    q_df["env_block"] = q_df["domain"].astype(str) + "_B" + q_df["block_design"].astype(str)
    z_hat_lobo = np.empty_like(z)
    for eb in q_df["env_block"].unique():
        tr = (q_df["env_block"] != eb).to_numpy()
        te = ~tr
        c_tr, *_ = np.linalg.lstsq(A[tr], z[tr], rcond=None)
        z_hat_lobo[te] = A[te] @ c_tr

    r_lobo, p_lobo = pearsonr(z, z_hat_lobo)
    r2_lobo = float(1.0 - np.sum((z - z_hat_lobo) ** 2) / np.sum((z - z.mean()) ** 2))
    rmse_lobo = float(np.sqrt(np.mean((z - z_hat_lobo) ** 2)))

    # Also test Leave-One-Domain-Out (3 folds: 2023_F1, 2024_F1, 2024_F2)
    z_hat_lodo = np.empty_like(z)
    for dom in q_df["domain"].unique():
        tr = (q_df["domain"] != dom).to_numpy()
        te = ~tr
        c_tr, *_ = np.linalg.lstsq(A[tr], z[tr], rcond=None)
        z_hat_lodo[te] = A[te] @ c_tr
    r_lodo, p_lodo = pearsonr(z, z_hat_lodo)
    r2_lodo = float(1.0 - np.sum((z - z_hat_lodo) ** 2) / np.sum((z - z.mean()) ** 2))

    # Summarize quadrant stats
    q_summary = []
    for q_name, grp in q_df.groupby("quadrant"):
        q_summary.append({
            "quadrant": q_name,
            "n_plots": len(grp),
            "mean_N_kg_ha": round(float(grp.N_kg_ha.mean()), 1),
            "n_N0_plots": int((grp.N_kg_ha == 0).sum()),
            "mean_I_Leaf_Phys": round(float(grp.I_Leaf_Phys.mean()), 3),
            "mean_I_Gap_Struct": round(float(grp.I_Gap_Struct.mean()), 3),
            "mean_obs_blk_anom_Mg_ha": round(float(grp.yield_blk_anom.mean() / 1000.0), 3),
            "sd_obs_blk_anom_Mg_ha": round(float(grp.yield_blk_anom.std() / 1000.0), 3),
            "mean_obs_within_NB_Mg_ha": round(float(grp.yield_within_NB.mean() / 1000.0), 3),
            "mean_in_sample_pred_Mg_ha": round(float(z_hat_in[grp.index].mean()), 3),
            "mean_lobo_pred_Mg_ha": round(float(z_hat_lobo[grp.index].mean()), 3),
            "mean_lodo_pred_Mg_ha": round(float(z_hat_lodo[grp.index].mean()), 3),
        })
    df_q_sum = pd.DataFrame(q_summary)
    df_q_sum["in_sample_R2"] = round(r2_in, 4)
    df_q_sum["in_sample_r"] = round(float(r_in), 4)
    df_q_sum["lobo_R2"] = round(r2_lobo, 4)
    df_q_sum["lobo_r"] = round(float(r_lobo), 4)
    df_q_sum["lobo_p"] = float(p_lobo)
    df_q_sum["lodo_R2"] = round(r2_lodo, 4)
    df_q_sum["lodo_r"] = round(float(r_lodo), 4)
    df_q_sum["in_sample_RMSE_Mg_ha"] = round(rmse_in, 4)
    df_q_sum["lobo_RMSE_Mg_ha"] = round(rmse_lobo, 4)
    df_q_sum.to_csv(Q1_DIR / "diagnostic_item9_quadrant_lobo.csv", index=False)
    print(df_q_sum.to_string(index=False))
    return df_q_sum


def compute_morans_i(values: np.ndarray, W: np.ndarray, n_perm: int = 9999, seed: int = 20261110):
    """Compute global Moran's I and two-sided permutation p-value given weight matrix W."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    z = v - v.mean()
    denom = float(np.sum(z ** 2))
    w_sum = float(np.sum(W))
    if denom < 1e-12 or w_sum < 1e-12:
        return np.nan, np.nan, -1.0 / (n - 1)
    obs_i = float((n / w_sum) * (z @ W @ z) / denom)
    ei = -1.0 / (n - 1)
    rng = np.random.default_rng(seed)
    perm_i = np.empty(n_perm, dtype=float)
    for k in range(n_perm):
        zp = rng.permutation(z)
        perm_i[k] = (n / w_sum) * (zp @ W @ zp) / denom
    p_val = float((np.sum(np.abs(perm_i - ei) >= np.abs(obs_i - ei)) + 1) / (n_perm + 1))
    return obs_i, p_val, ei


def run_item13b_spatial_residuals_2024F1():
    print("\n=== ITEM 13b: 2024 F1 Plot-Level Spatial Residuals & Moran's I ===")
    gdf = gpd.read_file(GPKG, layer="inward_buffer_plot")
    gdf_24f1 = gdf[(gdf.year == 2024) & (gdf.field == "F1")].copy().sort_values("plot_id").reset_index(drop=True)
    # Project to UTM Zone 45N (EPSG:32645) for exact metric centroids and distances
    gdf_utm = gdf_24f1.to_crs(epsg=32645)
    coords = np.column_stack([gdf_utm.geometry.centroid.x, gdf_utm.geometry.centroid.y])
    pids = gdf_24f1.plot_id.astype(int).to_numpy()

    # Load calibrated (+3N0) and uncalibrated (k=0) predictions from rich_fig5_paradox_predictions.csv
    n0_preds = pd.read_csv(Q1_DIR / "rich_fig5_paradox_predictions.csv")
    tgt_24f1 = n0_preds[n0_preds.target == "2024_F1"].copy()
    tgt_24f1["is_N0_check"] = tgt_24f1["N"] == 0.0

    # Build spatial weight matrices for any subset of plot_ids
    def build_weights(sub_pids):
        idx = [np.flatnonzero(pids == p)[0] for p in sub_pids]
        xy = coords[idx]
        dist = np.sqrt(np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2))
        # Inverse distance weights (row-standardized)
        with np.errstate(divide="ignore"):
            w_inv = np.where(dist > 0, 1.0 / dist, 0.0)
        rs = w_inv.sum(axis=1, keepdims=True)
        w_inv_std = np.where(rs > 0, w_inv / rs, 0.0)
        # Queen-like nearest neighbor contiguity within 12 meters (adjacent plots along or across blocks)
        w_nn = ((dist > 0) & (dist <= 12.0)).astype(float)
        rs_nn = w_nn.sum(axis=1, keepdims=True)
        w_nn_std = np.where(rs_nn > 0, w_nn / rs_nn, 0.0)
        return w_inv_std, w_nn_std

    moran_rows = []
    plot_res_rows = []

    for src in ["2024_F2", "2023_F1"]:
        for spec, label in [("N_Leaf_T", "N+Leaf+T"), ("N_ResNet50", "N+ResNet-50"), ("N", "N-only")]:
            sub = tgt_24f1[(tgt_24f1.source == src) & (tgt_24f1.specification == spec)].sort_values("plot_id")
            if sub.empty:
                continue
            # All 19 plots and 16 non-N0 evaluation plots
            for subset_name, mask_cond in [
                ("Non-N0 holdout (n=16)", ~sub.is_N0_check),
                ("All plots (n=19)", np.ones(len(sub), dtype=bool)),
            ]:
                s_use = sub[mask_cond].sort_values("plot_id")
                w_inv, w_nn = build_weights(s_use.plot_id.to_numpy(int))
                for cal_mode, col_pred in [("k=0 (uncalibrated)", "pred_zero"), ("+3N0 calibrated", "pred_plus_3N0")]:
                    res = (s_use[col_pred] - s_use["observed"]).to_numpy(float)
                    rmse = float(np.sqrt(np.mean(res ** 2)))
                    bias = float(np.mean(res))
                    mi_inv, p_inv, ei = compute_morans_i(res, w_inv)
                    mi_nn, p_nn, _ = compute_morans_i(res, w_nn)
                    moran_rows.append({
                        "source": src,
                        "target": "2024_F1",
                        "model": label,
                        "calibration": cal_mode,
                        "subset": subset_name,
                        "n": len(s_use),
                        "RMSE_kg_ha": round(rmse, 1),
                        "mean_bias_kg_ha": round(bias, 1),
                        "morans_I_inv_dist": round(mi_inv, 4),
                        "p_value_inv_dist": round(p_inv, 4),
                        "morans_I_queen_12m": round(mi_nn, 4),
                        "p_value_queen_12m": round(p_nn, 4),
                        "expected_I": round(ei, 4),
                    })
            # Save plot-level residuals for all 19 plots
            for r in sub.itertuples():
                plot_res_rows.append({
                    "source": src,
                    "target": "2024_F1",
                    "model": label,
                    "plot_id": int(r.plot_id),
                    "block": int(r.block),
                    "N": float(r.N),
                    "is_N0_check": bool(r.is_N0_check),
                    "observed": float(r.observed),
                    "pred_uncal": float(r.pred_zero),
                    "pred_n0_cal": float(r.pred_plus_3N0),
                    "resid_uncal": float(r.pred_zero - r.observed),
                    "resid_n0_cal": float(r.pred_plus_3N0 - r.observed),
                })

    df_moran = pd.DataFrame(moran_rows)
    df_plot_res = pd.DataFrame(plot_res_rows)
    df_moran.to_csv(Q1_DIR / "spatial_residuals_2024F1_morans_i.csv", index=False)
    df_plot_res.to_csv(Q1_DIR / "spatial_residuals_2024F1_plot_level.csv", index=False)
    print(df_moran.to_string(index=False))
    return df_moran, df_plot_res


if __name__ == "__main__":
    run_item1_and_item2_audit()
    run_item3_boll_threshold_diagnostic()
    run_item5_rgb_only_audit()
    run_item9_quadrant_lobo_audit()
    run_item13b_spatial_residuals_2024F1()
