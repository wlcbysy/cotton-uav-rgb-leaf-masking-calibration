"""Module C: Few-shot target calibration, deep-feature calibration paradox, and active sampling curves.

Strictly read-only on 01_Data/raw and sat_full_figure_label_legend_sync_20260901.
Reads frozen baseline/deep predictions + Module B 3.3 mm predictions and writes to:
  10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/analysis/q1_spatial_calibration/
"""
from __future__ import annotations

from itertools import combinations, product
from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[3]
FORMAL_NEW = ROOT / '10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108'
FORMAL_OLD = FORMAL_NEW / 'analysis/frozen_baseline'
OUT = FORMAL_NEW / 'analysis/q1_spatial_calibration'

DOMAINS = ['2023_F1', '2024_F1', '2024_F2']
TRANSFER_TASKS = [
    ('2024_F1', '2024_F2', 'adjacent_field'),
    ('2024_F2', '2024_F1', 'adjacent_field'),
    ('2023_F1', '2024_F1', 'cross_year'),
    ('2023_F1', '2024_F2', 'cross_year'),
    ('2024_F1', '2023_F1', 'cross_year'),
    ('2024_F2', '2023_F1', 'cross_year'),
]


def load_unified_predictions() -> pd.DataFrame:
    """Assemble all 9-task plot-level predictions on the 64-plot common support without refitting."""
    sup = pd.read_csv(FORMAL_OLD / 'analysis/raw_rgb_mining/support_predictions.csv').query(
        "variant=='common_DAS' and algorithm=='PartialRidge' and evaluation in ['BlockCV','adjacent_field','cross_year']"
        " and specification in ['N','N_Cearly','N_C','N_C_T']"
    )[['source', 'target', 'evaluation', 'specification', 'plot_id', 'block', 'N', 'observed', 'predicted']]

    rgb52 = pd.read_csv(FORMAL_OLD / 'analysis/raw_rgb_mining/methodology_current/matched_RGB_predictions.csv').query(
        "specification=='N_RGB52_common4'"
    )[['source', 'target', 'evaluation', 'specification', 'plot_id', 'block', 'N', 'observed', 'predicted']]

    resnet = pd.read_csv(FORMAL_OLD / 'analysis/method_strengthening/resnet50_partial_ridge_predictions.csv').query(
        "specification=='N_ResNet50_PCA52_common4'"
    )[['source', 'target', 'evaluation', 'specification', 'plot_id', 'block', 'N', 'observed', 'predicted']]

    mod_b = pd.read_csv(OUT / 'multiscale_predictions.csv').query(
        "prefix=='3.3 mm' and specification in ['N_Leaf_T','N_Gap_early','N_Leaf','N_Micro_early','N_Micro_T','N_C_T_Braw','N_C_T_Bmorph']"
    )[['source', 'target', 'evaluation', 'specification', 'plot_id', 'block', 'N', 'observed', 'predicted']]

    df = pd.concat([sup, rgb52, resnet, mod_b], ignore_index=True)
    return df


def error_components(err: np.ndarray, n_arr: np.ndarray) -> dict[str, float]:
    """Exact orthogonal MSE decomposition: MSE = Bias^2 + Between-N MSE + Within-N MSE."""
    total_mse = float(np.mean(err ** 2))
    bias = float(np.mean(err))
    e_cent = err - bias
    c_mse = float(np.mean(e_cent ** 2))
    e_wn = err.copy()
    for lv in np.unique(n_arr):
        m = n_arr == lv
        e_wn[m] -= e_wn[m].mean()
    wn_mse = float(np.mean(e_wn ** 2))
    bn_mse = max(0.0, c_mse - wn_mse)
    return {
        'Total': math.sqrt(total_mse),
        'Bias': bias,
        'Centered': math.sqrt(c_mse),
        'Between-N': math.sqrt(bn_mse),
        'Within-N': math.sqrt(wn_mse),
    }


def block_resample_components(err: np.ndarray, n_arr: np.ndarray, blk_arr: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """Fixed-center 27 ordered target-block conditional 95% intervals matching REFRAME_STEP1_INTERVALS protocol."""
    base = error_components(err, n_arr)
    bias_fixed = float(np.mean(err))
    e_cent_fixed = err - bias_fixed
    e_wn_fixed = err.copy()
    for lv in np.unique(n_arr):
        m = n_arr == lv
        e_wn_fixed[m] -= e_wn_fixed[m].mean()

    uniq_b = sorted(set(blk_arr))
    draws = {k: [] for k in base}
    for bs in product(uniq_b, repeat=3):
        ix = np.concatenate([np.flatnonzero(blk_arr == b) for b in bs])
        tot = math.sqrt(float(np.mean(err[ix] ** 2)))
        bi = float(np.mean(err[ix]))
        ce = math.sqrt(float(np.mean(e_cent_fixed[ix] ** 2)))
        wn = math.sqrt(float(np.mean(e_wn_fixed[ix] ** 2)))
        bn = math.sqrt(max(0.0, ce * ce - wn * wn))
        draws['Total'].append(tot)
        draws['Bias'].append(bi)
        draws['Centered'].append(ce)
        draws['Between-N'].append(bn)
        draws['Within-N'].append(wn)

    return {
        k: (base[k], float(np.quantile(draws[k], 0.025)), float(np.quantile(draws[k], 0.975)))
        for k in base
    }


def run_hybrid_transfer_decomposition(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Experiment C1: Zero-label vs +3 N0 vs +3 Farmer-N225 vs Oracle across all 6 transfer directions."""
    core_specs = [
        'N', 'N_Cearly', 'N_Gap_early', 'N_C_T', 'N_Leaf_T', 'N_Leaf',
        'N_Micro_early', 'N_Micro_T', 'N_C_T_Braw', 'N_C_T_Bmorph',
        'N_RGB52_common4', 'N_ResNet50_PCA52_common4',
    ]
    rows = []
    contrast_rows = []

    for src, tgt, ev in TRANSFER_TASKS:
        sub = preds[(preds.source == src) & (preds.target == tgt) & (preds.evaluation == ev)]
        # Non-N0 holdout (21, 16, 18 plots)
        spec_errors_holdout = {}

        for spec in core_specs:
            d = sub[sub.specification == spec].sort_values('plot_id')
            d_n0 = d[d.N == 0].sort_values('plot_id')
            d_ho = d[d.N > 0].sort_values('plot_id')
            assert len(d_n0) == 3

            y_ho = d_ho.observed.to_numpy(float)
            p_ho = d_ho.predicted.to_numpy(float)
            n_ho = d_ho.N.to_numpy(float)
            b_ho = d_ho.block.to_numpy(int)

            # 1. Zero-label
            e_zero = p_ho - y_ho
            spec_errors_holdout[f'{spec}__zero'] = (e_zero, n_ho, b_ho)

            # 2. +3 N0 check-plot calibrated
            shift_n0 = float(np.mean(d_n0.observed.to_numpy(float) - d_n0.predicted.to_numpy(float)))
            e_n0 = (p_ho + shift_n0) - y_ho
            spec_errors_holdout[f'{spec}__plus_3N0'] = (e_n0, n_ho, b_ho)

            # 3. Single-shift Oracle on non-N0 holdout
            shift_ora = float(np.mean(y_ho - p_ho))
            e_ora = (p_ho + shift_ora) - y_ho
            spec_errors_holdout[f'{spec}__oracle'] = (e_ora, n_ho, b_ho)

            for regime, err_vec, shift_val in [
                ('zero_label', e_zero, 0.0),
                ('plus_3N0', e_n0, shift_n0),
                ('oracle', e_ora, shift_ora),
            ]:
                comp = block_resample_components(err_vec, n_ho, b_ho)
                for metric, (est, lo, hi) in comp.items():
                    rows.append({
                        'source': src, 'target': tgt, 'evaluation': ev,
                        'support': 'non_N0_held_out', 'n': len(d_ho),
                        'specification': spec, 'calibration_regime': regime,
                        'model_id': f'{spec}__{regime}',
                        'intercept_shift': shift_val,
                        'measure': metric, 'estimate': est, 'low_95': lo, 'high_95': hi,
                    })

        # Compute paired Total RMSE contrasts vs N__zero and vs N__plus_3N0
        for base_id in ('N__zero', 'N__plus_3N0'):
            e_base, n_ho, b_ho = spec_errors_holdout[base_id]
            uniq_b = sorted(set(b_ho))
            rmse_base = math.sqrt(float(np.mean(e_base ** 2)))
            for cand_id, (e_cand, _, _) in spec_errors_holdout.items():
                rmse_cand = math.sqrt(float(np.mean(e_cand ** 2)))
                gain = rmse_base - rmse_cand
                draws = []
                for bs in product(uniq_b, repeat=3):
                    ix = np.concatenate([np.flatnonzero(b_ho == b) for b in bs])
                    draws.append(math.sqrt(float(np.mean(e_base[ix] ** 2))) - math.sqrt(float(np.mean(e_cand[ix] ** 2))))
                lo, hi = np.quantile(draws, [0.025, 0.975])
                contrast_rows.append({
                    'source': src, 'target': tgt, 'evaluation': ev, 'n': len(e_base),
                    'baseline': base_id, 'candidate': cand_id,
                    'baseline_RMSE': rmse_base, 'candidate_RMSE': rmse_cand,
                    'RMSE_reduction': gain, 'low_95': float(lo), 'high_95': float(hi),
                })

    return pd.DataFrame(rows), pd.DataFrame(contrast_rows)


def select_uav_active_plots(d_tgt: pd.DataFrame, vigor_col: np.ndarray, k: int) -> list[int]:
    """Deterministic yield-blind spatial-block + quantile stratified plot selector."""
    df = d_tgt.copy()
    df['vigor'] = vigor_col
    m = len(df)
    if k <= 0:
        return []
    if k >= m:
        return df.sort_values('plot_id').plot_id.astype(int).tolist()

    # Order spatial blocks by median optical vigor: [low_block, mid_block, high_block]
    blk_order = df.groupby('block')['vigor'].median().sort_values().index.tolist()
    if k == 1:
        # Representative median-vigor plot inside the median spatial block
        mid_blk = blk_order[len(blk_order) // 2]
        sub_b = df[df.block == mid_blk].sort_values(['vigor', 'plot_id'])
        target_v = float(df['vigor'].median())
        return [int(sub_b.iloc[(sub_b['vigor'] - target_v).abs().argsort().iloc[0]].plot_id)]

    # Target global quantiles evenly spanning (0, 1), assigned round-robin across spatial blocks for spatial balance
    qs = np.linspace(1.0 / (k + 1), k / (k + 1), k)
    blk_cycle = (blk_order * ((k // len(blk_order)) + 2))[:k]
    if k == 2:
        blk_cycle = [blk_order[0], blk_order[-1]]
    chosen = []
    avail = df.copy()
    for q, b_pref in zip(qs, blk_cycle):
        target_v = float(np.quantile(df['vigor'], q))
        cand = avail[avail.block == b_pref]
        if cand.empty:
            cand = avail
        pick_row = cand.iloc[(cand['vigor'] - target_v).abs().argsort().iloc[0]]
        chosen.append(int(pick_row.plot_id))
        avail = avail[avail.plot_id != pick_row.plot_id]
    return chosen


def run_fewshot_budget_curves(preds: pd.DataFrame, plot64: pd.DataFrame) -> pd.DataFrame:
    """Experiment C2: Sample budget curves k=0..6 across field-deployable strategies (strictly out-of-sample)."""
    specs = ['N', 'N_Cearly', 'N_C_T', 'N_Leaf_T', 'N_ResNet50_PCA52_common4']
    vigor_lookup = plot64.set_index(['domain', 'plot_id'])['T_auc_mean'].to_dict()
    # Pre-harvest zero-label N_Leaf_T predicted yield rank for management+UAV stratified active sampling
    rng = np.random.default_rng(20261108)
    rows = []

    for src, tgt, ev in TRANSFER_TASKS:
        sub_task = preds[(preds.source == src) & (preds.target == tgt) & (preds.evaluation == ev)]
        d_guide = sub_task[sub_task.specification == 'N_Leaf_T'].set_index('plot_id')['predicted'].to_dict()

        for spec in specs:
            d = sub_task[sub_task.specification == spec].sort_values('plot_id').reset_index(drop=True)
            y_all = d.observed.to_numpy(float)
            p_all = d.predicted.to_numpy(float)
            n_all = d.N.to_numpy(float)
            pids = d.plot_id.to_numpy(int)
            vigor = np.array([vigor_lookup[(tgt, int(pid))] for pid in pids], dtype=float)
            pred_guide = np.array([d_guide[int(pid)] for pid in pids], dtype=float)
            m_tot = len(d)

            def eval_subset(cal_idx: list[int], eval_fertilized_only: bool = False, affine: bool = False) -> tuple[float, float]:
                cal_mask = np.zeros(m_tot, dtype=bool)
                cal_mask[cal_idx] = True
                te_mask = (~cal_mask) & (n_all > 0) if eval_fertilized_only else (~cal_mask)
                if len(cal_idx) == 0:
                    p_cal = p_all[te_mask]
                elif affine and len(cal_idx) >= 3 and np.std(p_all[cal_mask]) > 1e-6:
                    # Regularized 2-parameter recalibration shrinking slope toward 1.0 (ridge penalty lambda=2.0)
                    xc = p_all[cal_mask] - p_all[cal_mask].mean()
                    yc_resid = (y_all[cal_mask] - p_all[cal_mask])
                    delta_slope = float((xc @ (yc_resid - yc_resid.mean())) / (xc @ xc + 2.0 * np.var(p_all)))
                    slope = np.clip(1.0 + delta_slope, 0.5, 1.5)
                    intercept = float(y_all[cal_mask].mean() - slope * p_all[cal_mask].mean())
                    p_cal = intercept + slope * p_all[te_mask]
                else:
                    shift = float(np.mean(y_all[cal_mask] - p_all[cal_mask]))
                    p_cal = p_all[te_mask] + shift
                err = p_cal - y_all[te_mask]
                rmse = math.sqrt(float(np.mean(err ** 2)))
                bias = float(np.mean(err))
                return rmse, abs(bias)

            # Evaluate on TWO clean out-of-sample supports:
            # (A) 'unsampled_all': all remaining n - k target plots
            # (B) 'unsampled_fertilized': all remaining unsampled N > 0 commercial plots
            for sup_name, fert_only in [('unsampled_all', False), ('unsampled_fertilized', True)]:
                for k in range(0, 7):
                    if k == 0:
                        rm0, ab0 = eval_subset([], fert_only)
                        for strat in ('Random_all', 'Block_Rand_fert', 'N0_check', 'Farmer_N225', 'Farmer_N300', 'UAV_Active_fert', 'UAV_Active_all', 'UAV_Stratified_Pred'):
                            rows.append({
                                'source': src, 'target': tgt, 'evaluation': ev, 'support': sup_name,
                                'specification': spec, 'strategy': strat, 'k': 0,
                                'mean_RMSE': rm0, 'q10_RMSE': rm0, 'q90_RMSE': rm0, 'mean_abs_bias': ab0,
                            })
                        continue

                    # 1. Uninformed random k plots from full field (exact if C(m,k) <= 2500 else 2500 MC draws)
                    n_comb = math.comb(m_tot, k)
                    if n_comb <= 2500:
                        combs = list(combinations(range(m_tot), k))
                    else:
                        combs = [tuple(rng.choice(m_tot, size=k, replace=False)) for _ in range(2500)]
                    rand_res = [eval_subset(list(c), fert_only) for c in combs]
                    rm_vals = [x[0] for x in rand_res]
                    ab_vals = [x[1] for x in rand_res]
                    rows.append({
                        'source': src, 'target': tgt, 'evaluation': ev, 'support': sup_name,
                        'specification': spec, 'strategy': 'Random_all', 'k': k,
                        'mean_RMSE': float(np.mean(rm_vals)),
                        'q10_RMSE': float(np.quantile(rm_vals, 0.10)),
                        'q90_RMSE': float(np.quantile(rm_vals, 0.90)),
                        'mean_abs_bias': float(np.mean(ab_vals)),
                    })

                    # 2. N0 check strips (k = 1, 2, 3)
                    n0_idx = np.flatnonzero(n_all == 0).tolist()
                    if k <= len(n0_idx):
                        c_n0 = [eval_subset(list(c), fert_only) for c in combinations(n0_idx, k)]
                        rows.append({
                            'source': src, 'target': tgt, 'evaluation': ev, 'support': sup_name,
                            'specification': spec, 'strategy': 'N0_check', 'k': k,
                            'mean_RMSE': float(np.mean([x[0] for x in c_n0])),
                            'q10_RMSE': float(np.min([x[0] for x in c_n0])),
                            'q90_RMSE': float(np.max([x[0] for x in c_n0])),
                            'mean_abs_bias': float(np.mean([x[1] for x in c_n0])),
                        })

                    # 3. Commercial farmer-rate strips N225 and N300 (k = 1, 2, 3 balanced across all 6 tasks)
                    if k <= 3:
                        for rate_val, rate_label in [(225.0, 'Farmer_N225'), (300.0, 'Farmer_N300')]:
                            r_idx = np.flatnonzero(n_all == rate_val).tolist()
                            if k <= len(r_idx):
                                c_r = [eval_subset(list(c), fert_only) for c in combinations(r_idx, k)]
                                rows.append({
                                    'source': src, 'target': tgt, 'evaluation': ev, 'support': sup_name,
                                    'specification': spec, 'strategy': rate_label, 'k': k,
                                    'mean_RMSE': float(np.mean([x[0] for x in c_r])),
                                    'q10_RMSE': float(np.min([x[0] for x in c_r])),
                                    'q90_RMSE': float(np.max([x[0] for x in c_r])),
                                    'mean_abs_bias': float(np.mean([x[1] for x in c_r])),
                                })

                    # 4. UAV canopy-guided stratified active sampling:
                    # (a) UAV_Active_fert: select k plots strictly from fertilized commercial plots (no N0 needed!)
                    d_fert = d[d.N > 0].copy()
                    v_fert = vigor[n_all > 0]
                    chosen_fert_pids = select_uav_active_plots(d_fert, v_fert, k)
                    chosen_fert_idx = [int(np.flatnonzero(pids == pid)[0]) for pid in chosen_fert_pids]
                    rm_uf, ab_uf = eval_subset(chosen_fert_idx, fert_only)
                    rows.append({
                        'source': src, 'target': tgt, 'evaluation': ev, 'support': sup_name,
                        'specification': spec, 'strategy': 'UAV_Active_fert', 'k': k,
                        'mean_RMSE': rm_uf, 'q10_RMSE': rm_uf, 'q90_RMSE': rm_uf, 'mean_abs_bias': ab_uf,
                    })

                    # (b) UAV_Active_all: select k plots across all plots in field using UAV vigor quantiles
                    chosen_all_pids = select_uav_active_plots(d, vigor, k)
                    chosen_all_idx = [int(np.flatnonzero(pids == pid)[0]) for pid in chosen_all_pids]
                    rm_ua, ab_ua = eval_subset(chosen_all_idx, fert_only)
                    rows.append({
                        'source': src, 'target': tgt, 'evaluation': ev, 'support': sup_name,
                        'specification': spec, 'strategy': 'UAV_Active_all', 'k': k,
                        'mean_RMSE': rm_ua, 'q10_RMSE': rm_ua, 'q90_RMSE': rm_ua, 'mean_abs_bias': ab_ua,
                    })

                    # (c) UAV_Stratified_Pred: select k fertilized commercial plots spanning pre-harvest zero-label N+Leaf+T yield quantiles
                    p_fert_guide = pred_guide[n_all > 0]
                    chosen_sp_pids = select_uav_active_plots(d_fert, p_fert_guide, k)
                    chosen_sp_idx = [int(np.flatnonzero(pids == pid)[0]) for pid in chosen_sp_pids]
                    rm_sp, ab_sp = eval_subset(chosen_sp_idx, fert_only)
                    rows.append({
                        'source': src, 'target': tgt, 'evaluation': ev, 'support': sup_name,
                        'specification': spec, 'strategy': 'UAV_Stratified_Pred', 'k': k,
                        'mean_RMSE': rm_sp, 'q10_RMSE': rm_sp, 'q90_RMSE': rm_sp, 'mean_abs_bias': ab_sp,
                    })

                    # (d) Block_Rand_fert: spatially block-balanced random sampling on fertilized commercial plots (N > 0)
                    # Isolates spatial block balancing from UAV optical quantile selection (DA-M2).
                    blk_fert_groups = [
                        [int(np.flatnonzero(pids == int(pid))[0]) for pid in grp_b.plot_id]
                        for _, grp_b in d_fert.groupby('block')
                    ]
                    n_b = len(blk_fert_groups)
                    blk_combs = []
                    if k == 1:
                        blk_combs = [[i] for g_b in blk_fert_groups for i in g_b]
                    elif k == 2 and n_b == 3:
                        for b1, b2 in combinations(range(3), 2):
                            for i1 in blk_fert_groups[b1]:
                                for i2 in blk_fert_groups[b2]:
                                    blk_combs.append([i1, i2])
                    elif k == 3 and n_b == 3:
                        for c_tup in product(*blk_fert_groups):
                            blk_combs.append(list(c_tup))
                    elif k == 4 and n_b == 3:
                        for b_two in range(3):
                            b_ones = [b for b in range(3) if b != b_two]
                            for c2 in combinations(blk_fert_groups[b_two], min(2, len(blk_fert_groups[b_two]))):
                                for i1 in blk_fert_groups[b_ones[0]]:
                                    for i2 in blk_fert_groups[b_ones[1]]:
                                        blk_combs.append(list(c2) + [i1, i2])
                    elif k == 5 and n_b == 3:
                        for b_one in range(3):
                            b_twos = [b for b in range(3) if b != b_one]
                            for i1 in blk_fert_groups[b_one]:
                                for c2a in combinations(blk_fert_groups[b_twos[0]], min(2, len(blk_fert_groups[b_twos[0]]))):
                                    for c2b in combinations(blk_fert_groups[b_twos[1]], min(2, len(blk_fert_groups[b_twos[1]]))):
                                        blk_combs.append([i1] + list(c2a) + list(c2b))
                    elif k == 6 and n_b == 3:
                        c_lists = [list(combinations(g_b, min(2, len(g_b)))) for g_b in blk_fert_groups]
                        for c_tup in product(*c_lists):
                            blk_combs.append(list(c_tup[0]) + list(c_tup[1]) + list(c_tup[2]))
                    if len(blk_combs) > 2500:
                        rng_blk = np.random.default_rng(20261109 + k)
                        sel_ix = rng_blk.choice(len(blk_combs), size=2500, replace=False)
                        blk_combs = [blk_combs[i] for i in sel_ix]
                    if blk_combs:
                        br_res = [eval_subset(c_b, fert_only) for c_b in blk_combs]
                        br_rm = [x[0] for x in br_res]
                        br_ab = [x[1] for x in br_res]
                        rows.append({
                            'source': src, 'target': tgt, 'evaluation': ev, 'support': sup_name,
                            'specification': spec, 'strategy': 'Block_Rand_fert', 'k': k,
                            'mean_RMSE': float(np.mean(br_rm)),
                            'q10_RMSE': float(np.quantile(br_rm, 0.10)),
                            'q90_RMSE': float(np.quantile(br_rm, 0.90)),
                            'mean_abs_bias': float(np.mean(br_ab)),
                        })

    return pd.DataFrame(rows)


def run_calibrated_intervals(preds: pd.DataFrame) -> pd.DataFrame:
    """Experiment C3: Fixed-center pooled source-OOF 90% prediction intervals before and after 3-N0 calibration."""
    models = [
        ('N-only', 'N', False),
        ('N+C+T', 'N_C_T', False),
        ('N+Leaf+T', 'N_Leaf_T', False),
        ('N+ResNet-50', 'N_ResNet50_PCA52_common4', False),
        ('N+3N0', 'N', True),
        ('N+C+T+3N0', 'N_C_T', True),
        ('N+Leaf+T+3N0', 'N_Leaf_T', True),
        ('N+ResNet-50+3N0', 'N_ResNet50_PCA52_common4', True),
    ]
    rows = []
    for src in DOMAINS:
        for label, spec, use_n0 in models:
            cal = preds[
                (preds.source == src) & (preds.target == src) & (preds.evaluation == 'BlockCV')
                & (preds.specification == spec) & (preds.N > 0)
            ].sort_values('plot_id')
            n_cal = len(cal)
            rank = math.ceil(0.90 * (n_cal + 1))
            q = float(np.sort(np.abs(cal.observed.to_numpy(float) - cal.predicted.to_numpy(float)))[rank - 1])

            for tgt in DOMAINS:
                if src == tgt:
                    continue
                ev = 'adjacent_field' if src[:4] == tgt[:4] else 'cross_year'
                d_tgt = preds[
                    (preds.source == src) & (preds.target == tgt) & (preds.evaluation == ev)
                    & (preds.specification == spec)
                ].sort_values('plot_id')
                d_n0 = d_tgt[d_tgt.N == 0]
                z = d_tgt[d_tgt.N > 0].copy().sort_values('plot_id')
                offset = float(np.mean(d_n0.observed.to_numpy(float) - d_n0.predicted.to_numpy(float))) if use_n0 else 0.0
                center = z.predicted.to_numpy(float) + offset
                obs = z.observed.to_numpy(float)
                covered = (obs >= center - q) & (obs <= center + q)
                blk_arr = z.block.to_numpy(int)
                boot = []
                for bs in product([1, 2, 3], repeat=3):
                    ix = np.concatenate([np.flatnonzero(blk_arr == b) for b in bs])
                    boot.append(float(np.mean(covered[ix])))
                lo, hi = np.quantile(boot, [0.025, 0.975])
                rows.append({
                    'source': src, 'target': tgt, 'evaluation': ev,
                    'model': label, 'specification': spec, 'calibrated_3N0': use_n0,
                    'calibration_n': n_cal, 'target_n': len(z),
                    'offset_kg_ha': offset, 'radius_q90': q, 'mean_width_kg_ha': 2.0 * q,
                    'coverage': float(np.mean(covered)),
                    'coverage_low_95': float(lo), 'coverage_high_95': float(hi),
                })

    return pd.DataFrame(rows)


def main():
    preds = load_unified_predictions()
    plot64 = pd.read_csv(OUT / 'plot64_millimeter_features.csv')

    decomp_df, contrast_df = run_hybrid_transfer_decomposition(preds)
    decomp_df.to_csv(OUT / 'hybrid_transfer_decomposition.csv', index=False)
    contrast_df.to_csv(OUT / 'hybrid_transfer_contrasts.csv', index=False)

    curves_df = run_fewshot_budget_curves(preds, plot64)
    curves_df.to_csv(OUT / 'fewshot_calibration_curves.csv', index=False)

    intervals_df = run_calibrated_intervals(preds)
    intervals_df.to_csv(OUT / 'calibrated_interval_coverage.csv', index=False)

    print('=== C1. Six-Task Mean Total / Centered / Between-N / Within-N RMSE on Non-N0 Holdout ===')
    piv = decomp_df.groupby(['calibration_regime', 'specification', 'measure'])['estimate'].mean().unstack('measure')
    print(piv[['Total', 'Bias', 'Centered', 'Between-N', 'Within-N']].round(1).to_string())

    print('\n=== C1b. Task-by-Task Total RMSE for Zero-Label vs +3 N0 ===')
    t_piv = decomp_df[
        (decomp_df.measure == 'Total')
        & (decomp_df.calibration_regime.isin(['zero_label', 'plus_3N0', 'oracle']))
        & (decomp_df.specification.isin(['N', 'N_C_T', 'N_Leaf_T', 'N_ResNet50_PCA52_common4']))
    ].pivot(index=['source', 'target'], columns='model_id', values='estimate')
    print(t_piv.round(1).to_string())

    print('\n=== C2. Six-Task Mean Few-Shot Budget Curves (unsampled_fertilized support, k=0..6) ===')
    c_piv = curves_df[
        (curves_df.support == 'unsampled_fertilized')
        & (curves_df.specification.isin(['N', 'N_C_T', 'N_Leaf_T', 'N_ResNet50_PCA52_common4']))
    ].groupby(['specification', 'strategy', 'k'])['mean_RMSE'].mean().unstack('k')
    print(c_piv.round(1).to_string())

    print('\n=== C3. Six-Task Mean Empirical 90% Coverage & Interval Width ===')
    i_piv = intervals_df.groupby('model')[['coverage', 'mean_width_kg_ha']].mean()
    i_piv['coverage_pct'] = 100.0 * i_piv['coverage']
    print(i_piv[['coverage_pct', 'mean_width_kg_ha']].round(1).to_string())


if __name__ == '__main__':
    main()
