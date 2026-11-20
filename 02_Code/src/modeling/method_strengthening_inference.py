"""Conditional blocked inference and intercept diagnostics from frozen predictions."""
from itertools import product

import numpy as np
import pandas as pd

from method_strengthening_integrity import MINING, OUT, SEED

TASK = ["source", "target", "evaluation"]
PLOT = ["plot_id", "block", "N", "observed"]
N_PERM = 99_999
MODELS = ("N", "N_Cearly", "N_C", "N_C_T", "N_B", "N_C_B", "N_C_T_B",
          "N_RGB13_S5b", "N_RGB52_common4", "N_RGB13_S6_matched")


def read(name):
    return pd.read_csv(MINING / name)


def frozen_predictions():
    canopy = read("support_predictions.csv").query(
        "variant == 'common_DAS' and algorithm == 'PartialRidge' "
        "and evaluation in ['BlockCV', 'adjacent_field', 'cross_year']"
    )
    boll = read("ai_reference/AI_source_predictions.csv").query("specification != 'N'")
    rgb = read("methodology_current/matched_RGB_predictions.csv")
    p = pd.concat([canopy, boll, rgb], ignore_index=True)
    assert set(p.specification) == set(MODELS)
    for _, d in p.groupby(TASK):
        for _, z in d.groupby("specification"):
            ref = d[d.specification == "N"].sort_values("plot_id")
            assert np.allclose(z.sort_values("plot_id")[PLOT], ref[PLOT], atol=1e-8, rtol=0)
    return p


def within_errors(e, rates):
    return e - pd.Series(e).groupby(rates).transform("mean").to_numpy()


def rmse(e):
    return np.sqrt(np.mean(e**2))


def holm(p):
    values = np.asarray(p)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted


def ci(a, b):
    gain = a - b
    return (*np.quantile(gain, [.025, .975]), *np.quantile(100 * gain / a, [.025, .975]))


def permutation_p(null, observed, monte_carlo=False):
    extreme = np.count_nonzero(abs(null) >= abs(observed) - 1e-8)
    return (extreme + monte_carlo) / (len(null) + monte_carlo)


def inference(pred):
    frozen = read("source_data_current/relative_RMSE_gains.csv")
    rgb = read("source_data_current/matched_relative_RMSE_CI.csv").query(
        "specification == 'N_RGB52_common4'"
    ).rename(columns={"specification": "model"}).assign(base="N", measure="Total", existing_gate=False)
    spec = pd.concat([frozen, rgb], ignore_index=True)
    extras = spec.query("base == 'N' and model == 'N_C_T'")[TASK + ["base", "measure"]].copy()
    extras["model"] = "N_C_T_B"
    spec = pd.concat([spec, extras], ignore_index=True)
    assert len(spec) == 135
    rows = []
    for task, requests in spec.groupby(TASK, sort=False):
        mask = np.logical_and.reduce([pred[key].eq(value) for key, value in zip(TASK, task)])
        p = pred[mask].sort_values("plot_id")
        ref = p[p.specification == "N"]
        rates, blocks = ref.N.to_numpy(), ref.block.to_numpy()
        unique = np.unique(blocks)
        assert len(unique) == 3
        members = [np.flatnonzero(blocks == b) for b in unique]
        resamples = [np.concatenate(group) for group in product(members, repeat=len(members))]
        block_signs = np.asarray(list(product([-1, 1], repeat=len(members))))
        rng = np.random.default_rng(SEED)
        signs = np.empty((N_PERM, len(ref)), dtype=np.int8)
        for indexes in members:
            # Swap the paired loss labels within the observed block, without
            # moving plots or N rates across blocks. This assumes plot-pair exchangeability.
            signs[:, indexes] = 2 * rng.integers(0, 2, (N_PERM, len(indexes)), dtype=np.int8) - 1
        plot_indices = rng.integers(len(ref), size=(2000, len(ref)))
        errors = {model: (d.predicted - d.observed).to_numpy() for model, d in p.groupby("specification")}
        for req in requests.itertuples(index=False):
            a, b = errors[req.base], errors[req.model]
            if req.measure == "Within-N":
                a, b = within_errors(a, rates), within_errors(b, rates)
            sa, sb = a**2, b**2
            d = sa - sb
            gain = rmse(a) - rmse(b)
            cluster_ci = ci(np.array([rmse(a[ix]) for ix in resamples]),
                            np.array([rmse(b[ix]) for ix in resamples]))
            old_missing = pd.isna(req.absolute_ci_low)
            if old_missing:
                plot_ci = ci(np.sqrt(sa[plot_indices].mean(axis=1)), np.sqrt(sb[plot_indices].mean(axis=1)))
            else:
                assert np.isclose(gain, req.gain_absolute, rtol=0, atol=1e-6)
                plot_ci = (req.absolute_ci_low, req.absolute_ci_high, req.relative_ci_low, req.relative_ci_high)
            is_canopy = req.model in ("N_Cearly", "N_C", "N_C_T")
            net_b = req.model == "N_C_T_B" and req.base == "N_C_T"
            fig4 = is_canopy or req.model == "N_RGB52_common4"
            fig5 = req.model in ("N_B", "N_C_B") or net_b
            fig6 = req.base == "N" and req.model in ("N_C_T", "N_C_T_B")
            figure_marker = ""
            if (fig4 or fig5) and not old_missing and bool(req.existing_gate):
                figure_marker = "†" if req.measure == "Within-N" else "*"
            table_marker = figure_marker if (is_canopy or net_b) else ""
            if req.model == "N_Cearly" and req.measure == "Within-N":
                table_marker = ""
            if req.model == "N_C" and req.measure == "Within-N" and task[:2] != ("2024_F2", "2024_F1"):
                table_marker = ""
            rows.append(dict(zip(TASK, task), base=req.base, model=req.model, measure=req.measure,
                             table2_family=is_canopy, table3_family=net_b,
                             fig4_family=fig4, fig5_family=fig5, fig6_family=fig6,
                             n=len(ref), target_blocks=len(unique), baseline_RMSE=rmse(a),
                             model_RMSE=rmse(b), gain_absolute=gain, gain_relative_pct=100 * gain / rmse(a),
                             plot_ci_source="new Fig6 comparator" if old_missing else "frozen plot-level",
                             plot_low=plot_ci[0], plot_high=plot_ci[1],
                             plot_relative_low=plot_ci[2], plot_relative_high=plot_ci[3],
                             cluster_low=cluster_ci[0], cluster_high=cluster_ci[1],
                             cluster_relative_low=cluster_ci[2], cluster_relative_high=cluster_ci[3],
                             p_within_block=permutation_p(signs @ d, d.sum(), monte_carlo=True),
                             p_whole_block=permutation_p(block_signs @ np.array([d[i].sum() for i in members]), d.sum()),
                             table_marker_original=table_marker, figure_marker_original=figure_marker))
    out = pd.DataFrame(rows)
    for family in ("table2", "table3", "fig4", "fig5", "fig6"):
        keep = out[f"{family}_family"]
        for method in ("within_block", "whole_block"):
            out.loc[keep, f"{family}_{method}_p_holm"] = holm(out.loc[keep, f"p_{method}"])
        positive = keep & out.gain_absolute.ge(15) & out.cluster_low.gt(0)
        for method in ("within_block", "whole_block"):
            valid = positive & out[f"{family}_{method}_p_holm"].le(.05)
            out[f"{family}_{method}_gate"] = np.where(valid, np.where(out.measure.eq("Total"), "*", "†"), "")
    for family in ("table2", "table3", "fig4", "fig5"):
        original = out["table_marker_original" if family.startswith("table") else "figure_marker_original"]
        marked = out[f"{family}_family"] & original.ne("")
        for method in ("within_block", "whole_block"):
            gate = out[f"{family}_{method}_gate"]
            out[f"{family}_{method}_original_marker_status"] = np.where(
                marked, np.where(gate.eq(original), "retained", "removed"), ""
            )
    assert not out.filter(regex="whole_block_gate").ne("").any().any()
    out.to_csv(OUT / "blocked_inference.csv", index=False)


def error_stats(e, n):
    return {"RMSE": rmse(e), "bias": e.mean(), "bias_squared": e.mean()**2,
            "centered_RMSE": rmse(e - e.mean()), "within_N_RMSE": rmse(within_errors(e, n))}


def transfer(pred):
    rows = []
    for task, d in pred.query("source != target").groupby(TASK, sort=False):
        ref = d[d.specification == "N"].sort_values("plot_id")
        y, n = ref.observed.to_numpy(), ref.N.to_numpy()
        errors = {model: z.sort_values("plot_id").predicted.to_numpy() - y
                  for model, z in d.groupby("specification")}
        offset = -errors["N"][n == 0].mean()
        assert (n == 0).sum() == 3
        for name, test in (("all_target_descriptive", n >= 0), ("non_N0_held_out", n != 0)):
            base = errors["N"][test]
            oracle = base - base.mean()
            stats = error_stats(base, n[test])
            oracle_gain = stats["RMSE"] - rmse(oracle)
            compare = {**{k: v[test] for k, v in errors.items()}, "N_oracle": oracle, "N_plus_N0": base + offset}
            for model, e in compare.items():
                values = error_stats(e, n[test])
                centered_gain = stats["centered_RMSE"] - values["centered_RMSE"]
                within_gain = stats["within_N_RMSE"] - values["within_N_RMSE"]
                if model in ("N_oracle", "N_plus_N0"):
                    assert abs(within_gain) < 1e-8 and abs(centered_gain) < 1e-8
                rows.append(dict(zip(TASK, task), support=name, model=model, n=test.sum(),
                                 N0_calibration_n=(n == 0).sum(),
                                 N0_intercept=offset, oracle_intercept=-base.mean(), **values,
                                 RMSE_gain_vs_N=stats["RMSE"]-values["RMSE"],
                                 RMSE_gain_vs_N_oracle=rmse(oracle)-values["RMSE"],
                                 RMSE_gain_vs_N0=rmse(base+offset)-values["RMSE"],
                                 pct_oracle_bias_MSE_recovered=100*(1-values["bias_squared"]/stats["bias_squared"]),
                                 pct_oracle_RMSE_gain_recovered=100*(stats["RMSE"]-values["RMSE"])/oracle_gain,
                                 centered_gain_vs_N=centered_gain, within_N_gain_vs_N=within_gain))
    assert len(rows) == 6 * 2 * (len(MODELS) + 2)
    pd.DataFrame(rows).to_csv(OUT / "transfer_intercept_diagnostics.csv", index=False)


if __name__ == "__main__":
    predictions = frozen_predictions()
    inference(predictions)
    transfer(predictions)
