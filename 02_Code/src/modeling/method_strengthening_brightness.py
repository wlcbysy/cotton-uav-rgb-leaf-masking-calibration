"""S6 sensitivity to yield-blind scaling using field-access bare-ground patches."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from shapely.geometry import box

from method_strengthening_inference import TASK
from method_strengthening_integrity import MINING, OUT, ROOT, SEED, sha256

sys.path.insert(0, str(ROOT / "02_Code/utilities"))
import analyze_human_rgb_reference as reference
from extract_raw_rgb_information import GPKG, arrays, crop

# Geometry was selected from S6 overviews with only plot outlines displayed.
# Each affine maps 1,000-pixel-wide overview coordinates to the raw raster CRS.
# Three disjoint small patches sample the exposed access strip south of the trial.
PATCHES = {
    "2023_F1": (
        Affine(4.1707511977051586e-7, 0, 81.31410052096427, 0, -4.170123999083756e-7, 40.60618866348714),
        [(305, 933, 368, 940), (465, 934, 521, 942), (631, 936, 690, 944)],
    ),
    "2024_F1": (
        Affine(.0263484, 0, 526577.5597102039, 0, -.02634213698630137, 4495086.528458049),
        [(381, 1537, 442, 1543), (518, 1537, 571, 1543), (755, 1539, 806, 1545)],
    ),
    "2024_F2": (
        Affine(.0268056, 0, 526537.4395697666, 0, -.02681135776974703, 4495085.787904234),
        [(265, 1465, 317, 1472), (435, 1467, 483, 1474), (738, 1468, 784, 1474)],
    ),
}
REFERENCE_DN = 128.0


def white_masks(saturation, brightness, valid):
    return {f"white_b{b}_s{s}": (brightness >= b) & (saturation <= s/100) & valid
            for b in (120, 140, 160, 180) for s in (20, 30, 40)}


def scale_for_raster(src, geom, domain):
    tf, patches = PATCHES[domain]
    samples, audit = [], []
    for i, (x1, y1, x2, y2) in enumerate(patches, 1):
        west, north = tf * (x1, y1)
        east, south = tf * (x2, y2)
        roi = box(west, south, east, north)
        assert not geom.intersects(roi).any(), (domain, i)
        rgb, valid = crop(src, roi)
        assert valid.mean() > .95 and rgb[valid].max() <= 255, (domain, i)
        *_, br = arrays(rgb)
        pixels = br[valid]
        samples.append(pixels)
        audit.append(dict(domain=domain, raster_file=Path(src.name).name,
                          patch_id=i, crs=src.crs.to_string(),
                          west=west, south=south, east=east, north=north,
                          valid_pixels=len(pixels), valid_fraction=valid.mean(),
                          median_DN=np.median(pixels),
                          q25_DN=np.quantile(pixels, .25), q75_DN=np.quantile(pixels, .75)))
    # Equal patch weight prevents slight differences in area dominating scale.
    median = float(np.median([np.median(p) for p in samples]))
    factor = REFERENCE_DN / median
    for r in audit:
        r.update(flight_reference_median_DN=median, common_reference_DN=REFERENCE_DN, scale=factor)
    return factor, audit


def build_inputs():
    raw = pd.read_csv(MINING / "direct_image_features.csv")
    norm = raw.copy()
    points = pd.read_csv(MINING / "ai_reference/AI_reference_points.csv")
    orig = points.copy()
    assert len(points) == 2808 and not points.point_id.duplicated().any()
    label_path = MINING / "ai_reference/ai_labels.json"
    label_json = json.loads(label_path.read_text())
    assert points.label.eq(points.point_id.map(label_json["labels"])).all()
    assert len(label_json["labels"]) == len(points)
    assert sha256(label_path) == json.loads((MINING/"ai_reference/analysis_status.json").read_text())["label_file_sha256"]
    geometry = gpd.read_file(GPKG, layer="inward_buffer_plot").set_index(["year", "field", "plot_id"])
    audits, all_b = [], []
    point_n, raw_max = 0, 0.0
    for path, rows in raw[raw.DAS.eq(181)].groupby("raster_path"):
        year, field = rows.iloc[0][["year", "field"]]
        domain = f"{year}_{field}"
        with rasterio.open(path) as src:
            geom = geometry.to_crs(src.crs)
            factor, patch_rows = scale_for_raster(src, geom.loc[(year, field)].geometry, domain)
            audits.extend(patch_rows)
            for index, row in rows.iterrows():
                rgb, valid = crop(src, geom.loc[(year, field, row.plot_id)].geometry)
                assert valid.sum() == row.valid_pixels and rgb[valid].max() <= 255
                *_, sat, br = arrays(rgb)
                matched = points.index[points.domain.eq(domain) & points.plot_id.eq(row.plot_id) & points.phase.eq("cotton")]
                yy, xx = points.loc[matched, "y"].to_numpy(), points.loc[matched, "x"].to_numpy()
                assert valid[yy, xx].all()
                point_n += len(matched)
                old = white_masks(sat, br, valid)
                # A common positive multiplier on R, G and B leaves saturation
                # unchanged. Work on float brightness to avoid clipping/rounding.
                scaled = white_masks(sat, br.astype(float)*factor, valid)
                for rule, mask in scaled.items():
                    raw_b = old[rule].sum()/valid.sum()
                    raw_max = max(raw_max, abs(raw_b-row[rule]))
                    assert np.array_equal(old[rule][yy, xx], orig.loc[matched, rule].to_numpy())
                    points.loc[matched, rule] = mask[yy, xx]
                    new_b = mask.sum()/valid.sum()
                    norm.loc[index, rule] = new_b
                    all_b.append(dict(domain=domain, plot_id=row.plot_id, block=row.block_design,
                                      N=row.N_kg_ha, valid_pixels=row.valid_pixels, rule=rule,
                                      B_raw=raw_b, B_normalized=new_b, change_pp=100*(new_b-raw_b)))
        print("S6", domain, "factor", factor, "ROIs", len(rows), flush=True)
    assert len(audits) == 9 and len(all_b) == 66*12 and point_n == 1404 and raw_max < 1e-9
    assert points[~points.phase.eq("cotton")].equals(orig[~points.phase.eq("cotton")])
    pd.DataFrame(audits).to_csv(OUT/"brightness_reference_patches.csv", index=False)
    pd.DataFrame(all_b).to_csv(OUT/"brightness_B_by_plot_rule.csv", index=False)
    points.to_csv(OUT/"brightness_reference_points.csv", index=False)
    return raw, norm, orig, points, raw_max


def validation(before, after):
    frozen = pd.read_csv(MINING/"ai_reference/AI_mask_validation_metrics.csv")
    records = []
    for source in sorted(before.domain.unique()):
        source_plots = before.loc[before.domain.eq(source), "plot_id"].unique()
        for target in sorted(before.domain.unique()):
            for phase, positive in (("canopy", "g"), ("cotton", "c")):
                old_rule = "green_canopeo" if phase == "canopy" else reference.select_rule(before, source, source_plots)[0]
                new_rule = "green_canopeo" if phase == "canopy" else reference.select_rule(after, source, source_plots)[0]
                d = before[before.domain.eq(target) & before.phase.eq(phase) & before.role.eq("validation")]
                known = d[d.label.ne("u")]
                raw = reference.metrics(known.label.eq(positive), known[old_rule], known.area_weight)
                new = reference.metrics(known.label.eq(positive), after.loc[known.index, new_rule], known.area_weight)
                baseline = frozen.query("calibration_source == @source and validation_domain == @target and phase == @phase").iloc[0]
                assert baseline.mask_rule == old_rule and abs(baseline.F1-raw["F1"]) < 1e-10
                records.append(dict(calibration_source=source, target=target, phase=phase,
                                    validation_points=len(d), uncertain=len(d)-len(known),
                                    raw_rule=old_rule, normalized_rule=new_rule,
                                    **{f"raw_{k}": v for k, v in raw.items()},
                                    **{f"normalized_{k}": v for k, v in new.items()},
                                    F1_change_pp=100*(new["F1"]-raw["F1"])))
    pd.DataFrame(records).to_csv(OUT/"brightness_validation_matrix.csv", index=False)


def nested(points, direct, dest):
    common = pd.read_csv(MINING/"common_plot_features_common_DAS.csv")
    # Use the same production threshold selector, role logic and PartialRidge
    # nesting on the frozen common-DAS feature matrix; only B is recomputed.
    with patch.object(reference, "OUT", dest), patch.object(reference, "PREFIX", "brightness"), \
         patch.object(reference.repair, "support_tables", return_value={"common_DAS": common}):
        receipt = reference.nested_cotton_models(points, direct, None)
    assert receipt == dict(prediction_rows=768, complete_metric_sets=36, unavailable_jobs=0), receipt
    return receipt


def main():
    raw, scaled, before, after, feature_replay = build_inputs()
    validation(before, after)
    scratch = ROOT/"90_Scratch/method_strengthening"
    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=scratch) as tmp:
        path = Path(tmp)
        nested(before, raw, path)
        replay = pd.read_csv(path/"brightness_source_predictions.csv")
    frozen = pd.read_csv(MINING/"ai_reference/AI_source_predictions.csv")
    keys = TASK + ["specification", "plot_id", "block", "N", "observed"]
    d = frozen.merge(replay, on=keys, validate="one_to_one")
    replay_max = float(abs(d.predicted_x-d.predicted_y).max())
    assert len(d) == len(frozen) == len(replay) and replay_max < 1e-5
    assert d.mask_rule_x.eq(d.mask_rule_y).all()
    receipt = nested(after, scaled, OUT)
    normalized = pd.read_csv(OUT/"brightness_source_predictions.csv")
    b_raw = raw[raw.DAS.eq(181)].assign(domain=lambda d: d.year.astype(str)+"_"+d.field).set_index(["domain","plot_id"])
    b_new = scaled[scaled.DAS.eq(181)].assign(domain=lambda d: d.year.astype(str)+"_"+d.field).set_index(["domain","plot_id"])
    k = TASK + ["specification", "plot_id"]
    support = frozen[frozen.specification.ne("N")].merge(normalized, on=k, validate="one_to_one")
    support["B_raw"] = [b_raw.loc[(r.target, r.plot_id), r.mask_rule_x] for r in support.itertuples()]
    support["B_normalized"] = [b_new.loc[(r.target, r.plot_id), r.mask_rule_y] for r in support.itertuples()]
    support[k+["mask_rule_x", "mask_rule_y", "B_raw", "B_normalized"]].rename(
        columns={"mask_rule_x":"raw_rule", "mask_rule_y":"normalized_rule"}
    ).to_csv(OUT/"brightness_selected_B_by_task_plot.csv", index=False)
    (OUT/"brightness_receipt.json").write_text(json.dumps(
        dict(seed=SEED, deterministic=True, **receipt, reference_points=2808, changed_phase_points=1404,
             raw_feature_max_difference=feature_replay, raw_prediction_max_difference=replay_max),
        indent=2)+"\n")


if __name__ == "__main__":
    main()
