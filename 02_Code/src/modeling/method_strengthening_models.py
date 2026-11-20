"""Fit the nested RF and frozen ImageNet representation comparators."""
import argparse
from itertools import product
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from threadpoolctl import threadpool_limits

import mine_raw_rgb_information as legacy
from method_strengthening_inference import TASK
from method_strengthening_integrity import MINING, OUT, ROOT, SEED, sha256

RF_SEED = 20261025
RF_GRID = list(product([2, 3, None], [1, 3, 5], [100, 300]))
RF_FIXED = (3, 3, 100)
PC_PER_STAGE = 13
SCRATCH = ROOT / "90_Scratch/method_strengthening"
RF_SPECS = ["N", "N_Cearly", "N_C", "N_C_T"]
MODERN_SPEC = "N_ResNet50_PCA52_common4"


def sets():
    common = pd.read_csv(MINING / "common_plot_features_common_DAS.csv")
    assert len(common) == 64
    return {k: d.reset_index(drop=True) for k, d in common.groupby("domain")}


def source_splits(d):
    ii = np.arange(len(d))
    g = d.block_design.to_numpy()
    return [(ii[g != b], int(b)) for b in np.unique(g)] + [(ii, 0)]


def jobs_for_split(frames, source, held):
    if held:
        return [(source, np.flatnonzero(frames[source].block_design.eq(held)), "BlockCV")]
    return [(target, np.arange(len(d)), "adjacent_field" if target[:4] == source[:4] else "cross_year")
            for target, d in frames.items() if target != source]


def records(predict, frames, source, held, train, spec, alg):
    result = []
    train_ids = ";".join(map(str, frames[source].iloc[train].plot_id))
    for target, idx, evaluation in jobs_for_split(frames, source, held):
        d = frames[target].iloc[idx]
        y_hat = predict(target, idx)
        assert len(y_hat) == len(d) and np.isfinite(y_hat).all()
        if held:
            assert set(d.plot_id).isdisjoint(frames[source].iloc[train].plot_id)
        result.extend(
            dict(variant="method_strengthening_common_DAS", source=source, target=target,
                 algorithm=alg, specification=spec, evaluation=evaluation,
                 plot_id=r.plot_id, block=r.block_design, N=r.N_kg_ha, observed=r.yield_kg_ha,
                 predicted=value, source_block_omitted=held, train_n=len(train), train_plot_ids=train_ids)
            for r, value in zip(d.itertuples(), y_hat)
        )
    return result


def inner_splits(d, train):
    g = d.block_design.to_numpy()
    return [(train[g[train] != b], g[train] == b) for b in np.unique(g[train])]


def score(e):
    return float(np.sqrt(np.mean(e**2))), float(np.mean(abs(e)))


def save_predictions(rows, prefix, expected):
    pp = pd.DataFrame(rows)
    keys = TASK + ["specification", "plot_id"]
    assert not pp.duplicated(keys).any() and len(pp) == expected
    met, _ = legacy.summarize(pp)
    pp.to_csv(OUT / f"{prefix}_predictions.csv", index=False)
    met.to_csv(OUT / f"{prefix}_metrics.csv", index=False)
    return met


def rf_model(params):
    depth, leaf, trees = params
    model = legacy.model("RF")
    model.regressor.set_params(max_depth=depth, min_samples_leaf=leaf, n_estimators=trees,
                               random_state=RF_SEED)
    assert isinstance(model.regressor, RandomForestRegressor)
    return model


def random_forest():
    frames = sets()
    rows, fixed_rows, tuning = [], [], []
    for source, d in frames.items():
        y = d.yield_kg_ha.to_numpy()
        for spec in RF_SPECS:
            x = {dom: legacy.design(td, spec) for dom, td in frames.items()}
            for train, held in source_splits(d):
                splits = inner_splits(d, train)
                scores = []
                for depth, leaf, trees in RF_GRID:
                    params = (depth, leaf, trees)
                    pred = np.empty(len(train))
                    for subtrain, test in splits:
                        model = rf_model(params).fit(x[source][subtrain], y[subtrain])
                        pred[test] = model.predict(x[source][train[test]])
                    rmse, mae = score(pred - y[train])
                    scores.append((rmse, mae, depth if depth is not None else np.inf, -leaf, trees, params))
                best = min(scores)[-1]
                for rmse, mae, _, _, _, params in scores:
                    depth, leaf, trees = params
                    tuning.append(dict(source=source, specification=spec, held_block=held,
                                       train_n=len(train), max_depth=depth if depth is not None else "unlimited",
                                       min_samples_leaf=leaf, n_estimators=trees, inner_RMSE=rmse,
                                       inner_MAE=mae, selected=params == best, random_state=RF_SEED))
                final = rf_model(best).fit(x[source][train], y[train])
                rows.extend(records(lambda dom, idx: final.predict(x[dom][idx]),
                                    frames, source, held, train, spec, "RF_nested"))
                fixed = rf_model(RF_FIXED).fit(x[source][train], y[train])
                fixed_rows.extend(records(lambda dom, idx: fixed.predict(x[dom][idx]),
                                          frames, source, held, train, spec, "RF"))
            print("RF", source, spec, flush=True)
    frozen = pd.read_csv(MINING / "support_predictions.csv").query(
        "variant == 'common_DAS' and algorithm == 'RF' "
        "and evaluation in ['BlockCV', 'adjacent_field', 'cross_year']"
    )
    keys = TASK + ["specification", "plot_id"]
    parity = frozen.merge(pd.DataFrame(fixed_rows), on=keys, validate="one_to_one")
    max_error = abs(parity.predicted_x - parity.predicted_y).max()
    assert len(parity) == len(frozen) == len(rows) == 64*3*4 and max_error < 1e-5, max_error
    print("Fixed RF max prediction difference", max_error, flush=True)
    save_predictions(rows, "nested_rf", 64*3*4)
    pd.DataFrame(tuning).to_csv(OUT / "nested_rf_tuning.csv", index=False)
    (OUT / "nested_rf_replay.json").write_text(json.dumps(
        {"frozen_fixed_RF_replay_n": len(parity), "max_absolute_prediction_difference": max_error},
        indent=2) + "\n")


def extract_resnet():
    import geopandas as gpd
    from PIL import Image
    import rasterio
    import torch
    import torchvision
    from torchvision.models import ResNet50_Weights, resnet50

    sys.path.insert(0, str(ROOT / "02_Code/src/data"))
    from extract_raw_rgb_information import GPKG, crop

    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.hub.set_dir(str(SCRATCH / "torch/hub"))
    weights = ResNet50_Weights.IMAGENET1K_V2
    net = resnet50(weights=weights).requires_grad_(False).eval()
    net.fc = torch.nn.Identity()
    transforms = weights.transforms()
    mean = np.array(transforms.mean, dtype=np.float32)
    std = np.array(transforms.std, dtype=np.float32)
    geometry = gpd.read_file(GPKG, layer="inward_buffer_plot").set_index(["year", "field", "plot_id"])
    direct = pd.read_csv(MINING / "direct_image_features.csv")
    direct["domain"] = direct.year.astype(str) + "_" + direct.field
    selected = []
    for dom, d in sets().items():
        anchors = [92, 107, 130, 142] if dom.startswith("2023") else [92, 106, 128, 141]
        selected.append(direct[(direct.domain == dom) & direct.DAS.isin(anchors) & direct.plot_id.isin(d.plot_id)])
    selected = pd.concat(selected).sort_values(["domain", "plot_id", "DAS"]).reset_index(drop=True)
    assert len(selected) == 256 and selected.groupby(["domain", "plot_id"]).size().eq(4).all()
    vectors = np.empty((len(selected), 2048), np.float32)
    for raster_path, rows in selected.groupby("raster_path"):
        batch = []
        with rasterio.open(raster_path) as src:
            g = geometry.to_crs(src.crs)
            for i, row in rows.iterrows():
                rgb, valid = crop(src, g.loc[(row.year, row.field, row.plot_id)].geometry)
                assert valid.sum() == row.valid_pixels and rgb[valid].max() <= 255
                # Resize the entire inward ROI with preserved aspect ratio. Fill
                # outside-ROI pixels and letterbox with neutral ImageNet means.
                rgb = np.where(valid[..., None], rgb, np.round(mean * 255)).clip(0, 255).astype(np.uint8)
                h, w = valid.shape
                scale = 224 / max(h, w)
                h, w = max(1, round(h * scale)), max(1, round(w * scale))
                tile = np.asarray(Image.fromarray(rgb).resize((w, h), Image.Resampling.BILINEAR),
                                  dtype=np.float32) / 255
                canvas = np.zeros((224, 224, 3), dtype=np.float32)
                y, x = (224-h)//2, (224-w)//2
                canvas[y:y+h, x:x+w] = (tile-mean)/std
                batch.append(canvas.transpose(2, 0, 1))
        with torch.inference_mode():
            for start in range(0, len(batch), 8):
                idx = rows.index[start:start+8]
                vectors[idx] = net(torch.from_numpy(np.stack(batch[start:start+8]))).numpy()
        print("ResNet", raster_path, len(batch), flush=True)
    assert np.isfinite(vectors).all()
    np.savez_compressed(OUT / "resnet50_roi_embeddings.npz", vectors=vectors,
                        domain=selected.domain.to_numpy(dtype=str), plot_id=selected.plot_id.to_numpy(),
                        DAS=selected.DAS.to_numpy())
    checkpoint = torch.hub.get_dir() + "/checkpoints/" + weights.url.rsplit("/", 1)[-1]
    (OUT / "resnet50_embedding_receipt.json").write_text(json.dumps(
        dict(weight=weights.name, url=weights.url, weight_sha256=sha256(Path(checkpoint)),
             torch=torch.__version__, torchvision=torchvision.__version__,
             roi_dates=len(selected), features_per_date=vectors.shape[1], seed=SEED, device="cpu",
             preproc="Full inward ROI, bilinear aspect-preserving resize and neutral padding to 224x224; ImageNet mean/std"),
        indent=2) + "\n")


def modern():
    frames = sets()
    with np.load(OUT / "resnet50_roi_embeddings.npz") as z:
        data = {(dom, int(p), int(t)): v.astype(np.float64)
                for dom, p, t, v in zip(z["domain"], z["plot_id"], z["DAS"], z["vectors"])}
    optical = {}
    n_design = {}
    for dom, d in frames.items():
        dates = [92, 107, 130, 142] if dom.startswith("2023") else [92, 106, 128, 141]
        optical[dom] = np.array([[data[dom, p, t] for t in dates] for p in d.plot_id])
        n_design[dom] = legacy.design(d, "N")

    def project(pca, dom, indexes):
        z = optical[dom][indexes]
        components = pca.transform(z.reshape(-1, z.shape[-1])).reshape(len(indexes), 4*PC_PER_STAGE)
        return np.c_[n_design[dom][indexes], components]

    def fit_pca(dom, idx):
        z = optical[dom][idx]
        matrix = z.reshape(-1, z.shape[-1])
        assert len(matrix) > PC_PER_STAGE
        return PCA(n_components=PC_PER_STAGE, svd_solver="full", random_state=SEED).fit(matrix)

    rows, tuning = [], []
    for dom, d in frames.items():
        y = d.yield_kg_ha.to_numpy()
        for train, held in source_splits(d):
            inner = []
            for subtrain, test in inner_splits(d, train):
                pca = fit_pca(dom, subtrain)
                inner.append((project(pca, dom, subtrain), y[subtrain],
                              project(pca, dom, train[test]), test, subtrain))
            rank = []
            for alpha in legacy.ALPHAS:
                pred = np.empty(len(train))
                for x, yy, test_x, test, _ in inner:
                    pred[test] = legacy.PartialRidge(alpha).fit(x, yy).predict(test_x)
                rank.append((*score(pred - y[train]), -alpha))
            best = -min(rank)[2]
            pca = fit_pca(dom, train)
            final = legacy.PartialRidge(best).fit(project(pca, dom, train), y[train])
            rows.extend(records(lambda target, idx: final.predict(project(pca, target, idx)),
                                frames, dom, held, train, MODERN_SPEC, "PartialRidge"))
            for rmse, mae, neg_alpha in rank:
                tuning.append(dict(source=dom, held_block=held, train_n=len(train),
                                   inner_train_plot_counts=";".join(str(len(z[-1])) for z in inner),
                                   alpha=-neg_alpha, inner_RMSE=rmse, inner_MAE=mae,
                                   selected=-neg_alpha == best, pca_explained_fraction=pca.explained_variance_ratio_.sum()))
    save_predictions(rows, "resnet50_partial_ridge", 64*3)
    pd.DataFrame(tuning).to_csv(OUT / "resnet50_partial_ridge_tuning.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=["rf", "resnet", "resnet_fit"])
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.run == "rf":
            random_forest()
        else:
            if args.run == "resnet":
                extract_resnet()
            modern()


if __name__ == "__main__":
    main()
