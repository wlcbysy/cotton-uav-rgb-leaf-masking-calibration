"""Verify protected inputs before and after the isolated method analyses."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FAMILY = ROOT / "10_Results/cotton_rgb"
POINTER = FAMILY / "CURRENT_FORMAL_VERSION.txt"
FORMAL = FAMILY / POINTER.read_text().strip()
MINING = FORMAL / "analysis/raw_rgb_mining"
OUT = FORMAL / "analysis/method_strengthening"
SEED = 20261106


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot():
    frozen = json.loads((FORMAL / "qa/FIGURE_NUMERICAL_BASELINE_CURRENT.json").read_text())
    assert len(frozen) == 86
    groups = {
        "raw": [p for p in (ROOT / "01_Data/raw").rglob("*") if p.is_file()],
        "714_slot_inputs": [
            ROOT / "01_Data/processed/model_ready/cotton_rgb_formal_714_slots.csv",
            *sorted((FORMAL / "authority_inputs").glob("*714*.csv")),
        ],
        "label_and_yield_authority": [
            MINING / "ai_reference/ai_labels.json",
            MINING / "ai_reference/AI_reference_points.csv",
            MINING / "human_reference/human_labels.json",
            MINING / "human_reference/private_sampling_manifest.csv",
            MINING / "human_reference/review_dataset.json",
            FORMAL / "authority_inputs/yield_authority.csv",
            ROOT / "01_Data/processed/mappings/plot_yield_mapping_current.csv",
            ROOT / "01_Data/processed/spatial/cotton_rgb/field_authority.gpkg",
        ],
        "frozen_numeric": [ROOT / name for name in frozen],
        "formal_release": [
            POINTER,
            *(p for p in (ROOT / "08_Submission_Package").rglob("*") if p.is_file()),
            *(p for folder in ("manuscript", "captions", "editable", "submission_package", "repository_deposition")
              for p in (FORMAL / folder).rglob("*") if p.is_file()),
            *(p for p in (MINING / "figures_v2_yield_prediction").rglob("*") if p.is_file()),
        ],
    }
    # Check every existing CSV in the project and every directly used input,
    # including files outside the historical 86-file manifest.
    groups["other_csv"] = [
        p for top in ("00_Project", "01_Data", "02_Code", "03_Figures", "04_Tables",
                      "05_Supplementary", "08_Submission_Package", "10_Results")
        for p in (ROOT / top).rglob("*.csv")
        if p.is_file() and OUT not in p.parents and p not in groups["frozen_numeric"]
    ]
    cache = {}
    records = {}
    for group, paths in groups.items():
        records[group] = {}
        for path in sorted(paths):
            name = path.relative_to(ROOT).as_posix()
            if name not in cache:
                cache[name] = sha256(path)
            records[group][name] = cache[name]
    assert records["frozen_numeric"] == frozen, "Historical 86-file baseline mismatch"
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["before", "after"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = OUT / "protected_sha256.json"
    if args.mode == "before" and baseline.exists():
        raise FileExistsError(baseline)
    current = snapshot()
    counts = {key: len(files) for key, files in current.items()}
    if args.mode == "before":
        baseline.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")
    else:
        before = json.loads(baseline.read_text())
        differences = {
            group: [name for name in (files.keys() | before[group].keys())
                    if files.get(name) != before[group].get(name)]
            for group, files in current.items()
        }
        assert not any(differences.values()), differences
        (OUT / "integrity_check.json").write_text(
            json.dumps({"status": "PASS", "counts": counts,
                        "unique_file_count": len(set().union(*(set(d) for d in current.values()))),
                        "baseline_manifest_sha256": sha256(baseline),
                        "frozen_historical_sha256_mismatches": 0, "changes": differences},
                       indent=2) + "\n"
        )
    print(args.mode, counts, flush=True)


if __name__ == "__main__":
    main()
