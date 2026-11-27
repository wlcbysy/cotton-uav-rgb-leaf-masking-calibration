"""Canonical project-relative paths.

Operational code must import locations from this module instead of embedding
machine-specific or legacy directory names.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROJECT_DIR = ROOT / "00_Project"
DATA_DIR = ROOT / "01_Data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_OBSERVATIONS_DIR = RAW_DIR / "observations"
ORTHOMOSAIC_DIR = RAW_DIR / "remote_sensing" / "orthomosaics"
PLOT_BOUNDARIES_DIR = RAW_DIR / "spatial" / "plot_boundaries"
CHINA_PROVINCES_PATH = DATA_DIR / "external" / "geospatial" / "china_provinces.geojson"
WEATHER_PATH = PROCESSED_DIR / "weather" / "alar_2023_2024_daily_weather.csv"
MODEL_READY_DIR = PROCESSED_DIR / "model_ready"
MAPPING_USED_PATH = PROCESSED_DIR / "mappings" / "plot_yield_mapping_current.csv"

RESULTS_DIR = ROOT / "10_Results" / "cotton_rgb"
FORMAL_POINTER = RESULTS_DIR / "CURRENT_FORMAL_VERSION.txt"
FORMAL_VERSION = FORMAL_POINTER.read_text(encoding="utf-8").splitlines()[0].strip()
if not FORMAL_VERSION:
    raise RuntimeError(f"Empty formal-version pointer: {FORMAL_POINTER}")
FORMAL_RESULT_DIR = RESULTS_DIR / FORMAL_VERSION
FORMAL_MAIN_FIGURE_DIR = FORMAL_RESULT_DIR / "figures" / "main"
FORMAL_SUPPLEMENTARY_FIGURE_DIR = FORMAL_RESULT_DIR / "figures" / "supplementary"
FORMAL_TABLE_DIR = FORMAL_RESULT_DIR / "tables"
MODEL_RESULTS_DIR = FORMAL_RESULT_DIR / "analysis"
MODEL_AUDIT_DIR = FORMAL_RESULT_DIR / "qa"
DATE_PAIR_COMPLETENESS_PATH = MODEL_AUDIT_DIR / "date_pair_completeness.csv"
FIGURE_DIR = FORMAL_MAIN_FIGURE_DIR
SUPPLEMENTARY_FIGURE_DIR = FORMAL_SUPPLEMENTARY_FIGURE_DIR
TABLE_DIR = FORMAL_TABLE_DIR
MANUSCRIPT_DIR = FORMAL_RESULT_DIR / "manuscript"

DELIVERABLES_DIR = FORMAL_RESULT_DIR / "submission_package"
CURRENT_SUBMISSION_DIR = FORMAL_RESULT_DIR / "submission_package"
REPOSITORY_DEPOSITION_DIR = FORMAL_RESULT_DIR / "repository_deposition"
SCRATCH_DIR = ROOT / "90_Scratch"
RENDER_QA_DIR = SCRATCH_DIR / "render_qa" / "cotton_rgb"
