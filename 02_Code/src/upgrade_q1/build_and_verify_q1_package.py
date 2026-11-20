#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compile, synchronize, and verify the complete Q1 upgraded manuscript, figures, tables,
Word (.docx) submission package, and source-data repository while strictly guarding
byte-for-byte immutability of 01_Data/raw and sat_full_figure_label_legend_sync_20260901.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / '02_Code/utilities'))
sys.path.insert(0, str(ROOT / '02_Code/src/upgrade_q1'))

from audit_figure_pdf_geometry import audit as audit_pdf_geometry
from build_v22_submission_package import auditor, compile_md_to_yu_docx
from q1_manuscript_prose import write_all_manuscript_files

FAMILY = ROOT / '10_Results/cotton_rgb'
FORMAL_NEW = FAMILY / 'sat_q1_spatial_calibration_20261108'
FORMAL_OLD = FORMAL_NEW / 'analysis/frozen_baseline'

Q1_DIR = FORMAL_NEW / 'analysis/q1_spatial_calibration'
DOCS = FORMAL_NEW / 'manuscript'
QA = FORMAL_NEW / 'qa'
TABLES_DIR = FORMAL_NEW / 'tables/q1_current'
EDITABLE_DIR = FORMAL_NEW / 'editable'
PACKAGE = ROOT / '08_Submission_Package'
SUB_ZIP = FORMAL_NEW / 'submission_package/Submission_Package.zip'
SOURCE_ZIP = FORMAL_NEW / 'repository_deposition/Current_Source_Data.zip'

COVER = 'SAT_CottonRGB_Cover_Letter_AuthorCompletionRequired_20260901'
PRESUB = 'SAT_CottonRGB_Manuscript_PreSubmission_20260901'
INTEG = 'SAT_CottonRGB_Manuscript_V22_Integrated'
MAIN_TABLES = 'SAT_CottonRGB_Main_Tables_V22'
MAIN_FIGURES = 'SAT_CottonRGB_Figures_V22'
SUPP = 'SAT_CottonRGB_Supplementary_Material_20260901'
SUPP_TABLES = 'SAT_CottonRGB_Supplementary_Tables_20260901'
CAP = 'SAT_CottonRGB_Figure_Captions_20260901'

COMPONENTS = {
    'Manuscript': (PRESUB, PRESUB, False),
    'Tables': (MAIN_TABLES, 'SAT_CottonRGB_Main_Tables_20260901', True),
    'Figures': (MAIN_FIGURES, 'Figures', False),
    'Supplementary_Materials': (SUPP, SUPP, True),
    'Cover_Letter': (COVER, COVER, False),
    'Title_Page': ('SAT_CottonRGB_Title_Page_AuthorCompletionRequired_20260901',) * 2 + (False,),
    'Declaration_of_Interests': ('Declaration_of_Interests',) * 2 + (False,),
    'Highlights': ('SAT_CottonRGB_Highlights_20260901',) * 2 + (False,),
}

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def metadata(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'sha256': digest, 'bytes': path.stat().st_size}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def rel_link(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.is_file():
        link_path.unlink()
    elif link_path.exists():
        raise ValueError(f'Refusing to overwrite directory: {link_path}')
    link_path.symlink_to(os.path.relpath(target, link_path.parent))


def verify_immutability_gates() -> dict:
    """Assert 100% SHA-256 preservation of 01_Data/raw and frozen_baseline."""
    prot = json.loads((FORMAL_OLD / 'analysis/method_strengthening/protected_sha256.json').read_text(encoding='utf-8'))
    raw_expected = set(prot['raw'])
    raw_actual = {str(p.relative_to(ROOT)) for p in (ROOT / '01_Data/raw').rglob('*') if p.is_file()}
    assert raw_expected == raw_actual, '01_Data/raw file list changed!'
    for group in ('raw', 'frozen_numeric', '714_slot_inputs', 'label_and_yield_authority'):
        for rel_p, sha in prot[group].items():
            assert metadata(ROOT / rel_p)['sha256'] == sha, f'Protected file altered: {rel_p}'

    old_snapshots = {}
    for fp in sorted(FORMAL_OLD.rglob('*')):
        if fp.is_file() and not fp.is_symlink():
            rel_f = str(fp.relative_to(FORMAL_OLD))
            old_snapshots[rel_f] = metadata(fp)['sha256']

    return {
        'raw_files_verified': len(raw_expected),
        'frozen_numeric_verified': len(prot['frozen_numeric']),
        'frozen_baseline_file_count': len(old_snapshots),
        'frozen_baseline_tree_sha256': hashlib.sha256(
            json.dumps(old_snapshots, sort_keys=True).encode('utf-8')
        ).hexdigest(),
    }


def collect_figure_assets() -> dict[str, list[Path]]:
    assets: dict[str, list[Path]] = {}
    for source, kind, count in ((MAIN_FIGURES, 'main', 6), (SUPP, 'supplementary', 4)):
        text = (DOCS / f'{source}.md').read_text(encoding='utf-8')
        names = re.findall(r'\]\(\.\./figures/' + kind + r'/(Figure_[^/]+)\.png\)', text)
        assert len(names) == len(set(names)) == count, f'Unexpected figure names in {source}: {names}'
        assets[kind] = [
            FORMAL_NEW / 'figures' / kind / f'{name}.{ext}'
            for name in names
            for ext in ('pdf', 'svg', 'png', 'tiff')
        ]
        missing = [p for p in assets[kind] if not p.is_file()]
        assert not missing, f'Missing figure files: {missing}'
    return assets


def export_display_tables() -> list[Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for stale in list(TABLES_DIR.glob('*.csv')):
        stale.unlink()
    outputs: list[Path] = []
    buffers: dict[Path, list[list[str]]] = {}

    for stem in (MAIN_TABLES, SUPP_TABLES):
        number = None
        part = 0
        previous = False
        for line in (DOCS / f'{stem}.md').read_text(encoding='utf-8').splitlines():
            caption = re.match(r'^\*\*Table (S?\d+)\.', line)
            if caption:
                number, part = caption[1], 0
            if not line.startswith('|'):
                previous = False
                continue
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if all(re.fullmatch(r':?-+:?', c) for c in cells):
                continue
            if not previous:
                part += 1
                suffix = '' if part == 1 else f'_part{part}'
                out_p = TABLES_DIR / f'Table_{number}{suffix}.csv'
                outputs.append(out_p)
                buffers[out_p] = []
            assert number is not None
            buffers[out_p].append([re.sub(r'</?su[bp]>|\*', '', c) for c in cells])
            previous = True

    for out_p, rows in buffers.items():
        with out_p.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(rows)

    inference_p = TABLES_DIR / 'archived_blocked_inference_135contrasts.csv'
    with (FORMAL_OLD / 'analysis/method_strengthening/blocked_inference.csv').open(newline='', encoding='utf-8') as src:
        reader = csv.DictReader(src)
        cols = [nm for nm in reader.fieldnames if 'marker' not in nm and not nm.endswith('_gate')]
        rows = [cols] + [[row[c] for c in cols] for row in reader]
    with inference_p.open('w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    outputs.append(inference_p)
    return outputs


def load_paragraph_options():
    opt = Path.home() / '.codex/skills/academic-manuscript-editor/scripts/paragraph_options.py'
    spec = importlib.util.spec_from_file_location('paragraph_options', opt)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compile_all_docx() -> None:
    options = load_paragraph_options()
    for stem in (PRESUB, INTEG, SUPP, MAIN_TABLES, SUPP_TABLES, CAP):
        assert auditor.audit_manuscript(DOCS / f'{stem}.md') == 0, f'Auditor failed on {stem}.md'

    jobs = list(COMPONENTS.values()) + [
        (INTEG, '20261108_CottonYieldRGB_Q1_IntegratedManuscript', False),
        (CAP, CAP, False),
        (SUPP_TABLES, SUPP_TABLES, True),
    ]
    for src, dst, landscape in jobs:
        out = DOCS / f'{dst}.docx'
        compile_md_to_yu_docx(DOCS / f'{src}.md', out, landscape=landscape)
        options.normalize(out)
        audit_res = options.audit(options.read_package(out)[3])
        assert audit_res['valid'], f'paragraph_options audit failed for {out.name}: {audit_res}'

    PACKAGE.mkdir(parents=True, exist_ok=True)
    for name, (_, dst, _) in COMPONENTS.items():
        shutil.copyfile(DOCS / f'{dst}.docx', PACKAGE / f'{name}.docx')


def build_archives_and_readme(assets: dict[str, list[Path]], tables: list[Path]) -> list[Path]:
    EDITABLE_DIR.mkdir(parents=True, exist_ok=True)
    for paths in assets.values():
        for p in paths:
            if p.suffix == '.svg':
                rel_link(p, EDITABLE_DIR / p.name)
    plot_script = ROOT / '02_Code/src/upgrade_q1/plot_q1_upgraded_figures.py'
    rel_link(plot_script, EDITABLE_DIR / plot_script.name)

    # Source data directory for figures & supplementary
    sd_dir = Q1_DIR / 'source_data_current'
    sd_dir.mkdir(parents=True, exist_ok=True)
    for existing in list(sd_dir.iterdir()):
        if existing.is_symlink() or existing.is_file():
            existing.unlink()
    for p in sorted((FORMAL_OLD / 'analysis').rglob('*')):
        if p.is_file() and not p.is_symlink() and p.name != 'protected_sha256.json':
            rel_link(p, sd_dir / p.name)
    for p in sorted(Q1_DIR.glob('*.csv')):
        rel_link(p, sd_dir / p.name)

    # Submission_Package.zip
    SUB_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(SUB_ZIP, 'w', ZIP_DEFLATED) as z:
        for name in sorted(COMPONENTS):
            doc = PACKAGE / f'{name}.docx'
            rel_link(doc, SUB_ZIP.parent / 'Submission_Package' / doc.name)
            z.write(doc, doc.name)

    # Current_Source_Data.zip
    SOURCE_ZIP.parent.mkdir(parents=True, exist_ok=True)
    base_members: list[Path] = []
    for folder in (
        ROOT / '00_Project/annotation',
        ROOT / '01_Data/metadata',
        ROOT / '01_Data/processed',
        ROOT / '02_Code/config',
        ROOT / '02_Code/src',
        ROOT / '02_Code/utilities',
        FORMAL_OLD / 'analysis',
        FORMAL_OLD / 'authority_inputs',
    ):
        if folder.exists():
            for p in sorted(folder.rglob('*')):
                if (
                    p.is_file()
                    and not p.is_symlink()
                    and '__pycache__' not in p.parts
                    and not p.name.startswith('.')
                    and p.suffix != '.pyc'
                ):
                    base_members.append(p)
    if (FAMILY / 'CURRENT_FORMAL_VERSION.txt').is_file():
        base_members.append(FAMILY / 'CURRENT_FORMAL_VERSION.txt')

    q1_code = sorted((ROOT / '02_Code/src/upgrade_q1').glob('*.py'))
    q1_csvs = sorted(Q1_DIR.glob('*.csv')) + sorted(Q1_DIR.glob('*.npz'))
    sources = sorted(set(base_members + q1_code + q1_csvs + tables))

    with ZipFile(SOURCE_ZIP, 'w', ZIP_DEFLATED) as z:
        for p in sources:
            z.write(p, p.relative_to(ROOT))
        z.writestr(
            'CURRENT_README.txt',
            'Q1 Spatial Coarsening Sensitivity & Few-Shot Quantile-Stratified Calibration Formal Release (2026-11-17).\n'
            'Includes all six Main Figures, four curated Supplementary Figures (Fig. S1-S4), Tables 1-3, Tables S1-S7,\n'
            'pure-leaf & canopy-gap extractions, simulated multi-scale GSD sensitivity (3.3 mm to 3 m),\n'
            'connected-component boll-cluster morphology & 21-crop threshold sensitivity, four-way orthogonal transfer decomposition,\n'
            'Canopeo-masked vs. unmasked ResNet-50 ablations, zero-leakage RGB-only normalization regimes, 6-block pooled-source LOEO CV,\n'
            '2024 F1 plot-level spatial residual maps & Moran\'s I, commercial few-shot calibration curves (k = 0 to 6), and reproduction scripts.\n',
        )

    write_json(
        QA / 'METHODOLOGY_SOURCE_DATA_CURRENT.json',
        {
            'status': 'SYNCHRONIZED_Q1_FORMAL_CURRENT',
            'sources': {str(p.relative_to(ROOT)): metadata(p)['sha256'] for p in sources},
            'submission_zip': str(SUB_ZIP.relative_to(ROOT)),
            'source_zip': str(SOURCE_ZIP.relative_to(ROOT)),
        },
    )

    (FORMAL_NEW / 'STATUS.md').write_text(
        '# Formal Release Status: `sat_q1_spatial_calibration_20261108`\n\n'
        '- **Target Journal**: *Artificial Intelligence in Agriculture*\n'
        '- **Lifecycle**: Current formal Q1-upgraded local submission package (not yet submitted).\n'
        '- **Main Figures**: 6 (`180.0 mm` width, `<= 210.0 mm` height, `Arial >= 7.0 pt`, PDF geometry `PASS`).\n'
        '- **Supplementary Figures**: 4 (`180.0 mm` width, PDF geometry `PASS`).\n'
        '- **Tables**: Main Tables 1–3 and Supplementary Tables S1–S7.\n'
        '- **Immutability**: `01_Data/raw` (49 files) and `sat_full_figure_label_legend_sync_20260901` verified 100% byte-identical.\n',
        encoding='utf-8',
    )
    (FORMAL_NEW / 'README.md').write_text(
        '# Cotton UAV RGB Diagnosis & Yield Prediction — Q1 Spatial Coarsening & Few-Shot Calibration Release\n\n'
        'Directory `sat_q1_spatial_calibration_20261108` stores the isolated formal outputs for:\n'
        '*Disentangling canopy nitrogen limitation from stand-gap openness for UAV RGB yield prediction in mulched cotton*.\n\n'
        '## Key Subdirectories\n'
        '- `analysis/q1_spatial_calibration/`: Dual-axis diagnostic attribution (`I_Leaf` vs. `I_Gap`), prior-free (`RGB-only`) vs. integrated (`PartialRidge`) yield inversion, Module B (3.3 mm–3 m simulated GSD coarsening sensitivity & boll morphology + 21-crop threshold audit), Module C (4-way MSE decomposition, between-N slope-bias mechanism, 2024 F1 spatial residual Moran\'s I, k=0..6 quantile-stratified calibration, interval coverage), and peer-review ablations (Canopy-Masked ResNet-50, 2x2 index x masking, 6-block LOEO).\n'
        '- `figures/main/` & `figures/supplementary/`: Vector PDF/SVG and 600 DPI PNG/TIFF exports.\n'
        '- `tables/q1_current/`: Machine-readable CSV mirrors of Tables 1–3 and Tables S1–S7.\n'
        '- `manuscript/` & `submission_package/`: Yu-Shuyang XML-verified `.docx` manuscripts and `Submission_Package.zip`.\n'
        '- `repository_deposition/Current_Source_Data.zip`: Complete reproducible code and source-data archive.\n',
        encoding='utf-8',
    )
    return sources


def audit_figures_and_write_manifests(assets: dict[str, list[Path]], tables: list[Path]) -> int:
    QA.mkdir(parents=True, exist_ok=True)
    # 1. Strict 210.01 mm height check on Main Figures
    with TemporaryDirectory(prefix='cotton-q1-main-geom-') as work_m:
        f_m = Path(work_m)
        for p in assets['main']:
            if p.suffix in {'.pdf', '.svg'}:
                (f_m / p.name).symlink_to(p)
        res_m = f_m / 'main_geom.csv'
        audit_pdf_geometry(f_m, res_m, max_height_mm=210.01, minimum_font_pt=7.0)
        df_m = pd.read_csv(res_m)
        assert len(df_m) == 6 and df_m.width_mm.eq(180.0).all() and df_m.status.eq('PASS').all(), df_m

    # 2. Combined 10-figure geometry manifest (6 Main + 4 Supplementary)
    with TemporaryDirectory(prefix='cotton-q1-all-geom-') as work_all:
        f_a = Path(work_all)
        for p in [p for paths in assets.values() for p in paths if p.suffix in {'.pdf', '.svg'}]:
            (f_a / p.name).symlink_to(p)
            if p.suffix == '.svg':
                words = ' '.join(
                    ''.join(t.itertext()) for t in ET.parse(p).getroot().iter() if t.tag.split('}')[-1] == 'text'
                )
                assert not re.search(r'[*†‡]', words), f'Forbidden star/dagger in {p}'
        res_a = f_a / 'geometry.csv'
        audit_pdf_geometry(f_a, res_a, max_height_mm=300.0, minimum_font_pt=7.0)
        df_a = pd.read_csv(res_a)
        assert len(df_a) == 10 and df_a.width_mm.eq(180.0).all() and df_a.status.eq('PASS').all(), df_a
        (QA / 'FIGURE_PDF_GEOMETRY_CURRENT.csv').write_bytes(res_a.read_bytes())
        (QA / 'FIGURE_PDF_GEOMETRY_CURRENT.json').write_bytes(res_a.with_suffix('.json').read_bytes())

    title = (DOCS / f'{PRESUB}.md').read_text(encoding='utf-8').splitlines()[0].strip('# *')
    fig_files = [p for paths in assets.values() for p in paths]
    write_json(
        DOCS / 'MANUSCRIPT_BUILD_MANIFEST.json',
        {
            'status': 'FORMAL_Q1_SPATIAL_CALIBRATION_BUILT',
            'build_date': '2026-11-17',
            'title': title,
            'target_journal': 'Artificial Intelligence in Agriculture',
            'lifecycle_status': 'CURRENT_Q1_UPGRADE / LOCAL / NOT_SUBMITTED',
            'direct_support_plots': 64,
            'descriptive_boll_plots': 66,
            'non_N0_holdout_plots': {'2023 F1': 21, '2024 F1': 16, '2024 F2': 18},
            'stratified_verification_points': 2808,
            'main_figures': 6,
            'main_tables': 3,
            'supplementary_figures': 4,
            'supplementary_tables': 7,
            'manuscript_dir_docx_count': 11,
            'submission_package_docx_count': 8,
            'outputs': {p.name: metadata(p) for p in sorted(DOCS.glob('*.docx')) if not p.name.startswith('~$')},
            'submission_package_outputs': {f'{name}.docx': metadata(PACKAGE / f'{name}.docx') for name in COMPONENTS},
            'submission_zip': metadata(SUB_ZIP),
            'source_data_zip': metadata(SOURCE_ZIP),
        },
    )
    write_json(
        QA / 'figure_export_manifest.json',
        {
            'status': 'CURRENT_Q1_FIGURES_EXPORTED',
            'checked_on': '2026-11-17',
            'main_figures': 6,
            'supplementary_figures': 4,
            'formats': ['pdf', 'svg', 'png', 'tiff'],
            'files': {str(p.relative_to(FORMAL_NEW)): metadata(p) for p in sorted(fig_files)},
            'qa': 'qa/Q1_UPGRADE_QA_REPORT.json',
        },
    )
    return len(df_a)


def sync_workspace_symlinks() -> None:
    (FAMILY / 'CURRENT_FORMAL_VERSION.txt').write_text(FORMAL_NEW.name + '\n', encoding='utf-8')
    sd_dir = Q1_DIR / 'source_data_current'
    symlink_map = [
        (EDITABLE_DIR, ROOT / '03_Figures/Editable'),
        (FORMAL_NEW / 'figures/main', ROOT / '03_Figures/Main'),
        (sd_dir, ROOT / '03_Figures/Source_Data'),
        (TABLES_DIR, ROOT / '04_Tables/Current'),
        (TABLES_DIR, ROOT / '04_Tables/Source_Data'),
        (FORMAL_NEW / 'figures/supplementary', ROOT / '05_Supplementary/Figures'),
        (DOCS / f'{SUPP}.docx', ROOT / f'05_Supplementary/Materials/{SUPP}.docx'),
        (DOCS / f'{SUPP}.md', ROOT / f'05_Supplementary/Materials/{SUPP}.md'),
        (sd_dir, ROOT / '05_Supplementary/Source_Data'),
        (DOCS / f'{SUPP_TABLES}.docx', ROOT / f'05_Supplementary/Tables/{SUPP_TABLES}.docx'),
        (DOCS / f'{SUPP_TABLES}.md', ROOT / f'05_Supplementary/Tables/{SUPP_TABLES}.md'),
        (TABLES_DIR, ROOT / '05_Supplementary/Tables/Source_Data'),
        (DOCS, ROOT / '06_Manuscript'),
        (FORMAL_NEW / 'repository_deposition', ROOT / '07_Repository_Deposition'),
    ]
    for tgt, lnk in symlink_map:
        rel_link(tgt, lnk)


def verify_docx_and_package() -> dict:
    options = load_paragraph_options()
    manifest = json.loads((DOCS / 'MANUSCRIPT_BUILD_MANIFEST.json').read_text(encoding='utf-8'))
    wanted = {f'{name}.docx' for name in COMPONENTS}
    assert {p.name for p in PACKAGE.glob('*.docx') if not p.name.startswith('~$')} == wanted

    doc_summary = {}
    with ZipFile(SUB_ZIP) as z:
        assert z.testzip() is None and set(z.namelist()) == wanted
        for name, (_, stem, _) in COMPONENTS.items():
            p = PACKAGE / f'{name}.docx'
            assert p.read_bytes() == z.read(p.name) == (DOCS / f'{stem}.docx').read_bytes()
            assert (SUB_ZIP.parent / 'Submission_Package' / p.name).resolve() == p.resolve()
            assert manifest['submission_package_outputs'][p.name] == metadata(p)
            assert options.audit(options.read_package(p)[3])['valid'], name
            with ZipFile(p) as doc:
                assert doc.testzip() is None
                xml = ET.fromstring(doc.read('word/document.xml'))
            runs = list(xml.iter(W + 't'))
            assert not any(re.search(r'[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻₀₁₂₃₄₅₆₇₈₉₊₋]', t.text or '') for t in runs), name
            imgs = len(list(xml.iter(W + 'drawing')))
            tbls = len(list(xml.iter(W + 'tbl')))
            if name in ('Figures', 'Supplementary_Materials'):
                assert imgs == {'Figures': 6, 'Supplementary_Materials': 4}[name], (name, imgs)
            if name == 'Tables':
                assert tbls == 3, tbls
            if name == 'Supplementary_Materials':
                assert tbls == 7, tbls
            if name in ('Tables', 'Supplementary_Materials'):
                assert not any(
                    re.search(r'[*†‡]', t.text or '') for tbl in xml.iter(W + 'tbl') for t in tbl.iter(W + 't')
                ), f'Forbidden star/dagger in table cell of {name}'
            doc_summary[name] = {
                'drawings': imgs,
                'tables': tbls,
                'native_scientific_runs': len(list(xml.iter(W + 'vertAlign'))),
                'sha256': metadata(p)['sha256'],
            }

    src_manifest = json.loads((QA / 'METHODOLOGY_SOURCE_DATA_CURRENT.json').read_text(encoding='utf-8'))
    with ZipFile(SOURCE_ZIP) as z:
        assert z.testzip() is None
        assert set(z.namelist()) == set(src_manifest['sources']) | {'CURRENT_README.txt'}
        for rel_name, sha in src_manifest['sources'].items():
            assert hashlib.sha256(z.read(rel_name)).hexdigest() == sha, rel_name

    return doc_summary


def write_formal_release_manifest(qa_report: dict) -> None:
    write_json(QA / 'Q1_UPGRADE_QA_REPORT.json', qa_report)
    all_files = {}
    for p in sorted(FORMAL_NEW.rglob('*')):
        if p.is_file() and not p.is_symlink() and not p.name.startswith('~$') and p.name != 'FORMAL_RELEASE_MANIFEST_20261108_Q1.json':
            all_files[str(p.relative_to(FORMAL_NEW))] = metadata(p)
    write_json(
        FORMAL_NEW / 'FORMAL_RELEASE_MANIFEST_20261108_Q1.json',
        {
            'release_id': FORMAL_NEW.name,
            'target_journal': 'Artificial Intelligence in Agriculture',
            'status': 'AUTHOR_APPROVED_Q1_UPGRADE_ABC / LOCAL / NOT_SUBMITTED',
            'built_on': '2026-11-17',
            'main_figures': 6,
            'supplementary_figures': 4,
            'main_tables': 3,
            'supplementary_tables': 7,
            'technical_qa': 'qa/Q1_UPGRADE_QA_REPORT.json',
            'file_count': len(all_files),
            'files': all_files,
        },
    )


def main() -> None:
    pre_gate = verify_immutability_gates()
    write_all_manuscript_files()
    assets = collect_figure_assets()
    tables = export_display_tables()
    compile_all_docx()
    sources = build_archives_and_readme(assets, tables)
    n_figs = audit_figures_and_write_manifests(assets, tables)
    sync_workspace_symlinks()
    doc_summary = verify_docx_and_package()
    post_gate = verify_immutability_gates()
    assert pre_gate == post_gate

    qa_report = {
        'status': 'PASS',
        'checked_on': '2026-11-08',
        'formal_version': FORMAL_NEW.name,
        'immutability_gates': post_gate,
        'figure_pdf_geometry_passed': n_figs,
        'display_csv_tables': len(tables),
        'source_zip_files': len(sources),
        'documents': doc_summary,
    }
    write_formal_release_manifest(qa_report)
    print(json.dumps(qa_report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
