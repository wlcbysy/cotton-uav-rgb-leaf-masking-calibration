#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build & Synchronize Complete V2.2 Cotton Yield Prediction Manuscript & Submission Package
Enforces 100% of Yu Shuyang (yu-shuyang-docx-engine) formatting and copy-editing rules:
- XML-locked Times New Roman (western + East Asian), pure black (#000000)
- Native superscript/subscript conversion (eliminating Songti line-spacing ghosts)
- Native Three-Line Tables with repeating <w:tblHeader/> across pages
- Landscape sections automatically applied for wide multi-column tables in Tables.docx & Supplementary_Materials.docx
- Zero Chinese characters, zero body bold, zero Figure-as-subject, zero P=0.000, bidirectional citation check
"""

import hashlib
import json
import os
import re
import sys
import importlib.util
import shutil
from pathlib import Path
import pandas as pd
import docx
from docx import Document
from docx.oxml import OxmlElement
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING, WD_BREAK
from docx.shared import Inches, Pt, Mm

# Import core XML locking & table builders from yu-shuyang-docx-engine
ENGINE_DIR = Path.home() / ".gemini/config/skills/yu-shuyang-docx-engine/scripts"
sys.path.insert(0, str(ENGINE_DIR))
import render_yu_docx as yu
import audit_yu_manuscript as auditor

# AI-assisted image extraction is an actual research method in this project.
# A legacy language template must not suppress this confirmed provenance.
auditor.AI_DISCLOSURE_PATTERNS = [
    pattern for pattern in auditor.AI_DISCLOSURE_PATTERNS
    if pattern != r"\bAI-assisted\b"
]

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT_DIR = (ROOT / "06_Manuscript").resolve()
SUBMISSION_DIR = ROOT / "08_Submission_Package"
MINING_DIR = ROOT / "10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/analysis/frozen_baseline/analysis/raw_rgb_mining"
FIG_SRC_DIR = MINING_DIR / "figures_v2_yield_prediction"
TAB_SRC_DIR = MINING_DIR / "tables_v2_yield_prediction"


def ensure_fig_symlink():
    link_path = MANUSCRIPT_DIR / "figures_v2"
    rel_target = os.path.relpath(FIG_SRC_DIR.resolve(), MANUSCRIPT_DIR)
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_symlink():
            link_path.unlink()
    link_path.symlink_to(rel_target)


def clean_academic_markdown(md_text: str) -> str:
    """Replace absolute Chinese-containing figure paths with pure-ASCII relative paths and convert inline backticks."""
    old_prefix = "10_Results/cotton_rgb/sat_q1_spatial_calibration_20261108/analysis/frozen_baseline/analysis/raw_rgb_mining/figures_v2_yield_prediction/"
    txt = md_text.replace(old_prefix, "figures_v2/")

    # Replace specific technical backtick tokens with clean academic notation (no backticks in Word)
    replacements = {
        "`[1, N, N²]`": "[1, *N*, *N*<sup>2</sup>]",
        "`X_N = [1, N, N²]`": "*X*<sub>*N*</sub> = [1, *N*, *N*<sup>2</sup>]",
        "`N+Cearly`": "*N* + *C*<sub>early</sub>",
        "`N+C`": "*N* + *C*",
        "`N+C+T`": "*N* + *C* + *T*",
        "`N+B`": "*N* + *B*",
        "`N+C+B`": "*N* + *C* + *B*",
        "`N+C+T+B`": "*N* + *C* + *T* + *B*",
        "`N+RGB13`": "*N* + RGB13",
        "`N`-only": "*N*-only",
        "`N`": "*N*",
        "`T_auc_mean`": "*T*<sub>auc_mean</sub>",
        "`Bias²`": "Bias<sup>2</sup>",
        "`MSE = Bias² + Between-N + Within-N`": "MSE = Bias<sup>2</sup> + *B*<sub>err</sub> + *W*<sub>err</sub>",
        "`MSE = Bias² + BetweenN + WithinN`": "MSE = Bias<sup>2</sup> + *B*<sub>err</sub> + *W*<sub>err</sub>",
        "`BetweenN`": "*B*<sub>err</sub>",
        "`WithinN`": "*W*<sub>err</sub>",
        "`within-N RMSE`": "within-*N* RMSE",
        "`r`<sub>adj</sub>": "*r*<sub>adj</sub>",
        "`Z`<sub>⊥</sub>": "*Z*<sub>⊥</sub>",
        "`r`": "*r*",
        "`n`": "*n*",
        "`F1`": "*F*<sub>1</sub>",
        "`337.7 kg ha⁻¹`": "337.7 kg ha<sup>-1</sup>",
        "`460.2 kg ha⁻¹`": "460.2 kg ha<sup>-1</sup>",
        "`282.0`": "282.0",
        "`703.3 kg ha⁻¹`": "703.3 kg ha<sup>-1</sup>",
        "`1360`": "1360",
        "`2489 °C·d`": "2489 °C·d",
        "`92`": "92",
        "`181 DAS`": "181 DAS",
        "`B ≥ 140, S ≤ 0.20`": "*B* ≥ 140, *S* ≤ 0.20",
        "`*`": "\\*",
        "`†`": "†",
    }
    for k, v in replacements.items():
        txt = txt.replace(k, v)

    # Strip any remaining inline backticks `foo` -> foo
    txt = re.sub(r"`([^`]+)`", r"\1", txt)
    return txt


def append_rich_text_ext(p, raw_text: str, size_pt: float = 11.0, force_bold: bool = False, allow_inline_bold: bool = False):
    """Extended rich-text renderer supporting both Main (Figure 1 / Table 1) and Supplementary (Figure S1 / Table S1) caption prefixes."""
    escaped_text = raw_text.replace("\\*", "\x00STAR\x00")
    text = yu.normalize_unicode_sup_sub(escaped_text)
    text = re.sub(r"(?<![\*A-Za-z])([PFtn])(?=\s*[=<>])", r"*\1*", text)
    text = re.sub(r"(?<![\*A-Za-z])R(?=<sup>2</sup>)", r"*R*", text)

    parts = yu.TOKEN_PATTERN.split(text)
    for part in parts:
        if not part:
            continue
        if part.startswith("<sup>") and part.endswith("</sup>"):
            inner = part[5:-6]
            for seg in re.split(r"(\*[^\*]+\*)", inner):
                if not seg:
                    continue
                italic_flag = seg.startswith("*") and seg.endswith("*") and len(seg) > 2
                seg_txt = (seg[1:-1] if italic_flag else seg).replace("\x00STAR\x00", "*")
                r = p.add_run(seg_txt)
                yu.lock_run_font(r, size_pt=size_pt, bold=force_bold, italic=italic_flag, sup=True)
        elif part.startswith("<sub>") and part.endswith("</sub>"):
            inner = part[5:-6]
            for seg in re.split(r"(\*[^\*]+\*)", inner):
                if not seg:
                    continue
                italic_flag = seg.startswith("*") and seg.endswith("*") and len(seg) > 2
                seg_txt = (seg[1:-1] if italic_flag else seg).replace("\x00STAR\x00", "*")
                r = p.add_run(seg_txt)
                yu.lock_run_font(r, size_pt=size_pt, bold=force_bold, italic=italic_flag, sub=True)
        elif part.startswith("***") and part.endswith("***") and len(part) > 6:
            inner = part[3:-3].replace("\x00STAR\x00", "*")
            use_bold = force_bold or allow_inline_bold
            r = p.add_run(inner)
            yu.lock_run_font(r, size_pt=size_pt, bold=use_bold, italic=True)
        elif part.startswith("**") and part.endswith("**") and len(part) > 4:
            inner = part[2:-2]
            is_allowed_prefix = bool(
                re.match(r"^(Figure\s+S?\d+|Fig\.\s+S?\d+|Table\s+S?\d+|Keywords|Declarations|Highlights)[:\.]?$", inner.replace("\x00STAR\x00", "*").strip(), re.I)
            )
            use_bold = force_bold or allow_inline_bold or is_allowed_prefix
            if "<sup>" in inner or "<sub>" in inner or "*" in inner:
                append_rich_text_ext(p, inner.replace("\x00STAR\x00", "\\*"), size_pt=size_pt, force_bold=use_bold, allow_inline_bold=use_bold)
            else:
                r = p.add_run(inner.replace("\x00STAR\x00", "*"))
                yu.lock_run_font(r, size_pt=size_pt, bold=use_bold)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            inner = part[1:-1]
            if "<sup>" in inner or "<sub>" in inner:
                sub_parts = re.split(r"(<sup>.*?</sup>|<sub>.*?</sub>)", inner)
                for sp in sub_parts:
                    if not sp:
                        continue
                    if sp.startswith("<sup>") and sp.endswith("</sup>"):
                        r = p.add_run(sp[5:-6].replace("\x00STAR\x00", "*"))
                        yu.lock_run_font(r, size_pt=size_pt, bold=force_bold, italic=True, sup=True)
                    elif sp.startswith("<sub>") and sp.endswith("</sub>"):
                        r = p.add_run(sp[5:-6].replace("\x00STAR\x00", "*"))
                        yu.lock_run_font(r, size_pt=size_pt, bold=force_bold, italic=True, sub=True)
                    else:
                        r = p.add_run(sp.replace("\x00STAR\x00", "*"))
                        yu.lock_run_font(r, size_pt=size_pt, bold=force_bold, italic=True)
            else:
                r = p.add_run(inner.replace("\x00STAR\x00", "*"))
                yu.lock_run_font(r, size_pt=size_pt, bold=force_bold, italic=True)
        else:
            r = p.add_run(part.replace("\x00STAR\x00", "*"))
            yu.lock_run_font(r, size_pt=size_pt, bold=force_bold)


def render_three_line_table_ext(doc, table_lines, font_pt: float = None):
    rows_data = []
    for line in table_lines:
        stripped = line.strip().strip("|")
        cells = [c.strip() for c in stripped.split("|")]
        if all(re.match(r"^:?-{2,}:?$", c) for c in cells if c):
            continue
        rows_data.append(cells)
    if not rows_data:
        return

    n_rows = len(rows_data)
    n_cols = max(len(r) for r in rows_data)
    # Adaptive font size inside dense multi-column tables (Times New Roman 9.5pt for >=8 cols, 10.5pt otherwise)
    cell_pt = font_pt if font_pt is not None else (9.0 if n_cols >= 8 else 10.0)

    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for r_idx, row_vals in enumerate(rows_data):
        is_header = r_idx == 0
        # Keep a scientific data row intact without enabling paragraph options.
        tr_pr = table.rows[r_idx]._tr.get_or_add_trPr()
        no_split = OxmlElement("w:cantSplit")
        tr_pr.append(no_split)
        row_cells = table.rows[r_idx].cells
        for c_idx in range(n_cols):
            val = row_vals[c_idx] if c_idx < len(row_vals) else ""
            val = re.sub(r"^\*\*|\*\*$", "", val.strip())
            cell_p = row_cells[c_idx].paragraphs[0]
            align = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            yu.format_paragraph_spacing(
                cell_p, align=align, space_before_pt=2.0, space_after_pt=2.0, line_spacing=WD_LINE_SPACING.SINGLE
            )
            append_rich_text_ext(cell_p, val, size_pt=cell_pt, force_bold=is_header, allow_inline_bold=is_header)

    yu.apply_three_line_table_xml(table)
    spacer = doc.add_paragraph()
    yu.format_paragraph_spacing(spacer, space_before_pt=0, space_after_pt=6.0)


def compile_md_to_yu_docx(md_path: Path, out_path: Path, landscape: bool = False, img_width_in: float = 6.0):
    doc = Document()
    normal_style = doc.styles["Normal"]
    normal_style.font.name = yu.FONT_NAME
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = yu.BLACK

    for section in doc.sections:
        if landscape:
            section.orientation = WD_ORIENT.LANDSCAPE
            section.page_width = Inches(11.0)
            section.page_height = Inches(8.5)
            section.top_margin = Inches(0.8)
            section.bottom_margin = Inches(0.8)
            section.left_margin = Inches(0.8)
            section.right_margin = Inches(0.8)
        else:
            section.top_margin = Inches(1.0)
            section.bottom_margin = Inches(1.0)
            section.left_margin = Inches(1.0)
            section.right_margin = Inches(1.0)

    full_scale_figures = md_path.name == 'SAT_CottonRGB_Figures_V22.md' or 'Supplementary_Material' in md_path.stem
    if full_scale_figures:
        for section in doc.sections:
            section.orientation=WD_ORIENT.PORTRAIT
            section.page_width=Mm(210); section.page_height=Mm(297)
            section.left_margin=Mm(15); section.right_margin=Mm(15)
            section.top_margin=Mm(15); section.bottom_margin=Mm(15)
    lines = md_path.read_text(encoding="utf-8").splitlines()
    in_references = False
    title_rendered = False
    table_buffer = []

    def explicit_page_break():
        # A table spacer belongs after the table, even though doc.paragraphs
        # omits table nodes. Keep it and give it a minimal break-only line.
        p = doc.paragraphs[-1] if doc.paragraphs else doc.add_paragraph()
        if not p.text and not p._p.xpath('.//w:drawing'):
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = Pt(1)
            run = p.add_run()
            run.font.size = Pt(1)
        else:
            run = p.add_run()
        run.add_break(WD_BREAK.PAGE)

    image_count = 0
    table_caption_count = 0
    cover_letter = 'Cover_Letter' in md_path.stem
    title_page = 'Title_Page' in md_path.stem
    integrated = 'Integrated' in md_path.stem
    short_document = cover_letter or title_page

    supplementary_figures = 'Supplementary_Material' in md_path.stem
    for raw_line in lines:
        stripped = raw_line.strip()
        if table_buffer and not stripped.startswith("|"):
            render_three_line_table_ext(doc, table_buffer)
            table_buffer = []

        if not stripped or stripped == "---":
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            table_buffer.append(stripped)
            continue

        img_match = re.match(r"^!\[(.*?)\]\((.*?)\)$", stripped)
        if img_match:
            caption_text, img_rel_path = img_match.group(1).strip(), img_match.group(2).strip()
            img_full = (md_path.parent / img_rel_path).resolve() if not Path(img_rel_path).is_absolute() else Path(img_rel_path)
            if image_count:
                # Explicit page-break runs preserve boundaries with all six
                # paragraph pagination options disabled.
                explicit_page_break()
            if not img_full.is_file():
                raise FileNotFoundError(img_full)
            p_img = doc.add_paragraph()
            yu.format_paragraph_spacing(p_img, align=WD_ALIGN_PARAGRAPH.CENTER, space_before_pt=6.0, space_after_pt=2.0, line_spacing=WD_LINE_SPACING.SINGLE)
            run_img = p_img.add_run()
            from PIL import Image
            with Image.open(img_full) as im:
                aspect = im.width / im.height
            width = 180 / 25.4 if full_scale_figures or integrated else min(img_width_in, (4.8 if landscape else 6.7) * aspect)
            run_img.add_picture(str(img_full), width=Inches(width))
            if caption_text:
                p_cap = doc.add_paragraph()
                yu.format_paragraph_spacing(p_cap, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before_pt=2.0, space_after_pt=10.0, line_spacing=WD_LINE_SPACING.SINGLE)
                append_rich_text_ext(p_cap, caption_text, size_pt=11.0, force_bold=False, allow_inline_bold=True)
            image_count += 1
            continue

        h_match = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if h_match:
            level = len(h_match.group(1))
            h_text = re.sub(r"^\*\*|\*\*$", "", h_match.group(2).strip())
            if cover_letter and level == 1:
                title_rendered = True
                continue
            if h_text.startswith('1. Supplementary Figures') and supplementary_figures:
                explicit_page_break()
            if integrated and h_text in ('Main Tables', 'Main Figures'):
                section = doc.add_section(WD_SECTION.NEW_PAGE)
                landscape_section = h_text == 'Main Tables'
                section.orientation = WD_ORIENT.LANDSCAPE if landscape_section else WD_ORIENT.PORTRAIT
                section.page_width = Inches(11) if landscape_section else Mm(210)
                section.page_height = Inches(8.5) if landscape_section else Mm(297)
                margin = Inches(.8) if landscape_section else Mm(15)
                section.left_margin = section.right_margin = section.top_margin = section.bottom_margin = margin
            if h_text.startswith('2. Supplementary Tables'):
                if supplementary_figures:
                    section=doc.add_section(WD_SECTION.NEW_PAGE)
                    section.orientation=WD_ORIENT.LANDSCAPE
                    section.page_width=Inches(11);section.page_height=Inches(8.5)
                    section.left_margin=section.right_margin=section.top_margin=section.bottom_margin=Inches(.8)
                else:
                    explicit_page_break()
                image_count = 0
            if level == 1 and not title_rendered:
                p = doc.add_paragraph()
                yu.format_paragraph_spacing(p, align=WD_ALIGN_PARAGRAPH.LEFT, space_before_pt=0.0, space_after_pt=10.0,
                                            line_spacing=WD_LINE_SPACING.ONE_POINT_FIVE if title_page else WD_LINE_SPACING.DOUBLE)
                append_rich_text_ext(p, h_text, size_pt=14.0, force_bold=True, allow_inline_bold=True)
                title_rendered = True
                continue

            if re.search(r"\bReferences\b", h_text, re.I):
                in_references = True

            p = doc.add_paragraph()
            if level <= 2:
                yu.format_paragraph_spacing(p, align=WD_ALIGN_PARAGRAPH.LEFT, space_before_pt=0.0, space_after_pt=0.0)
                append_rich_text_ext(p, h_text, size_pt=12.0, force_bold=True, allow_inline_bold=True)
            elif level == 3:
                yu.format_paragraph_spacing(p, align=WD_ALIGN_PARAGRAPH.LEFT, space_before_pt=0.0, space_after_pt=0.0)
                append_rich_text_ext(p, h_text, size_pt=11.0, force_bold=True, allow_inline_bold=True)
            else:
                yu.format_paragraph_spacing(p, align=WD_ALIGN_PARAGRAPH.LEFT, space_before_pt=10.0, space_after_pt=0.0)
                append_rich_text_ext(p, h_text, size_pt=11.0, force_bold=True, allow_inline_bold=True)
            continue

        if re.match(r"^(\*\*)?Table\s+S?\d+[\.:]", stripped, re.I):
            if table_caption_count or image_count:
                explicit_page_break()
            table_caption_count += 1
            p = doc.add_paragraph()
            yu.format_paragraph_spacing(p, align=WD_ALIGN_PARAGRAPH.LEFT, space_before_pt=6.0, space_after_pt=2.0)
            append_rich_text_ext(p, stripped, size_pt=11.0, force_bold=False, allow_inline_bold=False)
            continue

        if re.match(r"^(\*\*)?(Figure|Fig\.)\s+S?\d+[\.:]", stripped, re.I):
            if md_path.name == "SAT_CottonRGB_Figure_Captions_20260901.md" and stripped.startswith("**Figure 4."):
                explicit_page_break()
            p = doc.add_paragraph()
            yu.format_paragraph_spacing(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before_pt=2.0, space_after_pt=10.0)
            append_rich_text_ext(p, stripped, size_pt=10.0, force_bold=False, allow_inline_bold=False)
            continue

        if 'Highlights' in md_path.stem and stripped.startswith('- '):
            p = doc.add_paragraph(style='List Bullet')
            yu.format_paragraph_spacing(p, align=WD_ALIGN_PARAGRAPH.LEFT, space_after_pt=6.0)
            append_rich_text_ext(p, stripped[2:], size_pt=11.0)
            continue

        eq_match = re.match(r"^(.*\S)\s+\((\d+)\)$", stripped)
        if eq_match and "=" in eq_match.group(1) and not stripped.startswith(("where ", "Where ")):
            eq_body, eq_num = eq_match.group(1).strip(), eq_match.group(2)
            p = doc.add_paragraph()
            yu.format_paragraph_spacing(p, align=WD_ALIGN_PARAGRAPH.LEFT, space_before_pt=4.0, space_after_pt=6.0)
            avail_in = (11.0 - 1.6) if landscape else (8.5 - 2.0)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(avail_in / 2.0), docx.enum.text.WD_TAB_ALIGNMENT.CENTER)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(avail_in), docx.enum.text.WD_TAB_ALIGNMENT.RIGHT)
            append_rich_text_ext(p, f"\t{eq_body}\t({eq_num})", size_pt=11.0, force_bold=False, allow_inline_bold=False)
            continue

        p = doc.add_paragraph()
        if in_references:
            yu.format_paragraph_spacing(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before_pt=0.0, space_after_pt=6.0, hanging_pt=21.0)
        elif short_document:
            letter_left = stripped.startswith(('Dear ', 'Sincerely,', 'Zhenggui Zhang', 'State Key Laboratory',
                                               'Institute of Cotton', 'Anyang ', 'Email:'))
            yu.format_paragraph_spacing(
                p, align=WD_ALIGN_PARAGRAPH.LEFT if letter_left or title_page else WD_ALIGN_PARAGRAPH.JUSTIFY,
                space_before_pt=0, space_after_pt=6,
                line_spacing=WD_LINE_SPACING.ONE_POINT_FIVE if cover_letter else WD_LINE_SPACING.SINGLE,
            )
        else:
            yu.format_paragraph_spacing(p, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before_pt=0.0, space_after_pt=10.0)
        append_rich_text_ext(p, stripped, size_pt=11.0, force_bold=False, allow_inline_bold=False)

    if table_buffer:
        render_three_line_table_ext(doc, table_buffer)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))
    print(f"[OK] Compiled DOCX: {out_path}")


def fmt_signed_md(val: float, decimals: int = 1) -> str:
    thr = 0.5 * (10 ** (-decimals))
    if abs(val) < thr:
        return f"+{0.0:.{decimals}f}"
    sign = "+" if val > 0 else "−"
    return f"{sign}{abs(val):.{decimals}f}"


def fmt_num_md(val: float, decimals: int = 2) -> str:
    thr = 0.5 * (10 ** (-decimals))
    if abs(val) < thr:
        return f"{0.0:.{decimals}f}"
    return f"−{abs(val):.{decimals}f}" if val < 0 else f"{val:.{decimals}f}"


TASK_ORDER = [
    ("2023 F1", "2023 F1"), ("2024 F1", "2024 F1"), ("2024 F2", "2024 F2"),
    ("2024 F1", "2024 F2"), ("2024 F2", "2024 F1"), ("2023 F1", "2024 F1"),
    ("2023 F1", "2024 F2"), ("2024 F1", "2023 F1"), ("2024 F2", "2023 F1"),
]

def publication_table_text(text: str) -> str:
    """Normalize and order Markdown display rows without touching frozen CSVs."""
    lines = text.splitlines()
    for j, line in enumerate(lines):
        if line.startswith('|') or re.match(r'^\*\*Table (?:S?\d+)\.', line):
            line = line.replace('common_DAS', 'common DAS')
            line = line.replace('cross_year', 'Cross-year').replace('adjacent_field', 'Adjacent-field')
            lines[j] = re.sub(r'matched_holdout_([123])', lambda m: f'Matched holdout (omit B{m[1]})', line)
    table = None
    i = 0
    while i < len(lines):
        m = re.match(r'^\*\*Table (S?\d+)\.', lines[i])
        if m: table = m[1]
        if table == '2' and lines[i].startswith('Note:') and 'optical coefficients' not in lines[i]:
            lines[i] += ' ‡ indicates optical coefficients shrunk to the N baseline (α at the grid maximum).'
        if table in {'2', '3', 'S3', 'S7'} and lines[i].startswith('|'):
            start = i
            while i < len(lines) and lines[i].startswith('|'): i += 1
            block = lines[start:i]
            if len(block) > 2:
                def rank(line):
                    for n, (src, tgt) in enumerate(TASK_ORDER):
                        if f'{src} → {tgt}' in line: return n
                    return len(TASK_ORDER)
                rows = sorted(block[2:], key=rank)
                if table == '2':
                    rows = [re.sub(r'(\+0\.0 \[\+0\.0, \+0\.1\])', r'\1‡', row) if ('2023 F1 → 2024 F1' in row or '2023 F1 → 2024 F2' in row) and '‡' not in row else row for row in rows]
                lines[start:i] = block[:2] + rows
            continue
        i += 1
    return '\n'.join(lines) + ('\n' if text.endswith('\n') else '')

def build_supplementary_markdown() -> str:
    ts1 = pd.read_csv(TAB_SRC_DIR / "Table_S1_single_date_timing_screen.csv")
    ts2 = pd.read_csv(TAB_SRC_DIR / "Table_S2_segmentation_3x3_and_uncertain_bounds.csv")
    ts3 = pd.read_csv(TAB_SRC_DIR / "Table_S3_algorithm_and_support_4way_metrics.csv")
    ts4 = pd.read_csv(TAB_SRC_DIR / "Table_S4_leave_one_source_block_sensitivity.csv")

    dom_map = {"2023_F1": "2023 F1", "2024_F1": "2024 F1", "2024_F2": "2024 F2"}
    rule_map = {
        "green_canopeo": "Canopeo green",
        "white_b140_s20": "*b*<sub>min</sub> = 140, *S* ≤ 0.20",
        "white_b120_s20": "*b*<sub>min</sub> = 120, *S* ≤ 0.20",
        "white_b140_s30": "*b*<sub>min</sub> = 140, *S* ≤ 0.30",
    }

    # Format Table S1 (BlockCV + same-year adjacent_field single-date Ridge N+C screening)
    s1_sub = ts1[(ts1.algorithm == "Ridge") & (ts1.evaluation.isin(["BlockCV", "adjacent_field"]))].copy()
    s1_lines = [
        "| Flight DAS | Source → Target | Evaluation | *n* | RMSE (kg ha<sup>-1</sup>) | *R*<sup>2</sup> | Within-*N* RMSE | ΔTotal vs. OLS *N* | ΔWithin-*N* vs. OLS *N* |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in s1_sub.sort_values(["evaluation", "source", "target", "DAS"]).iterrows():
        s1_lines.append(
            f"| {int(r.DAS)} | {dom_map[r.source]} → {dom_map[r.target]} | {r.evaluation} | {int(r.n)} | "
            f"{r.RMSE:.1f} | {fmt_num_md(r.R2, 2)} | {r.within_N_RMSE:.1f} | "
            f"{fmt_signed_md(r.gain_OLS_RMSE, 1)} | {fmt_signed_md(r.gain_OLS_within_N_RMSE, 1)} |"
        )

    # Format Table S2 (Full 3x3 segmentation consistency & ROI validation)
    s2_lines = [
        "| Phase | Calibration → Validation | Selected mask rule | Precision (%) [Bounds] | Recall (%) [Bounds] | *F*<sub>1</sub> (%) | Uncertain pts | ROI in-interval (%) | Mean exceedance (percentage points) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in ts2.sort_values(["phase", "calibration_source", "validation_domain"]).iterrows():
        rule_clean = rule_map.get(str(r.mask_rule), str(r.mask_rule).replace("_", " "))
        s2_lines.append(
            f"| {r.phase.capitalize()} | {dom_map[r.calibration_source]} → {dom_map[r.validation_domain]} | {rule_clean} | "
            f"{r.precision*100:.1f} [{r.precision_lower*100:.0f}–{r.precision_upper*100:.0f}] | "
            f"{r.recall*100:.1f} [{r.recall_lower*100:.0f}–{r.recall_upper*100:.0f}] | "
            f"{r.F1*100:.1f} | {int(r.unknown_point_n)} | {r.within_interval_pct:.1f} | {r.mean_interval_exceedance_pct:.2f} |"
        )

    # Format Table S3 (4-algorithm comparison on common_DAS + 4-support PartialRidge N+C+T comparison)
    s3_alg = ts3[(ts3.variant == "common_DAS") & (
        ((ts3.algorithm == "OLS") & (ts3.specification == "N")) |
        ((ts3.algorithm.isin(["PartialRidge", "Ridge", "RF"])) & (ts3.specification == "N_C_T"))
    )].copy()
    s3_lines = [
        "| Source → Target (Role) | Algorithm & Spec | *n* | RMSE (kg ha<sup>-1</sup>) | MAE | *R*<sup>2</sup> | Bias | Within-*N* RMSE | ΔTotal vs. OLS *N* | ΔWithin-*N* |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in s3_alg.sort_values(["source", "target", "algorithm"]).iterrows():
        r = r.copy()
        base = s3_alg[(s3_alg.source == r.source) & (s3_alg.target == r.target) & (s3_alg.evaluation == r.evaluation) & (s3_alg.algorithm == "OLS")].iloc[0]
        r["gain_matched_RMSE"] = base.RMSE - r.RMSE
        r["gain_matched_within_N_RMSE"] = base.within_N_RMSE - r.within_N_RMSE
        sp_disp = "*N*-only" if r.specification == "N" else "*N*+*C*+*T*"
        s3_lines.append(
            f"| {dom_map[r.source]} → {dom_map[r.target]} ({r.evaluation}) | {r.algorithm} ({sp_disp}) | {int(r.n)} | "
            f"{r.RMSE:.1f} | {r.MAE:.1f} | {fmt_num_md(r.R2, 2)} | {fmt_signed_md(r.Bias, 1)} | "
            f"{r.within_N_RMSE:.1f} | {fmt_signed_md(r.gain_matched_RMSE, 1)} | {fmt_signed_md(r.gain_matched_within_N_RMSE, 1)} |"
        )

    # Format Table S4 (27-fold leave-one-source-block-out sensitivity for PartialRidge N+C+T)
    s4_sub = ts4[ts4.specification == "N_C_T"].copy()
    s4_lines = [
        "| Source domain | Omitted block | Train *n* | Target domain | Evaluation role | Eval *n* | *N*+*C*+*T* RMSE (kg ha<sup>-1</sup>) | Paired ΔRMSE vs. *N*-only (kg ha<sup>-1</sup>) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in s4_sub.sort_values(["source", "source_block_omitted", "target"]).iterrows():
        s4_lines.append(
            f"| {dom_map[r.source]} | Block {int(r.source_block_omitted)} | {int(r.train_n)} | {dom_map[r.target]} | {r.evaluation} | {int(r.evaluation_n)} | {r.RMSE:.1f} | {fmt_signed_md(r.gain_RMSE, 1)} |"
        )

    supp_md = f"""# **Supplementary Materials for: Management-conditioned cotton yield prediction from UAV RGB canopy trajectories and pre-harvest boll proxies**

Shu-Yang Yu<sup>a,b,c,d</sup>, Cundong Li<sup>c</sup>, Pengcheng Li<sup>a,b,d</sup>, Zhanlei Pan<sup>a,b,d</sup>, Jian Wang<sup>a,b,d</sup>, Guilan Sun<sup>a,b,d</sup>, Yaopeng Zhang<sup>a,b,d</sup>, Junhong Li<sup>a,b,d</sup>, Menghua Zhai<sup>a,b,d</sup>, Wenqi Zhao<sup>a,b,d</sup>, Xiaolei Ren<sup>a,b,d</sup>, Ningning Zhang<sup>a,b,c,d</sup>, Xin Li<sup>a,b,c,d</sup>, Zhenggui Zhang<sup>a,b,d,*</sup>

\\* Corresponding author: Zhenggui Zhang (Email: [corresponding author email])

## **1. Supplementary Figures S1-S5**

![**Figure S1.** Retrospective single-date canopy yield prediction screening across all 33 UAV flights. Plot support varies with flight date. (a) Within-field spatial BlockCV single-date *N* + *C* RMSE gain relative to OLS *N* across all flights in 2023 Field 1, 2024 Field 1, and 2024 Field 2. (b) Same-year adjacent-field transfer between 2024 Field 1 and Field 2 across shared flight dates for total RMSE gain (solid lines) and within-*N* RMSE gain (dotted lines), with the common reproductive window indicated (shaded 92–142 DAS; vertical dotted guides mark growth stages S3–S6). Colors and symbols identify the validation target environment.](figures_v2/Figure_S1_retrospective_single_date_screening.png)

![**Figure S2.** Cross-domain 3 × 3 segmentation transfer matrix and full-ROI class-assignment bounds from an AI-extracted, author-verified 2,808-point reference. (a-b) Pre-harvest open-boll segmentation precision (%) and *F*<sub>1</sub>-score (%) across all nine source-calibration to target-validation combinations, with uncertain-point lower and upper sensitivity bounds in brackets and dashed boxes highlighting self-domain calibration. (c-d) Full-ROI predicted green canopy fraction *C* (%) and exposed boll fraction *B* (%) benchmarked against 36-point stratified reference intervals across independent validation ROIs (sorted across ranks 1–30).](figures_v2/Figure_S2_cross_domain_segmentation_and_area_validation.png)

![**Figure S3.** Marginal canopy-integral association and treatment-specific errors. (a) Raw canopy-integral versus yield scatter and environment-specific OLS lines with 95% mean-response intervals, corresponding to the added-variable display in Fig. 3g. (b) Treatment-centered RMSE for 2024 F2 → 2024 F1, comparing *N*-only, *N*+*C*+*T* and *N*+*C*+*B*. Within-*N* centering is retrospective, not prospective recalibration or a ranking metric.](figures_v2/Figure_S3_marginal_association_and_treatment_errors.png)

![**Figure S4.** Algorithm comparisons across all nine tasks on the same 64-plot direct-observation support. (a–b) Total and within-*N* RMSE for OLS *N*-only, PartialRidge *N*+*C*+*T*, joint Ridge *N*+*C*+*T* and the fixed Random Forest. ](figures_v2/Figure_S4_algorithm_comparison.png)

![**Figure S5.** Operational robustness across nine tasks. (a) *N*+*C*+*T* gain over *N*-only under common versus dense within-window support and DAS versus GDD coordinates; dense support uses four, six or seven, and seven dates in 2023 F1, 2024 F1 and 2024 F2, respectively. (b) Source-block omission sensitivity: 27 evaluations from nine reduced-source fits, with fill indicating omitted block and vertical bars showing full-support references. (c–d) Canopy and boll threshold perturbation ranges. Grey segments connect minimum and maximum RMSE and have no significance-color interpretation.](figures_v2/Figure_S5_robustness.png)

## **Supplementary methods**

Extending the α grid to 10<sup>7</sup> left both 2023 F1-source specifications (*N*+*C* and *N*+*C*+*T*) at the new upper bound, changed target RMSE by less than 0.05 kg ha<sup>-1</sup> (maximum absolute change 0.048744 kg ha<sup>-1</sup>), and did not change the conclusions (numerical source: branch codex/alpha-grid-1e7-experiment, commit aa3192d, 10_Results/alpha_grid_experiment/alpha_grid_metrics.csv).

## **2. Supplementary Tables S1-S8**

**Table S1.** Retrospective single-date canopy (*N* + *C*) yield prediction screening across UAV flight dates under spatial BlockCV and same-year adjacent-field transfer.

{chr(10).join(s1_lines)}

**Table S2.** Complete 3 × 3 cross-domain segmentation consistency, uncertain-point sensitivity bounds, and full-ROI interval agreement across canopy and pre-harvest boll phases (2,808 author-audited points).

{chr(10).join(s2_lines)}

**Table S3.** Multi-algorithm yield prediction benchmark comparing OLS (*N*-only), PartialRidge, conventional joint Ridge, and fixed Random Forest on the unified 64-plot direct-observation support (common_DAS).

{chr(10).join(s3_lines)}

**Table S4.** Leave-one-source-block-out training sensitivity from nine reduced-source fits evaluated in 27 source–target roles for PartialRidge (*N* + *C* + *T*, common_DAS).

{chr(10).join(s4_lines)}

**Table S5.** Mathematical definitions and invalid-denominator rules for the empirical 13-index visible suite (RGB13) benchmarked in cross-year extrapolation tests.

| Index | Formula (chromaticity *r*, *g*, *b*) | Channels | Invalid denominator rule | Primary reference |
| --- | --- | --- | --- | --- |
| ExB | 1.4*b* − *g* | *g*, *b* | Not applicable | Mesas-Carrascosa et al. (2020) |
| ExG | 2*g* − *r* − *b* | *r*, *g*, *b* | Not applicable | Woebbecke et al. (1995) |
| ExR | 1.4*r* − *g* | *r*, *g* | Not applicable | Meyer and Neto (2008) |
| ExGR | 3*g* − 2.4*r* − *b* | *r*, *g*, *b* | Not applicable | Meyer and Neto (2008) |
| GBRI | *g* / *b* | *g*, *b* | *b* = 0 masked as missing | Bendig et al. (2015) |
| GLI | (2*g* − *r* − *b*) / (2*g* + *r* + *b*) | *r*, *g*, *b* | 2*g* + *r* + *b* = 0 masked | Bendig et al. (2015) |
| IKAW | (*r* − *b*) / (*r* + *b*) | *r*, *b* | *r* + *b* = 0 masked | Kawashima and Nakatani (1998) |
| MGRVI | (*g*<sup>2</sup> − *r*<sup>2</sup>) / (*g*<sup>2</sup> + *r*<sup>2</sup>) | *r*, *g* | *g*<sup>2</sup> + *r*<sup>2</sup> = 0 masked | Bendig et al. (2015) |
| NGBDI | (*g* − *b*) / (*g* + *b*) | *g*, *b* | *g* + *b* = 0 masked | Zhi et al. (2025) |
| NGRDI | (*g* − *r*) / (*g* + *r*) | *r*, *g* | *g* + *r* = 0 masked | Tucker (1979) |
| RGBVI | (*g*<sup>2</sup> − *rb*) / (*g*<sup>2</sup> + *rb*) | *r*, *g*, *b* | *g*<sup>2</sup> + *rb* = 0 masked | Bendig et al. (2015) |
| RGRI | *r* / *g* | *r*, *g* | *g* = 0 masked | Bendig et al. (2015) |
| VARI | (*g* − *r*) / (*g* + *r* − *b*) | *r*, *g*, *b* | *g* + *r* − *b* = 0 masked | Gitelson et al. (2002) |

**Table S6.** Archive completeness ledger for the 15 incomplete plot-date slots in 2024 Field 1, explaining why plots 20 and 24 were excluded from the strict 64-plot direct-observation common trajectory support.

| Calendar date (2024) | DAS | Affected physical plot IDs | Observed bracketing dates | Treatment in primary V2.2 analysis |
| --- | --- | --- | --- | --- |
| 2024-07-16 | 92 | Plot 24 | 2024-07-02 to 2024-08-21 | Excluded from *n* = 64 complete-case direct support |
| 2024-07-30 | 106 | Plots 20, 24 | 2024-07-02 to 2024-08-21 | Excluded from *n* = 64 complete-case direct support |
| 2024-08-06 | 113 | Plots 20, 24 | 2024-07-02 to 2024-08-21 | Non-anchor dense flight; Plots 20 & 24 excluded |
| 2024-08-14 | 121 | Plots 3, 4, 7, 12, 20, 24 | 2024-08-06 to 2024-08-21 | Non-anchor dense flight (not used in common_DAS) |
| 2024-09-08 | 146 | Plots 4, 12, 20, 24 | 2024-09-03 to 2024-09-19 | Post-S5b dense flight (not used in common_DAS) |
"""
    matched = pd.read_csv(MINING_DIR / "methodology_current/matched_RGB_metrics.csv")
    sup = pd.read_csv(MINING_DIR / "support_metrics.csv")
    sup = sup[(sup.variant == 'common_DAS') & (sup.algorithm == 'PartialRidge') & (~sup.evaluation.str.startswith('matched'))]
    ai = pd.read_csv(MINING_DIR / "ai_reference/AI_source_model_metrics.csv")
    extra = ["", "**Table S7.** Matched feature comparisons on identical direct plots, source splits, and all-valid inward-ROI pixel support. RMSE is in kg ha<sup>-1</sup>. N+RGB52 and N+C+T use the same four dates; S5b and S6 single-date comparisons are listed separately. Every specification includes the quadratic N baseline.", "", "| Source → Target (role) | n | N+C (S5b) | N+RGB13 (S5b) | N+C+T (4 dates) | N+RGB52 (4 dates) | N+B (S6) | N+RGB13 (S6) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (src,tgt,ev), d in matched.groupby(['source','target','evaluation'],sort=False):
        def rm(spec): return d[d.specification == spec].iloc[0].RMSE
        base = sup[(sup.source == src) & (sup.target == tgt) & (sup.evaluation == ev)]
        nb = ai[(ai.source == src) & (ai.target == tgt) & (ai.evaluation == ev) & (ai.specification == 'N_B')].iloc[0].RMSE
        nc = base[base.specification == 'N_C'].iloc[0].RMSE
        nt = base[base.specification == 'N_C_T'].iloc[0].RMSE
        extra.append(f"| {dom_map[src]} → {dom_map[tgt]} ({ev}) | {int(d.iloc[0].n)} | {nc:.1f} | {rm('N_RGB13_S5b'):.1f} | {nt:.1f} | {rm('N_RGB52_common4'):.1f} | {nb:.1f} | {rm('N_RGB13_S6_matched'):.1f} |")
    extra.extend(["", "**Table S8.** Model-based inference. N tests adjust for block; the pooled boll model adjusts for environment and nested block. Partial-correlation degrees of freedom account for the full control-matrix rank. Pairwise letters in Figure 3 use Holm-adjusted model contrasts.", "", "| Response / environment | n | F | Numerator df | Residual df | P |", "| --- | --- | --- | --- | --- | --- |"])
    tests = pd.read_csv(MINING_DIR / 'methodology_current/blocked_omnibus_tests.csv')
    for r in tests.itertuples():
        pv = '<0.001' if r.P < .001 else f'{r.P:.3f}'
        extra.append(f"| {r.response} / {dom_map.get(r.domain,r.domain)} | {r.n} | {r.F:.2f} | {r.df_num} | {int(r.df_den)} | {pv} |")
    extra.extend(["", "| Partial boll-yield association | n | Control rank (including intercepts) | Residual df | r | P |", "| --- | --- | --- | --- | --- | --- |"])
    for r in pd.read_csv(MINING_DIR / 'methodology_current/partial_correlations.csv').itertuples():
        extra.append(f"| {dom_map.get(r.domain,r.domain)} | {r.n} | {r.control_rank} | {r.df_resid} | {r.r:+.3f} | {r.P:.3f} |")
    extra.extend(["", "Note: The pooled high-N boll slope was −1.718 percentage points per 100 kg N ha<sup>-1</sup> (95% interval −2.382 to −1.054), adjusted for environment and nested block. The fixed descriptive boll rule was brightness 140/saturation 0.20. Full contrasts, adjusted means, partial residuals, matched extracted indices, predictions, and tuning records accompany the source-data package.", "", "The point reference comprises 39 plots × two phases × 36 stratified points. Calibration and validation roles are assigned by whole plot. Known-class metrics exclude uncertain points; class-assignment bounds do not include finite sampling error. Labels were AI-extracted and author-verified. The conditional paired-error intervals retain fixed predictions and original N-group error centers. They do not include model refitting uncertainty, preserve spatial-block dependence, or adjust for multiple comparisons.", ""])
    supp_md += "\n".join(extra)
    return publication_table_text(supp_md)


def main():
    ensure_fig_symlink()

    # 1. Clean & save master integrated manuscript
    integ_path = MANUSCRIPT_DIR / "SAT_CottonRGB_Manuscript_V22_Integrated.md"
    raw_integ = integ_path.read_text(encoding="utf-8")
    clean_integ = publication_table_text(clean_academic_markdown(raw_integ))
    old = 'circles, squares and triangles distinguish the three comparisons.'
    new = ('purple circles, magenta squares and coral triangles distinguish the three comparisons. '
           'In (b), an extreme negative within-N gain for 2023 F1 → 2024 F2 is shown with a truncation arrow and its numerical estimate and interval to preserve resolution over the main range.')
    clean_integ = clean_integ.replace(old, new)
    clean_integ = clean_integ.replace('an extreme negative within-N gain for 2023 F1 → 2024 F2 is shown', 'extreme negative within-N gains for 2023 F1 → 2024 F2 are shown')
    clean_integ = clean_integ.replace('with a truncation arrow and its numerical estimate and interval', 'with truncation arrows and numerical estimates and intervals')
    clean_integ = clean_integ.replace('Open black points in a separate right-hand column show total RMSE on an independent axis.', 'Points in a separate right-hand column show total RMSE on an independent axis: open grey for *N*-only, filled green for *N*+*C*+*T*, and filled pink for *N*+*C*+*T*+*B*.')
    zh_path = MANUSCRIPT_DIR / 'Figure_Captions_ZH.md'
    zh = zh_path.read_text(encoding='utf-8')
    zh = zh.replace('三种比较分别用圆、方、三角标记。', '三种比较分别用紫色圆形、洋红色方形和珊瑚色三角形标记。面板b中，2023 F1 → 2024 F2 的极端负向处理内增益使用截断箭头并直接标注数值及区间，以保持主要区间的分辨率。')
    zh = zh.replace('右侧独立横轴窄列的黑色空心点表示总RMSE', '右侧独立横轴窄列表示总RMSE：灰色空心圆、绿色实心圆和粉色实心圆分别对应三个模型')
    zh_path.write_text(zh, encoding='utf-8')
    integ_path.write_text(clean_integ, encoding="utf-8")

    # 2. Extract standalone Tables, Figures, and Text-only Elsevier Manuscript
    lines = clean_integ.splitlines()
    text_only_lines = []
    tables_only_lines = ["# **Main Tables**", ""]
    figures_only_lines = ["# **Main Figures**", ""]
    fig_captions_end = ["", "## **Figure captions**", ""]

    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if re.match(r"^\*\*Table\s+\d+\.", s):
            tables_only_lines.append(s)
            tables_only_lines.append("")
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith("|") or lines[i].strip().startswith("Note:")):
                tables_only_lines.append(lines[i])
                i += 1
            tables_only_lines.append("")
            continue
        img_m = re.match(r"^!\[(.*?)\]\((.*?)\)$", s)
        if img_m:
            cap, rel_p = img_m.group(1), img_m.group(2)
            figures_only_lines.append(f"![{cap}]({rel_p})")
            figures_only_lines.append("")
            fig_captions_end.append(cap)
            fig_captions_end.append("")
            i += 1
            continue
        text_only_lines.append(lines[i])
        i += 1

    text_only_md = "\n".join(text_only_lines + fig_captions_end).strip() + "\n"
    presub_md_path = MANUSCRIPT_DIR / "SAT_CottonRGB_Manuscript_PreSubmission_20260901.md"
    presub_md_path.write_text(text_only_md, encoding="utf-8")

    tables_md_path = MANUSCRIPT_DIR / "SAT_CottonRGB_Main_Tables_V22.md"
    tables_md_path.write_text("\n".join(tables_only_lines).strip() + "\n", encoding="utf-8")

    figures_md_path = MANUSCRIPT_DIR / "SAT_CottonRGB_Figures_V22.md"
    figures_md_path.write_text("\n".join(figures_only_lines).strip() + "\n", encoding="utf-8")

    fig_cap_md_path = MANUSCRIPT_DIR / "SAT_CottonRGB_Figure_Captions_20260901.md"
    fig_cap_md_path.write_text("# **Figure captions**\n\n" + "\n".join(fig_captions_end[3:]).strip() + "\n", encoding="utf-8")

    # Plain-text synchronized figure captions (.txt)
    plain_caps = re.sub(r"<sup>-1</sup>", "⁻¹", "\n".join(fig_captions_end[3:]).strip())
    plain_caps = re.sub(r"<sup>2</sup>", "²", plain_caps)
    plain_caps = re.sub(r"</?su[bp]>", "", plain_caps)
    plain_caps = plain_caps.replace("\\*", "\x00STAR\x00").replace("**", "").replace("*", "").replace("\x00STAR\x00", "*")
    (MANUSCRIPT_DIR / "SAT_CottonRGB_Figure_Captions_20260901.txt").write_text("Figure captions\n\n" + plain_caps + "\n", encoding="utf-8")

    # 3. Build Supplementary Materials Markdown (full + standalone Supplementary Tables)
    supp_md = build_supplementary_markdown()
    supp_md_path = MANUSCRIPT_DIR / "SAT_CottonRGB_Supplementary_Material_20260901.md"
    supp_md_path.write_text(supp_md, encoding="utf-8")

    supp_tables_only = "# **Supplementary Tables S1-S8**\n\n" + supp_md.split("## **2. Supplementary Tables S1-S8**\n\n", 1)[1]
    supp_tables_md_path = MANUSCRIPT_DIR / "SAT_CottonRGB_Supplementary_Tables_20260901.md"
    supp_tables_md_path.write_text(supp_tables_only, encoding="utf-8")

    abstract = clean_integ.split('## **Abstract**',1)[1].split('## **Keywords**',1)[0].strip()
    zh_summary_md = f"""# 管理条件下基于无人机RGB冠层轨迹与采前吐絮代理的棉花产量预测

## English title

Management-conditioned cotton yield prediction from UAV RGB canopy trajectories and pre-harvest boll proxies

## 中文摘要（内部对照参考）

本研究基于两季三个棉田环境、7个施氮梯度和33幅正射影像，在64个具备共同四时期直接观测的小区上，评价RGB特征超越施氮管理基准的预测增量及其跨环境适用性。PartialRidge保留无惩罚二次施氮响应，并对光学残差正则化。另选39个小区的78幅完整ROI，以2808个分层随机点进行AI提取后作者核验。控制施氮量分类与区组后，吐絮代理与产量的合并偏相关为0.30，P值为0.058。2024 F1区组验证的冠层轨迹RMSE由687.7降至655.6 kg/ha；早期冠层特征将2024 F1至F2邻田RMSE由691.8降至460.2 kg/ha。在相同64小区、四观测日期、ROI像元与来源训练划分下，冠层轨迹在9项任务中的6项优于RGB52，另外3项经验指数更好；2024 F1至2023 F1的对应RMSE分别为948.0和892.7 kg/ha。加入采前吐絮代理在部分任务中进一步降低总误差与同氮水平内误差，但在其他任务中出现损失。误差分解作为辅助诊断，分别描述总体偏差、施氮处理间误差与处理内去均值误差的变化；评价集内去均值不用于实际预测校准，也不直接表征排序准确率或因果机制。2024 F2至2023 F1的冠层轨迹及加入吐絮代理模型虽相对基准改善，预测R²仍为负。六项RMSE较低是描述性计数，并非跨独立环境的总体优越性检验。结论限定于任务相关的RGB信息增量及其迁移边界。

## Abstract

{abstract}

**Keywords:** *Gossypium hirsutum*; Unmanned aerial systems; Nitrogen fertilization; Orthogonalized Ridge regression; Model transferability
"""
    (MANUSCRIPT_DIR / "SAT_CottonRGB_Title_Abstract_Keywords_ZH.md").write_text(zh_summary_md, encoding="utf-8")

    # 4. Build Cover Letter, Highlights, Title Page, and Declaration of Interests
    cover_md = """# **Cover Letter**

Dear Editor-in-Chief,

We are pleased to submit our manuscript entitled "Management-conditioned cotton yield prediction from UAV RGB canopy trajectories and pre-harvest boll proxies" for consideration as a Research Article in *Artificial Intelligence in Agriculture*.

Known nitrogen management provides a strong baseline against which to evaluate the additional predictive value of UAV RGB observations. Across three environments and 64 plots with common direct observations, we tested canopy trajectories and pre-harvest boll features using source-only tuning, within-field block validation, and frozen-model transfer across fields and years. Matched four-date comparisons gave lower RMSE for canopy trajectories in six of nine evaluation roles and for empirical indices in three. Boll features added information in selected transfers, with losses in other tasks. Complementary error decomposition characterized the scales of these gains, while cross-year performance defined their transfer boundaries. Segmentation assessment used AI-extracted labels subsequently verified by the author. Evaluating prediction gains against recorded management and across environments addresses interpretable crop sensing and agricultural machine learning within the scope of *Artificial Intelligence in Agriculture*.

We confirm that this manuscript is our original work, has not been published previously, and is not currently under consideration for publication elsewhere. All authors have approved the manuscript and agree with its submission to your journal.

Thank you for your time and consideration. We look forward to your positive response.

Sincerely,

Zhenggui Zhang, Ph.D., Professor (on behalf of all co-authors)
State Key Laboratory of Cotton Bio-breeding and Integrated Utilization
Institute of Cotton Research, Chinese Academy of Agricultural Sciences
Anyang 455000, Henan, China
Email: [corresponding author email]
"""
    cover_md_path = MANUSCRIPT_DIR / "SAT_CottonRGB_Cover_Letter_AuthorCompletionRequired_20260901.md"
    cover_md_path.write_text(cover_md, encoding="utf-8")

    highlights_md = """# **Highlights**

- PartialRidge tests UAV RGB yield information conditioned on known N management.
- Canopy trajectories cut 2024 error over N and gave lower RMSE than RGB52 (6/9).
- Controlling for N reverses negative boll-yield trends; boll gains vary by domain.
- Orthogonal MSE decomposition separates global bias shifts from within-N error.
- Cross-year gains over the N baseline can coexist with negative predictive R<sup>2</sup>.
"""
    highlights_md_path = MANUSCRIPT_DIR / "SAT_CottonRGB_Highlights_20260901.md"
    highlights_md_path.write_text(highlights_md, encoding="utf-8")

    decl_md = """# **Declaration of Competing Interest**

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in the manuscript entitled "Management-conditioned cotton yield prediction from UAV RGB canopy trajectories and pre-harvest boll proxies".

Authors: Shu-Yang Yu, Cundong Li, Pengcheng Li, Zhanlei Pan, Jian Wang, Guilan Sun, Yaopeng Zhang, Junhong Li, Menghua Zhai, Wenqi Zhao, Xiaolei Ren, Ningning Zhang, Xin Li, and Zhenggui Zhang.
"""
    decl_md_path = MANUSCRIPT_DIR / "Declaration_of_Interests.md"
    decl_md_path.write_text(decl_md, encoding="utf-8")

    title_page_md = """# **Management-conditioned cotton yield prediction from UAV RGB canopy trajectories and pre-harvest boll proxies**

Article type: Research Article

Target journal: *Artificial Intelligence in Agriculture*

## **Authors and affiliations**

Shu-Yang Yu<sup>a,b,c,d</sup>, Cundong Li<sup>c</sup>, Pengcheng Li<sup>a,b,d</sup>, Zhanlei Pan<sup>a,b,d</sup>, Jian Wang<sup>a,b,d</sup>, Guilan Sun<sup>a,b,d</sup>, Yaopeng Zhang<sup>a,b,d</sup>, Junhong Li<sup>a,b,d</sup>, Menghua Zhai<sup>a,b,d</sup>, Wenqi Zhao<sup>a,b,d</sup>, Xiaolei Ren<sup>a,b,d</sup>, Ningning Zhang<sup>a,b,c,d</sup>, Xin Li<sup>a,b,c,d</sup>, Zhenggui Zhang<sup>a,b,d,*</sup>

<sup>a</sup> State Key Laboratory of Cotton Bio-breeding and Integrated Utilization, Institute of Cotton Research of the Chinese Academy of Agricultural Sciences, Anyang, Henan 455000, China

<sup>b</sup> Xinjiang Key Laboratory for Crop Gene Editing and Germplasm Innovation, Institute of Western Agriculture, Chinese Academy of Agricultural Sciences, Changji, Xinjiang 831100, China

<sup>c</sup> College of Agronomy, Hebei Agricultural University, Baoding, Hebei 071001, China

<sup>d</sup> Changji Station, Chinese Agrosystem Long-Term Observation Network, Changji, Xinjiang 831100, China

## **Corresponding author**

\\* Corresponding author: Zhenggui Zhang (Email: [corresponding author email]), State Key Laboratory of Cotton Bio-breeding and Integrated Utilization, Institute of Cotton Research, Chinese Academy of Agricultural Sciences, Anyang 455000, Henan, China.

## **Declarations**

Funding: This research was supported by the Natural Science Foundation of Xinjiang Uygur Autonomous Region (2024D01B91, 2024D01B92), the Tianshan Talent Training Program (2024TSYCQNTJ0026), the Tingzhou Science and Technology Innovation Team (2023CT09), the China Postdoctoral Science Foundation (2024T171024, 2024M763621), and the Chinese Agrosystem Long-Term Observation Network (CALTON-CJ) Changji Station.

Competing interests: The authors declare no conflicts of interest.

Data availability: Processed plot-level optical features, model predictions, 2,808-point stratified verification records, and the code for model-based inference and matched feature comparisons are provided in the accompanying source-data archive. Raw GeoTIFF orthomosaics are available from the corresponding author upon reasonable request.
"""
    title_md_path = MANUSCRIPT_DIR / "SAT_CottonRGB_Title_Page_AuthorCompletionRequired_20260901.md"
    title_md_path.write_text(title_page_md, encoding="utf-8")

    # 5. Audit Integrated & Text-Only Manuscripts via Yu Shuyang Auditor
    rc1 = auditor.audit_manuscript(integ_path)
    rc2 = auditor.audit_manuscript(presub_md_path)
    if rc1 != 0 or rc2 != 0:
        raise RuntimeError("Manuscript failed Yu Shuyang hard audit rules!")

    # 6. Compile all DOCX files into 06_Manuscript and 08_Submission_Package
    compile_md_to_yu_docx(integ_path, MANUSCRIPT_DIR / "20261027_CottonYieldRGB_IntegratedManuscript.docx", landscape=False, img_width_in=6.0)
    compile_md_to_yu_docx(presub_md_path, MANUSCRIPT_DIR / "SAT_CottonRGB_Manuscript_PreSubmission_20260901.docx", landscape=False)
    compile_md_to_yu_docx(tables_md_path, MANUSCRIPT_DIR / "SAT_CottonRGB_Main_Tables_20260901.docx", landscape=True)
    compile_md_to_yu_docx(fig_cap_md_path, MANUSCRIPT_DIR / "SAT_CottonRGB_Figure_Captions_20260901.docx", landscape=False)
    compile_md_to_yu_docx(supp_md_path, MANUSCRIPT_DIR / "SAT_CottonRGB_Supplementary_Material_20260901.docx", landscape=True, img_width_in=7.8)
    compile_md_to_yu_docx(supp_tables_md_path, MANUSCRIPT_DIR / "SAT_CottonRGB_Supplementary_Tables_20260901.docx", landscape=True)
    compile_md_to_yu_docx(cover_md_path, MANUSCRIPT_DIR / "SAT_CottonRGB_Cover_Letter_AuthorCompletionRequired_20260901.docx", landscape=False)
    compile_md_to_yu_docx(highlights_md_path, MANUSCRIPT_DIR / "SAT_CottonRGB_Highlights_20260901.docx", landscape=False)
    compile_md_to_yu_docx(title_md_path, MANUSCRIPT_DIR / "SAT_CottonRGB_Title_Page_AuthorCompletionRequired_20260901.docx", landscape=False)
    compile_md_to_yu_docx(decl_md_path, MANUSCRIPT_DIR / "Declaration_of_Interests.docx", landscape=False)

    # Synchronize 08_Submission_Package
    compile_md_to_yu_docx(presub_md_path, SUBMISSION_DIR / "Manuscript.docx", landscape=False)
    compile_md_to_yu_docx(tables_md_path, SUBMISSION_DIR / "Tables.docx", landscape=True)
    compile_md_to_yu_docx(figures_md_path, SUBMISSION_DIR / "Figures.docx", landscape=False, img_width_in=6.2)
    compile_md_to_yu_docx(supp_md_path, SUBMISSION_DIR / "Supplementary_Materials.docx", landscape=True, img_width_in=7.8)
    compile_md_to_yu_docx(cover_md_path, SUBMISSION_DIR / "Cover_Letter.docx", landscape=False)
    compile_md_to_yu_docx(title_md_path, SUBMISSION_DIR / "Title_Page.docx", landscape=False)
    compile_md_to_yu_docx(decl_md_path, SUBMISSION_DIR / "Declaration_of_Interests.docx", landscape=False)
    compile_md_to_yu_docx(highlights_md_path, SUBMISSION_DIR / "Highlights.docx", landscape=False)

    # Enforce the standing six disabled paragraph options after all authoring.
    opt_path = Path.home() / '.codex/skills/academic-manuscript-editor/scripts/paragraph_options.py'
    opt_spec = importlib.util.spec_from_file_location('current_paragraph_options', opt_path)
    options = importlib.util.module_from_spec(opt_spec)
    opt_spec.loader.exec_module(options)
    for folder in [MANUSCRIPT_DIR, SUBMISSION_DIR]:
        for path in folder.glob('*.docx'):
            if path.name.startswith(('~$', '.~')):
                continue
            options.normalize(path)
            assert options.audit(options.read_package(path)[3])['valid']
    # Identical component files have identical bytes in both supported entries.
    for src, dst in {
        'SAT_CottonRGB_Manuscript_PreSubmission_20260901.docx': 'Manuscript.docx',
        'SAT_CottonRGB_Main_Tables_20260901.docx': 'Tables.docx',
        'SAT_CottonRGB_Supplementary_Material_20260901.docx': 'Supplementary_Materials.docx',
        'SAT_CottonRGB_Cover_Letter_AuthorCompletionRequired_20260901.docx': 'Cover_Letter.docx',
        'SAT_CottonRGB_Highlights_20260901.docx': 'Highlights.docx',
        'SAT_CottonRGB_Title_Page_AuthorCompletionRequired_20260901.docx': 'Title_Page.docx',
        'Declaration_of_Interests.docx': 'Declaration_of_Interests.docx',
    }.items():
        shutil.copyfile(MANUSCRIPT_DIR / src, SUBMISSION_DIR / dst)
    shutil.copyfile(SUBMISSION_DIR / 'Figures.docx', MANUSCRIPT_DIR / 'Figures.docx')

    # 7. Update MANUSCRIPT_BUILD_MANIFEST.json
    manifest_outputs = {}
    for p in sorted(MANUSCRIPT_DIR.glob("*.docx")):
        if p.name.startswith(('~$', '.~')):
            continue
        data = p.read_bytes()
        manifest_outputs[p.name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    submission_outputs = {}
    for p in sorted(SUBMISSION_DIR.glob("*.docx")):
        if p.name.startswith(('~$', '.~')):
            continue
        data = p.read_bytes()
        submission_outputs[p.name] = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    manifest = {
        "status": "V22_PRESUBMISSION_MANUSCRIPT_SUITE_BUILT",
        "build_date": "2026-10-28",
        "title": "Management-conditioned cotton yield prediction from UAV RGB canopy trajectories and pre-harvest boll proxies",
        "lifecycle_status": "CURRENT_N64_MATCHED_REANALYSIS / DOCX_QA_RECORDED_SEPARATELY / NOT_SUBMITTED",
        "target_journal": "Artificial Intelligence in Agriculture",
        "direct_support_plots": 64,
        "stratified_verification_points": 2808,
        "main_figures": 6,
        "main_tables": 3,
        "supplementary_figures": 5,
        "supplementary_tables": 8,
        "reference_style": "Harvard author-year (35 bidirectionally verified entries)",
        "manuscript_dir_docx_count": len(manifest_outputs),
        "submission_package_docx_count": len(submission_outputs),
        "outputs": manifest_outputs,
        "submission_package_outputs": submission_outputs,
    }
    (MANUSCRIPT_DIR / "MANUSCRIPT_BUILD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("[OK] Updated MANUSCRIPT_BUILD_MANIFEST.json")


if __name__ == "__main__":
    main()
