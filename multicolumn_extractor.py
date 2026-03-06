"""
pdf_to_markdown.py
━━━━━━━━━━━━━━━━━━
Convert a PDF to Markdown.

Text + layout + reading order → Docling  (doc.export_to_markdown())
Tables                        → pdfplumber (handles complex/cross-page better)

Docling's export_to_markdown already handles:
  - multi-column reading order
  - headings, lists, captions
  - correct document structure

We only override tables: suppress Docling's table output and inject
pdfplumber's cleaner table markdown at the right positions.

Usage:
    python pdf_to_markdown.py input.pdf [output_dir]
"""

from __future__ import annotations
import re
import sys
from pathlib import Path

import pdfplumber
import pandas as pd
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions


# ══════════════════════════════════════════════════════════════════════════════
#  TABLES  —  pdfplumber
# ══════════════════════════════════════════════════════════════════════════════

def _clean_cell(val) -> str:
    if val is None:
        return ""
    return " ".join(str(val).replace("\n", " ").split())


def _is_paragraph_block(rows: list[list]) -> bool:
    if not rows or {len(r) for r in rows} != {1}:
        return False
    return (sum(len(str(r[0])) for r in rows) / len(rows)) > 40


def _header_repeats(row, prev_header) -> bool:
    if len(row) != len(prev_header):
        return False
    matches = sum(
        _clean_cell(a).lower() == _clean_cell(b).lower()
        for a, b in zip(row, prev_header)
    )
    return matches >= len(row) * 0.6


def _rows_to_markdown(rows: list[list]) -> str:
    cleaned = [[_clean_cell(c) for c in row] for row in rows]
    width   = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    df = pd.DataFrame(cleaned[1:], columns=cleaned[0])
    df = df.loc[:, (df != "").any(axis=0)]
    return df.to_markdown(index=False)


def extract_tables(pdf_path: str) -> list[dict]:
    """
    Extract tables with position info.
    Returns [{page, top, bottom, segments, markdown}, ...]
    """
    raw = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for t in page.find_tables():
                rows = t.extract()
                if not rows or _is_paragraph_block(rows):
                    continue
                raw.append({
                    "page": page_num, "top": t.bbox[1],
                    "bottom": t.bbox[3], "rows": rows,
                })

    if not raw:
        return []

    # Merge cross-page continuations
    groups = [{"rows": raw[0]["rows"],
               "segs": [{"page": raw[0]["page"], "top": raw[0]["top"],
                         "bottom": raw[0]["bottom"]}]}]
    for i in range(1, len(raw)):
        prev, curr = raw[i - 1], raw[i]
        same_width = len(prev["rows"][0]) == len(curr["rows"][0])
        close_page = curr["page"] - prev["page"] <= 1
        repeats    = _header_repeats(curr["rows"][0], prev["rows"][0])
        if same_width and close_page and not repeats:
            groups[-1]["rows"].extend(curr["rows"])
            groups[-1]["segs"].append({"page": curr["page"], "top": curr["top"],
                                       "bottom": curr["bottom"]})
        else:
            groups.append({"rows": curr["rows"],
                           "segs": [{"page": curr["page"], "top": curr["top"],
                                     "bottom": curr["bottom"]}]})

    return [
        {
            "page"    : g["segs"][0]["page"],
            "top"     : g["segs"][0]["top"],
            "bottom"  : g["segs"][-1]["bottom"],
            "segments": g["segs"],
            "markdown": _rows_to_markdown(g["rows"]),
        }
        for g in groups
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_pdf(pdf_path: str, output_dir: str = "output") -> None:
    pdf_path    = Path(pdf_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n📄 PDF    : {pdf_path.name}")
    print( "🔍 Text   : Docling  (export_to_markdown)")
    print( "🔍 Tables : pdfplumber\n")

    # Step 1: Docling converts the whole document
    print("Step 1 — Converting with Docling...")
    opts = PdfPipelineOptions()
    opts.do_ocr = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    doc = converter.convert(str(pdf_path)).document
    docling_md = doc.export_to_markdown()
    print("         Done.")

    # Step 2: pdfplumber extracts tables
    print("Step 2 — Extracting tables (pdfplumber)...")
    tables = extract_tables(str(pdf_path))
    print(f"         {len(tables)} table(s) found")

    # Step 3: Build outputs
    print("Step 3 — Building Markdown...\n")

    if tables:
        # Replace Docling's table blocks with pdfplumber's versions (in order)
        table_pattern = re.compile(r'(\|[^\n]+\|\n)+', re.MULTILINE)
        pdfplumber_mds = [t["markdown"] for t in tables]
        idx = [0]

        def replacer(m):
            if idx[0] < len(pdfplumber_mds):
                out = pdfplumber_mds[idx[0]] + "\n"
                idx[0] += 1
                return out
            return m.group(0)

        full_md = table_pattern.sub(replacer, docling_md)
    else:
        full_md = docling_md

    tables_md = "\n\n".join(
        f"## Table {i}  *(page {t['page']})*\n\n{t['markdown']}"
        for i, t in enumerate(tables, 1)
    ) or "*No tables found.*"

    text_only_md = re.sub(r'(\|[^\n]+\|\n)+', '', docling_md).strip()

    # Step 4: Save
    for filename, content in {
        "full_document.md": full_md,
        "tables_only.md"  : tables_md,
        "text_only.md"    : text_only_md,
    }.items():
        p = output_path / filename
        p.write_text(content, encoding="utf-8")
        print(f"  ✅ {p}")

    print(f"\n📦 Done → {output_path}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:   python pdf_to_markdown.py <input.pdf> [output_dir]")
        sys.exit(1)
    parse_pdf(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output")