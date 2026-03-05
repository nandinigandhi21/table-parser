"""
pdf_parser.py  ← run this
Layout-aware PDF → Markdown parser.

Architecture:
    pdfplumber  → tables only (geometry-based, reliable for complex/cross-page tables)
    Docling     → text layout (headings, paragraphs, lists, captions — semantically
                  classified). Falls back to pdfplumber+column-detection if Docling
                  is not installed.
    layout_merger → combines both by (page, col, y-position) into one Markdown

Usage:
    python pdf_parser.py <path_to_pdf> [output_dir]

Output files in <output_dir>/:
    full_document.md  — layout-aware markdown (text + tables in reading order)
    tables_only.md    — only tables
    text_only.md      — only text blocks
"""

import sys
from pathlib import Path
from collections import Counter

from table_extractor import extract_tables
from text_extractor  import extract_text_blocks
from layout_merger   import build_markdown, KIND_TO_MD


# ── Build table zones from all segments of all tables ────────────────────────

def build_table_zones(tables: list[dict]) -> dict[int, list[tuple]]:
    """
    Build {page_num: [(top, bottom), ...]} covering EVERY page each table spans.
    Used to suppress table cell text from the text extractor.
    """
    zones: dict[int, list[tuple]] = {}
    for t in tables:
        for seg in t.get("segments", [{"page": t["page"],
                                        "top" : t["top"],
                                        "bottom": t["bottom"]}]):
            zones.setdefault(seg["page"], []).append((seg["top"], seg["bottom"]))
    return zones


# ── Markdown-only renderers ───────────────────────────────────────────────────

def tables_to_markdown(tables: list[dict]) -> str:
    parts = []
    for i, t in enumerate(tables, 1):
        parts.append(f"## Table {i}  *(page {t['page']})*\n")
        parts.append(t["markdown"])
        parts.append("")
    return "\n".join(parts)


def text_to_markdown(text_blocks: list[dict]) -> str:
    parts = []
    for b in text_blocks:
        template = KIND_TO_MD.get(b["kind"], "{text}")
        parts.append(template.format(text=b["text"].strip()))
        parts.append("")
    return "\n".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str, output_dir: str = "output") -> None:
    pdf_path    = Path(pdf_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n📄 PDF      : {pdf_path.name}")
    print( "🔍 Tables   : pdfplumber (geometry-based)")
    print( "🔍 Text     : Docling if installed, else pdfplumber+column-detection\n")

    # ── Step 1: Extract tables with pdfplumber ────────────────────────────────
    print("Step 1 — Extracting tables (pdfplumber)...")
    tables      = extract_tables(str(pdf_path))
    table_zones = build_table_zones(tables)
    n_zones     = sum(len(v) for v in table_zones.values())
    print(f"         Found {len(tables)} logical table(s) across "
          f"{len(table_zones)} page(s) ({n_zones} zone(s) total)")

    # ── Step 2: Extract text, suppressing table zones ─────────────────────────
    print("Step 2 — Extracting text blocks...")
    text_blocks = extract_text_blocks(str(pdf_path), table_zones)
    kind_counts = Counter(b["kind"] for b in text_blocks)
    print(f"         Found {len(text_blocks)} text block(s): "
          + ", ".join(f"{k}={v}" for k, v in kind_counts.items()))

    # ── Step 3: Merge into layout-aware Markdown ──────────────────────────────
    print("Step 3 — Merging into layout-aware Markdown...\n")
    full_md   = build_markdown(text_blocks, tables)
    tables_md = tables_to_markdown(tables)
    text_md   = text_to_markdown(text_blocks)

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    outputs = {
        "full_document.md": full_md,
        "tables_only.md"  : tables_md,
        "text_only.md"    : text_md,
    }
    for filename, content in outputs.items():
        path = output_path / filename
        path.write_text(content, encoding="utf-8")
        print(f"  ✅ {path}")

    print(f"\n📦 Done → {output_path}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:   python pdf_parser.py <path_to_pdf> [output_dir]")
        print("Example: python pdf_parser.py report.pdf output/")
        sys.exit(1)

    parse_pdf(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output")