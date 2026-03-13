"""
pdf_parser.py  ← run this
Layout-aware PDF → Markdown parser.

Architecture:
    pdfplumber      → tables  (geometry-based, reliable for complex/cross-page tables)
    Docling         → text layout (headings, paragraphs, lists, captions).
                      Falls back to pdfplumber+column-detection if Docling
                      is not installed.
    PyMuPDF (fitz)  → images  (embedded image extraction with position info)
    Docling         → formulas (formula enrichment pipeline → LaTeX output)
    layout_merger   → combines all four by (page, y-position) into one Markdown

Usage:
    python pdf_parser.py <path_to_pdf> [output_dir]

Output in <output_dir>/:
    full_document.md      — layout-aware markdown (text + tables + images + formulas)
    tables_only.md        — only tables
    text_only.md          — only text blocks
    images_only.md        — only image references
    formulas_only.md      — only extracted formulas (LaTeX)
    images/               — all extracted image files
        page1_img1.png
        page2_img1.png
        ...
"""

import sys
from pathlib import Path
from collections import Counter

from table_extractor   import extract_tables
from text_extractor    import extract_text_blocks
from image_extractor   import extract_images
from formula_extractor import extract_formulas, formula_extraction_available
from layout_merger     import build_markdown, KIND_TO_MD


# ── Build table zones ─────────────────────────────────────────────────────────

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
    if not tables:
        return "*No tables found.*"
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


def images_to_markdown(images: list[dict]) -> str:
    if not images:
        return "*No images found.*"
    parts = []
    for img in images:
        caption = img.get("caption", "").strip()
        alt     = caption if caption else (
            f"Page {img['page']} — Image {img['index']} "
            f"({img['width']}×{img['height']}px)"
        )
        parts.append(f"![{alt}]({img['path']})")
        if caption:
            parts.append(f"*{caption}*")
        parts.append("")
    return "\n".join(parts)


def formulas_to_markdown(formulas: list[dict]) -> str:
    if not formulas:
        return "*No formulas found.*"
    parts = []
    for i, f in enumerate(formulas, 1):
        parts.append(f"### Formula {i}  *(page {f['page']})*\n")
        latex = f.get("latex", "").strip()
        text  = f.get("text",  "").strip()
        if latex:
            parts.append(f"$$\n{latex}\n$$")
        else:
            parts.append(f"`{text}`")
        parts.append("")
    return "\n".join(parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str, output_dir: str = "output") -> None:
    pdf_path    = Path(pdf_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n📄 PDF       : {pdf_path.name}")
    print( "🔍 Tables    : pdfplumber (geometry-based)")
    print( "🔍 Text      : Docling if installed, else pdfplumber+column-detection")
    print( "🔍 Images    : PyMuPDF")
    print( "🔍 Formulas  : Docling (formula enrichment)\n")

    # ── Step 1: Extract tables ────────────────────────────────────────────────
    print("Step 1 — Extracting tables (pdfplumber)...")
    tables      = extract_tables(str(pdf_path))
    table_zones = build_table_zones(tables)
    n_zones     = sum(len(v) for v in table_zones.values())
    print(f"         Found {len(tables)} logical table(s) across "
          f"{len(table_zones)} page(s) ({n_zones} zone(s) total)")

    # ── Step 2: Extract text blocks ───────────────────────────────────────────
    print("Step 2 — Extracting text blocks...")
    text_blocks = extract_text_blocks(str(pdf_path), table_zones)
    kind_counts = Counter(b["kind"] for b in text_blocks)
    print(f"         Found {len(text_blocks)} text block(s): "
          + ", ".join(f"{k}={v}" for k, v in kind_counts.items()))

    # ── Step 3: Extract images ────────────────────────────────────────────────
    print("Step 3 — Extracting images (PyMuPDF)...")
    images = extract_images(str(pdf_path), output_dir=str(output_path))
    print(f"         Found {len(images)} image(s) → saved to {output_path}/images/")

    # ── Step 4: Extract formulas ──────────────────────────────────────────────
    print("Step 4 — Extracting formulas (locate → crop → LaTeX)...")
    try:
        formulas = extract_formulas(str(pdf_path), output_dir=str(output_path))
    except Exception as e:
        print(f"         Formula extraction failed ({type(e).__name__}: {e}), skipping")
        formulas = []

    # ── Step 4b: Suppress text blocks that overlap formula zones ─────────────
    # Formula text (e.g. '3x 2 + 5y = 6') leaks as plain text blocks because
    # text_extractor runs before formula_extractor. Post-filter here now that
    # we have formula bboxes.
    if formulas:
        formula_zones: dict[int, list[tuple]] = {}
        for f in formulas:
            # Expand zone downward to capture nearby sub/superscripts and
            # annotation artefacts (footnote numbers, noise glyphs) that sit
            # just outside the formula bbox but belong to the same region.
            formula_zones.setdefault(f["page"], []).append(
                (f["top"] - 5, f["bottom"] + 30)
            )

        def _in_formula_zone(page: int, top: float, bottom: float) -> bool:
            for ft, fb in formula_zones.get(page, []):
                if min(bottom, fb) - max(top, ft) > 2:
                    return True
            return False

        before      = len(text_blocks)
        text_blocks = [b for b in text_blocks
                       if not _in_formula_zone(b["page"], b["top"], b["bottom"])]
        suppressed  = before - len(text_blocks)
        if suppressed:
            print(f"         Suppressed {suppressed} text block(s) inside formula zones")

    # ── Step 5: Merge into layout-aware Markdown ──────────────────────────────
    print("Step 5 — Merging into layout-aware Markdown...\n")
    full_md     = build_markdown(text_blocks, tables, images, formulas)
    tables_md   = tables_to_markdown(tables)
    text_md     = text_to_markdown(text_blocks)
    images_md   = images_to_markdown(images)
    formulas_md = formulas_to_markdown(formulas)

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    outputs = {
        "full_document.md" : full_md,
        "tables_only.md"   : tables_md,
        "text_only.md"     : text_md,
        "images_only.md"   : images_md,
        "formulas_only.md" : formulas_md,
    }
    for filename, content in outputs.items():
        path = output_path / filename
        path.write_text(content, encoding="utf-8")
        print(f"  ✅ {path}")

    print(f"\n📦 Done → {output_path}/")
    print(f"   images/  : {len(images)} file(s)")
    print(f"   tables   : {len(tables)}")
    print(f"   text blk : {len(text_blocks)}")
    print(f"   formulas : {len(formulas)}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:   python pdf_parser.py <path_to_pdf> [output_dir]")
        print("Example: python pdf_parser.py report.pdf output/")
        sys.exit(1)

    parse_pdf(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output")