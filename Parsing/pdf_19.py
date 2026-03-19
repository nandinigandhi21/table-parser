"""
pdf_parser_hybrid.py
────────────────────
Hybrid PDF → Markdown parser combining:
  • Docling   — primary engine for text, layout, reading order, simple tables
  • pdfplumber (table_extractor.py)            — complex bordered tables
  • borderless_table_extractor.py              — borderless tables Docling misses
  • PyMuPDF   (image_extractor.py)             — embedded images with captions

Architecture:
  Step 1  Docling converts the PDF → base markdown
  Step 2  pdfplumber extracts ALL bordered tables with proper geometry
  Step 3  borderless_table_extractor finds whitespace-aligned tables
  Step 4  PyMuPDF extracts all embedded images + captions
  Step 5  Compare each pdfplumber table against Docling's version:
            • If pdfplumber's table is richer → replace Docling's version
            • If Docling's version is fine → keep it
  Step 6  Inject borderless tables at their correct page positions
  Step 7  Inject image references with captions at their correct positions
  Step 8  Write full_document.md

Usage:
  python pdf_parser_hybrid.py <pdf_path> [output_dir]

Output:
  <output_dir>/full_document.md   — final hybrid markdown
  <output_dir>/images/            — all extracted image files

Requirements:
  pip install docling pdfplumber pandas tabulate pymupdf pillow
"""

from __future__ import annotations

import re
import sys
import warnings
import logging
from pathlib import Path

import pandas as pd

from table_extractor            import extract_tables
from borderless_table_extractor import extract_borderless_tables
from image_extractor            import extract_images

# ── Silence all third-party noise ────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


# ══════════════════════════════════════════════════════════════════════════════
#  DOCLING CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def _docling_available() -> bool:
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        return False


def convert_with_docling(pdf_path: str) -> tuple[str, list[dict]]:
    """
    Convert PDF using Docling.

    Returns:
        (markdown_str, docling_tables)
        docling_tables: list of {page, top, bottom, markdown, n_cols, n_rows}
    """
    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr             = False
    opts.do_table_structure = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

    result   = converter.convert(pdf_path)
    doc      = result.document
    markdown = doc.export_to_markdown()

    # Extract Docling table positions for comparison
    docling_tables: list[dict] = []
    page_heights: dict[int, float] = {}
    for page_no, page_obj in doc.pages.items():
        try:
            page_heights[int(page_no)] = float(page_obj.size.height)
        except Exception:
            page_heights[int(page_no)] = 842.0

    for item, _ in doc.iterate_items():
        label = getattr(item, "label", None)
        if label is None:
            continue
        label_str = label.value if hasattr(label, "value") else str(label)
        if label_str != "table":
            continue
        try:
            prov    = item.prov[0]
            page_no = int(prov.page_no)
            bbox    = prov.bbox
            h       = page_heights.get(page_no, 842.0)
            top     = h - bbox.t
            bottom  = h - bbox.b
            top, bottom = min(top, bottom), max(top, bottom)
        except Exception:
            continue

        # Get the exported markdown for this table
        try:
            tbl_md = item.export_to_dataframe().to_markdown(index=False)
            n_cols = item.export_to_dataframe().shape[1]
            n_rows = item.export_to_dataframe().shape[0]
        except Exception:
            tbl_md = ""
            n_cols = 0
            n_rows = 0

        docling_tables.append({
            "page"    : page_no,
            "top"     : top,
            "bottom"  : bottom,
            "markdown": tbl_md,
            "n_cols"  : n_cols,
            "n_rows"  : n_rows,
        })

    result = None; doc = None; converter = None  # cleanup
    return markdown, docling_tables


# ══════════════════════════════════════════════════════════════════════════════
#  TABLE QUALITY SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _table_quality_score(df: pd.DataFrame) -> float:
    """
    Score a DataFrame's quality 0.0–1.0.
    Higher = more complete, fewer empty cells, better column names.

    Used to decide whether pdfplumber's table is better than Docling's.
    """
    if df.empty:
        return 0.0

    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 0.0

    # Penalise empty cells
    empty_cells = (df == "").sum().sum() + df.isnull().sum().sum()
    fill_ratio  = 1.0 - (empty_cells / total_cells)

    # Reward proper column names (not "Col1", "Col2"...)
    generic_headers = sum(
        1 for c in df.columns
        if re.fullmatch(r'Col\d+|Unnamed.*|\d+', str(c))
    )
    header_score = 1.0 - (generic_headers / max(len(df.columns), 1))

    # Reward more columns (more structure preserved)
    col_score = min(df.shape[1] / 10, 1.0)

    return fill_ratio * 0.5 + header_score * 0.3 + col_score * 0.2


def _pdfplumber_beats_docling(pp_df: pd.DataFrame,
                               dl_table: dict) -> bool:
    """
    Return True if the pdfplumber table is significantly better than
    Docling's version of the same table.

    Criteria:
      • pdfplumber has more columns (Docling merged/missed columns)
      • pdfplumber has fewer empty cells
      • pdfplumber's quality score is meaningfully higher
    """
    try:
        dl_df = _markdown_to_df(dl_table["markdown"])
    except Exception:
        return True  # Docling couldn't even produce valid markdown → use pdfplumber

    pp_score = _table_quality_score(pp_df)
    dl_score = _table_quality_score(dl_df)

    # pdfplumber wins if:
    # 1. It has more columns (captured structure Docling missed)
    more_cols = pp_df.shape[1] > dl_df.shape[1]
    # 2. Its quality score is meaningfully better (>10% improvement)
    better_quality = pp_score > dl_score + 0.10

    return more_cols or better_quality


def _markdown_to_df(md: str) -> pd.DataFrame:
    """Parse a markdown table string into a DataFrame."""
    lines = [l for l in md.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return pd.DataFrame()

    def parse_row(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    headers = parse_row(lines[0])
    rows    = []
    for line in lines[2:]:   # skip separator line
        if re.match(r'^\|[-| :]+\|?$', line.strip()):
            continue
        rows.append(parse_row(line))

    if not rows:
        return pd.DataFrame(columns=headers)

    # Pad rows to header width
    w = len(headers)
    rows = [r + [""] * (w - len(r)) for r in rows]
    rows = [r[:w] for r in rows]
    return pd.DataFrame(rows, columns=headers)


# ══════════════════════════════════════════════════════════════════════════════
#  TABLE MATCHING  (pdfplumber ↔ Docling by page + position)
# ══════════════════════════════════════════════════════════════════════════════

def _bbox_overlap_ratio(a_top: float, a_bot: float,
                         b_top: float, b_bot: float) -> float:
    """Return vertical overlap ratio between two bboxes (0–1)."""
    overlap = min(a_bot, b_bot) - max(a_top, b_top)
    if overlap <= 0:
        return 0.0
    union = max(a_bot, b_bot) - min(a_top, b_top)
    return overlap / union if union > 0 else 0.0


def _match_tables(pp_tables: list[dict],
                   dl_tables: list[dict],
                   overlap_threshold: float = 0.5
                   ) -> list[tuple[dict, dict | None]]:
    """
    For each pdfplumber table, find the best-matching Docling table
    on the same page by vertical bbox overlap.

    Returns list of (pp_table, dl_table_or_None).
    """
    pairs: list[tuple[dict, dict | None]] = []

    for pp in pp_tables:
        best_dl   = None
        best_ratio = 0.0

        for dl in dl_tables:
            if dl["page"] != pp["page"]:
                continue
            ratio = _bbox_overlap_ratio(
                pp["top"], pp["bottom"], dl["top"], dl["bottom"]
            )
            if ratio > best_ratio:
                best_ratio = ratio
                best_dl    = dl

        if best_ratio >= overlap_threshold:
            pairs.append((pp, best_dl))
        else:
            pairs.append((pp, None))   # no Docling match → pdfplumber only

    return pairs


# ══════════════════════════════════════════════════════════════════════════════
#  MARKDOWN TABLE BLOCK DETECTION
# ══════════════════════════════════════════════════════════════════════════════

# Matches a complete markdown table block (one or more header+separator+data rows)
_MD_TABLE_RE = re.compile(
    r'(\|[^\n]+\|\n\|[-| :]+\|\n(?:\|[^\n]+\|\n?)*)',
    re.MULTILINE,
)


def _extract_md_tables(markdown: str) -> list[tuple[int, int, str]]:
    """
    Find all markdown table blocks in a string.
    Returns list of (start_pos, end_pos, table_text).
    """
    results = []
    for m in _MD_TABLE_RE.finditer(markdown):
        results.append((m.start(), m.end(), m.group(0)))
    return results


def _tables_similar(md1: str, md2: str, threshold: float = 0.6) -> bool:
    """
    Return True if two markdown table strings are similar enough
    to be the same table (same number of columns, similar headers).
    """
    try:
        df1 = _markdown_to_df(md1)
        df2 = _markdown_to_df(md2)
    except Exception:
        return False

    if df1.empty or df2.empty:
        return False

    # Same column count
    if df1.shape[1] != df2.shape[1]:
        return False

    # Similar headers
    headers1 = [str(c).strip().lower() for c in df1.columns]
    headers2 = [str(c).strip().lower() for c in df2.columns]
    matches  = sum(a == b for a, b in zip(headers1, headers2))
    return matches / len(headers1) >= threshold


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4  —  REPLACE DOCLING TABLES WITH PDFPLUMBER WHERE BETTER
# ══════════════════════════════════════════════════════════════════════════════

def _inject_better_tables(markdown: str,
                           pp_tables: list[dict],
                           dl_tables: list[dict]) -> tuple[str, int]:
    """
    For each pdfplumber table that is better than Docling's version,
    find the corresponding markdown table block and replace it.

    Returns (updated_markdown, n_replacements).
    """
    pairs        = _match_tables(pp_tables, dl_tables)
    replacements = 0
    result_md    = markdown

    for pp_tbl, dl_tbl in pairs:
        if dl_tbl is None:
            # No Docling match — this table is completely missing from Docling's
            # output. It will be injected in Step 5 instead.
            continue

        pp_df = pp_tbl["dataframe"]
        if not _pdfplumber_beats_docling(pp_df, dl_tbl):
            continue   # Docling's version is fine — keep it

        # Find the Docling table block in the markdown by content similarity
        md_tables = _extract_md_tables(result_md)
        best_pos: tuple[int, int] | None = None
        best_sim  = 0.0

        for start, end, tbl_text in md_tables:
            sim = 1.0 if _tables_similar(tbl_text, dl_tbl["markdown"]) else 0.0
            if sim > best_sim:
                best_sim = sim
                best_pos = (start, end)

        if best_pos is None or best_sim < 0.6:
            continue   # Can't find it in the markdown — skip

        # Replace with pdfplumber's version
        new_table_md  = pp_tbl["markdown"]
        result_md     = (result_md[:best_pos[0]]
                         + new_table_md
                         + result_md[best_pos[1]:])
        replacements += 1

    return result_md, replacements


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5  —  INJECT BORDERLESS TABLES
# ══════════════════════════════════════════════════════════════════════════════

def _build_page_position_index(markdown: str) -> list[tuple[int, int]]:
    """
    Build a list of (char_pos, page_num) from page break markers in the markdown.
    Docling inserts markers like '<!-- Page N -->' or uses section structure.
    We use a heuristic: split on major section starts.

    Returns list of (char_position, page_number) sorted by char_position.
    Falls back to a single entry (0, 1) if no markers found.
    """
    # Try Docling-style page markers: <!-- page N --> or \n---\n*Page N*
    markers: list[tuple[int, int]] = []

    for pattern in [
        r'<!--\s*[Pp]age\s+(\d+)\s*-->',
        r'\*Page\s+(\d+)\*',
        r'<!-- Page break -->',
    ]:
        for m in re.finditer(pattern, markdown):
            try:
                page_num = int(m.group(1)) if m.lastindex else 0
                if page_num > 0:
                    markers.append((m.start(), page_num))
            except (ValueError, IndexError):
                pass

    if markers:
        markers.sort()
        return markers

    # No markers found — return a single anchor at position 0 for page 1
    return [(0, 1)]


def _find_insert_position(markdown: str,
                           target_page: int,
                           page_index: list[tuple[int, int]]) -> int:
    """
    Find the character position in the markdown where content from
    target_page should be inserted.

    Strategy: find the last markdown table block before the next page starts,
    and insert after it. If no table found, insert at the page start position.
    """
    # Find the start of target_page content
    page_start = 0
    page_end   = len(markdown)

    for i, (pos, pnum) in enumerate(page_index):
        if pnum == target_page:
            page_start = pos
            if i + 1 < len(page_index):
                page_end = page_index[i + 1][0]
            break
    else:
        # Page not in index — append at end
        return len(markdown)

    # Find the last table block within this page's range
    md_tables = _extract_md_tables(markdown[page_start:page_end])
    if md_tables:
        last_table_end = md_tables[-1][1]
        return page_start + last_table_end

    # No table on this page — insert at end of page section
    return page_end


def _inject_borderless_tables(markdown: str,
                                borderless: list[dict],
                                existing_tables: list[dict]) -> tuple[str, int]:
    """
    Insert borderless tables into the markdown at their correct page positions.

    Each borderless table is inserted after the last existing table on its page,
    or at the end of the page's content section if no tables exist on that page.

    Tables already covered by pdfplumber bordered extraction are skipped
    (they would have been passed as excluded_zones to the borderless extractor).

    Returns (updated_markdown, n_inserted).
    """
    if not borderless:
        return markdown, 0

    page_index = _build_page_position_index(markdown)
    result_md  = markdown
    inserted   = 0
    offset     = 0   # track position shifts as we insert text

    # Process borderless tables in page order
    for tbl in sorted(borderless, key=lambda t: (t["page"], t["top"])):
        tbl_md   = tbl["markdown"]
        tbl_page = tbl["page"]

        # Build the insertion block with a clear label
        insert_block = f"\n\n{tbl_md}\n\n"

        # Find insert position in the CURRENT (shifted) markdown
        current_page_index = _build_page_position_index(result_md)
        insert_pos = _find_insert_position(result_md, tbl_page,
                                            current_page_index)
        insert_pos += offset

        result_md = (result_md[:insert_pos]
                     + insert_block
                     + result_md[insert_pos:])
        offset   += len(insert_block)
        inserted += 1

    return result_md, inserted


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — INJECT IMAGES
# ══════════════════════════════════════════════════════════════════════════════

def _build_img_md(img: dict) -> str:
    """
    Build the markdown block for a single image.

    Format:
        ![Figure N](images/pageN_figN.png)
        *Full caption text.*

    Short alt text (Figure N) so screen readers and search get the label.
    Full caption as italic — exactly ONE occurrence in the document.
    """
    caption = img.get("caption", "").strip()
    path    = img["path"]
    page    = img["page"]

    m   = re.match(r'(fig(?:ure)?\s*\d+)', caption, re.IGNORECASE)
    alt = m.group(1) if m else f"Figure (page {page})"

    block = f"![{alt}]({path})"
    if caption:
        block += f"\n*{caption}*"
    return block


# ── Patterns for Docling caption placement ───────────────────────────────────
#
# Docling places figure captions in two ways depending on PDF type:
#
#   VECTOR figures (caption BEFORE placeholder):
#       Figure N. Caption text...
#
#       <!-- image -->
#
#   RASTER figures (caption AFTER placeholder):
#       <!-- image -->
#
#       Figure: Caption text    or    Caption text
#
# We remove both patterns to prevent the caption appearing twice —
# once as a plain paragraph and once inside our *italic* image block.

_CAP_BEFORE_IMG = re.compile(
    r'((?:^.+\n)+)'        # caption paragraph (one or more lines)
    r'(\n)'                # blank line
    r'(<!-- image -->)',    # placeholder
    re.MULTILINE,
)

_CAP_AFTER_IMG = re.compile(
    r'(<!-- image -->)'    # placeholder
    r'(\n\n)'              # blank line
    r'((?:^.+\n)+)',       # caption paragraph (one or more lines)
    re.MULTILINE,
)

_FIG_CAPTION_RE = re.compile(
    r'^(fig(?:ure)?[\s\:\.]|figure\s*\:)', re.IGNORECASE
)


def _remove_docling_caption(markdown: str) -> str:
    """
    Remove figure caption paragraphs that Docling emits alongside
    <!-- image --> placeholders, so the caption only appears once —
    inside our injected *italic* image block.

    Handles both orderings:
      • caption BEFORE placeholder (vector figures)
      • caption AFTER  placeholder (raster/embedded images)
    """

    def _strip_before(m: re.Match) -> str:
        cap_text    = m.group(1)
        blank       = m.group(2)
        placeholder = m.group(3)
        first_line  = cap_text.lstrip().split('\n')[0].strip()
        if _FIG_CAPTION_RE.match(first_line):
            return blank + placeholder     # remove caption, keep placeholder
        return m.group(0)

    def _strip_after(m: re.Match) -> str:
        placeholder = m.group(1)
        blank       = m.group(2)
        cap_text    = m.group(3)
        first_line  = cap_text.lstrip().split('\n')[0].strip()
        if _FIG_CAPTION_RE.match(first_line):
            return placeholder + blank     # remove caption, keep placeholder
        return m.group(0)

    result = _CAP_BEFORE_IMG.sub(_strip_before, markdown)
    result = _CAP_AFTER_IMG.sub(_strip_after,  result)
    return result


def _remove_injected_caption_duplicates(markdown: str,
                                         images: list[dict]) -> str:
    """
    After image injection, scan for any remaining loose caption paragraphs
    that exactly match an injected image's caption text and remove them.

    This catches captions that Docling placed far enough from the placeholder
    that the regex patterns above didn't catch them.
    """
    result = markdown
    for img in images:
        caption = img.get("caption", "").strip()
        if not caption:
            continue
        # Build a pattern that matches the caption as a standalone paragraph
        # (surrounded by blank lines) but NOT when it's inside *...*
        escaped = re.escape(caption)
        # Match the caption as a plain paragraph (not preceded by * or ![)
        pattern = re.compile(
            r'(?<!\*)\n\n' + escaped + r'\n\n',
            re.MULTILINE,
        )
        result = pattern.sub('\n\n', result)
    return result


def _inject_images(markdown: str,
                   images: list[dict]) -> tuple[str, int]:
    """
    Inject image blocks into the markdown, replacing Docling's
    <!-- image --> placeholders.

    Docling emits captions either before or after the placeholder
    depending on PDF type. Both are removed so the caption only
    appears once — inside the *italic* line of our image block.
    """
    if not images:
        return markdown, 0

    # Step 0: remove Docling's standalone caption paragraphs
    # (handles both before-placeholder and after-placeholder patterns)
    result_md   = _remove_docling_caption(markdown)
    inserted    = 0
    placeholder = "<!-- image -->"
    sorted_imgs = sorted(images, key=lambda x: (x["page"], x["top"]))
    placed_paths: set[str] = set()

    # ── Step A: Replace <!-- image --> placeholders in page order ─────────
    for img in sorted_imgs:
        if placeholder not in result_md:
            break
        result_md = result_md.replace(placeholder, _build_img_md(img), 1)
        placed_paths.add(img["path"])
        inserted += 1

    # ── Step B: Fallback for images with no matching placeholder ──────────
    for img in sorted_imgs:
        if img["path"] in placed_paths:
            continue
        img_block  = "\n\n" + _build_img_md(img) + "\n"
        page_index = _build_page_position_index(result_md)
        insert_pos = _find_insert_position(result_md, img["page"], page_index)
        result_md  = result_md[:insert_pos] + img_block + result_md[insert_pos:]
        placed_paths.add(img["path"])
        inserted  += 1

    # Step C: Final safety pass — remove any remaining loose caption
    # paragraphs that exactly match an injected image caption
    result_md = _remove_injected_caption_duplicates(result_md, images)

    return result_md, inserted


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def parse_pdf(pdf_path: str, output_dir: str = "output") -> dict:
    """
    Full hybrid pipeline: Docling + pdfplumber + borderless + images.

    Args:
        pdf_path   : Path to the PDF file.
        output_dir : Output folder.

    Returns:
        {
          "markdown"           : final full markdown string,
          "n_docling_replaced" : how many Docling tables were replaced,
          "n_borderless"       : how many borderless tables were injected,
          "n_images"           : how many images were extracted,
          "bordered_tables"    : list of pdfplumber table dicts,
          "borderless_tables"  : list of borderless table dicts,
          "images"             : list of image dicts,
        }
    """
    pdf_path    = Path(pdf_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ── Step 1: Docling ───────────────────────────────────────────────────────
    if not _docling_available():
        raise ImportError("Docling is not installed. Run: pip install docling")

    base_markdown, docling_tables = convert_with_docling(str(pdf_path))

    # ── Step 2: pdfplumber bordered tables ───────────────────────────────────
    bordered = extract_tables(str(pdf_path))

    # ── Step 3: Borderless tables ─────────────────────────────────────────────
    excl_zones: dict[int, list[tuple]] = {}
    for t in bordered:
        for seg in t.get("segments", [{"page": t["page"],
                                        "top" : t["top"],
                                        "bottom": t["bottom"]}]):
            excl_zones.setdefault(seg["page"], []).append(
                (seg["top"], seg["bottom"])
            )
    borderless = extract_borderless_tables(str(pdf_path),
                                            excluded_zones=excl_zones)

    # ── Step 4: Extract images ────────────────────────────────────────────────
    images = extract_images(str(pdf_path), output_dir=str(output_path))

    # ── Step 5: Replace Docling tables where pdfplumber is better ─────────────
    markdown, n_replaced = _inject_better_tables(
        base_markdown, bordered, docling_tables
    )

    # ── Step 6: Inject borderless tables ─────────────────────────────────────
    markdown, n_injected = _inject_borderless_tables(
        markdown, borderless, bordered
    )

    # ── Step 7: Inject images ─────────────────────────────────────────────────
    markdown, n_images = _inject_images(markdown, images)

    # ── Step 8: Save output ───────────────────────────────────────────────────
    (output_path / "full_document.md").write_text(markdown, encoding="utf-8")

    return {
        "markdown"           : markdown,
        "n_docling_replaced" : n_replaced,
        "n_borderless"       : n_injected,
        "n_images"           : n_images,
        "bordered_tables"    : bordered,
        "borderless_tables"  : borderless,
        "images"             : images,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:   python pdf_parser_hybrid.py <pdf_path> [output_dir]")
        print("Example: python pdf_parser_hybrid.py report.pdf output/")
        sys.exit(1)

    out = sys.argv[2] if len(sys.argv) > 2 else "output"
    result = parse_pdf(sys.argv[1], out)

    print(f"✅ Done → {out}/full_document.md  "
          f"| tables replaced: {result['n_docling_replaced']} "
          f"| borderless: {result['n_borderless']} "
          f"| images: {result['n_images']}")