"""
text_extractor.py

PRIMARY  : Docling layout model — handles multi-column, semantic heading
           classification, lists, captions, footnotes, reading order.
FALLBACK : pdfplumber with column detection — used automatically if Docling
           is not installed.

Both backends accept table_zones {page: [(top, bottom), ...]} from pdfplumber
to suppress any text that physically overlaps with a known table area.

Returns each block as a dict:
    {page, top, bottom, kind, text, col}
    kind in {TITLE, H1, H2, H3, PARAGRAPH, LIST_ITEM, CAPTION, FOOTNOTE}
"""

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED: TABLE ZONE SUPPRESSION
# ══════════════════════════════════════════════════════════════════════════════

def _inside_table(page: int, top: float, bottom: float,
                  table_zones: dict[int, list[tuple]]) -> bool:
    """
    Return True if this text block overlaps with any known table zone.
    Uses > 2pt overlap threshold to avoid suppressing adjacent captions.
    """
    for t_top, t_bottom in table_zones.get(page, []):
        overlap = min(bottom, t_bottom) - max(top, t_top)
        if overlap > 2:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  DOCLING BACKEND
# ══════════════════════════════════════════════════════════════════════════════

def _docling_available() -> bool:
    try:
        import docling  # noqa: F401
        return True
    except ImportError:
        return False


def _extract_with_docling(pdf_path: str,
                          table_zones: dict[int, list[tuple]]) -> list[dict]:
    """
    Extract text blocks using Docling's layout model.
    table_zones suppresses text that lives inside pdfplumber-detected tables.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = False  # tables handled by pdfplumber
    pipeline_options.do_ocr             = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    try:
        result = converter.convert(pdf_path)
        doc    = result.document

        # Page height map (Docling uses bottom-left origin; we flip to top-down)
        page_heights: dict[int, float] = {}
        for page_no, page_obj in doc.pages.items():
            try:
                page_heights[int(page_no)] = float(page_obj.size.height)
            except Exception:
                page_heights[int(page_no)] = 842.0

        # Label → kind mapping
        _LABEL_MAP = {
            "title"       : "TITLE",
            "text"        : "PARAGRAPH",
            "list_item"   : "LIST_ITEM",
            "caption"     : "CAPTION",
            "footnote"    : "FOOTNOTE",
            "formula"     : "PARAGRAPH",
            "code"        : "PARAGRAPH",
            "page_header" : None,   # skip
            "page_footer" : None,   # skip
            "page_number" : None,   # skip
            "table"       : None,   # skip — pdfplumber handles
            "picture"     : None,   # skip
        }

        def label_to_kind(item) -> str | None:
            label_val = getattr(item, "label", None)
            if label_val is None:
                return None
            label_str = label_val.value if hasattr(label_val, "value") else str(label_val)
            if label_str == "section_header":
                level = getattr(item, "level", 1)
                return f"H{min(level, 3)}"
            return _LABEL_MAP.get(label_str, "PARAGRAPH")

        def get_pos(item, heights):
            try:
                prov    = item.prov[0]
                page_no = prov.page_no
                bbox    = prov.bbox
                h       = heights.get(page_no, 842.0)
                # Docling bbox.t / bbox.b are in bottom-left origin → flip
                top    = h - bbox.t
                bottom = h - bbox.b
                return page_no, min(top, bottom), max(top, bottom)
            except Exception:
                return None

        blocks = []
        skipped = 0
        for item, _level in doc.iterate_items():
            kind = label_to_kind(item)
            if kind is None:
                continue
            text = getattr(item, "text", "").strip()
            if not text:
                continue
            pos = get_pos(item, page_heights)
            if pos is None:
                continue
            page_no, top, bottom = pos

            # ── Suppress text inside known table zones ────────────────────────────
            if _inside_table(page_no, top, bottom, table_zones):
                skipped += 1
                continue

            blocks.append({
                "page": page_no, "top": top, "bottom": bottom,
                "kind": kind,    "text": text, "col": 0,
            })

        if skipped:
            print(f"         Suppressed {skipped} text block(s) inside table zones")

        return blocks
    finally:
        # Ensure proper cleanup to avoid memory leaks
        if 'result' in locals():
            result = None
        if 'doc' in locals():
            doc = None
        if 'converter' in locals():
            converter = None


# ══════════════════════════════════════════════════════════════════════════════
#  PDFPLUMBER FALLBACK WITH MULTI-COLUMN DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _detect_columns(words: list[dict], page_width: float,
                    min_words_per_col: int = 3) -> list[float]:
    """
    Detect column boundary x-coordinates by finding significant gaps
    in the x0 distribution of words.

    A gap qualifies as a column separator only when:
      1. Gap width > 6% of page width
      2. At least `min_words_per_col` words exist on EACH side of the gap

    Returns sorted list of x-positions that mark the START of each column.
    e.g. [57.0, 318.0] = two columns starting at x=57 and x=318.
    """
    COLUMN_GAP_RATIO = 0.06   # gap > 6% of page width = column separator candidate

    min_gap = page_width * COLUMN_GAP_RATIO
    x0s     = sorted(set(round(w["x0"]) for w in words if w.get("x0") is not None))
    if not x0s:
        return [0.0]

    # Find candidate gaps
    candidate_splits = []
    for i in range(1, len(x0s)):
        if x0s[i] - x0s[i - 1] >= min_gap:
            split_x = float(x0s[i])
            # Count words on each side
            left_count  = sum(1 for w in words if w["x0"] <  split_x)
            right_count = sum(1 for w in words if w["x0"] >= split_x)
            if left_count >= min_words_per_col and right_count >= min_words_per_col:
                candidate_splits.append(split_x)

    col_starts = [float(x0s[0])] + candidate_splits
    return col_starts


def _assign_column(x0: float, col_starts: list[float]) -> int:
    col = 0
    for i, start in enumerate(col_starts):
        if x0 >= start - 2:
            col = i
    return col


def _build_size_tiers(pdf_path: str) -> dict[float, str]:
    """
    Map font sizes to heading tiers.
    Most-frequent size = body (PARAGRAPH). Larger sizes = heading tiers.
    """
    import pdfplumber
    from collections import Counter

    counts: Counter = Counter()
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for ch in page.chars:
                counts[round(ch["size"], 1)] += 1

    if not counts:
        return {}

    body_size = max(
        (s for s in counts if s >= 6),
        key=lambda s: counts[s],
    )
    heading_sizes = sorted([s for s in counts if s > body_size], reverse=True)
    tier_names    = ["TITLE", "H1", "H2", "H3"]

    tiers: dict[float, str] = {}
    for i, size in enumerate(heading_sizes):
        tiers[size] = tier_names[i] if i < len(tier_names) else "H3"
    for size in counts:
        if size <= body_size:
            tiers[size] = "BODY"
    return tiers


def _extract_with_pdfplumber(pdf_path: str,
                              table_zones: dict[int, list[tuple]]) -> list[dict]:
    """
    Fallback text extractor with column-aware reading order.
    Suppresses words inside table_zones before processing.
    """
    import re
    import pdfplumber

    tiers = _build_size_tiers(pdf_path)

    def get_kind(size: float, fontname: str) -> str:
        tier    = tiers.get(round(size, 1), "BODY")
        is_bold = bool(re.search(r"bold|Black|Heavy|Semibold", fontname, re.IGNORECASE))
        if tier in ("TITLE", "H1", "H2", "H3"):
            return tier
        return "BOLD" if is_bold else "PARAGRAPH"

    def group_into_lines(words: list[dict], gap: float = 3.0) -> list[list[dict]]:
        if not words:
            return []
        lines, cur = [], [words[0]]
        for w in words[1:]:
            if abs(w["top"] - cur[-1]["top"]) <= gap:
                cur.append(w)
            else:
                lines.append(sorted(cur, key=lambda x: x["x0"]))
                cur = [w]
        lines.append(sorted(cur, key=lambda x: x["x0"]))
        return lines

    def line_text(line: list[dict]) -> str:
        result = ""
        for i, w in enumerate(line):
            if i > 0 and (w["x0"] - line[i-1]["x1"]) > 1.5:
                result += " "
            result += w["text"]
        return result.strip()

    all_blocks: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size", "fontname"])
            if not words:
                continue

            # ── Filter out words inside table zones ───────────────────────────
            zones = table_zones.get(page_num, [])
            if zones:
                words = [
                    w for w in words
                    if not _inside_table(page_num, w["top"], w["bottom"], table_zones)
                ]
            if not words:
                continue

            # ── Detect columns using only non-table words ─────────────────────
            # IMPORTANT: use filtered words so table cell x-positions don't
            # create false column boundaries
            col_starts = _detect_columns(words, page.width)
            n_cols     = len(col_starts)

            for w in words:
                w["_col"] = _assign_column(w["x0"], col_starts)

            # ── Group lines per column ────────────────────────────────────────
            line_tuples = []
            for col_idx in range(n_cols):
                col_words = sorted(
                    [w for w in words if w["_col"] == col_idx],
                    key=lambda w: (round(w["top"], 1), w["x0"]),
                )
                for line in group_into_lines(col_words):
                    if not line:
                        continue
                    top    = min(w["top"]    for w in line)
                    bottom = max(w["bottom"] for w in line)
                    text   = line_text(line)
                    if not text.strip():
                        continue
                    dominant = max(line, key=lambda w: w["size"])
                    kind = get_kind(dominant["size"], dominant["fontname"])
                    line_tuples.append((text, kind, top, bottom, col_idx))

            # ── Merge consecutive same-kind/col lines into blocks ─────────────
            if not line_tuples:
                continue

            ct, ck, ctop, cbot, ccol = line_tuples[0]
            for text, kind, top, bottom, col in line_tuples[1:]:
                if kind == ck and col == ccol and (top - cbot) < 10.0:
                    ct   += " " + text
                    cbot  = bottom
                else:
                    all_blocks.append({
                        "page": page_num, "top": ctop, "bottom": cbot,
                        "kind": ck, "text": ct, "col": ccol,
                    })
                    ct, ck, ctop, cbot, ccol = text, kind, top, bottom, col

            all_blocks.append({
                "page": page_num, "top": ctop, "bottom": cbot,
                "kind": ck, "text": ct, "col": ccol,
            })

    return all_blocks


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def extract_text_blocks(pdf_path: str,
                        table_zones: dict[int, list[tuple]] | None = None
                        ) -> list[dict]:
    """
    Extract all text blocks with correct multi-column reading order.
    Text inside known table areas is suppressed via table_zones.

    Args:
        pdf_path   : Path to the PDF file.
        table_zones: {page_num: [(top, bottom), ...]} of table areas to suppress.
                     Build this from table_extractor.extract_tables() segments.

    Returns list of dicts:
        {page, top, bottom, kind, text, col}
    """
    table_zones = table_zones or {}

    if _docling_available():
        try:
            print("         Backend: Docling (semantic layout + multi-column)")
            return _extract_with_docling(pdf_path, table_zones)
        except Exception as e:
            print(f"         Docling failed ({type(e).__name__}: {e}), falling back to pdfplumber")
            return _extract_with_pdfplumber(pdf_path, table_zones)
    else:
        print("         Backend: pdfplumber (column-detection fallback)")
        return _extract_with_pdfplumber(pdf_path, table_zones)