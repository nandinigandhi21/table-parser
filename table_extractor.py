"""
table_extractor.py
Extracts all tables from a PDF using pdfplumber.
Returns each table as a dict with:
  - page     : page number (1-based)
  - top      : y-coordinate of table top (for layout ordering)
  - bottom   : y-coordinate of table bottom
  - dataframe: cleaned pandas DataFrame
  - markdown : table rendered as Markdown string
"""

import re
import pdfplumber
import pandas as pd


# ── Cell cleaning ─────────────────────────────────────────────────────────────

def fix_wrapped_word(text: str) -> str:
    """
    Fix mid-word line-breaks caused by narrow PDF columns.
    e.g. 'Amo\\nunt' → 'Amount',  'NET\\nINCOME' → 'NET INCOME'
    """
    if text is None:
        return ""
    parts = str(text).split("\n")
    result = parts[0]
    for part in parts[1:]:
        if result and part:
            last_char  = result[-1]
            first_char = part[0]
            joined_len = len(result) + len(part)
            if "_" in result or "@" in result:
                result += part
            elif last_char.isdigit() and re.match(r"[\w]", first_char):
                result += part
            elif first_char.islower() and joined_len <= 8:
                result += part
            else:
                result += " " + part
        else:
            result += " " + part
    return result.strip()


def clean_cell(val) -> str:
    return " ".join(fix_wrapped_word(val).split())


# ── False-positive filter ─────────────────────────────────────────────────────

def is_paragraph_block(rows: list[list]) -> bool:
    """Single-column block where every cell is a long sentence → not a table."""
    if not rows:
        return False
    if {len(r) for r in rows} != {1}:
        return False
    avg_len = sum(len(str(r[0])) for r in rows) / len(rows)
    return avg_len > 40


# ── Cross-page continuation ───────────────────────────────────────────────────

def looks_like_header_repeat(row: list, prev_header: list) -> bool:
    if len(row) != len(prev_header):
        return False
    matches = sum(
        clean_cell(a).lower() == clean_cell(b).lower()
        for a, b in zip(row, prev_header)
    )
    return matches >= len(row) * 0.6


def is_continuation(prev_rows, curr_rows, page_diff: int) -> bool:
    if page_diff > 1:
        return False
    pw = len(prev_rows[0]) if prev_rows else 0
    cw = len(curr_rows[0]) if curr_rows else 0
    if pw != cw or pw == 0:
        return False
    if looks_like_header_repeat(curr_rows[0], prev_rows[0]):
        return False
    return True


# ── DataFrame builder ─────────────────────────────────────────────────────────

def rows_to_df(rows: list[list]) -> pd.DataFrame:
    cleaned = [[clean_cell(c) for c in row] for row in rows]
    width   = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (width - len(r)) for r in cleaned]
    df = pd.DataFrame(cleaned[1:], columns=cleaned[0])
    return df.loc[:, (df != "").any(axis=0)]


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_tables(pdf_path: str) -> list[dict]:
    """
    Extract all logical tables from a PDF.

    Returns list of dicts:
        {page, top, bottom, dataframe, markdown}
    """
    raw: list[dict] = []   # {page, top, bottom, rows}

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for t in page.find_tables():
                rows = t.extract()
                if not rows or is_paragraph_block(rows):
                    continue
                raw.append({
                    "page"  : page_num,
                    "top"   : t.bbox[1],    # y-top of table bbox
                    "bottom": t.bbox[3],    # y-bottom of table bbox
                    "rows"  : rows,
                })

    if not raw:
        return []

    # Merge cross-page continuations
    # Each logical table tracks ALL segments (page, top, bottom) it spans
    first   = raw[0]
    logical: list[dict] = [{
        "segments": [{"page": first["page"], "top": first["top"], "bottom": first["bottom"]}],
        "all_rows": first["rows"],
    }]

    for i in range(1, len(raw)):
        prev, curr = raw[i - 1], raw[i]
        if is_continuation(prev["rows"], curr["rows"], curr["page"] - prev["page"]):
            logical[-1]["all_rows"].extend(curr["rows"])
            logical[-1]["segments"].append({
                "page": curr["page"], "top": curr["top"], "bottom": curr["bottom"],
            })
        else:
            logical.append({
                "segments": [{"page": curr["page"], "top": curr["top"], "bottom": curr["bottom"]}],
                "all_rows": curr["rows"],
            })

    # Build output dicts
    results = []
    for entry in logical:
        df       = rows_to_df(entry["all_rows"])
        segments = entry["segments"]
        results.append({
            "page"     : segments[0]["page"],
            "top"      : segments[0]["top"],
            "bottom"   : segments[-1]["bottom"],
            "segments" : segments,
            "dataframe": df,
            "markdown" : df.to_markdown(index=False),
        })

    return results