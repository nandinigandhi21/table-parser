"""
borderless_table_extractor.py
Extracts borderless (whitespace-aligned) tables from digital PDFs using
pdfplumber word-coordinate analysis.

Algorithm:
  1. Extract all words with bounding boxes from each page.
  2. Group words into visual rows (words within ROW_SNAP_PT vertically).
  3. Split rows into contiguous blocks separated by vertical gaps.
  4. For each block, detect column boundaries by clustering word x0
     positions that appear consistently across multiple rows.
  5. Assign every word to a column, build a 2D grid.
  6. Merge adjacent columns that are spuriously split (e.g. "X-Ray" → two cols).
  7. Merge orphan wrapped rows (e.g. "Pathology\\nSlides" → "Pathology Slides").
  8. Apply filters to reject false positives (paragraph/body-column text).
  9. Convert valid grids to pandas DataFrames and Markdown strings.

Standalone usage:
  python borderless_table_extractor.py <pdf_path>

Integration with table_extractor.py:
  from borderless_table_extractor import extract_borderless_tables
  tables = extract_borderless_tables("report.pdf", excluded_zones={...})
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

ROW_SNAP_PT         = 4.0   # words within this many pts vertically → same row
COL_SNAP_PT         = 6.0   # x0 positions within this many pts → same column
MIN_COL_PRESENCE    = 0.50  # column boundary must appear in ≥ 50% of rows
MIN_COLS            = 2     # minimum columns to qualify as a table
MIN_ROWS            = 2     # minimum data rows (excluding header)
MAX_ROWS            = 30    # tables rarely exceed 30 rows; more = body text
MAX_CELL_LEN        = 60    # cells longer than this are likely paragraph text
MAX_LONG_CELL_RATIO = 0.20  # if >20% of cells are long → reject as paragraph
MIN_SHORT_CELL_RATIO= 0.50  # ≥50% of filled cells must be ≤30 chars
MIN_COL_CONSISTENCY = 0.70  # ≥70% of rows must use most columns
COL_MERGE_GAP_PT    = 50    # adjacent columns closer than this are candidates


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — ROW GROUPING
# ══════════════════════════════════════════════════════════════════════════════

def _group_into_rows(words: list[dict]) -> list[list[dict]]:
    """
    Group words into visual rows by clustering on their 'top' coordinate.
    Words within ROW_SNAP_PT of each other vertically belong to the same row.
    """
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    rows: list[list[dict]] = []
    current_row: list[dict] = [sorted_words[0]]
    current_top: float = sorted_words[0]["top"]
    for w in sorted_words[1:]:
        if abs(w["top"] - current_top) <= ROW_SNAP_PT:
            current_row.append(w)
        else:
            rows.append(sorted(current_row, key=lambda x: x["x0"]))
            current_row = [w]
            current_top = w["top"]
    if current_row:
        rows.append(sorted(current_row, key=lambda x: x["x0"]))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — SPLIT INTO CONTIGUOUS BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

def _find_table_blocks(rows: list[list[dict]]) -> list[list[list[dict]]]:
    """
    Split page rows into contiguous blocks. A vertical gap larger than
    1.5× median row height signals a block boundary.
    """
    if not rows:
        return []
    if len(rows) == 1:
        return [rows]
    row_tops = [min(w["top"]    for w in row) for row in rows]
    row_bots = [max(w["bottom"] for w in row) for row in rows]
    heights  = [row_bots[i] - row_tops[i] for i in range(len(rows))]
    median_h = sorted(heights)[len(heights) // 2] if heights else 12.0
    gap_threshold = max(median_h * 1.5, ROW_SNAP_PT * 3)
    blocks: list[list[list[dict]]] = []
    current: list[list[dict]] = [rows[0]]
    for i in range(1, len(rows)):
        if row_tops[i] - row_bots[i - 1] > gap_threshold:
            blocks.append(current)
            current = [rows[i]]
        else:
            current.append(rows[i])
    if current:
        blocks.append(current)
    return blocks


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — COLUMN BOUNDARY DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _detect_column_boundaries(rows: list[list[dict]]) -> list[float]:
    """
    Find x0 positions that appear consistently across rows — left edges of columns.
    Cluster nearby x0s and keep those present in ≥ MIN_COL_PRESENCE of rows.
    """
    n_rows = len(rows)
    if n_rows == 0:
        return []
    all_x0s = sorted(float(w["x0"]) for row in rows for w in row)
    if not all_x0s:
        return []
    clusters: list[list[float]] = []
    current: list[float] = [all_x0s[0]]
    for x in all_x0s[1:]:
        if x - current[-1] <= COL_SNAP_PT:
            current.append(x)
        else:
            clusters.append(current)
            current = [x]
    clusters.append(current)
    boundaries: list[float] = []
    for cluster in clusters:
        rep = sorted(cluster)[len(cluster) // 2]
        count = sum(
            1 for row in rows
            if any(abs(float(w["x0"]) - rep) <= COL_SNAP_PT for w in row)
        )
        if count / n_rows >= MIN_COL_PRESENCE:
            boundaries.append(rep)
    return sorted(boundaries)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — ASSIGN WORDS TO COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def _assign_columns(rows: list[list[dict]],
                    boundaries: list[float]) -> list[list[str]]:
    """
    Assign each word to the column whose boundary is the largest x0 ≤ word.x0.
    Returns a 2D list of strings: grid[row][col].
    """
    n_cols = len(boundaries)
    grid: list[list[str]] = []
    for row in rows:
        cells: list[list[str]] = [[] for _ in range(n_cols)]
        for w in row:
            col_idx = 0
            for i, bx in enumerate(boundaries):
                if float(w["x0"]) >= bx - COL_SNAP_PT:
                    col_idx = i
            cells[col_idx].append(w["text"])
        grid.append([" ".join(cell_words).strip() for cell_words in cells])
    return grid


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — MERGE SPURIOUSLY SPLIT COLUMNS
# ══════════════════════════════════════════════════════════════════════════════

def _merge_split_columns(grid: list[list[str]],
                          boundaries: list[float]
                          ) -> tuple[list[list[str]], list[float]]:
    """
    Merge adjacent columns that are close together (gap < COL_MERGE_GAP_PT)
    AND where one of the two is mostly empty (< 40% filled).

    This fixes cases like "X-Ray" being split into "X" | "Ray" because
    the hyphenated word tokens have different x0 positions.
    """
    if len(boundaries) < 2:
        return grid, boundaries
    changed = True
    while changed:
        changed = False
        for i in range(len(boundaries) - 1):
            if boundaries[i + 1] - boundaries[i] > COL_MERGE_GAP_PT:
                continue
            n_rows = len(grid)
            left_filled  = sum(
                1 for row in grid if i < len(row) and row[i].strip()
            ) / n_rows
            right_filled = sum(
                1 for row in grid if i+1 < len(row) and row[i+1].strip()
            ) / n_rows
            if min(left_filled, right_filled) < 0.40:
                new_grid = []
                for row in grid:
                    new_row = list(row)
                    lv = new_row[i]     if i   < len(new_row) else ""
                    rv = new_row[i + 1] if i+1 < len(new_row) else ""
                    new_row[i] = (lv + " " + rv).strip()
                    if i + 1 < len(new_row):
                        new_row.pop(i + 1)
                    new_grid.append(new_row)
                grid = new_grid
                boundaries = boundaries[:i + 1] + boundaries[i + 2:]
                changed = True
                break
    return grid, boundaries


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — MERGE ORPHAN WRAPPED ROWS
# ══════════════════════════════════════════════════════════════════════════════

def _merge_orphan_rows(grid: list[list[str]]) -> list[list[str]]:
    """
    Merge orphan rows — rows where only one cell has content.
    This happens when a multi-word cell value wraps to a new line in the PDF
    (e.g. "Pathology" on one line, "Slides" on the next line below it).
    The orphan word is appended to the same-column cell in the previous row.
    """
    if len(grid) < 2:
        return grid
    result: list[list[str]] = [grid[0]]
    for row in grid[1:]:
        filled = [(i, c) for i, c in enumerate(row) if c.strip()]
        if len(filled) == 1:
            col_idx, val = filled[0]
            prev = result[-1]
            if col_idx < len(prev) and prev[col_idx].strip():
                new_prev = list(prev)
                new_prev[col_idx] = (new_prev[col_idx] + " " + val).strip()
                result[-1] = new_prev
                continue
        result.append(row)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — FALSE POSITIVE FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def _looks_like_header(row: list[str]) -> bool:
    """
    Heuristic: first row is a header if cells are short and not purely numeric.
    """
    filled = [c for c in row if c.strip()]
    if not filled:
        return False
    avg_len = sum(len(c) for c in filled) / len(filled)
    if avg_len > 40:
        return False
    numeric = sum(
        1 for c in filled if re.fullmatch(r'[\d\.\,\%\$\-\+]+', c.strip())
    )
    return numeric / len(filled) <= 0.6


def _is_valid_table(grid: list[list[str]], n_cols: int) -> bool:
    """
    Return True if the grid looks like a real table.
    Rejects paragraph text, body columns, and degenerate grids.
    """
    n_rows = len(grid)
    if n_rows < MIN_ROWS + 1:
        return False
    if n_cols < MIN_COLS:
        return False
    if n_rows > MAX_ROWS:
        return False

    # Column consistency
    consistent = sum(
        1 for row in grid
        if sum(1 for c in row if c.strip()) >= max(2, n_cols - 1)
    )
    if consistent / n_rows < MIN_COL_CONSISTENCY:
        return False

    all_cells = [c for row in grid for c in row if c.strip()]
    if not all_cells:
        return False

    # Reject if too many long cells
    long_cells = sum(1 for c in all_cells if len(c) > MAX_CELL_LEN)
    if long_cells / len(all_cells) > MAX_LONG_CELL_RATIO:
        return False

    # Reject if not enough short cells
    short_cells = sum(1 for c in all_cells if len(c) <= 30)
    if short_cells / len(all_cells) < MIN_SHORT_CELL_RATIO:
        return False

    # Average words per cell
    avg_words = sum(len(c.split()) for c in all_cells) / len(all_cells)
    if avg_words > 8:
        return False

    # Sentence-fragment detection
    _FRAG_ENDINGS = (
        ',', ' and', ' the', ' of', ' in', ' to',
        ' a', ' is', ' or', ' for', ' on', ' by', ' with', ' at',
    )
    fragment_cells = sum(
        1 for c in all_cells if any(c.endswith(e) for e in _FRAG_ENDINGS)
    )
    if fragment_cells / len(all_cells) > 0.15:
        return False

    # At least two columns must be well-filled
    filled_cols = sum(
        1 for col_idx in range(n_cols)
        if sum(
            1 for row in grid
            if col_idx < len(row) and row[col_idx].strip()
        ) / n_rows >= 0.4
    )
    if filled_cols < MIN_COLS:
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — BUILD DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════

def _grid_to_dataframe(grid: list[list[str]], n_cols: int) -> pd.DataFrame:
    """
    Convert a 2D string grid to a pandas DataFrame.
    Uses first row as header if it looks like one, else generates Col1, Col2...
    """
    padded = [row + [""] * (n_cols - len(row)) for row in grid]
    if _looks_like_header(padded[0]):
        headers = [h.strip() if h.strip() else f"Col{i+1}"
                   for i, h in enumerate(padded[0])]
        data = padded[1:]
    else:
        headers = [f"Col{i+1}" for i in range(n_cols)]
        data = padded
    df = pd.DataFrame(data, columns=headers)
    return df.loc[:, (df != "").any(axis=0)]


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def extract_borderless_tables(
    pdf_path: str,
    excluded_zones: dict[int, list[tuple]] | None = None,
) -> list[dict]:
    """
    Extract borderless (whitespace-aligned) tables from a PDF.

    Args:
        pdf_path       : Path to the PDF file.
        excluded_zones : {page_num: [(top, bottom), ...]} regions to skip.
                         Pass bordered table bboxes here to avoid duplicates.

    Returns:
        List of dicts sorted by (page, top):
            {page, top, bottom, x0, x1, dataframe, markdown, method}
    """
    pdf_path       = Path(pdf_path)
    excluded_zones = excluded_zones or {}
    results: list[dict] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size", "fontname"])
            if not words:
                continue

            # Exclude words inside known bordered-table zones
            excl = excluded_zones.get(page_num, [])
            if excl:
                words = [
                    w for w in words
                    if not any(
                        min(float(w["bottom"]), ez_bot)
                        - max(float(w["top"]), ez_top) > 2
                        for ez_top, ez_bot in excl
                    )
                ]
            if not words:
                continue

            all_rows = _group_into_rows(words)
            if not all_rows:
                continue

            for block_rows in _find_table_blocks(all_rows):
                if len(block_rows) < MIN_ROWS + 1:
                    continue

                boundaries = _detect_column_boundaries(block_rows)
                n_cols = len(boundaries)
                if n_cols < MIN_COLS:
                    continue

                grid = _assign_columns(block_rows, boundaries)
                grid, boundaries = _merge_split_columns(grid, boundaries)
                n_cols = len(boundaries)
                if n_cols < MIN_COLS:
                    continue

                grid = _merge_orphan_rows(grid)
                if len(grid) < MIN_ROWS + 1:
                    continue

                if not _is_valid_table(grid, n_cols):
                    continue

                block_words = [w for row in block_rows for w in row]
                tbl_top    = min(float(w["top"])    for w in block_words)
                tbl_bottom = max(float(w["bottom"]) for w in block_words)
                tbl_x0     = min(float(w["x0"])     for w in block_words)
                tbl_x1     = max(float(w["x1"])     for w in block_words)

                df = _grid_to_dataframe(grid, n_cols)
                if df.empty or len(df.columns) < MIN_COLS:
                    continue

                results.append({
                    "page"     : page_num,
                    "top"      : tbl_top,
                    "bottom"   : tbl_bottom,
                    "x0"       : tbl_x0,
                    "x1"       : tbl_x1,
                    "dataframe": df,
                    "markdown" : df.to_markdown(index=False),
                    "method"   : "borderless",
                })

    results.sort(key=lambda t: (t["page"], t["top"]))
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE TEST ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "borderless_tables-continued.pdf"
    print(f"\n🔍 Borderless table extraction: {pdf}\n")
    tables = extract_borderless_tables(pdf)
    if not tables:
        print("No borderless tables found.")
    else:
        print(f"Found {len(tables)} borderless table(s):\n")
        for i, t in enumerate(tables, 1):
            print(f"{'─' * 60}")
            print(f"[{i}] Page {t['page']} | "
                  f"top={t['top']:.1f}  bottom={t['bottom']:.1f} | "
                  f"cols={t['dataframe'].shape[1]}  rows={t['dataframe'].shape[0]}")
            print()
            print(t["markdown"])
            print()