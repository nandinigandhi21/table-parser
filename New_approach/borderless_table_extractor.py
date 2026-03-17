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
  6. Merge adjacent columns that are spuriously split:
       A) One column is sparse (< 40% filled), OR
       B) Right column always co-occurs with left column in the same row
          (e.g. "Chest" + "X-Ray" always appear together → same logical column)
  7. Merge orphan wrapped rows (e.g. "Pathology" + "Slides" on next line).
  8. Apply filters to reject false positives (paragraph/body-column text).
     Special case: small blocks at page top are kept as cross-page continuation
     candidates even if they have fewer rows than the minimum.
  9. Cross-page merge: consecutive tables on adjacent pages with matching
     column headers and no new header row are merged into one logical table.
  10. Convert valid grids to pandas DataFrames and Markdown strings.

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
PAGE_TOP_PT         = 150   # blocks starting within this y from top = page-top


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — ROW GROUPING
# ══════════════════════════════════════════════════════════════════════════════

def _group_into_rows(words: list[dict]) -> list[list[dict]]:
    """Group words into visual rows by clustering on 'top' coordinate."""
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
    """Split rows into blocks using vertical gap > 1.5× median row height."""
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
    """Find x0 positions appearing in ≥ MIN_COL_PRESENCE of rows."""
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
    """Assign each word to its column. Returns grid[row][col]."""
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
    Merge adjacent columns that belong to the same logical column.

    Condition A — one column is sparse (< 40% filled).
    Condition B — every row that has content in the RIGHT column also has
                  content in the LEFT column (co-occurrence = same logical column).
                  This catches "Chest" + "X-Ray", "ViT" + "Base" etc.
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

            # Condition A: sparse column
            cond_a = min(left_filled, right_filled) < 0.40

            # Condition B: right always co-occurs with left
            right_rows = [
                row for row in grid if i+1 < len(row) and row[i+1].strip()
            ]
            cond_b = (
                len(right_rows) > 0
                and all(i < len(row) and row[i].strip() for row in right_rows)
            )

            if cond_a or cond_b:
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
    Merge single-content rows into the previous row's same column.
    Fixes PDF line-wrapping: "Pathology" on line N, "Slides" on line N+1.
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
    """True if row looks like a table header: short, mostly non-numeric."""
    filled = [c for c in row if c.strip()]
    if not filled:
        return False
    if sum(len(c) for c in filled) / len(filled) > 40:
        return False
    numeric = sum(
        1 for c in filled if re.fullmatch(r'[\d\.\,\%\$\-\+\/]+', c.strip())
    )
    return numeric / len(filled) <= 0.6


def _is_page_top_continuation(grid: list[list[str]], tbl_top: float,
                               page_num: int = 1) -> bool:
    """
    True if this small block is likely the tail of a cross-page table:
    - NOT on page 1
    - Within PAGE_TOP_PT of page top
    - Very few rows (≤ 3) — a genuine continuation is just leftover rows
    - No long cells, no sentence-fragment endings
    """
    if page_num <= 1:
        return False
    if tbl_top >= PAGE_TOP_PT:
        return False
    if len(grid) > 3:   # more than 3 rows = probably its own table, not a tail
        return False
    if len(grid) < 1:
        return False
    all_cells = [c for row in grid for c in row if c.strip()]
    if not all_cells:
        return False
    # No long cells
    if any(len(c) > MAX_CELL_LEN for c in all_cells):
        return False
    # No sentence-fragment endings
    _FRAG = (',', ' and', ' the', ' of', ' in', ' to', ' a', ' is',
             ' or', ' for', ' on', ' by', ' with', ' at')
    if any(any(c.endswith(e) for e in _FRAG) for c in all_cells):
        return False
    return True


def _is_valid_table(grid: list[list[str]], n_cols: int) -> bool:
    """
    True if grid looks like a real table (not paragraph or body-column text).
    """
    n_rows = len(grid)
    if n_rows < MIN_ROWS + 1 or n_cols < MIN_COLS or n_rows > MAX_ROWS:
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

    if sum(1 for c in all_cells if len(c) > MAX_CELL_LEN) / len(all_cells) > MAX_LONG_CELL_RATIO:
        return False
    if sum(1 for c in all_cells if len(c) <= 30) / len(all_cells) < MIN_SHORT_CELL_RATIO:
        return False
    if sum(len(c.split()) for c in all_cells) / len(all_cells) > 8:
        return False

    # Sentence-fragment endings indicate body text
    _FRAG = (',', ' and', ' the', ' of', ' in', ' to', ' a', ' is',
             ' or', ' for', ' on', ' by', ' with', ' at')
    if sum(1 for c in all_cells if any(c.endswith(e) for e in _FRAG)) / len(all_cells) > 0.15:
        return False

    # ── Concatenated-word detection ───────────────────────────────────────────
    # PDF encoding artifacts produce words with no internal spaces, e.g.
    # "onthetrainingsetandevaluatedonthetestset." — avg word length > 15
    # signals this is not a real table, just garbled body text.
    avg_word_len = sum(
        len(word)
        for c in all_cells
        for word in c.split()
    ) / max(sum(len(c.split()) for c in all_cells), 1)
    if avg_word_len > 15:
        return False

    # ── Email / institution / header block detection ──────────────────────────
    # Author blocks and institution names look like tables to the column
    # detector (short aligned tokens) but contain email patterns or
    # institution-style ALL-CAPS/mixed words with @ or domain suffixes.
    email_cells = sum(
        1 for c in all_cells
        if '@' in c or c.endswith('.com') or c.endswith('.edu')
        or c.endswith('.org') or c.endswith('.net')
    )
    if email_cells > 0:
        return False

    # Reject if cells look like institution/author names:
    # all cells are short (≤20 chars) AND contain only capitalized words
    # AND no numeric content — classic author header pattern
    if n_rows <= 3:
        name_like = sum(
            1 for c in all_cells
            if len(c) <= 25
            and all(w[0].isupper() for w in c.split() if w and w[0].isalpha())
            and not any(ch.isdigit() for ch in c)
            and not any(ch in c for ch in ['%', '=', '+', '-', '_'])
        )
        if name_like / len(all_cells) > 0.80:
            return False

    filled_cols = sum(
        1 for ci in range(n_cols)
        if sum(1 for row in grid if ci < len(row) and row[ci].strip()) / n_rows >= 0.4
    )
    return filled_cols >= MIN_COLS


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — BUILD DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════

def _grid_to_dataframe(grid: list[list[str]], n_cols: int) -> pd.DataFrame:
    """Convert 2D string grid to DataFrame. Auto-detects header row."""
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
#  STEP 9 — CROSS-PAGE MERGE
# ══════════════════════════════════════════════════════════════════════════════

def _looks_like_standalone_header(row: list) -> bool:
    """True if a row looks like a new table header (short, non-numeric cells)."""
    cells = [str(c).strip() for c in row if c and str(c).strip()]
    if not cells:
        return False
    if sum(len(c) for c in cells) / len(cells) > 30:
        return False
    numeric = sum(1 for c in cells if re.fullmatch(r'[\d\.\,\%\$\-\+\/]+', c))
    return numeric / len(cells) <= 0.4


def _merge_cross_page(tables: list[dict]) -> list[dict]:
    """
    Merge borderless tables that span across pages.

    Two consecutive tables are merged when ALL of:
      1. On adjacent pages (page_diff == 1)
      2. Second table is a page-top continuation candidate OR column names match
      3. Second table's first data row does NOT look like a new header
    """
    if len(tables) < 2:
        return tables

    merged: list[dict] = []

    # Clean up internal fields helper
    def _clean(t: dict) -> dict:
        return {k: v for k, v in t.items()
                if not k.startswith("_")}

    merged.append(tables[0])

    for curr in tables[1:]:
        prev      = merged[-1]
        page_diff = curr["page"] - prev["page"]

        # Rebuild curr DataFrame using prev's column headers if it's a
        # raw continuation block whose auto-detected headers are wrong
        if curr.get("_is_cont") and curr.get("_raw_grid") is not None:
            prev_cols = list(prev["dataframe"].columns)
            raw_grid  = curr["_raw_grid"]
            n_cols    = curr["_n_cols"]
            # Pad/trim rows to match prev column count
            padded = [row + [""] * (len(prev_cols) - len(row))
                      for row in raw_grid]
            padded = [row[:len(prev_cols)] for row in padded]
            curr_df = pd.DataFrame(padded, columns=prev_cols)
            curr_df = curr_df.loc[:, (curr_df != "").any(axis=0)]
        else:
            curr_df = curr["dataframe"]

        same_cols = curr_df.shape[1] == prev["dataframe"].shape[1]
        col_match = list(curr_df.columns) == list(prev["dataframe"].columns)

        # A table with proper column headers (not Col1, Col2...) is a new table
        has_real_headers = not all(
            re.fullmatch(r'Col\d+', h) for h in curr_df.columns
        )
        headers_look_like_table = _looks_like_standalone_header(
            list(curr_df.columns)
        )
        is_new_table = has_real_headers and headers_look_like_table

        can_merge = (
            page_diff == 1
            and same_cols
            and (col_match or curr.get("_is_cont"))
            and not is_new_table
        )

        if can_merge:
            combined_df = pd.concat(
                [prev["dataframe"], curr_df], ignore_index=True
            )
            merged[-1] = {
                **_clean(prev),
                "bottom"   : curr["bottom"],
                "dataframe": combined_df,
                "markdown" : combined_df.to_markdown(index=False),
            }
        else:
            merged.append(_clean(curr))

    # Clean internal fields from all results
    return [_clean(t) for t in merged]


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
                         Pass bordered table bboxes to avoid duplicates.

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
                # Compute bbox first — needed for page-top detection
                block_words = [w for row in block_rows for w in row]
                tbl_top    = min(float(w["top"])    for w in block_words)
                tbl_bottom = max(float(w["bottom"]) for w in block_words)
                tbl_x0     = min(float(w["x0"])     for w in block_words)
                tbl_x1     = max(float(w["x1"])     for w in block_words)

                # Allow small blocks near page top through the row-count gate
                # so they can participate in cross-page merging.
                # We check with an empty grid here (grid not built yet) —
                # just use position + page number to decide.
                is_page_top_block = (page_num > 1 and tbl_top < PAGE_TOP_PT)
                min_rows_required = 2 if is_page_top_block else MIN_ROWS + 1
                if len(block_rows) < min_rows_required:
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

                # Validate — page-top continuation blocks bypass full validation
                is_continuation_candidate = _is_page_top_continuation(
                    grid, tbl_top, page_num
                )
                if not _is_valid_table(grid, n_cols):
                    if not is_continuation_candidate:
                        continue

                df = _grid_to_dataframe(grid, n_cols)
                if df.empty or len(df.columns) < MIN_COLS:
                    continue

                results.append({
                    "page"        : page_num,
                    "top"         : tbl_top,
                    "bottom"      : tbl_bottom,
                    "x0"          : tbl_x0,
                    "x1"          : tbl_x1,
                    "dataframe"   : df,
                    "markdown"    : df.to_markdown(index=False),
                    "method"      : "borderless",
                    # Store raw grid for continuation blocks so cross-page
                    # merge can re-build the DataFrame with correct headers
                    "_raw_grid"   : grid if is_continuation_candidate else None,
                    "_n_cols"     : n_cols,
                    "_is_cont"    : is_continuation_candidate,
                })

    results.sort(key=lambda t: (t["page"], t["top"]))
    results = _merge_cross_page(results)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE TEST ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "report.pdf"
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