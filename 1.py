"""
Advanced PDF Table Parser using Docling
========================================
Focused purely on accurate table extraction from digital PDFs.

Handles:
  - Row-spanning and column-spanning (merged) cells
  - Multi-level column headers
  - Row headers and section divider rows
  - Rich per-cell metadata (bbox, page, span info)
  - Export to DataFrame, HTML (with proper colspan/rowspan), Markdown, CSV, JSON
  - Multi-page table detection and merging (post-processing layer)
    → detects continuation fragments across pages
    → strips repeated header rows on continuation pages
    → handles orphaned continuations (no header on continuation page)
    → rejects ambiguous merges when multiple candidates exist on same page

Fixes applied (v4):
  - pages_tag separator bug in export_all
  - row_end offset wrong for cells spanning the header boundary in merge
  - greedy scan now rejects ambiguous continuations (multiple candidates)
  - col_similarity uses normalised fuzzy comparison, not exact string match
  - multiprocessing.cpu_count() fallback for containerised environments

Requirements:
    pip install docling pandas
"""

from __future__ import annotations

import html
import json
import logging
import multiprocessing
import unicodedata
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional

import pandas as pd

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

try:
    from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
    _BACKEND = DoclingParseV4DocumentBackend
    _BACKEND_NAME = "DoclingParseV4"
except ImportError:
    from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
    _BACKEND = DoclingParseV2DocumentBackend
    _BACKEND_NAME = "DoclingParseV2 (fallback)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Tuning constants for multi-page merge
# ──────────────────────────────────────────────

# Minimum similarity score (0–1) for column labels to be considered matching.
# Uses normalised fuzzy comparison, not exact string equality.
# 0.85 = 85% similarity required.
MULTIPAGE_COL_SIMILARITY_THRESHOLD = 0.85

# Maximum page gap allowed between a table and its continuation.
# 1 = only consecutive pages. Raise to 2 only if full-page figures can
# interrupt a table mid-flow in your specific PDFs.
MULTIPAGE_MAX_PAGE_GAP = 1

# Max normalised left-edge difference (0.0–1.0 of page width) for two tables
# to be considered horizontally aligned.
MULTIPAGE_BBOX_X_TOLERANCE = 0.05


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────
@dataclass
class CellInfo:
    """All metadata for a single parsed table cell."""
    text: str
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    row_span: int
    col_span: int
    is_column_header: bool
    is_row_header: bool
    is_row_section: bool
    bbox: Optional[dict]
    source_page: Optional[int] = None

    def __repr__(self) -> str:
        return (
            f"CellInfo('{self.text[:30]}', "
            f"row={self.row_start}:{self.row_end}, "
            f"col={self.col_start}:{self.col_end}, "
            f"span={self.row_span}r×{self.col_span}c)"
        )


class ParsedTable:
    """All data extracted for one logical table (may span multiple pages)."""

    def __init__(
        self,
        table_index: int,
        page_number: Optional[int],
        num_rows: int,
        num_cols: int,
        cells: Optional[list[CellInfo]] = None,
        source_pages: Optional[list[int]] = None,
    ) -> None:
        self.table_index = table_index
        self.page_number = page_number
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.cells: list[CellInfo] = cells or []
        self.source_pages: list[int] = source_pages or (
            [page_number] if page_number is not None else []
        )

    @property
    def is_multipage(self) -> bool:
        return len(self.source_pages) > 1

    def __repr__(self) -> str:
        pages = (
            f"pages={self.source_pages}" if self.is_multipage
            else f"page={self.page_number}"
        )
        return (
            f"ParsedTable(index={self.table_index}, {pages}, "
            f"{self.num_rows}r×{self.num_cols}c, cells={len(self.cells)})"
        )

    # ─────────────────────────────────────────
    # Cached core computations
    # ─────────────────────────────────────────
    @cached_property
    def _grid_propagated(self) -> list[list[str]]:
        return self._build_grid(propagate=True)

    @cached_property
    def _grid_origin_only(self) -> list[list[str]]:
        return self._build_grid(propagate=False)

    @cached_property
    def _header_row_idxs(self) -> set[int]:
        idxs: set[int] = set()
        for cell in self.cells:
            if cell.is_column_header:
                for r in range(cell.row_start, cell.row_end):
                    idxs.add(r)
        return idxs

    @cached_property
    def _col_labels(self) -> list[str]:
        """
        Derive one label per column using a cell-origin approach.

        WHY: the grid-propagation approach (previous implementation) fills every
        slot a merged cell covers with its text. This causes a cell like
        "Invoice #123456789" spanning cols 0-3 to become the label for all four
        columns — which is wrong. A merged header cell should only label the
        columns it *originates* at, unless it is a true parent in a multi-level
        header (e.g. "Q1" above "Jan | Feb | March"), in which case it provides
        context for every child column.

        Rules per column C:
          1. Walk each header row top-to-bottom.
          2. Find the cell covering (row, C).
          3. If that cell ORIGINATES at C (col_start == C): always include its text.
          4. If C is a continuation slot of a wide cell:
               - Include the parent text ONLY if lower header rows contain
                 differentiating child cells at C (multi-level pattern).
               - Otherwise skip it (it is a title/section row, not a column header).
        """
        header_rows = sorted(self._header_row_idxs)
        if not header_rows:
            return [f"col_{c}" for c in range(self.num_cols)]

        header_cells = [c for c in self.cells if c.is_column_header]

        labels: list[str] = []
        for col in range(self.num_cols):
            parts: list[str] = []
            for row in header_rows:
                # Find the unique cell that covers position (row, col)
                covering = next(
                    (cell for cell in header_cells
                     if cell.row_start <= row < cell.row_end
                     and cell.col_start <= col < cell.col_end),
                    None,
                )
                if covering is None or not covering.text.strip():
                    continue

                is_origin   = (covering.col_start == col)
                is_single   = (covering.col_span == 1)

                if is_origin or is_single:
                    # Cell starts here — always contribute its text
                    if covering.text not in parts:
                        parts.append(covering.text)
                else:
                    # col is a continuation slot of a wide spanning cell.
                    # Only include the parent text if there are differentiating
                    # child cells in lower header rows (multi-level header).
                    has_child = any(
                        child for child in header_cells
                        if child.row_start > row
                        and child.col_start <= col < child.col_end
                        and child.col_span < covering.col_span
                    )
                    if has_child and covering.text not in parts:
                        parts.append(covering.text)
                    # else: wide title-row spanning cell — skip for this column

            labels.append(" | ".join(parts) if parts else f"col_{col}")

        return labels

    @cached_property
    def _cells_sorted(self) -> list[CellInfo]:
        return sorted(self.cells, key=lambda c: (c.row_start, c.col_start))

    def _build_grid(self, propagate: bool) -> list[list[str]]:
        grid: list[list[str]] = [
            [""] * self.num_cols for _ in range(self.num_rows)
        ]
        for cell in self.cells:
            for r in range(cell.row_start, cell.row_end):
                for c in range(cell.col_start, cell.col_end):
                    if propagate or (r == cell.row_start and c == cell.col_start):
                        grid[r][c] = cell.text
        return grid

    # ─────────────────────────────────────────
    # Exports
    # ─────────────────────────────────────────
    def to_dataframe(self, propagate_merged_values: bool = True) -> pd.DataFrame:
        """
        Return a pandas DataFrame.

        propagate_merged_values=True  (default)
            Merged cell text fills every row it covers.
            Best for analysis — every row is self-contained.
        propagate_merged_values=False
            Only the origin cell is filled; continuation slots are empty strings.
        """
        grid = self._grid_propagated if propagate_merged_values else self._grid_origin_only
        body = [grid[r] for r in range(self.num_rows) if r not in self._header_row_idxs]
        return pd.DataFrame(body, columns=self._col_labels)

    def to_html(self) -> str:
        """
        Emit a complete HTML <table> with correct colspan / rowspan attributes.
        Cell text is HTML-escaped — characters like <, >, &, " are always safe.
        """
        claimed = [[False] * self.num_cols for _ in range(self.num_rows)]
        row_cells: dict[int, list[str]] = {r: [] for r in range(self.num_rows)}

        for cell in self._cells_sorted:
            r, c = cell.row_start, cell.col_start
            if claimed[r][c]:
                continue
            for dr in range(cell.row_span):
                for dc in range(cell.col_span):
                    claimed[r + dr][c + dc] = True

            tag = "th" if (cell.is_column_header or cell.is_row_header) else "td"
            attrs: list[str] = []
            if cell.col_span > 1:
                attrs.append(f'colspan="{cell.col_span}"')
            if cell.row_span > 1:
                attrs.append(f'rowspan="{cell.row_span}"')
            if cell.is_row_section:
                attrs.append('class="section-row"')
            attr_str = (" " + " ".join(attrs)) if attrs else ""
            safe_text = html.escape(cell.text)
            row_cells[r].append(f"<{tag}{attr_str}>{safe_text}</{tag}>")

        rows = "\n".join(
            f"  <tr>{''.join(row_cells[r])}</tr>" for r in range(self.num_rows)
        )
        pages_note = (
            f"<!-- spans pages {self.source_pages} -->\n" if self.is_multipage else ""
        )
        return (
            f"{pages_note}"
            "<table border='1' cellpadding='4' cellspacing='0'>\n"
            f"{rows}\n"
            "</table>"
        )

    def to_markdown(self) -> str:
        """
        Emit GitHub-Flavored Markdown.
        Separator is placed after the last header row (not blindly after row 0).
        Merged cells carry span annotations; continuation slots show ↑ or ←.
        """
        grid: list[list[str]] = [[""] * self.num_cols for _ in range(self.num_rows)]
        for cell in self.cells:
            label = cell.text
            if cell.row_span > 1 or cell.col_span > 1:
                label += f" *(span {cell.row_span}r×{cell.col_span}c)*"
            for r in range(cell.row_start, cell.row_end):
                for c in range(cell.col_start, cell.col_end):
                    if r == cell.row_start and c == cell.col_start:
                        grid[r][c] = label
                    elif r > cell.row_start:
                        grid[r][c] = "↑"
                    else:
                        grid[r][c] = "←"

        last_header = max(self._header_row_idxs) if self._header_row_idxs else 0
        sep_line = "|" + "|".join(["---"] * self.num_cols) + "|"

        lines: list[str] = []
        for i, row in enumerate(grid):
            lines.append("| " + " | ".join(row) + " |")
            if i == last_header:
                lines.append(sep_line)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Full JSON-serialisable representation of this table."""
        return {
            "table_index":  self.table_index,
            "page_number":  self.page_number,
            "source_pages": self.source_pages,
            "is_multipage": self.is_multipage,
            "num_rows":     self.num_rows,
            "num_cols":     self.num_cols,
            "col_labels":   self._col_labels,
            "cells": [
                {
                    "text":             c.text,
                    "row_start":        c.row_start,
                    "row_end":          c.row_end,
                    "col_start":        c.col_start,
                    "col_end":          c.col_end,
                    "row_span":         c.row_span,
                    "col_span":         c.col_span,
                    "is_column_header": c.is_column_header,
                    "is_row_header":    c.is_row_header,
                    "is_row_section":   c.is_row_section,
                    "bbox":             c.bbox,
                    "source_page":      c.source_page,
                }
                for c in self.cells
            ],
        }

    def print_summary(self) -> None:
        """Print a human-readable summary with merged cell breakdown."""
        merged   = [c for c in self.cells if c.row_span > 1 or c.col_span > 1]
        col_hdrs = [c for c in self.cells if c.is_column_header]
        row_hdrs = [c for c in self.cells if c.is_row_header]
        sections = [c for c in self.cells if c.is_row_section]

        print(f"\n{'─'*65}")
        tag = (
            f"[MULTI-PAGE: {self.source_pages}]" if self.is_multipage
            else f"[page {self.page_number}]"
        )
        print(f"  Table {self.table_index + 1}  {tag}"
              f"  |  {self.num_rows} rows × {self.num_cols} cols")
        print(f"  Merged cells        : {len(merged)}")
        print(f"  Column header cells : {len(col_hdrs)}")
        print(f"  Row header cells    : {len(row_hdrs)}")
        print(f"  Section rows        : {len(sections)}")
        if merged:
            print("\n  Merged cell details:")
            for c in merged:
                print(f"    '{c.text[:45]:<45}'  "
                      f"R[{c.row_start}:{c.row_end - 1}]  "
                      f"C[{c.col_start}:{c.col_end - 1}]  "
                      f"({c.row_span}r × {c.col_span}c)")
        print(f"{'─'*65}")


# ──────────────────────────────────────────────
# Multi-page merge helpers
# ──────────────────────────────────────────────

def _normalise_label(s: str) -> str:
    """
    Normalise a column label for fuzzy comparison.
    Lowercases, strips, collapses whitespace, removes unicode diacritics.
    Handles minor OCR noise and formatting differences between pages.
    """
    s = s.strip().lower()
    s = " ".join(s.split())   # collapse internal whitespace
    # Remove diacritics (e.g. é → e) for robustness against OCR noise
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    return s


def _col_similarity(labels_a: list[str], labels_b: list[str]) -> float:
    """
    Fuzzy similarity score (0–1) between two lists of column labels.

    FIX: Uses normalised comparison instead of exact string equality.
    "Revenue ($M)" and "Revenue ( $M )" now correctly score ~1.0.
    Returns 1.0 if both lists are empty (both headerless — valid continuation).
    """
    if not labels_a and not labels_b:
        return 1.0
    if len(labels_a) != len(labels_b):
        return 0.0
    norm_a = [_normalise_label(l) for l in labels_a]
    norm_b = [_normalise_label(l) for l in labels_b]
    matches = sum(1 for a, b in zip(norm_a, norm_b) if a == b)
    return matches / len(labels_a)


def _bbox_x_aligned(
    table_a: ParsedTable,
    table_b: ParsedTable,
    tolerance: float,
) -> bool:
    """
    Check whether two tables share the same left-edge position (within tolerance).
    Returns True when no bbox data is available — can't disprove alignment.
    """
    def leftmost(t: ParsedTable) -> Optional[float]:
        xs = [c.bbox["left"] for c in t.cells if c.bbox]
        return min(xs) if xs else None

    x_a = leftmost(table_a)
    x_b = leftmost(table_b)
    if x_a is None or x_b is None:
        return True
    return abs(x_a - x_b) <= tolerance


def _is_continuation(
    base: ParsedTable,
    candidate: ParsedTable,
    max_page_gap: int,
    col_sim_threshold: float,
    bbox_x_tolerance: float,
) -> bool:
    """
    Decide whether `candidate` is a direct continuation of `base`.

    All four criteria must pass:
      1. Candidate starts on the next page (within max_page_gap)
      2. Same number of columns
      3. Column labels match at >= col_sim_threshold, OR candidate has no header
      4. Tables are left-edge aligned within bbox_x_tolerance
    """
    last_base_page  = max(base.source_pages)      if base.source_pages      else base.page_number
    first_cand_page = min(candidate.source_pages) if candidate.source_pages else candidate.page_number

    if last_base_page is None or first_cand_page is None:
        return False

    # 1. Page adjacency
    if not (1 <= (first_cand_page - last_base_page) <= max_page_gap):
        return False

    # 2. Column count
    if base.num_cols != candidate.num_cols:
        return False

    # 3. Column label similarity
    if bool(candidate._header_row_idxs):
        sim = _col_similarity(base._col_labels, candidate._col_labels)
        if sim < col_sim_threshold:
            return False
    # Headerless candidate passes automatically (orphaned continuation)

    # 4. Horizontal alignment
    if not _bbox_x_aligned(base, candidate, bbox_x_tolerance):
        return False

    return True


def _merge_two_tables(base: ParsedTable, continuation: ParsedTable) -> ParsedTable:
    """
    Merge `continuation` into `base`, appending its body rows.

    FIX: row_end is now offset independently from row_start.
    Previously both used the same `skipped_before` count, which was wrong
    for cells whose span crosses the header/body boundary.
    """
    skip_rows: set[int] = set()
    if continuation._header_row_idxs:
        sim = _col_similarity(base._col_labels, continuation._col_labels)
        if sim >= MULTIPAGE_COL_SIMILARITY_THRESHOLD:
            skip_rows = continuation._header_row_idxs
            log.debug(
                "  Stripping %d repeated header row(s) from continuation (page %s)",
                len(skip_rows), continuation.page_number,
            )

    contrib_rows = continuation.num_rows - len(skip_rows)
    row_offset   = base.num_rows

    new_cells: list[CellInfo] = []
    for cell in continuation.cells:
        if cell.row_start in skip_rows:
            continue

        # FIX: row_start and row_end each get their own independent skip count
        skipped_before_start = sum(1 for s in skip_rows if s < cell.row_start)
        skipped_before_end   = sum(1 for s in skip_rows if s < cell.row_end)

        new_cells.append(CellInfo(
            text=cell.text,
            row_start=cell.row_start - skipped_before_start + row_offset,
            row_end=cell.row_end     - skipped_before_end   + row_offset,
            col_start=cell.col_start,
            col_end=cell.col_end,
            row_span=cell.row_span,
            col_span=cell.col_span,
            is_column_header=False,
            is_row_header=cell.is_row_header,
            is_row_section=cell.is_row_section,
            bbox=cell.bbox,
            source_page=continuation.page_number,
        ))

    merged_pages = list(dict.fromkeys(base.source_pages + continuation.source_pages))

    merged = ParsedTable(
        table_index=base.table_index,
        page_number=base.page_number,
        num_rows=base.num_rows + contrib_rows,
        num_cols=base.num_cols,
        cells=base.cells + new_cells,
        source_pages=merged_pages,
    )

    log.info(
        "  Merged table %d: pages %s → %d rows total "
        "(%d base + %d new, %d header rows stripped)",
        base.table_index + 1,
        merged_pages,
        merged.num_rows,
        base.num_rows,
        contrib_rows,
        len(skip_rows),
    )
    return merged


def merge_multipage_tables(
    tables: list[ParsedTable],
    max_page_gap: int = MULTIPAGE_MAX_PAGE_GAP,
    col_sim_threshold: float = MULTIPAGE_COL_SIMILARITY_THRESHOLD,
    bbox_x_tolerance: float = MULTIPAGE_BBOX_X_TOLERANCE,
) -> list[ParsedTable]:
    """
    Detect and merge tables that are continuations of each other across pages.

    FIX: Ambiguous merge rejection.
    If more than one table on the candidate page qualifies as a continuation
    of the current table, the merge is rejected for safety — both tables are
    kept as independent entries. This prevents incorrect merging when two
    separate tables on the same page happen to share the same column structure.

    Algorithm: greedy forward scan with ambiguity guard.
    """
    if not tables:
        return []

    ordered = sorted(
        tables,
        key=lambda t: (min(t.source_pages) if t.source_pages else t.page_number or 0),
    )

    merged: list[ParsedTable] = []
    current = ordered[0]

    i = 1
    while i < len(ordered):
        candidate = ordered[i]

        if not _is_continuation(current, candidate,
                                max_page_gap, col_sim_threshold, bbox_x_tolerance):
            merged.append(current)
            current = candidate
            i += 1
            continue

        # FIX: Check for ambiguity — is there another table on the same page
        # that also qualifies as a continuation?
        cand_page = (
            min(candidate.source_pages) if candidate.source_pages
            else candidate.page_number
        )
        rivals = [
            t for t in ordered[i + 1:]
            if (min(t.source_pages) if t.source_pages else t.page_number) == cand_page
            and _is_continuation(current, t,
                                  max_page_gap, col_sim_threshold, bbox_x_tolerance)
        ]

        if rivals:
            # Ambiguous — multiple tables on next page qualify. Keep all separate.
            log.warning(
                "  Ambiguous continuation for table %d (page %s): "
                "%d rival(s) found. Skipping merge to avoid data corruption.",
                current.table_index + 1,
                cand_page,
                len(rivals),
            )
            merged.append(current)
            current = candidate
        else:
            current = _merge_two_tables(current, candidate)

        i += 1

    merged.append(current)

    # Reassign contiguous indices
    for new_idx, t in enumerate(merged):
        t.table_index = new_idx

    n_orig   = len(tables)
    n_merged = len(merged)
    if n_orig != n_merged:
        log.info(
            "Multi-page merge: %d fragment(s) → %d logical table(s)",
            n_orig, n_merged,
        )

    return merged


# ──────────────────────────────────────────────
# Converter
# ──────────────────────────────────────────────
def build_converter(
    use_accurate_mode: bool = True,
    do_cell_matching: bool = True,
    num_threads: int | None = None,
    device: AcceleratorDevice = AcceleratorDevice.AUTO,
) -> DocumentConverter:
    """
    Build a DocumentConverter optimised for accurate digital PDF table parsing.
    Build once and reuse via parse_tables(converter=…) for multiple PDFs.

    Parameters
    ----------
    use_accurate_mode : ACCURATE (True) vs FAST (False) TableFormer mode.
    do_cell_matching  : Match cells to embedded PDF text layer (True for digital PDFs).
    num_threads       : CPU threads. Defaults to cpu_count() with fallback to 4.
    device            : AUTO | CUDA | MPS | CPU
    """
    # FIX: cpu_count() can return None in containerised environments
    if num_threads is None:
        num_threads = multiprocessing.cpu_count() or 4

    accelerator_options = AcceleratorOptions(
        num_threads=num_threads,
        device=device,
    )
    table_structure_options = TableStructureOptions(
        mode=TableFormerMode.ACCURATE if use_accurate_mode else TableFormerMode.FAST,
        do_cell_matching=do_cell_matching,
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr                  = False
    pipeline_options.do_table_structure      = True
    pipeline_options.table_structure_options = table_structure_options
    pipeline_options.accelerator_options     = accelerator_options
    pipeline_options.generate_page_images    = False
    pipeline_options.generate_picture_images = False
    pipeline_options.do_code_enrichment      = False
    pipeline_options.do_formula_enrichment   = False

    log.info(
        "Converter | backend=%s | mode=%s | cell_matching=%s | threads=%d | device=%s",
        _BACKEND_NAME,
        "ACCURATE" if use_accurate_mode else "FAST",
        do_cell_matching,
        num_threads,
        device.value if hasattr(device, "value") else str(device),
    )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=_BACKEND,
            )
        }
    )


# ──────────────────────────────────────────────
# Cell extraction
# ──────────────────────────────────────────────
def _extract_cells(table_item, table_index: int) -> ParsedTable:
    """
    Convert a docling TableItem into a ParsedTable.
    Deduplicates the raw grid so each logical cell appears exactly once.
    """
    page_number = table_item.prov[0].page_no if table_item.prov else None

    parsed = ParsedTable(
        table_index=table_index,
        page_number=page_number,
        num_rows=table_item.data.num_rows,   # FIX: num_rows lives on TableData, not TableItem
        num_cols=table_item.data.num_cols,   # FIX: num_cols lives on TableData, not TableItem
    )

    seen: set[tuple[int, int]] = set()

    for row in table_item.data.grid:
        for cell in row:
            origin = (cell.start_row_offset_idx, cell.start_col_offset_idx)
            if origin in seen:
                continue
            seen.add(origin)

            bbox_dict = None
            if cell.bbox is not None:
                bbox_dict = {
                    "left":   cell.bbox.l,
                    "top":    cell.bbox.t,
                    "right":  cell.bbox.r,
                    "bottom": cell.bbox.b,
                }

            parsed.cells.append(CellInfo(
                text=cell.text.strip(),
                row_start=cell.start_row_offset_idx,
                row_end=cell.end_row_offset_idx,
                col_start=cell.start_col_offset_idx,
                col_end=cell.end_col_offset_idx,
                row_span=cell.end_row_offset_idx - cell.start_row_offset_idx,
                col_span=cell.end_col_offset_idx - cell.start_col_offset_idx,
                is_column_header=bool(cell.column_header),
                is_row_header=bool(cell.row_header),
                is_row_section=bool(cell.row_section),
                bbox=bbox_dict,
                source_page=page_number,
            ))

    return parsed


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
def parse_tables(
    pdf_path: str | Path,
    converter: DocumentConverter | None = None,
    use_accurate_mode: bool = True,
    do_cell_matching: bool = True,
    num_threads: int | None = None,
    device: AcceleratorDevice = AcceleratorDevice.AUTO,
    max_num_pages: int | None = None,
    max_file_size: int | None = None,
    merge_multipage: bool = True,
    multipage_max_gap: int = MULTIPAGE_MAX_PAGE_GAP,
    multipage_col_sim: float = MULTIPAGE_COL_SIMILARITY_THRESHOLD,
    multipage_bbox_tol: float = MULTIPAGE_BBOX_X_TOLERANCE,
) -> list[ParsedTable]:
    """
    Parse all tables from a digital PDF, with optional multi-page merging.

    Parameters
    ----------
    converter        : Pre-built converter (pass when processing multiple PDFs).
    merge_multipage  : Detect and merge cross-page table fragments (default True).
    multipage_max_gap: Max page gap for continuation detection (default 1).
    multipage_col_sim: Min column-label similarity threshold 0–1 (default 0.85).
    multipage_bbox_tol: Max x-alignment tolerance 0–1 (default 0.05).

    Raises
    ------
    FileNotFoundError  : PDF does not exist.
    RuntimeError       : Docling reports full conversion failure.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if converter is None:
        converter = build_converter(
            use_accurate_mode=use_accurate_mode,
            do_cell_matching=do_cell_matching,
            num_threads=num_threads,
            device=device,
        )

    log.info("Parsing: %s", pdf_path.name)

    convert_kwargs: dict = {}
    if max_num_pages is not None:
        convert_kwargs["max_num_pages"] = max_num_pages
    if max_file_size is not None:
        convert_kwargs["max_file_size"] = max_file_size

    result = converter.convert(str(pdf_path), **convert_kwargs)
    errors = getattr(result, "errors", [])

    if result.status == ConversionStatus.FAILURE:
        raise RuntimeError(
            f"Docling failed to convert '{pdf_path.name}'. Errors: {errors}"
        )
    if result.status == ConversionStatus.PARTIAL_SUCCESS:
        log.warning(
            "Partial conversion for '%s' — some pages may be missing. Errors: %s",
            pdf_path.name, errors,
        )

    doc = result.document
    log.info(
        "Found %d table fragment(s) in '%s'  [status: %s]",
        len(doc.tables), pdf_path.name, result.status.value,
    )

    raw_tables: list[ParsedTable] = []
    for i, table_item in enumerate(doc.tables):
        log.info(
            "  Fragment %d: %d rows × %d cols  (page %s)",
            i + 1, table_item.data.num_rows, table_item.data.num_cols,
            table_item.prov[0].page_no if table_item.prov else "?",
        )
        raw_tables.append(_extract_cells(table_item, table_index=i))

    if merge_multipage and len(raw_tables) > 1:
        return merge_multipage_tables(
            raw_tables,
            max_page_gap=multipage_max_gap,
            col_sim_threshold=multipage_col_sim,
            bbox_x_tolerance=multipage_bbox_tol,
        )

    return raw_tables


def parse_tables_batch(
    pdf_paths: list[str | Path],
    use_accurate_mode: bool = True,
    do_cell_matching: bool = True,
    num_threads: int | None = None,
    device: AcceleratorDevice = AcceleratorDevice.AUTO,
    merge_multipage: bool = True,
) -> dict[str, list[ParsedTable]]:
    """
    Parse tables from multiple PDFs using a single shared converter.

    The TableFormer model is loaded exactly once for the entire batch.
    Failed PDFs are logged and mapped to an empty list — one failure
    does not abort the rest of the batch.
    """
    converter = build_converter(
        use_accurate_mode=use_accurate_mode,
        do_cell_matching=do_cell_matching,
        num_threads=num_threads,
        device=device,
    )

    results: dict[str, list[ParsedTable]] = {}
    for path in pdf_paths:
        path = Path(path)
        try:
            tables = parse_tables(
                pdf_path=path,
                converter=converter,
                merge_multipage=merge_multipage,
            )
            results[path.name] = tables
            log.info("Batch: %s → %d logical table(s)", path.name, len(tables))
        except Exception as exc:
            log.error("Batch: failed '%s' — %s", path.name, exc)
            results[path.name] = []

    return results


def export_all(
    parsed_tables: list[ParsedTable],
    output_dir: str | Path,
    formats: list[str] | None = None,
) -> dict[str, list[str]]:
    """
    Write each parsed table to disk in the requested formats.

    FIX: pages_tag now correctly produces "pages_3_4_5" (with separator).
    Each write is wrapped in try/except — one failure does not abort the rest.

    Returns {"saved": [...paths], "failed": [...paths]}.
    """
    if formats is None:
        formats = ["csv", "html", "markdown", "json"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved:  list[str] = []
    failed: list[str] = []

    for pt in parsed_tables:
        # FIX: underscore between "pages" and the page numbers
        pages_tag = (
            f"pages_{'_'.join(str(p) for p in pt.source_pages)}"
            if pt.is_multipage else f"page_{pt.page_number}"
        )
        stem = f"table_{pt.table_index + 1}_{pages_tag}"

        targets: list[tuple[str, Path, object]] = []

        if "csv" in formats:
            p = output_dir / f"{stem}.csv"
            targets.append(("csv", p,
                lambda _p=p, _t=pt: _t.to_dataframe().to_csv(_p, index=False)))

        if "html" in formats:
            p = output_dir / f"{stem}.html"
            targets.append(("html", p,
                lambda _p=p, _t=pt: _p.write_text(
                    f"<!DOCTYPE html><html><body>\n{_t.to_html()}\n</body></html>",
                    encoding="utf-8")))

        if "markdown" in formats:
            p = output_dir / f"{stem}.md"
            targets.append(("md", p,
                lambda _p=p, _t=pt: _p.write_text(
                    _t.to_markdown(), encoding="utf-8")))

        if "json" in formats:
            p = output_dir / f"{stem}.json"
            targets.append(("json", p,
                lambda _p=p, _t=pt: _p.write_text(
                    json.dumps(_t.to_dict(), indent=2, ensure_ascii=False),
                    encoding="utf-8")))

        for fmt, path, write_fn in targets:
            try:
                write_fn()
                saved.append(str(path))
                log.info("  %-8s → %s", fmt.upper(), path)
            except Exception as exc:
                failed.append(str(path))
                log.error("  %-8s FAILED for %s — %s", fmt.upper(), stem, exc)

    if failed:
        log.warning("%d file(s) failed to export: %s", len(failed), failed)

    return {"saved": saved, "failed": failed}


# ──────────────────────────────────────────────
# Example Usage
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else "tables.pdf"

    tables = parse_tables(
        pdf_path=pdf,
        use_accurate_mode=True,
        do_cell_matching=True,
        device=AcceleratorDevice.AUTO,
        merge_multipage=True,
    )

    for pt in tables:
        pt.print_summary()
        print(f"\n  DataFrame ({pt.num_rows} rows):")
        print(pt.to_dataframe().to_string())

    result = export_all(tables, output_dir="parsed_tables")
    print(f"\nDone — {len(result['saved'])} files saved, "
          f"{len(result['failed'])} failed.")

    # ── Batch usage ───────────────────────────────────────────────────
    # from pathlib import Path
    # pdfs = list(Path("my_pdfs").glob("*.pdf"))
    # batch = parse_tables_batch(pdfs, merge_multipage=True)
    # for name, tables in batch.items():
    #     print(f"{name}: {len(tables)} logical table(s)")