"""
PDF Table Parser v4
Fixes:
  1. Broken wrapped words in cells/headers (Company_Na\\nme → Company_Name)
  2. Cross-page table merging (bank table, equity table)
  3. False-positive paragraph detection (single-column text blocks filtered out)
  4. Smarter continuation heuristic (doesn't require blank first cell)
"""

from pathlib import Path
import pdfplumber
import pandas as pd
import re


# ── Cell cleaning ─────────────────────────────────────────────────────────────

def fix_wrapped_word(text: str) -> str:
    """
    Fix mid-word line-breaks caused by narrow PDF columns.
    e.g. 'Company_Na\\nme' → 'Company_Name'
         'NET\\nINCOME\\n2021' → 'NET INCOME 2021'

    Strategy:
    - If the character before \\n is NOT a space/hyphen AND
      the character after \\n is lowercase or continues an underscore word,
      join without space (it's a broken word).
    - Otherwise join with a space (it's a real multi-line value).
    """
    if text is None:
        return ""
    # Replace \n with special marker, then decide per occurrence
    parts = str(text).split("\n")
    result = parts[0]
    for part in parts[1:]:
        if result and part:
            last_char  = result[-1]
            first_char = part[0]
            joined_len = len(result) + len(part)

            # Rule 1: underscore identifier or email/URL → join without space
            #   e.g. 'Company_Na' + 'me', 'r.malhotra@tec' + 'hnova.in'
            if "_" in result or "@" in result:
                result += part

            # Rule 2: alphanumeric code (digit → letter/digit) → join without space
            #   e.g. '07AABCT9988' + 'Q1Z5'
            elif last_char.isdigit() and re.match(r'[\w]', first_char):
                result += part

            # Rule 3: lowercase continuation of a SHORT fragment → broken word
            #   e.g. 'Amo' + 'unt' → 'Amount'  (joined ≤ 8 chars)
            elif first_char.islower() and joined_len <= 8:
                result += part

            # Rule 4: everything else → separate word or new token → add space
            #   e.g. 'Foreign' + 'currency', 'TechNova' + 'Systems', 'NET' + 'INCOME'
            else:
                result += " " + part
        else:
            result += " " + part
    return result.strip()


def clean_cell(val) -> str:
    """Full cell cleaner: fix wrapping, strip, normalise whitespace."""
    return " ".join(fix_wrapped_word(val).split())


# ── False-positive filter ─────────────────────────────────────────────────────

def is_paragraph_not_table(rows: list[list]) -> bool:
    """
    Return True if this 'table' is actually a paragraph/text block:
    - Only 1 column, AND
    - Every cell is a long sentence (>40 chars)
    """
    if not rows:
        return False
    widths = {len(r) for r in rows}
    if widths != {1}:
        return False
    avg_len = sum(len(str(r[0])) for r in rows) / len(rows)
    return avg_len > 40


# ── Cross-page continuation heuristic ────────────────────────────────────────

def looks_like_header(row: list, prev_header: list) -> bool:
    """True if this row repeats the previous table's header (column labels)."""
    if len(row) != len(prev_header):
        return False
    matches = sum(
        clean_cell(a).lower() == clean_cell(b).lower()
        for a, b in zip(row, prev_header)
    )
    return matches >= len(row) * 0.6   # 60%+ columns match → it's a repeated header


def is_continuation(prev_raw: list[list], curr_raw: list[list], page_diff: int) -> bool:
    """
    True if curr_raw is a continuation of prev_raw:
    - Same column count
    - Adjacent pages (diff <= 1)
    - First row of curr is NOT a repeated header
    """
    if page_diff > 1:
        return False
    prev_width = len(prev_raw[0]) if prev_raw else 0
    curr_width = len(curr_raw[0]) if curr_raw else 0
    if prev_width != curr_width or prev_width == 0:
        return False
    # Don't merge if first row is just repeating the header
    if looks_like_header(curr_raw[0], prev_raw[0]):
        return False
    return True


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_tables(pdf_path: str) -> list[pd.DataFrame]:
    """
    Extract all logical tables from a PDF.
    Returns a list of clean DataFrames.
    """
    # Step 1: collect all raw tables with page info
    raw_tables: list[tuple[int, list[list]]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for t in page.find_tables():
                rows = t.extract()
                if not rows:
                    continue
                if is_paragraph_not_table(rows):
                    print(f"  ✗  Page {page_num}: skipped single-column paragraph block")
                    continue
                raw_tables.append((page_num, rows))

    if not raw_tables:
        return []

    # Step 2: merge cross-page continuations
    logical: list[list[list]] = [raw_tables[0][1]]

    for i in range(1, len(raw_tables)):
        prev_page, prev_rows = raw_tables[i - 1]
        curr_page, curr_rows = raw_tables[i]

        if is_continuation(prev_rows, curr_rows, curr_page - prev_page):
            print(f"  ↔  Page {curr_page}: merged into previous table "
                  f"(+{len(curr_rows)} rows)")
            logical[-1].extend(curr_rows)
        else:
            logical.append(curr_rows)

    # Step 3: build clean DataFrames
    dataframes = []
    for rows in logical:
        cleaned = [[clean_cell(c) for c in row] for row in rows]
        # Uniform column count
        width   = max(len(r) for r in cleaned)
        cleaned = [r + [""] * (width - len(r)) for r in cleaned]

        header = cleaned[0]
        df     = pd.DataFrame(cleaned[1:], columns=header)
        # Drop fully-empty columns
        df = df.loc[:, (df != "").any(axis=0)]
        dataframes.append(df)

    return dataframes


# ── Save output ───────────────────────────────────────────────────────────────

def parse_tables(pdf_path: str, output_dir: str = "output") -> None:
    pdf_path    = Path(pdf_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n📄 Parsing : {pdf_path.name}")
    print( "🔍 Engine  : pdfplumber v4 (wrap-fix + false-positive filter + smart merge)\n")

    dataframes = extract_tables(str(pdf_path))

    if not dataframes:
        print("⚠️  No tables found.")
        return

    print(f"\n✅ Logical tables: {len(dataframes)}\n")

    all_parts = [f"# Tables extracted from: `{pdf_path.name}`\n"]

    for i, df in enumerate(dataframes, start=1):
        print(f"  → Table {i}: {df.shape[0]} rows × {df.shape[1]} cols")
        md = df.to_markdown(index=False)

        single_path = output_path / f"table_{i}.md"
        single_path.write_text(f"# Table {i}\n\n{md}\n", encoding="utf-8")
        print(f"     Saved → {single_path}")

        all_parts.append(f"## Table {i}\n\n{md}\n")

    combined = output_path / "all_tables.md"
    combined.write_text("\n".join(all_parts), encoding="utf-8")
    print(f"\n📦 Combined → {combined}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:   python table_parser_v4.py <path_to_pdf> [output_dir]")
        sys.exit(1)
    parse_tables(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "output")