"""
Docling Layout-Aware Hybrid Chunker
=====================================
Tuned precisely to Docling's markdown output format based on actual document analysis.

Docling-specific patterns handled:
  - NO page break markers (Docling doesn't embed them in .md)
  - Figures  : ![AltText](images/pageN_figM.png)  ← image filename encodes page number
  - Captions : *italic line immediately after image*  ← merged into figure chunk
  - Formulas : <!-- formula-not-decoded -->  ← kept as formula block
  - Headings : ## H2 used for both top-level and sub-sections (Docling flattens H1→H2)
  - Tables   : standard markdown tables, kept whole
  - Footnotes: short lines with superscript numbers — detected and attached to parent chunk
  - Floating  : orphan lines (author names, axis labels) — merged into nearest paragraph

Block types produced:
  paragraph | heading | table | figure | formula | list | footnote

Metadata per chunk:
  chunk_id       : uuid
  source_file    : stem of .md filename
  page_number    : inferred from image filenames in chunk (e.g. page5_fig1 → page 5)
                   or estimated by character offset ratio if no image present
  heading_path   : ["H2 title", "H3 subtitle", ...] breadcrumb
  block_type     : dominant block type
  image_refs     : list of image paths found in chunk
  has_formula    : True if chunk contains a formula placeholder
  char_start     : character offset in original markdown
  char_end       : character offset end
  token_count    : approximate token count
  chunk_index    : sequential index for ordering
"""

import re
import json
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

MAX_TOKENS      = 512   # max tokens per chunk
OVERLAP_TOKENS  = 50    # overlap between adjacent paragraph chunks
CHARS_PER_TOKEN = 4     # approximation

MAX_CHARS     = MAX_TOKENS * CHARS_PER_TOKEN
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN

# Short line threshold: lines shorter than this are likely orphan fragments
# (footnote numbers, axis labels, author names, etc.)
SHORT_LINE_CHARS = 60


# ─────────────────────────────────────────────
# Regex patterns  (Docling-specific)
# ─────────────────────────────────────────────

RE_HEADING       = re.compile(r'^(#{1,6})\s+(.+)$')
RE_TABLE_ROW     = re.compile(r'^\|')
RE_IMAGE         = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
RE_ITALIC_LINE   = re.compile(r'^\*(.+)\*$')          # Docling figure captions
RE_FORMULA       = re.compile(r'^<!--\s*formula-not-decoded\s*-->')
RE_LIST_ITEM     = re.compile(r'^(\s*[-*+]|\s*\d+[.)]) ')
RE_FOOTNOTE_NUM  = re.compile(r'^\d+\s+\S')           # "1 http://..." or "2 This..."
RE_PAGE_FROM_IMG = re.compile(r'page(\d+)', re.I)     # extract page from "images/page5_fig1.png"


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class Block:
    block_type:   str
    content:      str
    heading_level: int  = 0
    image_refs:   list  = field(default_factory=list)
    has_formula:  bool  = False
    char_start:   int   = 0
    char_end:     int   = 0
    # page inferred later from image refs
    page_number:  int   = 0


@dataclass
class Chunk:
    chunk_id:     str
    chunk_index:  int
    content:      str
    source_file:  str
    page_number:  int
    heading_path: list
    block_type:   str
    image_refs:   list
    has_formula:  bool
    char_start:   int
    char_end:     int
    token_count:  int


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def infer_page_from_images(image_refs: list, total_chars: int, char_pos: int) -> int:
    """
    Docling embeds page number in image filenames: images/page5_fig1.png → 5.
    If no image in chunk, estimate page by character offset ratio.
    """
    for ref in image_refs:
        m = RE_PAGE_FROM_IMG.search(ref)
        if m:
            return int(m.group(1))
    # fallback: estimate from char position (assumes ~3000 chars/page)
    return max(1, char_pos // 3000 + 1)


def split_at_sentence_boundary(text: str, max_chars: int) -> list[str]:
    """Split text at sentence boundaries to stay under max_chars."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    pieces, current = [], ""
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                pieces.append(current)
            if len(sent) > max_chars:
                # hard split at word boundary
                words, sub = sent.split(), ""
                for w in words:
                    if len((sub + " " + w).strip()) <= max_chars:
                        sub = (sub + " " + w).strip()
                    else:
                        if sub:
                            pieces.append(sub)
                        sub = w
                current = sub
            else:
                current = sent
    if current:
        pieces.append(current)
    return pieces if pieces else [text]


# ─────────────────────────────────────────────
# Step 1 — Parse markdown → Block list
# ─────────────────────────────────────────────

def parse_docling_blocks(text: str) -> list[Block]:
    """
    Parse Docling markdown into typed Block objects.

    Key Docling quirks handled:
    - Italic caption lines (*...*) immediately after an image → merged into figure block
    - <!-- formula-not-decoded --> → formula block
    - Short orphan lines (axis labels, author names) → merged into adjacent paragraph
    - Table separator rows are part of the table, not standalone
    """
    lines = text.splitlines(keepends=True)
    blocks: list[Block] = []

    in_table   = False
    in_list    = False
    buffer_lines: list[str] = []
    buffer_type = "paragraph"
    buffer_start = 0
    char_offset  = 0
    last_block_was_image = False   # track to catch italic captions

    def flush(end_offset: int):
        nonlocal buffer_lines, buffer_type, buffer_start, in_table, in_list
        if not buffer_lines:
            return
        content = "".join(buffer_lines).strip()
        if not content:
            buffer_lines = []
            in_table = in_list = False
            return
        imgs = [m.group(2) for m in RE_IMAGE.finditer(content)]
        has_f = bool(RE_FORMULA.search(content))
        blocks.append(Block(
            block_type=buffer_type,
            content=content,
            image_refs=imgs,
            has_formula=has_f,
            char_start=buffer_start,
            char_end=end_offset,
        ))
        buffer_lines = []
        in_table = in_list = False

    for line in lines:
        line_start  = char_offset
        char_offset += len(line)
        stripped    = line.rstrip("\n")

        # ── Heading ──────────────────────────────────────────
        h_match = RE_HEADING.match(stripped)
        if h_match:
            flush(line_start)
            last_block_was_image = False
            level = len(h_match.group(1))
            blocks.append(Block(
                block_type="heading",
                content=stripped,
                heading_level=level,
                char_start=line_start,
                char_end=char_offset,
            ))
            buffer_start = char_offset
            buffer_type  = "paragraph"
            continue

        # ── Formula placeholder ───────────────────────────────
        if RE_FORMULA.match(stripped):
            flush(line_start)
            last_block_was_image = False
            blocks.append(Block(
                block_type="formula",
                content=stripped,
                has_formula=True,
                char_start=line_start,
                char_end=char_offset,
            ))
            buffer_start = char_offset
            buffer_type  = "paragraph"
            continue

        # ── Image line ────────────────────────────────────────
        if RE_IMAGE.match(stripped):
            flush(line_start)
            imgs = [m.group(2) for m in RE_IMAGE.finditer(stripped)]
            blocks.append(Block(
                block_type="figure",
                content=stripped,
                image_refs=imgs,
                char_start=line_start,
                char_end=char_offset,
            ))
            last_block_was_image = True
            buffer_start = char_offset
            buffer_type  = "paragraph"
            continue

        # ── Italic caption (Docling: *caption text*) ─────────
        # Must appear right after an image block
        if RE_ITALIC_LINE.match(stripped) and last_block_was_image:
            # Merge caption into the preceding figure block
            if blocks and blocks[-1].block_type == "figure":
                blocks[-1].content += "\n" + stripped
                blocks[-1].char_end = char_offset
            last_block_was_image = False
            buffer_start = char_offset
            continue

        last_block_was_image = False

        # ── Table ─────────────────────────────────────────────
        if RE_TABLE_ROW.match(stripped):
            if not in_table:
                flush(line_start)
                in_table     = True
                buffer_type  = "table"
                buffer_start = line_start
            buffer_lines.append(line)
            continue
        else:
            if in_table:
                flush(char_offset)
                buffer_type  = "paragraph"
                buffer_start = char_offset

        # ── List item ─────────────────────────────────────────
        if RE_LIST_ITEM.match(stripped):
            if not in_list:
                flush(line_start)
                in_list      = True
                buffer_type  = "list"
                buffer_start = line_start
            buffer_lines.append(line)
            continue
        else:
            if in_list:
                flush(char_offset)
                buffer_type  = "paragraph"
                buffer_start = char_offset

        # ── Empty line ────────────────────────────────────────
        if stripped == "":
            if buffer_type in ("list",):
                flush(char_offset)
                buffer_type  = "paragraph"
                buffer_start = char_offset
            else:
                buffer_lines.append(line)
            continue

        # ── Short orphan line detection ───────────────────────
        # Docling emits floating fragments: footnote numbers, axis labels,
        # author names between figure captions and paragraphs.
        # Attach them to the current paragraph buffer rather than starting
        # a new block.
        if (
            len(stripped) < SHORT_LINE_CHARS
            and buffer_type == "paragraph"
            and buffer_lines                  # only if we already have content
            and not RE_HEADING.match(stripped)
        ):
            buffer_lines.append(line)
            continue

        # ── Default: paragraph ────────────────────────────────
        if buffer_type != "paragraph":
            flush(line_start)
            buffer_type  = "paragraph"
            buffer_start = line_start
        buffer_lines.append(line)

    flush(char_offset)
    return blocks


# ─────────────────────────────────────────────
# Step 2 — Heading hierarchy tracker
# ─────────────────────────────────────────────

def heading_text(block: Block) -> str:
    return re.sub(r'^#+\s*', '', block.content).strip()


def update_heading_stack(stack: list[Block], new_h: Block) -> list[Block]:
    """Keep only headings strictly shallower than new_h, then append."""
    stack = [h for h in stack if h.heading_level < new_h.heading_level]
    stack.append(new_h)
    return stack


# ─────────────────────────────────────────────
# Step 3 — Assemble chunks
# ─────────────────────────────────────────────

def blocks_to_chunks(blocks: list[Block], source_file: str, total_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading_stack: list[Block] = []
    chunk_idx = 0

    # Rolling paragraph buffer
    pb_content  = ""
    pb_start    = 0
    pb_imgs:  list[str] = []
    pb_has_f    = False
    pb_hpath: list[str] = []
    overlap_tail = ""

    def current_hpath():
        return [heading_text(h) for h in heading_stack]

    def flush_para(end_char: int):
        nonlocal pb_content, overlap_tail, pb_start, pb_imgs, pb_has_f, chunk_idx
        if not pb_content.strip():
            return

        full = (overlap_tail + "\n\n" + pb_content).strip() if overlap_tail else pb_content.strip()

        pieces = split_at_sentence_boundary(full, MAX_CHARS) if len(full) > MAX_CHARS else [full]

        for piece in pieces:
            page = infer_page_from_images(pb_imgs, total_chars, pb_start)
            chunks.append(Chunk(
                chunk_id    = str(uuid.uuid4()),
                chunk_index = chunk_idx,
                content     = piece,
                source_file = source_file,
                page_number = page,
                heading_path= list(pb_hpath),
                block_type  = "paragraph",
                image_refs  = list(pb_imgs),
                has_formula = pb_has_f,
                char_start  = pb_start,
                char_end    = end_char,
                token_count = approx_tokens(piece),
            ))
            chunk_idx += 1

        last = pieces[-1] if pieces else ""
        overlap_tail = last[-OVERLAP_CHARS:] if len(last) > OVERLAP_CHARS else last

        pb_content = ""
        pb_imgs    = []
        pb_has_f   = False

    for block in blocks:

        # ── Heading: flush buffer, update stack, don't emit standalone chunk ──
        if block.block_type == "heading":
            flush_para(block.char_start)
            overlap_tail = ""           # section boundary resets overlap
            heading_stack = update_heading_stack(heading_stack, block)
            pb_hpath = current_hpath()
            pb_start = block.char_end
            continue

        # ── Atomic blocks: table, figure, formula ─────────────────────────────
        if block.block_type in ("table", "figure", "formula"):
            flush_para(block.char_start)
            overlap_tail = ""
            page = infer_page_from_images(block.image_refs, total_chars, block.char_start)
            chunks.append(Chunk(
                chunk_id    = str(uuid.uuid4()),
                chunk_index = chunk_idx,
                content     = block.content,
                source_file = source_file,
                page_number = page,
                heading_path= current_hpath(),
                block_type  = block.block_type,
                image_refs  = block.image_refs,
                has_formula = block.has_formula,
                char_start  = block.char_start,
                char_end    = block.char_end,
                token_count = approx_tokens(block.content),
            ))
            chunk_idx += 1
            pb_start = block.char_end
            continue

        # ── Paragraph / list: accumulate ──────────────────────────────────────
        if not pb_content:
            pb_start = block.char_start
            pb_hpath = current_hpath()

        pb_content += ("\n\n" + block.content) if pb_content else block.content
        pb_imgs.extend(block.image_refs)
        pb_has_f = pb_has_f or block.has_formula

        if len(pb_content) >= MAX_CHARS:
            flush_para(block.char_end)

    flush_para(total_chars)
    return chunks


# ─────────────────────────────────────────────
# Step 4 — Post-process: table caption linking
# ─────────────────────────────────────────────

def link_table_captions(chunks: list[Chunk]) -> list[Chunk]:
    """
    Docling often emits a short paragraph BEFORE or AFTER a table that serves
    as its caption (e.g. 'Table 3. Error rates on ImageNet validation.').
    Detect these and merge into the table chunk so the table is never retrieved
    without its label.
    """
    result: list[Chunk] = []
    i = 0
    caption_re = re.compile(r'^Table\s+\d+[\.\:]', re.I)

    while i < len(chunks):
        c = chunks[i]

        # Paragraph immediately before a table that looks like a caption
        if (
            c.block_type == "paragraph"
            and i + 1 < len(chunks)
            and chunks[i+1].block_type == "table"
            and approx_tokens(c.content) < 80
            and caption_re.search(c.content)
        ):
            table = chunks[i+1]
            merged_content = c.content + "\n\n" + table.content
            result.append(Chunk(
                chunk_id    = table.chunk_id,
                chunk_index = table.chunk_index,
                content     = merged_content,
                source_file = table.source_file,
                page_number = table.page_number,
                heading_path= table.heading_path,
                block_type  = "table",
                image_refs  = table.image_refs,
                has_formula = table.has_formula,
                char_start  = c.char_start,
                char_end    = table.char_end,
                token_count = approx_tokens(merged_content),
            ))
            i += 2
            continue

        # Table followed by a short caption paragraph
        if (
            c.block_type == "table"
            and i + 1 < len(chunks)
            and chunks[i+1].block_type == "paragraph"
            and approx_tokens(chunks[i+1].content) < 80
            and caption_re.search(chunks[i+1].content)
        ):
            caption = chunks[i+1]
            merged_content = c.content + "\n\n" + caption.content
            result.append(Chunk(
                chunk_id    = c.chunk_id,
                chunk_index = c.chunk_index,
                content     = merged_content,
                source_file = c.source_file,
                page_number = c.page_number,
                heading_path= c.heading_path,
                block_type  = "table",
                image_refs  = c.image_refs,
                has_formula = c.has_formula,
                char_start  = c.char_start,
                char_end    = caption.char_end,
                token_count = approx_tokens(merged_content),
            ))
            i += 2
            continue

        result.append(c)
        i += 1

    return result


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def chunk_docling_markdown(
    markdown_path: str,
    output_path:   str = "chunks.json",
    max_tokens:    int = MAX_TOKENS,
    overlap_tokens:int = OVERLAP_TOKENS,
) -> list[dict]:
    """
    Full pipeline: Docling .md file → list of chunk dicts with metadata.

    Args:
        markdown_path  : path to Docling-generated .md file
        output_path    : output JSON path (set to None to skip writing)
        max_tokens     : max tokens per paragraph chunk (default 512)
        overlap_tokens : overlap between adjacent paragraph chunks (default 50)

    Returns:
        list of dicts, each a fully-annotated chunk ready for embedding
    """
    global MAX_TOKENS, OVERLAP_TOKENS, MAX_CHARS, OVERLAP_CHARS
    MAX_TOKENS     = max_tokens
    OVERLAP_TOKENS = overlap_tokens
    MAX_CHARS      = MAX_TOKENS * CHARS_PER_TOKEN
    OVERLAP_CHARS  = OVERLAP_TOKENS * CHARS_PER_TOKEN

    md_path     = Path(markdown_path)
    source_file = md_path.stem
    text        = md_path.read_text(encoding="utf-8")
    total_chars = len(text)

    # 1. Parse → blocks
    blocks = parse_docling_blocks(text)

    # 2. Blocks → chunks
    chunks = blocks_to_chunks(blocks, source_file, total_chars)

    # 3. Link table captions
    chunks = link_table_captions(chunks)

    # Re-index after merges
    for i, c in enumerate(chunks):
        c.chunk_index = i

    # 4. Serialise
    result = [asdict(c) for c in chunks]

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(result)} chunks → {out}")

    return result


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Docling layout-aware chunker for RAG")
    ap.add_argument("markdown",               help="Docling .md file")
    ap.add_argument("--output",  default="chunks.json", help="Output JSON file")
    ap.add_argument("--max-tokens",  type=int, default=512)
    ap.add_argument("--overlap",     type=int, default=50)
    ap.add_argument("--preview", action="store_true", help="Print first 5 chunks to stdout")
    args = ap.parse_args()

    chunks = chunk_docling_markdown(
        markdown_path  = args.markdown,
        output_path    = args.output,
        max_tokens     = args.max_tokens,
        overlap_tokens = args.overlap,
    )

    # Summary
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c["block_type"]] = by_type.get(c["block_type"], 0) + 1

    print(f"\n{'─'*50}")
    print(f"Total chunks : {len(chunks)}")
    print(f"Block types  :")
    for bt, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {bt:12s} {n:4d}")
    print(f"{'─'*50}")

    if args.preview:
        print("\nFirst 5 chunks:\n")
        for c in chunks[:5]:
            print(f"[{c['chunk_index']}] {c['block_type']:10s} | page {c['page_number']:>3} "
                  f"| tokens {c['token_count']:>4} | path: {' > '.join(c['heading_path']) or '(top)'}")
            print(f"    {c['content'][:120].replace(chr(10),' ')}...")
            if c['image_refs']:
                print(f"    images: {c['image_refs']}")
            if c['has_formula']:
                print(f"    ⚠ contains formula placeholder")
            print()