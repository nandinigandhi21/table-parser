"""
Docling Layout-Aware Chunker — LangChain Edition
=================================================
Same logic as the custom chunker but implemented entirely with LangChain
primitives, producing standard LangChain Document objects ready to drop
into any LangChain RAG pipeline (FAISS, Chroma, Pinecone, etc.).

LangChain components used:
  - Document                        : standard chunk container (page_content + metadata)
  - MarkdownHeaderTextSplitter      : heading-hierarchy splitting + breadcrumb metadata
  - RecursiveCharacterTextSplitter  : size-guard splitting at sentence/word boundaries
  - TextSplitter (base)             : subclassed for the custom Docling block pre-pass

Why we EXTEND LangChain rather than replace it:
  LangChain's splitters handle heading hierarchy and size control well.
  What they can't do is Docling-specific pre-processing:
    • Detect figure blocks and merge italic captions into them
    • Detect formula placeholders (<!-- formula-not-decoded -->)
    • Keep tables as atomic blocks (never split)
    • Infer page numbers from image filenames
    • Absorb orphan short lines into adjacent paragraphs
  So the architecture is:
    [Pre-pass: DoclingBlockPreprocessor]   ← custom, handles Docling quirks
            ↓  emits clean section strings
    [MarkdownHeaderTextSplitter]           ← LangChain, extracts heading metadata
            ↓  emits Documents with heading breadcrumb
    [RecursiveCharacterTextSplitter]       ← LangChain, enforces token size limit
            ↓  emits final sized Documents
    [DoclingMetadataEnricher]              ← custom, adds block_type/images/formula/page
            ↓
    Final LangChain Document list (ready for .add_documents())

Output Document fields:
  page_content   : chunk text
  metadata:
    chunk_id       : uuid
    chunk_index    : sequential int
    source_file    : .md filename stem
    page_number    : inferred from image filenames or char offset
    heading_path   : "H2 > H3" string  (also split into Header 1/2/3 by LangChain)
    block_type     : paragraph | table | figure | formula | list
    image_refs     : comma-separated image paths (empty string if none)
    has_formula    : True/False
    token_count    : approximate token count
    char_start     : character offset in original file
    char_end       : character offset end
"""

import re
import uuid
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

MAX_TOKENS       = 512
OVERLAP_TOKENS   = 50
CHARS_PER_TOKEN  = 4          # GPT/Claude tokenizers ≈ 4 chars per token
SHORT_LINE_CHARS = 60         # lines shorter than this are likely orphan fragments

MAX_CHARS        = MAX_TOKENS    * CHARS_PER_TOKEN
OVERLAP_CHARS    = OVERLAP_TOKENS * CHARS_PER_TOKEN


# ─────────────────────────────────────────────────────────────
# Regex patterns — Docling-specific
# ─────────────────────────────────────────────────────────────

RE_HEADING      = re.compile(r'^(#{1,6})\s+(.+)$')
RE_TABLE_ROW    = re.compile(r'^\|')
RE_IMAGE        = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
RE_ITALIC_CAP   = re.compile(r'^\*(.+)\*$')       # Docling figure captions
RE_FORMULA      = re.compile(r'^<!--\s*formula-not-decoded\s*-->')
RE_LIST_ITEM    = re.compile(r'^(\s*[-*+]|\s*\d+[.)]) ')
RE_PAGE_FROM_IMG= re.compile(r'page(\d+)', re.I)


# ─────────────────────────────────────────────────────────────
# Step 1 — Internal block representation
# ─────────────────────────────────────────────────────────────

@dataclass
class _Block:
    """Internal typed block before LangChain processing."""
    block_type:  str
    content:     str
    image_refs:  list = field(default_factory=list)
    has_formula: bool = False
    char_start:  int  = 0
    char_end:    int  = 0


def _infer_page(image_refs: list[str], total_chars: int, char_pos: int) -> int:
    """Infer page number from Docling image filenames (images/page5_fig1.png → 5)."""
    for ref in image_refs:
        m = RE_PAGE_FROM_IMG.search(ref)
        if m:
            return int(m.group(1))
    return max(1, char_pos // 3000 + 1)   # fallback: ~3000 chars per page


# ─────────────────────────────────────────────────────────────
# Step 2 — Docling pre-processor
#          Converts raw .md text into clean section strings
#          with atomic blocks (table/figure/formula) fenced
#          so LangChain splitters never break them.
# ─────────────────────────────────────────────────────────────

# Sentinel wrappers — we temporarily wrap atomic blocks so
# RecursiveCharacterTextSplitter treats them as unsplittable units.
_TABLE_START   = "\n\n<!-- ATOMIC:table:start -->\n"
_TABLE_END     = "\n<!-- ATOMIC:table:end -->\n\n"
_FIGURE_START  = "\n\n<!-- ATOMIC:figure:start -->\n"
_FIGURE_END    = "\n<!-- ATOMIC:figure:end -->\n\n"
_FORMULA_START = "\n\n<!-- ATOMIC:formula:start -->\n"
_FORMULA_END   = "\n<!-- ATOMIC:formula:end -->\n\n"

_ATOMIC_RE = re.compile(
    r'<!-- ATOMIC:(\w+):start -->(.*?)<!-- ATOMIC:\w+:end -->',
    re.DOTALL
)


def _preprocess_docling_md(text: str) -> str:
    """
    Transform Docling markdown into a cleaned string where:
      - Figure image lines are merged with their italic caption
      - Tables are wrapped in ATOMIC sentinels
      - Formulas are wrapped in ATOMIC sentinels
      - Short orphan lines are merged into adjacent paragraphs

    The result is valid markdown that MarkdownHeaderTextSplitter
    can process, with atomic blocks guaranteed to stay whole.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    in_table = False

    while i < len(lines):
        stripped = lines[i].rstrip("\n")

        # ── Formula ──────────────────────────────────────────
        if RE_FORMULA.match(stripped):
            if in_table:
                out.append(_TABLE_END)
                in_table = False
            out.append(_FORMULA_START)
            out.append(lines[i])
            out.append(_FORMULA_END)
            i += 1
            continue

        # ── Image line → merge with next italic caption ───────
        if RE_IMAGE.match(stripped):
            if in_table:
                out.append(_TABLE_END)
                in_table = False
            figure_content = lines[i]
            # look ahead for italic caption on next non-empty line
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines) and RE_ITALIC_CAP.match(lines[j].rstrip("\n")):
                figure_content = lines[i].rstrip("\n") + "\n" + lines[j]
                i = j + 1
            else:
                i += 1
            out.append(_FIGURE_START)
            out.append(figure_content)
            out.append(_FIGURE_END)
            continue

        # ── Table row ─────────────────────────────────────────
        if RE_TABLE_ROW.match(stripped):
            if not in_table:
                out.append(_TABLE_START)
                in_table = True
            out.append(lines[i])
            i += 1
            continue
        else:
            if in_table:
                out.append(_TABLE_END)
                in_table = False

        # ── Short orphan line — merge into previous content ───
        # (axis labels, footnote numbers, author names)
        if (
            stripped != ""
            and len(stripped) < SHORT_LINE_CHARS
            and not RE_HEADING.match(stripped)
            and not RE_LIST_ITEM.match(stripped)
            and out
        ):
            # Append to last output line instead of starting new paragraph
            # Find the last non-empty out entry and append inline
            out.append(lines[i])
            i += 1
            continue

        out.append(lines[i])
        i += 1

    if in_table:
        out.append(_TABLE_END)

    return "".join(out)


# ─────────────────────────────────────────────────────────────
# Step 3 — LangChain MarkdownHeaderTextSplitter
#          Produces Documents with heading breadcrumb metadata
# ─────────────────────────────────────────────────────────────

HEADERS_TO_SPLIT_ON = [
    ("#",      "Header_1"),
    ("##",     "Header_2"),
    ("###",    "Header_3"),
    ("####",   "Header_4"),
    ("#####",  "Header_5"),
    ("######", "Header_6"),
]


def _split_by_headers(preprocessed_text: str) -> list[Document]:
    """
    Use LangChain's MarkdownHeaderTextSplitter to split on headings.
    Returns Documents with Header_1..Header_6 metadata keys populated.
    strip_headers=False keeps the heading text inside the chunk content
    so we don't lose context when chunks are retrieved standalone.
    """
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,          # keep headings in content for context
        return_each_line=False,
    )
    return splitter.split_text(preprocessed_text)


# ─────────────────────────────────────────────────────────────
# Step 4 — LangChain RecursiveCharacterTextSplitter
#          Enforces MAX_CHARS limit, respects sentence/word boundaries
#          NEVER splits inside an ATOMIC block
# ─────────────────────────────────────────────────────────────

def _build_recursive_splitter() -> RecursiveCharacterTextSplitter:
    """
    RecursiveCharacterTextSplitter with separators tuned to:
      1. Never split inside ATOMIC sentinel blocks (tables/figures/formulas)
      2. Prefer splitting at paragraph boundaries
      3. Fall back to sentence → word boundaries
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHARS,
        chunk_overlap=OVERLAP_CHARS,
        separators=[
            # Atomic block boundaries — highest priority
            _TABLE_END,
            _FIGURE_END,
            _FORMULA_END,
            # Paragraph boundary
            "\n\n",
            # Sentence boundary
            ". ",
            "! ",
            "? ",
            # Word boundary (last resort)
            " ",
            "",
        ],
        keep_separator=True,
        is_separator_regex=False,
        length_function=len,
        add_start_index=True,    # LangChain adds 'start_index' to metadata
    )


def _size_guard_split(header_docs: list[Document]) -> list[Document]:
    """
    Apply RecursiveCharacterTextSplitter to each header-split Document.
    Atomic blocks that individually exceed MAX_CHARS are kept whole
    (tables/figures should not be truncated even if large).
    """
    splitter = _build_recursive_splitter()
    result: list[Document] = []

    for doc in header_docs:
        content = doc.page_content

        # Check if this doc is entirely one atomic block — keep whole
        atomic_match = _ATOMIC_RE.fullmatch(content.strip())
        if atomic_match:
            result.append(doc)
            continue

        # Split the content
        sub_docs = splitter.create_documents(
            texts=[content],
            metadatas=[doc.metadata],
        )
        result.extend(sub_docs)

    return result


# ─────────────────────────────────────────────────────────────
# Step 5 — Metadata enricher
#          Detects block_type, image_refs, formula flag, page number
#          Adds chunk_id, chunk_index, heading_path string
# ─────────────────────────────────────────────────────────────

def _detect_block_type(content: str) -> str:
    """Detect dominant block type from chunk content."""
    c = content.strip()
    if "<!-- ATOMIC:table"   in c: return "table"
    if "<!-- ATOMIC:figure"  in c: return "figure"
    if "<!-- ATOMIC:formula" in c: return "formula"
    if RE_LIST_ITEM.search(c):     return "list"
    return "paragraph"


def _extract_images(content: str) -> list[str]:
    return [m.group(2) for m in RE_IMAGE.finditer(content)]


def _strip_atomic_sentinels(content: str) -> str:
    """Remove ATOMIC wrapper comments from final chunk content."""
    content = content.replace(_TABLE_START,   "")
    content = content.replace(_TABLE_END,     "")
    content = content.replace(_FIGURE_START,  "")
    content = content.replace(_FIGURE_END,    "")
    content = content.replace(_FORMULA_START, "")
    content = content.replace(_FORMULA_END,   "")
    return content.strip()


def _build_heading_path(metadata: dict) -> str:
    """Build 'H1 > H2 > H3' breadcrumb string from LangChain header metadata."""
    parts = []
    for key in ["Header_1", "Header_2", "Header_3", "Header_4", "Header_5", "Header_6"]:
        if key in metadata and metadata[key]:
            parts.append(metadata[key])
    return " > ".join(parts)


def _enrich_metadata(
    docs: list[Document],
    source_file: str,
    total_chars: int,
) -> list[Document]:
    """
    Add Docling-specific metadata to each LangChain Document:
      block_type, image_refs, has_formula, page_number,
      heading_path, chunk_id, chunk_index, token_count,
      char_start, char_end, source_file
    """
    enriched: list[Document] = []

    for idx, doc in enumerate(docs):
        raw_content = doc.page_content
        clean_content = _strip_atomic_sentinels(raw_content)

        if not clean_content.strip():
            continue

        block_type  = _detect_block_type(raw_content)
        image_refs  = _extract_images(clean_content)
        has_formula = bool(RE_FORMULA.search(clean_content))

        # char_start from LangChain's add_start_index feature
        char_start  = doc.metadata.get("start_index", 0)
        char_end    = char_start + len(clean_content)
        page_number = _infer_page(image_refs, total_chars, char_start)

        heading_path = _build_heading_path(doc.metadata)

        # Build enriched metadata dict
        new_metadata = {
            # LangChain heading fields (kept for compatibility with vector stores)
            **{k: v for k, v in doc.metadata.items()
               if k.startswith("Header_")},
            # Our enriched fields
            "chunk_id":     str(uuid.uuid4()),
            "chunk_index":  idx,
            "source_file":  source_file,
            "page_number":  page_number,
            "heading_path": heading_path,
            "block_type":   block_type,
            "image_refs":   ", ".join(image_refs),   # string for vector store compat
            "has_formula":  has_formula,
            "token_count":  max(1, len(clean_content) // CHARS_PER_TOKEN),
            "char_start":   char_start,
            "char_end":     char_end,
        }

        enriched.append(Document(
            page_content=clean_content,
            metadata=new_metadata,
        ))

    return enriched


# ─────────────────────────────────────────────────────────────
# Step 6 — Table caption linking
#          Merge "Table N. ..." label paragraphs with table chunks
# ─────────────────────────────────────────────────────────────

_CAPTION_RE = re.compile(r'^Table\s+\d+[\.\:]', re.I)


def _link_table_captions(docs: list[Document]) -> list[Document]:
    """
    If a short paragraph immediately before or after a table chunk
    looks like a table caption ('Table 3. Error rates...'),
    merge it into the table chunk.
    """
    result: list[Document] = []
    i = 0
    while i < len(docs):
        doc = docs[i]

        # Short paragraph BEFORE a table
        if (
            doc.metadata.get("block_type") == "paragraph"
            and i + 1 < len(docs)
            and docs[i+1].metadata.get("block_type") == "table"
            and doc.metadata.get("token_count", 999) < 80
            and _CAPTION_RE.search(doc.page_content)
        ):
            table = docs[i+1]
            merged = doc.page_content + "\n\n" + table.page_content
            new_meta = {**table.metadata,
                        "content": merged,
                        "token_count": max(1, len(merged) // CHARS_PER_TOKEN),
                        "char_start": doc.metadata.get("char_start", 0)}
            result.append(Document(page_content=merged, metadata=new_meta))
            i += 2
            continue

        # Table followed by short caption paragraph
        if (
            doc.metadata.get("block_type") == "table"
            and i + 1 < len(docs)
            and docs[i+1].metadata.get("block_type") == "paragraph"
            and docs[i+1].metadata.get("token_count", 999) < 80
            and _CAPTION_RE.search(docs[i+1].page_content)
        ):
            caption = docs[i+1]
            merged = doc.page_content + "\n\n" + caption.page_content
            new_meta = {**doc.metadata,
                        "token_count": max(1, len(merged) // CHARS_PER_TOKEN),
                        "char_end": caption.metadata.get("char_end", 0)}
            result.append(Document(page_content=merged, metadata=new_meta))
            i += 2
            continue

        result.append(doc)
        i += 1

    # Re-index after merges
    for idx, doc in enumerate(result):
        doc.metadata["chunk_index"] = idx

    return result


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def chunk_docling_markdown_langchain(
    markdown_path:  str,
    output_path:    str = "chunks_langchain.json",
    max_tokens:     int = MAX_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[Document]:
    """
    Full LangChain pipeline: Docling .md → list[Document]

    Pipeline:
      1. DoclingPreprocessor  — merge figure+captions, fence atomic blocks
      2. MarkdownHeaderTextSplitter  — split on headings, extract breadcrumb
      3. RecursiveCharacterTextSplitter  — enforce token size limit
      4. MetadataEnricher  — add block_type, page, images, formula flag
      5. TableCaptionLinker  — merge table labels into table chunks

    Args:
        markdown_path   : path to Docling-generated .md file
        output_path     : save chunks as JSON (None to skip)
        max_tokens      : max tokens per paragraph chunk (default 512)
        overlap_tokens  : sentence overlap between chunks (default 50)

    Returns:
        list[Document]  — drop directly into vectorstore.add_documents()
    """
    global MAX_CHARS, OVERLAP_CHARS
    MAX_CHARS    = max_tokens    * CHARS_PER_TOKEN
    OVERLAP_CHARS = overlap_tokens * CHARS_PER_TOKEN

    md_path     = Path(markdown_path)
    source_file = md_path.stem
    text        = md_path.read_text(encoding="utf-8")
    total_chars = len(text)

    print(f"[1/5] Pre-processing Docling markdown ({total_chars:,} chars)...")
    preprocessed = _preprocess_docling_md(text)

    print(f"[2/5] Splitting by headers (MarkdownHeaderTextSplitter)...")
    header_docs = _split_by_headers(preprocessed)
    print(f"      → {len(header_docs)} header-level sections")

    print(f"[3/5] Applying size guard (RecursiveCharacterTextSplitter, max {max_tokens} tokens)...")
    sized_docs = _size_guard_split(header_docs)
    print(f"      → {len(sized_docs)} sized chunks")

    print(f"[4/5] Enriching metadata (block_type, page, images, formula)...")
    enriched_docs = _enrich_metadata(sized_docs, source_file, total_chars)
    print(f"      → {len(enriched_docs)} enriched chunks")

    print(f"[5/5] Linking table captions...")
    final_docs = _link_table_captions(enriched_docs)
    print(f"      → {len(final_docs)} final chunks")

    # Save to JSON
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        serializable = [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in final_docs
        ]
        out.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8"
        )
        print(f"\nSaved → {out}")

    return final_docs


# ─────────────────────────────────────────────────────────────
# Vector store integration helpers
# ─────────────────────────────────────────────────────────────

def load_into_chroma(docs: list[Document], collection_name: str = "docling_rag"):
    """
    Example: load chunks into ChromaDB.
    pip install chromadb langchain-chroma
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
    )
    print(f"Loaded {len(docs)} chunks into ChromaDB collection '{collection_name}'")
    return vectorstore


def load_into_faiss(docs: list[Document]):
    """
    Example: load chunks into FAISS (in-memory).
    pip install faiss-cpu langchain-community
    """
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    print(f"Loaded {len(docs)} chunks into FAISS")
    return vectorstore


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Docling layout-aware chunker — LangChain edition"
    )
    ap.add_argument("markdown",                   help="Docling .md file")
    ap.add_argument("--output",  default="chunks_langchain.json")
    ap.add_argument("--max-tokens",  type=int, default=512)
    ap.add_argument("--overlap",     type=int, default=50)
    ap.add_argument("--preview",     action="store_true",
                    help="Print all chunks summary + first 3 full chunks")
    args = ap.parse_args()

    docs = chunk_docling_markdown_langchain(
        markdown_path  = args.markdown,
        output_path    = args.output,
        max_tokens     = args.max_tokens,
        overlap_tokens = args.overlap,
    )

    # ── Summary ──────────────────────────────────────────────
    by_type: dict[str, int] = {}
    for d in docs:
        bt = d.metadata.get("block_type", "unknown")
        by_type[bt] = by_type.get(bt, 0) + 1

    print(f"\n{'─'*60}")
    print(f"Total chunks  : {len(docs)}")
    print(f"Block types   :")
    for bt, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {bt:12s}  {n:4d}")
    print(f"{'─'*60}")

    if args.preview:
        print("\nAll chunks:\n")
        for d in docs:
            m = d.metadata
            path = m.get("heading_path", "")
            imgs = f"  imgs: {m['image_refs']}" if m.get("image_refs") else ""
            form = "  [FORMULA]" if m.get("has_formula") else ""
            print(
                f"[{m['chunk_index']:02d}] {m['block_type']:10s} | "
                f"pg {m['page_number']:>2} | "
                f"{m['token_count']:>4} tok | "
                f"{path[:55]}"
                f"{imgs}{form}"
            )

        print("\n\nFirst 3 full Documents:\n")
        for d in docs[:3]:
            print("=" * 55)
            print(f"page_content:\n{d.page_content[:300]}{'...' if len(d.page_content) > 300 else ''}")
            print(f"\nmetadata:")
            for k, v in d.metadata.items():
                if k not in ("chunk_id",):   # skip uuid noise in preview
                    print(f"  {k:15s}: {v}")
            print()