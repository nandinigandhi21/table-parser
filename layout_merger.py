"""
layout_merger.py
Merges text blocks (from Docling via text_extractor) and tables (from
pdfplumber via table_extractor) in correct reading order (page → y-position).

Produces a single layout-aware Markdown string.
"""


# ── Markdown renderers ────────────────────────────────────────────────────────

KIND_TO_MD = {
    "TITLE"    : "# {text}",
    "H1"       : "## {text}",
    "H2"       : "### {text}",
    "H3"       : "#### {text}",
    "BOLD"     : "**{text}**",
    "PARAGRAPH": "{text}",
    "LIST_ITEM": "- {text}",
    "CAPTION"  : "*{text}*",
    "FOOTNOTE" : "> {text}",
}


def render_element(element: dict) -> str:
    """Convert a single layout element (text block or table) to Markdown."""
    if element["type"] == "table":
        return element["markdown"]

    kind     = element.get("kind", "PARAGRAPH")
    text     = element["text"].strip()
    template = KIND_TO_MD.get(kind, "{text}")
    return template.format(text=text)


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_layout(text_blocks: list[dict], tables: list[dict]) -> list[dict]:
    """
    Merge text blocks and tables into a single list sorted by (page, top).
    """
    elements = []

    for block in text_blocks:
        elements.append({
            "type"  : "text",
            "page"  : block["page"],
            "top"   : block["top"],
            "bottom": block["bottom"],
            "kind"  : block["kind"],
            "text"  : block["text"],
        })

    for table in tables:
        elements.append({
            "type"    : "table",
            "page"    : table["page"],
            "top"     : table["top"],
            "bottom"  : table["bottom"],
            "markdown": table["markdown"],
        })

    elements.sort(key=lambda e: (e["page"], e["top"]))
    return elements


# ── Markdown builder ──────────────────────────────────────────────────────────

def build_markdown(text_blocks: list[dict], tables: list[dict]) -> str:
    """
    Build a layout-aware Markdown document from text blocks and tables.

    Args:
        text_blocks: output of text_extractor.extract_text_blocks()
        tables     : output of table_extractor.extract_tables()

    Returns:
        Single Markdown string with headings, paragraphs, lists, and tables
        in their original document reading order.
    """
    elements  = merge_layout(text_blocks, tables)
    lines     = []
    prev_page = None
    prev_kind = None

    for el in elements:
        # Page break marker
        if prev_page is not None and el["page"] != prev_page:
            lines.append(f"\n---\n*Page {el['page']}*\n")

        md = render_element(el)
        if not md.strip():
            prev_page = el["page"]
            prev_kind = el.get("kind")
            continue

        cur_kind = el.get("kind") if el["type"] == "text" else "table"

        # Blank line between different element types
        if lines and prev_kind != cur_kind:
            lines.append("")

        lines.append(md)
        lines.append("")

        prev_page = el["page"]
        prev_kind = cur_kind

    return "\n".join(lines).strip()