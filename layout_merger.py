"""
layout_merger.py
Merges text blocks (from Docling via text_extractor), tables (from
pdfplumber via table_extractor), images (from PyMuPDF via
image_extractor), and formulas (from Docling via formula_extractor)
in correct reading order (page → y-position).

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
    """
    Convert a single layout element to Markdown.
    Handles: text block, table, image, formula.
    """

    # ── Table ─────────────────────────────────────────────────────────────────
    if element["type"] == "table":
        return element["markdown"]

    # ── Image ─────────────────────────────────────────────────────────────────
    if element["type"] == "image":
        path    = element["path"]         # e.g. "images/page1_img1.png"
        w       = element.get("width",  "")
        h       = element.get("height", "")
        caption = element.get("caption", "").strip()

        # Alt text: use caption if present, else generic label
        alt = caption if caption else (
            f"Image (page {element['page']}, {w}×{h}px)"
            if w and h else f"Image (page {element['page']})"
        )

        img_md = f"![{alt}]({path})"

        # Render caption as italic line below the image
        if caption:
            img_md += f"\n*{caption}*"

        return img_md

    # ── Formula ───────────────────────────────────────────────────────────────
    if element["type"] == "formula":
        latex      = element.get("latex",     "").strip()
        text       = element.get("text",      "").strip()
        crop_path  = element.get("crop_path", "").strip()

        parts = []
        # Show crop image if available
        if crop_path:
            parts.append(f"![Formula (page {element['page']})]({crop_path})")
        # Render LaTeX block if available, else plain text fallback
        if latex:
            parts.append(f"$$\n{latex}\n$$")
        elif text:
            parts.append(f"`{text}`")

        return "\n".join(parts) if parts else ""

    # ── Text block ────────────────────────────────────────────────────────────
    kind     = element.get("kind", "PARAGRAPH")
    text     = element["text"].strip()
    template = KIND_TO_MD.get(kind, "{text}")
    return template.format(text=text)


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_layout(
    text_blocks: list[dict],
    tables     : list[dict],
    images     : list[dict] | None = None,
    formulas   : list[dict] | None = None,
) -> list[dict]:
    """
    Merge text blocks, tables, images, and formulas into a single list
    sorted by (page, top) for correct reading order.
    """
    elements: list[dict] = []

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

    for image in (images or []):
        elements.append({
            "type"   : "image",
            "page"   : image["page"],
            "top"    : image["top"],
            "bottom" : image["bottom"],
            "path"   : image["path"],
            "width"  : image.get("width",   0),
            "height" : image.get("height",  0),
            "caption": image.get("caption", ""),
        })

    for formula in (formulas or []):
        elements.append({
            "type"     : "formula",
            "page"     : formula["page"],
            "top"      : formula["top"],
            "bottom"   : formula["bottom"],
            "kind"     : "FORMULA",
            "latex"    : formula.get("latex",     ""),
            "text"     : formula.get("text",      ""),
            "crop_path": formula.get("crop_path", ""),
        })

    elements.sort(key=lambda e: (e["page"], e["top"]))
    return elements


# ── Markdown builder ──────────────────────────────────────────────────────────

def build_markdown(
    text_blocks: list[dict],
    tables     : list[dict],
    images     : list[dict] | None = None,
    formulas   : list[dict] | None = None,
) -> str:
    """
    Build a layout-aware Markdown document from text blocks, tables,
    images, and formulas.

    Args:
        text_blocks : output of text_extractor.extract_text_blocks()
        tables      : output of table_extractor.extract_tables()
        images      : output of image_extractor.extract_images()     (optional)
        formulas    : output of formula_extractor.extract_formulas() (optional)

    Returns:
        Single Markdown string with all elements in original reading order.
    """
    elements  = merge_layout(text_blocks, tables, images, formulas)
    lines: list[str] = []
    prev_page = None
    prev_kind = None

    for el in elements:
        # ── Page break marker ─────────────────────────────────────────────────
        if prev_page is not None and el["page"] != prev_page:
            lines.append(f"\n---\n*Page {el['page']}*\n")

        md = render_element(el)
        if not md.strip():
            prev_page = el["page"]
            prev_kind = el.get("kind") or el["type"]
            continue

        cur_kind = el.get("kind") if el["type"] == "text" else el["type"]

        # ── Blank line between different element types ────────────────────────
        if lines and prev_kind != cur_kind:
            lines.append("")

        lines.append(md)
        lines.append("")

        prev_page = el["page"]
        prev_kind = cur_kind

    return "\n".join(lines).strip()