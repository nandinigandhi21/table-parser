"""
image_extractor.py
Extracts all embedded images from a PDF using PyMuPDF (fitz).

Caption detection strategy:
  After locating each image's bounding box, we search the text blocks
  on the same page that appear JUST BELOW the image (within CAPTION_GAP_PT
  points). A block is treated as a caption if it:
    1. Starts with a known caption prefix  (Figure / Fig / Image / Plate / Chart)
    OR
    2. Is the nearest text block below the image AND is short (≤ CAPTION_MAX_CHARS)

Returns each image as a dict with:
  - page    : page number (1-based)
  - top     : y-coordinate of image top  (top-down)
  - bottom  : y-coordinate of image bottom (top-down)
  - path    : relative path  e.g. "images/page1_img1.png"
  - width   : image width  in pixels
  - height  : image height in pixels
  - index   : image index on that page (1-based)
  - caption : caption string or "" if none found

Images are saved to <output_dir>/images/.
Tiny images (likely icons / decorators) are filtered out automatically.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF


# ── Config ────────────────────────────────────────────────────────────────────

MIN_WIDTH  = 60    # px  — images smaller than this are likely icons/noise
MIN_HEIGHT = 60    # px

CAPTION_GAP_PT    = 30   # pt  — max vertical gap between image bottom and caption
CAPTION_MAX_CHARS = 200  # characters — loose upper bound for a caption line

# Regex: caption prefix patterns  (case-insensitive)
_CAPTION_PREFIX = re.compile(
    r"^\s*(figure|fig\.?|image|img\.?|plate|chart|diagram|illustration|photo|exhibit)\s*[\d\.\-:]?",
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bbox_to_topdown(rect: fitz.Rect) -> tuple[float, float]:
    """
    PyMuPDF already uses TOP-LEFT origin so y0 < y1 = top-down.
    Returns (top, bottom).
    """
    return float(rect.y0), float(rect.y1)


def _get_page_text_blocks(page: fitz.Page) -> list[dict]:
    """
    Extract all text blocks on the page as:
        {top, bottom, x0, x1, text}
    Uses PyMuPDF's get_text("blocks") for speed.
    Block format: (x0, y0, x1, y1, text, block_no, block_type)
    block_type 0 = text, 1 = image
    """
    blocks = []
    for b in page.get_text("blocks"):
        x0, y0, x1, y1, text, _bno, btype = b
        if btype != 0:          # skip image blocks
            continue
        text = text.strip()
        if not text:
            continue
        blocks.append({
            "top"   : float(y0),
            "bottom": float(y1),
            "x0"    : float(x0),
            "x1"    : float(x1),
            "text"  : text,
        })
    return blocks


def _find_caption(img_top: float, img_bottom: float,
                  img_x0: float,  img_x1: float,
                  text_blocks: list[dict]) -> str:
    """
    Search text_blocks for a caption that belongs to this image.

    Search zones (in priority order):
      1. BELOW the image — within CAPTION_GAP_PT pts of the image bottom.
         Preferred: block starts with a known caption prefix.
         Fallback : nearest short block (≤ CAPTION_MAX_CHARS) that is
                    horizontally centred under the image (overlap ≥ 40% of
                    block width, not just a 10px touch).
      2. INSIDE the image bbox (bottom 20% of the image height) — some PDFs
         typeset the caption text inside the image bounding box.
      3. ABOVE the image — within CAPTION_GAP_PT pts of the image top, only
         if it starts with a known caption prefix (avoids stealing headings).

    Returns the caption string, or "" if nothing qualifies.
    """
    img_width  = img_x1 - img_x0
    img_height = img_bottom - img_top
    candidates = []

    for blk in text_blocks:
        blk_top    = blk["top"]
        blk_bottom = blk["bottom"]
        blk_width  = blk["x1"] - blk["x0"]
        text       = blk["text"].replace("\n", " ").strip()
        if not text:
            continue

        has_prefix = bool(_CAPTION_PREFIX.match(text))
        is_short   = len(text) <= CAPTION_MAX_CHARS

        # ── Horizontal overlap helpers ────────────────────────────────────────
        h_overlap     = min(blk["x1"], img_x1) - max(blk["x0"], img_x0)
        # Overlap fraction relative to the block's own width (centring check)
        overlap_ratio = (h_overlap / blk_width) if blk_width > 0 else 0.0

        # ── Zone 1: BELOW the image ───────────────────────────────────────────
        if blk_top >= img_bottom - 2:
            gap = blk_top - img_bottom
            if gap <= CAPTION_GAP_PT and h_overlap > 0:
                if has_prefix:
                    # Caption prefix → accept with any overlap
                    candidates.append((0, gap, False, text))
                elif is_short and overlap_ratio >= 0.4:
                    # Short block centred below image → likely caption
                    candidates.append((0, gap, True, text))

        # ── Zone 2: INSIDE the image bbox (bottom 20% of image height) ───────
        elif (blk_top >= img_bottom - img_height * 0.20
              and blk_bottom <= img_bottom + 5
              and h_overlap > 0):
            if has_prefix or is_short:
                candidates.append((1, blk_top, not has_prefix, text))

        # ── Zone 3: ABOVE the image (prefix-only, avoids stealing headings) ──
        elif blk_bottom <= img_top + 2:
            gap = img_top - blk_bottom
            if gap <= CAPTION_GAP_PT and has_prefix and h_overlap > 0:
                candidates.append((2, gap, False, text))

    if not candidates:
        return ""

    # Sort: zone asc → gap/position asc → non-prefix last
    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    return candidates[0][3]


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_images(pdf_path: str, output_dir: str = "output") -> list[dict]:
    """
    Extract all embedded images from a PDF, detect captions, and save
    images to <output_dir>/images/.

    Args:
        pdf_path   : Path to the PDF file.
        output_dir : Root output folder (images saved to output_dir/images/).

    Returns:
        List of dicts sorted by (page, top):
            {page, top, bottom, path, width, height, index, caption}
        where `path` is relative to output_dir → "images/page1_img1.png"
    """
    pdf_path   = Path(pdf_path)
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    doc = fitz.open(str(pdf_path))

    try:
        for page_num in range(len(doc)):
            page       = doc[page_num]
            page_label = page_num + 1           # 1-based
            image_list = page.get_images(full=True)

            if not image_list:
                continue

            # Pre-fetch all text blocks on this page for caption detection
            text_blocks = _get_page_text_blocks(page)

            # Build xref → rects map for position lookup
            xref_to_rects: dict[int, list[fitz.Rect]] = {}
            for img_info in image_list:
                xref = img_info[0]
                try:
                    rects = page.get_image_rects(xref)
                    if rects:
                        xref_to_rects[xref] = rects
                except Exception:
                    xref_to_rects[xref] = [page.rect]  # fallback: full page

            img_index  = 0
            seen_xrefs: set[int] = set()

            for img_info in image_list:
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                # ── Extract raw bytes ─────────────────────────────────────────
                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    continue

                img_bytes = base_image.get("image")
                img_ext   = base_image.get("ext", "png")
                img_w     = base_image.get("width",  0)
                img_h     = base_image.get("height", 0)

                # ── Normalise extension to web-safe formats ───────────────────
                # PyMuPDF can return exotic types (jb2, jpx, ccitt, jxr) that
                # browsers / Markdown renderers don't support. Convert to png.
                _EXT_MAP = {
                    "jpg" : "jpeg",
                    "jb2" : "png",
                    "jpx" : "jpeg",
                    "jxr" : "png",
                    "ccitt": "png",
                    "smask": "png",
                }
                orig_ext = img_ext.lower()
                img_ext  = _EXT_MAP.get(orig_ext, orig_ext)
                if img_ext not in ("jpeg", "png", "gif", "webp"):
                    img_ext = "png"

                # If we changed the format, re-encode via Pillow to get valid bytes
                if img_ext != orig_ext and orig_ext not in ("jpg",):
                    try:
                        from PIL import Image as _PILImage
                        import io as _io
                        _pil = _PILImage.open(_io.BytesIO(img_bytes)).convert("RGB")
                        _buf = _io.BytesIO()
                        _pil.save(_buf, format="PNG")
                        img_bytes = _buf.getvalue()
                        img_ext   = "png"
                    except Exception:
                        pass   # keep original bytes if Pillow unavailable

                if not img_bytes:
                    continue

                # ── Filter tiny / noise images ────────────────────────────────
                if img_w < MIN_WIDTH or img_h < MIN_HEIGHT:
                    continue

                img_index += 1

                # ── Save image file ───────────────────────────────────────────
                filename  = f"page{page_label}_img{img_index}.{img_ext}"
                save_path = images_dir / filename
                relative  = f"images/{filename}"

                save_path.write_bytes(img_bytes)

                # ── Get position ──────────────────────────────────────────────
                rects = xref_to_rects.get(xref)
                if rects:
                    rect        = rects[0]
                    top, bottom = _bbox_to_topdown(rect)
                    img_x0      = float(rect.x0)
                    img_x1      = float(rect.x1)
                else:
                    top, bottom = 0.0, float(page.rect.height)
                    img_x0, img_x1 = 0.0, float(page.rect.width)

                # ── Detect caption ────────────────────────────────────────────
                caption = _find_caption(top, bottom, img_x0, img_x1, text_blocks)

                results.append({
                    "page"   : page_label,
                    "top"    : top,
                    "bottom" : bottom,
                    "path"   : relative,
                    "width"  : img_w,
                    "height" : img_h,
                    "index"  : img_index,
                    "caption": caption,
                })

    finally:
        doc.close()

    results.sort(key=lambda x: (x["page"], x["top"]))
    return results