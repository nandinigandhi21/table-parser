"""
image_extractor.py
──────────────────
Extracts ALL figures from a PDF using a two-pass strategy:

  PASS 1 — Raster images (PyMuPDF / fitz)
    Extracts embedded JPEG/PNG/etc objects directly from the PDF binary.
    Fast, lossless, exact pixels.
    Works for: scanned PDFs, papers with photos, any raster-embedded image.

  PASS 2 — Vector figures (pdf2image + pdfplumber)
    For figures drawn as PDF vector graphics (charts, diagrams, plots),
    there are no embedded image objects — only rects, curves and lines.
    We detect these by:
      a) Finding figure captions ("Figure N.", "Fig. N.") with pdfplumber.
      b) Clustering graphic objects (rects/curves/lines) on the page to
         find where the figure actually is.
      c) Matching each caption to its graphic cluster.
      d) Rendering that page region at 150 DPI using pdf2image and saving as PNG.
    Skips captions already covered by Pass 1.

Returns list of dicts:
    {
      page    : page number (1-based)
      top     : y-coordinate of figure top  (pt, top-down)
      bottom  : y-coordinate of figure bottom
      caption : caption string or ""
      path    : relative path  e.g. "images/page1_fig1.png"
      width   : saved image width  (px)
      height  : saved image height (px)
    }

Standalone:
    python image_extractor.py <pdf_path> [output_dir]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pdfplumber
import pypdfium2 as pdfium
from PIL import Image


# ── Config ────────────────────────────────────────────────────────────────────

# Raster extraction
RASTER_MIN_W  = 60    # px — skip images smaller than this (icons/noise)
RASTER_MIN_H  = 60

# Caption detection
CAPTION_GAP_PT    = 35   # pt — max gap between figure bottom and caption
CAPTION_MAX_CHARS = 250  # loose upper bound for a caption line

# Vector extraction
DPI              = 150   # render resolution for vector figures
CLUSTER_GAP_PT   = 60    # vertical gap that splits two graphic clusters
MIN_FIG_HEIGHT   = 30    # pt — skip figure regions shorter than this
MIN_FIG_WIDTH    = 40    # pt — skip figure regions narrower than this
PADDING_PT       = 6     # extra padding around cropped figure region

# Caption regex — matches "Figure 1.", "Fig.2:", "Figure3 -" etc.
_CAPTION_RE = re.compile(
    r'^(figure|fig\.?)\s*\d+[\.\:\-\s]',
    re.IGNORECASE,
)
_CAPTION_RE_CONCAT = re.compile(       # handles "Figure1." with no space
    r'^(figure|fig\.?)\d+[\.\:\-]',
    re.IGNORECASE,
)
# PyMuPDF-style caption prefix (broader — used in Pass 1)
_CAPTION_PREFIX = re.compile(
    r"^\s*(figure|fig\.?|image|img\.?|plate|chart|diagram|illustration|photo|exhibit)\s*[\d\.\-:]?",
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════════════════
#  PASS 1 — RASTER IMAGE EXTRACTION  (PyMuPDF)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_raster(pdf_path: str,
                    images_dir: Path) -> list[dict]:
    """
    Extract embedded raster images using PyMuPDF.
    Returns list of image dicts. Returns [] if fitz not available.
    """
    try:
        import fitz
    except ImportError:
        return []   # PyMuPDF not installed — skip Pass 1

    results: list[dict] = []
    doc = fitz.open(str(pdf_path))

    try:
        for page_num in range(len(doc)):
            page       = doc[page_num]
            page_label = page_num + 1
            image_list = page.get_images(full=True)
            if not image_list:
                continue

            # Text blocks for caption detection
            text_blocks = []
            for b in page.get_text("blocks"):
                x0, y0, x1, y1, text, _bno, btype = b
                if btype == 0 and text.strip():
                    text_blocks.append({
                        "top": float(y0), "bottom": float(y1),
                        "x0" : float(x0), "x1"    : float(x1),
                        "text": text.strip(),
                    })

            # xref → rect mapping
            xref_rects: dict[int, list] = {}
            for img_info in image_list:
                xref = img_info[0]
                try:
                    rects = page.get_image_rects(xref)
                    if rects:
                        xref_rects[xref] = rects
                except Exception:
                    xref_rects[xref] = [page.rect]

            seen: set[int] = set()
            img_idx = 0

            for img_info in image_list:
                xref = img_info[0]
                if xref in seen:
                    continue
                seen.add(xref)

                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    continue

                img_bytes = base_image.get("image")
                img_ext   = base_image.get("ext", "png")
                img_w     = base_image.get("width",  0)
                img_h     = base_image.get("height", 0)

                if not img_bytes:
                    continue
                if img_w < RASTER_MIN_W or img_h < RASTER_MIN_H:
                    continue

                img_idx += 1
                filename  = f"page{page_label}_img{img_idx}.{img_ext}"
                save_path = images_dir / filename
                save_path.write_bytes(img_bytes)

                rects = xref_rects.get(xref)
                if rects:
                    rect   = rects[0]
                    top    = float(rect.y0)
                    bottom = float(rect.y1)
                    x0     = float(rect.x0)
                    x1     = float(rect.x1)
                else:
                    top, bottom = 0.0, float(page.rect.height)
                    x0, x1 = 0.0, float(page.rect.width)

                caption = _caption_below(top, bottom, x0, x1, text_blocks)

                results.append({
                    "page"   : page_label,
                    "top"    : top,
                    "bottom" : bottom,
                    "caption": caption,
                    "path"   : f"images/{filename}",
                    "width"  : img_w,
                    "height" : img_h,
                })
    finally:
        doc.close()

    return results


def _caption_below(img_top: float, img_bottom: float,
                   img_x0: float,  img_x1: float,
                   text_blocks: list[dict]) -> str:
    """Find caption text block below an image."""
    candidates = []
    for blk in text_blocks:
        if blk["top"] < img_bottom - 2:
            continue
        gap = blk["top"] - img_bottom
        if gap > CAPTION_GAP_PT:
            continue
        h_overlap = min(blk["x1"], img_x1) - max(blk["x0"], img_x0)
        if h_overlap < 10:
            continue
        text = blk["text"].replace("\n", " ").strip()
        has_prefix = bool(_CAPTION_PREFIX.match(text))
        is_short   = len(text) <= CAPTION_MAX_CHARS
        if has_prefix or is_short:
            candidates.append((gap, not has_prefix, text))
    if not candidates:
        return ""
    candidates.sort(key=lambda c: (c[0], c[1]))
    return candidates[0][2]


# ══════════════════════════════════════════════════════════════════════════════
#  PASS 2 — VECTOR FIGURE EXTRACTION  (pdfplumber + pdf2image)
# ══════════════════════════════════════════════════════════════════════════════

def _get_lines(page: pdfplumber.page.Page) -> list[dict]:
    """Group words into visual lines (±3 pt vertical tolerance)."""
    words = page.extract_words(keep_blank_chars=False)
    if not words:
        return []

    lines: list[list[dict]] = []
    cur: list[dict]         = []
    cur_top: float | None   = None

    for w in sorted(words, key=lambda w: (round(w["top"]), w["x0"])):
        if cur_top is None or abs(w["top"] - cur_top) <= 3:
            cur.append(w)
            if cur_top is None:
                cur_top = w["top"]
        else:
            lines.append(cur)
            cur     = [w]
            cur_top = w["top"]
    if cur:
        lines.append(cur)

    result = []
    for ln in lines:
        result.append({
            "top"   : min(w["top"]    for w in ln),
            "bottom": max(w["bottom"] for w in ln),
            "x0"    : min(w["x0"]     for w in ln),
            "x1"    : max(w["x1"]     for w in ln),
            "text"  : " ".join(w["text"] for w in sorted(ln, key=lambda w: w["x0"])),
        })
    return sorted(result, key=lambda l: l["top"])


def _find_captions(page: pdfplumber.page.Page) -> list[dict]:
    """
    Return all figure-caption lines on a page, sorted by top.
    Handles captions at line start AND mid-line captions in two-column PDFs.
    Distinguishes actual captions from inline figure references by
    requiring the caption portion to be substantive (>20 chars).
    """
    _CAP_ANYWHERE = re.compile(
        r'(figure\s*\d+[\.\:\-]|fig\.?\s*\d+[\.\:\-])',
        re.IGNORECASE,
    )

    _XREF_START = re.compile(
        r'^(figure|fig\.?)\s*\d+[\.\:\-\s]'
        r'(\(|first|second|third|shows?|middle|left|right|top|bottom'
        r'|above|below|also|see|similar|here|note|both)',
        re.IGNORECASE,
    )

    _XREF_WORDS = re.compile(
        r'^(first|second|third|shows?|middle|left|right|top|bottom|above|below'
        r'|also|see|similar|here|note|both|this|the\s)',
        re.IGNORECASE,
    )

    results:   list[dict] = []
    seen_tops: set[int]   = set()

    for ln in _get_lines(page):
        text    = ln["text"].strip()
        top_key = int(ln["top"])

        # Normal case: caption at start of line
        is_cap_start = (
            (_CAPTION_RE.match(text) or _CAPTION_RE_CONCAT.match(text))
            and not _XREF_START.match(text)   # exclude cross-references
        )
        if is_cap_start:
            if top_key not in seen_tops:
                results.append(ln)
                seen_tops.add(top_key)
            continue

        # Mid-line case: caption embedded in concatenated text
        m = _CAP_ANYWHERE.search(text)
        if m:
            cap_text = text[m.start():].strip()
            # Only treat as a real caption if:
            # 1. The caption portion is substantive (not just "Fig. 4.")
            # 2. It's not a short inline reference like "Fig. 4 right"
            # A real caption has descriptive text: typically > 20 chars
            # after the "Figure N." prefix
            prefix_end = m.end()
            description = text[prefix_end:].strip() if prefix_end < len(text) else ""
            prefix_end  = m.end()
            description = text[prefix_end:].strip() if prefix_end < len(text) else ""
            if len(description) > 20 and not _XREF_WORDS.match(description) and not description.startswith("("):
                if top_key not in seen_tops:
                    results.append({**ln, "text": cap_text})
                    seen_tops.add(top_key)

    return sorted(results, key=lambda l: l["top"])


def _full_caption(lines: list[dict], cap: dict) -> str:
    """Collect full caption text including continuation lines."""
    text  = cap["text"]
    below = [
        ln for ln in lines
        if 0 < ln["top"] - cap["bottom"] <= 30
        and ln["x0"] >= cap["x0"] - 25
    ]
    for ln in below:
        if ln["text"] and ln["text"][0].isupper() and not text.endswith(","):
            break
        text += " " + ln["text"]
    return text.strip()


def _graphic_clusters(page: pdfplumber.page.Page) -> list[dict]:
    """
    Cluster all rects, curves and lines into contiguous figure regions.
    Returns list of {top, bottom, x0, x1} sorted by top.
    """
    objs = []
    for o in list(page.rects) + list(page.curves) + list(page.lines):
        h = o.get("height", abs(o.get("bottom", 0) - o.get("top", 0)))
        w = o.get("width",  abs(o.get("x1", 0)    - o.get("x0", 0)))
        if h > 0 or w > 0:
            objs.append({
                "top"   : o["top"],
                "bottom": o["bottom"],
                "x0"    : o["x0"],
                "x1"    : o["x1"],
            })

    if not objs:
        return []

    objs.sort(key=lambda o: o["top"])
    clusters: list[list[dict]] = []
    cur = [objs[0]]

    for o in objs[1:]:
        prev_bot = max(x["bottom"] for x in cur)
        if o["top"] - prev_bot > CLUSTER_GAP_PT:
            clusters.append(cur)
            cur = [o]
        else:
            cur.append(o)
    clusters.append(cur)

    result = []
    for cl in clusters:
        top  = min(o["top"]    for o in cl)
        bot  = max(o["bottom"] for o in cl)
        x0   = min(o["x0"]     for o in cl)
        x1   = max(o["x1"]     for o in cl)
        if (bot - top) >= MIN_FIG_HEIGHT and (x1 - x0) >= MIN_FIG_WIDTH:
            result.append({"top": top, "bottom": bot, "x0": x0, "x1": x1})

    return sorted(result, key=lambda c: c["top"])


def _match_cluster(cap: dict, clusters: list[dict]) -> dict | None:
    """
    Find the graphic cluster for this caption.
    Accepts clusters that START before the caption (cluster top < cap top)
    and are close enough above or overlapping the caption.
    """
    candidates = [
        cl for cl in clusters
        if cl["top"] <= cap["top"] + 10           # cluster starts before caption
        and cap["top"] - cl["top"] >= MIN_FIG_HEIGHT   # enough height above
    ]
    if not candidates:
        return None
    # Pick the cluster whose bottom is closest to (but not far above) caption top
    return max(candidates, key=lambda cl: cl["bottom"])


def _render_page(pdf_path: str, page_idx: int) -> Image.Image:
    """Render a single PDF page to a PIL image using pypdfium2."""
    doc    = pdfium.PdfDocument(str(pdf_path))
    page   = doc[page_idx]
    bitmap = page.render(scale=DPI / 72.0)   # 72 pt = 1 inch; scale to target DPI
    img    = bitmap.to_pil()
    doc.close()
    return img


def _crop_save(page_img: Image.Image,
               pw_pt: float, ph_pt: float,
               x0: float, y0: float,
               x1: float, y1: float,
               out_path: Path) -> dict | None:
    """Crop a region from a rendered page image and save as PNG."""
    W, H = page_img.size
    sx, sy = W / pw_pt, H / ph_pt

    px0 = max(0, int((x0 - PADDING_PT) * sx))
    py0 = max(0, int((y0 - PADDING_PT) * sy))
    px1 = min(W, int((x1 + PADDING_PT) * sx))
    py1 = min(H, int((y1 + PADDING_PT) * sy))

    if px1 - px0 < 10 or py1 - py0 < 10:
        return None

    crop = page_img.crop((px0, py0, px1, py1))

    # Skip nearly blank crops
    lo, hi = crop.convert("L").getextrema()
    if hi - lo < 15:
        return None

    crop.save(str(out_path), "PNG")
    return {"width": crop.width, "height": crop.height}


def _extract_vector(pdf_path: str,
                    images_dir: Path,
                    raster_covered: set[tuple]) -> list[dict]:
    """
    Extract vector figures by rendering page regions.
    Handles same-page figures AND cross-page figures where the graphic
    cluster ends near the bottom of page N but the caption is at the
    top of page N+1.
    """
    results: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)

        # Pre-build per-page data (captions, clusters, lines, dims)
        page_data = []
        for page_idx in range(n_pages):
            page = pdf.pages[page_idx]
            page_data.append({
                "page"    : page,
                "pw"      : page.width,
                "ph"      : page.height,
                "captions": _find_captions(page),
                "clusters": _graphic_clusters(page),
                "lines"   : _get_lines(page),
            })

        rendered: dict[int, Image.Image] = {}   # cache rendered pages

        def get_render(idx: int) -> Image.Image:
            if idx not in rendered:
                rendered[idx] = _render_page(pdf_path, idx)
            return rendered[idx]

        # ── Same-page figures ──────────────────────────────────────────────
        for page_idx, pd in enumerate(page_data):
            page_num = page_idx + 1
            pw, ph   = pd["pw"], pd["ph"]
            fig_num  = 0

            for cap in pd["captions"]:
                full_cap = _full_caption(pd["lines"], cap)
                cap_key  = (page_num, full_cap[:30])
                if cap_key in raster_covered:
                    continue

                cluster = _match_cluster(cap, pd["clusters"])
                if cluster is None:
                    continue

                fig_x0  = cluster["x0"]
                fig_x1  = cluster["x1"]
                fig_top = cluster["top"]
                fig_bot = cap["top"]

                if fig_bot - fig_top < MIN_FIG_HEIGHT:
                    continue
                if fig_x1 - fig_x0 < MIN_FIG_WIDTH:
                    continue

                fig_num += 1
                filename = f"page{page_num}_fig{fig_num}.png"
                out_file = images_dir / filename

                info = _crop_save(get_render(page_idx), pw, ph,
                                  fig_x0, fig_top, fig_x1, fig_bot,
                                  out_file)
                if info:
                    results.append({
                        "page"   : page_num,
                        "top"    : fig_top,
                        "bottom" : fig_bot,
                        "caption": full_cap,
                        "path"   : f"images/{filename}",
                        **info,
                    })

        # ── Cross-page figures ─────────────────────────────────────────────
        # A figure cluster on page N whose caption is at the top of page N+1.
        # This happens when a figure is large enough to fill a page and its
        # caption spills onto the next page.
        #
        saved_paths = {r["path"] for r in results}

        for page_idx in range(n_pages - 1):
            pd_curr  = page_data[page_idx]
            pd_next  = page_data[page_idx + 1]
            page_num = page_idx + 1
            pw, ph   = pd_curr["pw"], pd_curr["ph"]

            if not pd_curr["clusters"]:
                continue

            # Find clusters on this page that have NO same-page caption match
            unmatched_clusters = []
            for cl in pd_curr["clusters"]:
                matched = any(
                    _match_cluster(cap, [cl]) is not None
                    for cap in pd_curr["captions"]
                )
                if not matched:
                    unmatched_clusters.append(cl)

            if not unmatched_clusters:
                continue

            # Captions in the top 30% of next page
            next_top_caps = [
                cap for cap in pd_next["captions"]
                if cap["top"] <= pd_next["ph"] * 0.30
            ]
            if not next_top_caps:
                continue

            for cap in next_top_caps:
                full_cap = _full_caption(pd_next["lines"], cap)
                cap_key  = (page_num + 1, full_cap[:30])

                if cap_key in raster_covered:
                    continue
                # Skip if this caption was already matched same-page on page N+1
                if any(full_cap[:30] in r.get("caption", "") for r in results
                       if r["page"] == page_num + 1):
                    continue

                # Use the largest unmatched cluster
                cluster = max(unmatched_clusters,
                              key=lambda cl: (cl["bottom"] - cl["top"]) * (cl["x1"] - cl["x0"]))

                fig_x0  = cluster["x0"]
                fig_x1  = cluster["x1"]
                fig_top = cluster["top"]
                fig_bot = cluster["bottom"]   # crop to cluster bottom, not page bottom

                if fig_bot - fig_top < MIN_FIG_HEIGHT:
                    continue
                if fig_x1 - fig_x0 < MIN_FIG_WIDTH:
                    continue

                filename = f"page{page_num}_figX.png"
                out_file = images_dir / filename
                rel_path = f"images/{filename}"

                if rel_path in saved_paths:
                    continue

                info = _crop_save(get_render(page_idx), pw, ph,
                                  fig_x0, fig_top, fig_x1, fig_bot,
                                  out_file)
                if info:
                    results.append({
                        "page"   : page_num,
                        "top"    : fig_top,
                        "bottom" : fig_bot,
                        "caption": full_cap,
                        "path"   : rel_path,
                        **info,
                    })
                    saved_paths.add(rel_path)
                    break

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def extract_images(pdf_path: str,
                   output_dir: str = "output") -> list[dict]:
    """
    Extract all figures from a PDF and save as image files.

    Combines raster extraction (PyMuPDF) + vector rendering (pdf2image):
      - Raster images are saved in their native format (JPEG/PNG/etc.)
      - Vector figures are rendered and saved as PNG

    Args:
        pdf_path   : Path to the PDF file.
        output_dir : Root output dir. Images saved to output_dir/images/.

    Returns:
        List of dicts sorted by (page, top):
            {page, top, bottom, caption, path, width, height}
    """
    pdf_path   = str(pdf_path)
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: raster images
    raster = _extract_raster(pdf_path, images_dir)

    # Build set of already-covered captions for deduplication
    raster_covered = {
        (r["page"], r["caption"][:30])
        for r in raster if r["caption"]
    }

    # Pass 2: vector figures (only captions not covered by Pass 1)
    vector = _extract_vector(pdf_path, images_dir, raster_covered)

    all_images = sorted(
        raster + vector,
        key=lambda x: (x["page"], x["top"]),
    )
    return all_images


# ══════════════════════════════════════════════════════════════════════════════
#  STANDALONE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "report.pdf"
    out = sys.argv[2] if len(sys.argv) > 2 else "output"

    imgs = extract_images(pdf, out)
    if not imgs:
        print("No figures found.")
    else:
        print(f"Found {len(imgs)} figure(s) — saved to {out}/images/\n")
        for i, img in enumerate(imgs, 1):
            print(f"[{i}] Page {img['page']} | {img['width']}x{img['height']}px"
                  f" | {img['path']}")
            print(f"     {img['caption'][:90]}")