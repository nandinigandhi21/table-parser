"""
formula_extractor.py
Extracts mathematical formulas from a PDF in three stages:

  Stage 1 — LOCATE  (pdfplumber)
      Cluster words into formula bounding boxes using body-baseline
      gap detection. Filters non-math content via symbol regex.

  Stage 2 — CROP  (pypdfium2 + Pillow)
      Render each PDF page to a high-res image and crop the formula
      bbox with padding. Saved to <output_dir>/formulas/.

  Stage 3 — LaTeX  (pluggable backend, best available is used)
      Priority order:
        1. pix2tex   — ViT-based model, best quality  (needs torch)
        2. texify    — newer pix2tex variant           (needs torch)
        3. Docling   — VLM formula enrichment          (needs docling)
        4. plain     — pdfplumber Unicode text fallback (always works)

Returns each formula as a dict:
    {page, top, bottom, kind, latex, text, crop_path}

    - kind      : always "FORMULA"
    - latex     : LaTeX string if a model backend succeeded, else ""
    - text      : plain Unicode text (always filled as fallback)
    - crop_path : relative path to saved crop image e.g. "formulas/page6_f1.png"
"""

from __future__ import annotations

import re
import warnings
import logging
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("docling").setLevel(logging.ERROR)


# ── Config ────────────────────────────────────────────────────────────────────

RENDER_SCALE    = 3      # pypdfium2 render scale (3× = ~216 dpi)
CROP_PADDING    = 10     # pt padding around formula bbox before crop
BASELINE_GAP    = 8.0    # pt gap between body baselines → new formula group
SUB_SUPER_RATIO = 0.9    # expand span by body_size × this above/below

# ══════════════════════════════════════════════════════════════════════════════
#  LaTeX POST-PROCESSING CLEANUP
# ══════════════════════════════════════════════════════════════════════════════

def _fix_int_brace_superscript(latex: str) -> str:
    r"""
    Fix {\int_{a}^{b}}^{n}  →  \int_{a}^{b}

    pix2tex sometimes wraps an integral in braces and then appends a ^{n}
    that belonged to a term it lost (e.g. 5^n).  We use proper brace-matching
    to find and strip this artefact without breaking other expressions.
    """
    result = []
    i = 0
    while i < len(latex):
        # Detect  {  immediately followed by \int
        if latex[i] == '{' and latex[i + 1:i + 5] == r'\int':
            # Walk forward to find the matching closing brace
            depth = 1
            j = i + 1
            while j < len(latex) and depth > 0:
                if   latex[j] == '{': depth += 1
                elif latex[j] == '}': depth -= 1
                j += 1
            # j is now one past the matching '}'
            # Check whether a ^{...} or ^<char> follows immediately
            k = j
            if k < len(latex) and latex[k] == '^':
                k += 1
                if k < len(latex) and latex[k] == '{':
                    depth2 = 1
                    k += 1
                    while k < len(latex) and depth2 > 0:
                        if   latex[k] == '{': depth2 += 1
                        elif latex[k] == '}': depth2 -= 1
                        k += 1
                    # Emit the integral content without outer braces or ^{…}
                    result.append(latex[i + 1: j - 1])
                    i = k
                    continue
                elif k < len(latex) and latex[k].isalnum():
                    result.append(latex[i + 1: j - 1])
                    i = k + 1
                    continue
        result.append(latex[i])
        i += 1
    return ''.join(result)


def clean_latex(latex: str) -> str:
    """
    Fix common pix2tex / texify output artifacts:
      - Strip displaystyle / textstyle prefixes
      - Fix misread symbols: chi→x, lo\\alpha→\\log
      - Fix spaced-out function names: 'l o g' → \\log
      - Fix ALL mangled integral variants (binom, stackrel, underset wrappers)
      - Unwrap begin{array}...end{array} wrappers
      - Take first row when model wraps multiple formulas in one array
      - Unwrap redundant double/triple braces
      - Remove orphaned/unmatched braces and empty {}
    """
    if not latex:
        return latex

    # 1. Strip display-style prefixes
    latex = re.sub(r'\\(displaystyle|textstyle|scriptstyle)\s*', '', latex)

    # 2. Fix misread symbols before any structural changes
    latex = re.sub(r'\\chi\b',         'x',      latex)   # χ → x in algebra
    latex = re.sub(r'l\s*o\s*\\alpha', r'\\log', latex)   # lo\alpha → \log

    # 3. Fix spaced-out function names: 'l o g' → \log
    for pat, rep in [
        (r'(?<![a-zA-Z\\])l\s*o\s*g(?![a-zA-Z])', r'\\log'),
        (r'(?<![a-zA-Z\\])l\s*i\s*m(?![a-zA-Z])', r'\\lim'),
        (r'(?<![a-zA-Z\\])s\s*i\s*n(?![a-zA-Z])', r'\\sin'),
        (r'(?<![a-zA-Z\\])c\s*o\s*s(?![a-zA-Z])', r'\\cos'),
        (r'(?<![a-zA-Z\\])t\s*a\s*n(?![a-zA-Z])', r'\\tan'),
        (r'(?<![a-zA-Z\\])e\s*x\s*p(?![a-zA-Z])', r'\\exp'),
    ]:
        latex = re.sub(pat, rep, latex)

    # 4. Fix ALL mangled integral variants
    #    4a. \underset{LOW}{\binom{HIGH}{}} → \int_{LOW}^{HIGH}
    latex = re.sub(
        r'\\underset\{([^}]+)\}\{\\binom\{([^}]+)\}\{[^}]*\}\}',
        r'\\int_{\1}^{\2}', latex)
    #    4b. \overset{HIGH}{\underset{LOW}{\int}} → \int_{LOW}^{HIGH}
    latex = re.sub(
        r'\\overset\{([^}]+)\}\{\\underset\{([^}]+)\}\{\\int\}\}',
        r'\\int_{\2}^{\1}', latex)
    #    4c. {\underset{LOW}{ANYTHING}}^{NOISE} → \int_{LOW}
    #        catches \stackrel, \vdots, and other garbage the model hallucinates
    latex = re.sub(
        r'\{\\underset\{([^}]+)\}\{[^{}]*(?:\{[^{}]*\})*[^{}]*\}\}\^\{[^}]*\}',
        r'\\int_{\1}', latex)
    #    4d. bare {\underset{LOW}{...}} without superscript → \int_{LOW}
    latex = re.sub(
        r'\{\\underset\{([^}]+)\}\{[^{}]*(?:\{[^{}]*\})*[^{}]*\}\}',
        r'\\int_{\1}', latex)

    # 5. Unwrap \begin{array}...\end{array} (inside-out, up to 4 levels)
    for _ in range(4):
        prev  = latex
        latex = re.sub(
            r'\\begin\{array\}\{[^}]*\}\s*(.*?)\s*\\end\{array\}',
            lambda m: m.group(1).strip(), latex, flags=re.DOTALL)
        if latex == prev:
            break

    # 6. Take first non-trivial row when multi-row array is present
    rows = [r.strip() for r in re.split(r'\\\\', latex) if r.strip() and len(r.strip()) > 2]
    if len(rows) > 1:
        latex = rows[0]

    # 7. Unwrap redundant double-braces: {{expr}} → expr
    for _ in range(6):
        prev  = latex
        latex = re.sub(r'\{\{([^{}]+)\}\}', r'\1', latex)
        if latex == prev:
            break

    # 8. Remove outer brace wrapping the whole expression
    while latex.startswith('{') and latex.endswith('}'):
        inner = latex[1:-1]
        depth, balanced = 0, True
        for ch in inner:
            if   ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth < 0: balanced = False; break
        if balanced and depth == 0:
            latex = inner
        else:
            break

    # 9. Remove stray { before \int / \frac etc. (no \b — breaks before _)
    _INT_BRACE = re.compile(r'\{(\\(?:int|frac|sum|prod|sqrt))')
    latex = _INT_BRACE.sub(r'\1', latex)

    # 10. Clean empty braces and thin spaces; re-run stray-brace removal after
    latex = re.sub(r'\{\}\s*', '', latex)
    latex = re.sub(r'\\,',     ' ', latex)
    latex = _INT_BRACE.sub(r'\1', latex)

    # 11. Fix {\int_{a}^{b}}^{n} → \int_{a}^{b}  (brace-matched helper)
    latex = _fix_int_brace_superscript(latex)

    # 12. Strip double superscript: \int_{a}^{b}^{c} → \int_{a}^{b}
    latex = re.sub(r'(\\int_\{[^}]*\}\^\{[^}]*\})\^\{[^}]*\}', r'\1', latex)

    # 13. Remove stray }}^{...} or }^{...} after completed \int expression
    latex = re.sub(r'(\\int_\{[^}]*\}(?:\^\{[^}]*\})?)\}+\^\{[^}]*\}', r'\1', latex)

    # 14. Strip any remaining unmatched closing braces
    depth = 0
    cleaned = []
    for ch in latex:
        if   ch == '{': depth += 1; cleaned.append(ch)
        elif ch == '}':
            if depth > 0: depth -= 1; cleaned.append(ch)
            # else: drop the unmatched }
        else:
            cleaned.append(ch)
    latex = ''.join(cleaned)

    # 15. Normalise whitespace
    latex = re.sub(r'[ \t]{2,}', ' ', latex).strip()

    return latex


# Symbols / tokens that strongly indicate mathematical content
_MATH_RE = re.compile(
    r'[=+*/^\u222b\u2211\u220f\u221a\u221e\u00b1\u2260\u2264\u2265\u2202\u00f7\u00d7]'
    r'|[\u03b1-\u03c9\u0391-\u03a9]'           # Greek letters
    r'|[\U0001D400-\U0001D7FF]'                 # Math Alphanumeric block (𝑙𝑜𝑔 etc.)
    r'|\\frac|\\sum|\\int|\\sqrt'               # LaTeX macros
    r'|\blog\b|\bsin\b|\bcos\b|\btan\b|\blim\b|\bsup\b|\binf\b'
)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — LOCATE  (pdfplumber)
# ══════════════════════════════════════════════════════════════════════════════

def _locate_formulas(pdf_path: str) -> list[dict]:
    """
    Find formula bounding boxes on every page using pdfplumber.

    Returns list of dicts:
        {page_0, page, top, bottom, x0, x1, text}
    where page_0 is 0-based index and page is 1-based label.
    """
    import pdfplumber

    results: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_label = page_num + 1
            words = page.extract_words(extra_attrs=["size", "fontname"])
            if not words:
                continue

            # Dominant body font size
            size_counts = Counter(round(w["size"], 0) for w in words)
            body_size   = size_counts.most_common(1)[0][0]

            # Body-level baselines
            body_words = [w for w in words if abs(w["size"] - body_size) < 1.5]
            baselines  = sorted(set(round(w["top"], 0) for w in body_words))
            if not baselines:
                continue

            # Cluster baselines into formula groups
            formula_groups: list[list[float]] = []
            cur = [baselines[0]]
            for b in baselines[1:]:
                if b - cur[-1] <= BASELINE_GAP:
                    cur.append(b)
                else:
                    formula_groups.append(cur)
                    cur = [b]
            formula_groups.append(cur)

            # Collect words per group, expanding for sub/superscripts
            for gi, grp in enumerate(formula_groups):
                next_top = (
                    formula_groups[gi + 1][0]
                    if gi + 1 < len(formula_groups) else float("inf")
                )
                span_top    = grp[0]    - body_size * SUB_SUPER_RATIO
                span_bottom = min(grp[-1] + body_size * 1.5, next_top - 1)

                grp_words = [
                    w for w in words
                    if w["top"] >= span_top and w["top"] <= span_bottom
                ]
                if not grp_words:
                    continue

                top    = min(w["top"]    for w in grp_words)
                bottom = max(w["bottom"] for w in grp_words)
                x0     = min(w["x0"]    for w in grp_words)
                x1     = max(w["x1"]    for w in grp_words)

                # Reconstruct text left-to-right, top-to-bottom
                sorted_w = sorted(grp_words, key=lambda w: (round(w["top"] / 3) * 3, w["x0"]))
                text = " ".join(w["text"] for w in sorted_w)

                if not _MATH_RE.search(text):
                    continue

                results.append({
                    "page_0": page_num,
                    "page"  : page_label,
                    "top"   : top,
                    "bottom": bottom,
                    "x0"    : x0,
                    "x1"    : x1,
                    "text"  : text,
                })

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — CROP  (pypdfium2 + Pillow)
# ══════════════════════════════════════════════════════════════════════════════

def _pypdfium2_available() -> bool:
    try:
        import pypdfium2  # noqa: F401
        return True
    except ImportError:
        return False


def _fitz_available() -> bool:
    try:
        import fitz  # noqa: F401
        return True
    except ImportError:
        return False


def _crop_formula_image(pdf_path: str, page_0: int,
                        top: float, bottom: float,
                        x0: float,  x1: float,
                        page_w: float, page_h: float) -> "Image | None":
    """
    Render the PDF page and crop the formula region.
    Tries pypdfium2 first, then fitz (PyMuPDF).
    Returns a PIL Image or None if neither renderer is available.
    """
    from PIL import Image

    padding = CROP_PADDING
    scale   = RENDER_SCALE

    if _pypdfium2_available():
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(pdf_path)
        try:
            pg     = doc[page_0]
            bitmap = pg.render(scale=scale)
            img    = bitmap.to_pil()
        finally:
            doc.close()

    elif _fitz_available():
        import fitz
        doc = fitz.open(pdf_path)
        try:
            pg  = doc[page_0]
            mat = fitz.Matrix(scale, scale)
            pix = pg.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        finally:
            doc.close()
    else:
        return None

    iw, ih = img.size
    sx, sy = iw / page_w, ih / page_h

    px0 = max(0,  int((x0     - padding) * sx))
    py0 = max(0,  int((top    - padding) * sy))
    px1 = min(iw, int((x1     + padding) * sx))
    py1 = min(ih, int((bottom + padding) * sy))

    cropped = img.crop((px0, py0, px1, py1))

    # Ensure white background
    bg = Image.new("RGB", cropped.size, "white")
    bg.paste(cropped)
    return bg


def _save_crops(pdf_path: str, formulas: list[dict],
                output_dir: str) -> list[dict]:
    """
    For each formula dict, render + crop the image and save to
    <output_dir>/formulas/. Adds 'crop_path' and 'crop_img' to each dict.
    """
    import pdfplumber
    from PIL import Image

    formulas_dir = Path(output_dir) / "formulas"
    formulas_dir.mkdir(parents=True, exist_ok=True)

    if not (_pypdfium2_available() or _fitz_available()):
        print("         No PDF renderer (pypdfium2/fitz) — skipping crops")
        for f in formulas:
            f["crop_path"] = ""
            f["crop_img"]  = None
        return formulas

    # Build page size map
    page_sizes: dict[int, tuple[float, float]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_sizes[i] = (page.width, page.height)

    # Per-page formula index counter
    page_counters: dict[int, int] = {}

    for formula in formulas:
        p0     = formula["page_0"]
        pw, ph = page_sizes.get(p0, (612.0, 792.0))
        idx    = page_counters.get(p0, 0) + 1
        page_counters[p0] = idx

        img = _crop_formula_image(
            pdf_path, p0,
            formula["top"], formula["bottom"],
            formula["x0"],  formula["x1"],
            pw, ph,
        )

        filename  = f"page{formula['page']}_f{idx}.png"
        save_path = formulas_dir / filename
        relative  = f"formulas/{filename}"

        if img is not None:
            img.save(str(save_path))
            formula["crop_path"] = relative
            formula["crop_img"]  = img
        else:
            formula["crop_path"] = ""
            formula["crop_img"]  = None

    return formulas


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — LaTeX BACKEND  (pluggable)
# ══════════════════════════════════════════════════════════════════════════════

# ── pix2tex ───────────────────────────────────────────────────────────────────

def _pix2tex_available() -> bool:
    try:
        from pix2tex.cli import LatexOCR  # noqa: F401
        return True
    except ImportError:
        return False


def _latex_with_pix2tex(formulas: list[dict]) -> list[dict]:
    from pix2tex.cli import LatexOCR
    model = LatexOCR()
    for f in formulas:
        img = f.get("crop_img")
        if img is None:
            continue
        try:
            f["latex"] = clean_latex(model(img))
        except Exception as e:
            print(f"         pix2tex failed on formula (page {f['page']}): {e}")
    return formulas


# ── texify ────────────────────────────────────────────────────────────────────

def _texify_available() -> bool:
    try:
        from texify.inference import batch_inference  # noqa: F401
        from texify.model.model import load_model     # noqa: F401
        return True
    except ImportError:
        return False


def _latex_with_texify(formulas: list[dict]) -> list[dict]:
    from texify.inference import batch_inference
    from texify.model.model import load_model
    from texify.model.processor import load_processor

    model     = load_model()
    processor = load_processor()

    imgs = [f["crop_img"] for f in formulas if f.get("crop_img") is not None]
    idxs = [i for i, f in enumerate(formulas) if f.get("crop_img") is not None]

    if not imgs:
        return formulas

    try:
        results = batch_inference(imgs, model, processor)
        for i, latex in zip(idxs, results):
            formulas[i]["latex"] = clean_latex(latex.strip())
    except Exception as e:
        print(f"         texify batch inference failed: {e}")

    return formulas


# ── Docling ───────────────────────────────────────────────────────────────────

def _docling_available() -> bool:
    try:
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        return hasattr(PdfPipelineOptions(), "do_formula_enrichment")
    except ImportError:
        return False


def _latex_with_docling(pdf_path: str) -> list[dict]:
    """
    Run Docling formula enrichment on the whole PDF.
    Returns list of {page, top, bottom, latex, text} dicts.
    """
    import warnings, logging
    warnings.filterwarnings("ignore")
    logging.getLogger("docling").setLevel(logging.ERROR)

    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    opts = PdfPipelineOptions()
    opts.do_formula_enrichment = True
    opts.do_table_structure    = False
    opts.do_ocr                = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

    results: list[dict] = []
    try:
        result = converter.convert(pdf_path)
        doc    = result.document

        page_heights: dict[int, float] = {}
        for pno, pobj in doc.pages.items():
            try:
                page_heights[int(pno)] = float(pobj.size.height)
            except Exception:
                page_heights[int(pno)] = 792.0

        for item, _ in doc.iterate_items():
            lv = getattr(item, "label", None)
            if lv is None:
                continue
            ls = lv.value if hasattr(lv, "value") else str(lv)
            if ls.lower() not in ("formula", "equation", "math"):
                continue
            try:
                prov    = item.prov[0]
                page_no = int(prov.page_no)
                bbox    = prov.bbox
                h       = page_heights.get(page_no, 792.0)
                top     = h - bbox.t
                bottom  = h - bbox.b
                top, bottom = min(top, bottom), max(top, bottom)
            except Exception:
                continue

            orig  = (getattr(item, "orig", "") or "").strip()
            text  = (getattr(item, "text", "") or "").strip()
            latex = orig if ("\\" in orig or "{" in orig) else (
                    text if ("\\" in text or "{" in text) else orig)

            results.append({
                "page": page_no, "top": top, "bottom": bottom,
                "latex": clean_latex(latex), "text": text or latex,
            })
    finally:
        try:
            result = None; doc = None; converter = None  # noqa
        except Exception:
            pass

    return results


# ── Plain text fallback ───────────────────────────────────────────────────────

def _apply_latex_backend(pdf_path: str, formulas: list[dict]) -> list[dict]:
    """
    Try LaTeX backends in priority order and fill formula['latex'].
    Backends are tried in order: pix2tex → texify → Docling → plain text.
    """
    # Initialise latex field
    for f in formulas:
        f.setdefault("latex", "")

    if _pix2tex_available():
        print("         LaTeX backend: pix2tex")
        return _latex_with_pix2tex(formulas)

    if _texify_available():
        print("         LaTeX backend: texify")
        return _latex_with_texify(formulas)

    if _docling_available():
        print("         LaTeX backend: Docling formula enrichment")
        # Docling works on the whole PDF, not per-crop
        # Match results back to our located formulas by page + proximity
        try:
            docling_results = _latex_with_docling(pdf_path)
            for f in formulas:
                best = None
                best_dist = float("inf")
                for dr in docling_results:
                    if dr["page"] != f["page"]:
                        continue
                    dist = abs(dr["top"] - f["top"])
                    if dist < best_dist:
                        best_dist = dist
                        best = dr
                if best and best_dist < 20:
                    f["latex"] = best.get("latex", "")
        except Exception as e:
            print(f"         Docling failed: {e}")
        return formulas

    print("         LaTeX backend: none available — using plain text fallback")
    return formulas


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def extract_formulas(pdf_path: str, output_dir: str = "output") -> list[dict]:
    """
    Full pipeline: locate → crop → LaTeX.

    Args:
        pdf_path   : Path to the PDF file.
        output_dir : Root output folder. Crops saved to output_dir/formulas/.

    Returns:
        List of dicts sorted by (page, top):
            {page, top, bottom, kind, latex, text, crop_path}
    """
    # Stage 1: locate
    print("         Stage 1 — locating formula bboxes (pdfplumber)...")
    formulas = _locate_formulas(pdf_path)
    print(f"         Found {len(formulas)} formula region(s)")

    if not formulas:
        return []

    # Stage 2: crop
    renderer = "pypdfium2" if _pypdfium2_available() else ("fitz" if _fitz_available() else "none")
    print(f"         Stage 2 — cropping formula images ({renderer})...")
    formulas = _save_crops(pdf_path, formulas, output_dir)
    saved    = sum(1 for f in formulas if f["crop_path"])
    print(f"         Saved {saved} crop image(s) → {output_dir}/formulas/")

    # Stage 3: LaTeX
    print("         Stage 3 — converting to LaTeX...")
    formulas = _apply_latex_backend(pdf_path, formulas)
    with_latex = sum(1 for f in formulas if f.get("latex"))
    print(f"         LaTeX extracted for {with_latex}/{len(formulas)} formula(s)")

    # Clean up internal-only fields before returning
    for f in formulas:
        f.pop("crop_img",  None)
        f.pop("page_0",    None)
        f["kind"] = "FORMULA"

    formulas.sort(key=lambda x: (x["page"], x["top"]))
    return formulas


def formula_extraction_available() -> bool:
    """Always True — pdfplumber fallback is always available."""
    return True