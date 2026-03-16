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

RENDER_SCALE       = 3      # pypdfium2 render scale (3× = ~216 dpi)
CROP_PADDING_SIDE  = 12     # pt padding left/right and bottom around formula bbox
CROP_PADDING_TOP   = 3      # pt padding above formula top (small to avoid label bleed)
BASELINE_GAP       = 8.0    # pt gap between body baselines (legacy, not used in v2 locator)
SUB_SUPER_RATIO    = 0.9    # legacy constant kept for compatibility

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

    # 4e. \stackrel{N}\int_{LOW}^{WRONG} → \int_{LOW}^{N}
    #     pix2tex/texify puts the upper limit as \stackrel{N} before \int
    #     and then puts a wrong value in ^{...}. We extract N from \stackrel
    #     and use it as the correct upper limit, discarding the wrong ^{...}.
    #     e.g. \log n\stackrel{3}\int_{0}^{n} 5^{n} → \log n \int_{0}^{3} 5^{n}
    #     NOTE: latex strings contain literal single backslashes, so we need
    #     \\\\ in the pattern to match one literal backslash.
    latex = re.sub(
        r'\\stackrel\{([^}]+)\}\\int_\{([^}]+)\}\^\{[^}]+\}',
        lambda m: f'\\int_{{{m.group(2)}}}^{{{m.group(1)}}}',
        latex,
    )
    # 4e-fallback: \stackrel{N}\int with no bounds at all → just drop stackrel
    latex = re.sub(r'\\stackrel\{[^}]+\}(\\int)', r'\1', latex)

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

    # 15. Remove stray punctuation artifacts: {,}  {.}  {;}  at end of expression
    latex = re.sub(r'\{[,\.;]\}\s*$', '', latex)
    latex = re.sub(r'\{[,\.;]\}',     '', latex)   # also mid-expression

    # 16. Normalise whitespace
    latex = re.sub(r'[ \t]{2,}', ' ', latex).strip()

    return latex


# Math-italic Unicode → ASCII (for reconcile_latex)
_UNICODE_MATH_TRANS = str.maketrans({
    '𝑙':'l','𝑜':'o','𝑔':'g','𝑛':'n','𝑎':'a','𝑏':'b','𝑥':'x',
    '𝑦':'y','𝑖':'i','𝑗':'j','𝑘':'k','𝑚':'m','𝑝':'p','𝑞':'q',
    '𝑟':'r','𝑠':'s','𝑡':'t','𝑢':'u','𝑣':'v','𝑤':'w','𝑧':'z',
    '𝐴':'A','𝐵':'B','𝐶':'C','𝐷':'D','𝐸':'E','𝐹':'F',
})


def reconcile_latex(latex: str, words: list[dict]) -> str:
    """
    Recover integrand terms that pix2tex drops from integrals.

    pix2tex reliably recognises the integral sign and its bounds but
    sometimes loses the integrand (e.g. produces \\int_{0}^{3} instead of
    \\int_{0}^{3} 5^{n}).  This function uses the pdfplumber word list —
    which retains the integrand as part of the '∫X' token — to patch it back.

    Safe-guards:
      - Only fires when \\int is present in latex
      - Does nothing if the integrand is already in the latex
      - Does nothing if no '∫' token is found in words
    """
    if not latex or r'\int' not in latex:
        return latex

    def _norm(t: str) -> str:
        return t.translate(_UNICODE_MATH_TRANS).strip()

    # Find the word containing the integral sign
    int_word = next((w for w in words if '∫' in w['text']), None)
    if not int_word:
        return latex

    # The integrand is whatever is fused to ∫ in the token (e.g. '∫5' → '5')
    integrand = _norm(int_word['text'].replace('∫', '').strip())
    if not integrand:
        return latex

    # Already present in latex?
    if re.sub(r'[^a-zA-Z0-9]', '', integrand) in re.sub(r'[^a-zA-Z0-9]', '', latex):
        return latex

    # Look for a superscript word to the right of the ∫ token
    # (smaller top value = higher on page = superscript, x0 >= int_word.x1)
    body_size = int_word['size']
    body_tops = [w['top'] for w in words if abs(w['size'] - body_size) < 1.5]
    baseline  = sum(body_tops) / len(body_tops) if body_tops else int_word['top']

    sups = sorted(
        [w for w in words
         if w['top'] < baseline - 2 and w['x0'] >= int_word['x1'] - 2],
        key=lambda w: w['x0'],
    )
    integrand_latex = (
        f'{integrand}^{{{_norm(sups[0]["text"])}}}' if sups else integrand
    )

    # Append integrand right after \int_{...}^{...}
    return re.sub(
        r'(\\int(?:_\{[^}]*\})?(?:\^\{[^}]*\})?)',
        rf'\1 {integrand_latex}',
        latex,
        count=1,
    )


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

# Natural-language words that signal a label/caption before a formula,
# never math content themselves.
_LABEL_WORD_RE = re.compile(
    r'^(sigmoid|tanh|relu|softmax|activation|function|equation|formula|'
    r'where|let|given|note|proof|theorem|lemma|corollary|example|solution)$',
    re.IGNORECASE,
)


def _is_label_word(text: str) -> bool:
    """
    Return True if this word is a natural-language label token, not math.
    Guards: any math symbol disqualifies the word immediately.
    """
    if _MATH_RE.search(text):
        return False
    clean = text.rstrip(":").strip()
    if _LABEL_WORD_RE.fullmatch(clean):
        return True
    # Long pure-alpha word ending in ':' → label  (e.g. "Function:", "Definition:")
    if clean.isalpha() and len(clean) > 4 and text.endswith(":"):
        return True
    return False


def _locate_formulas(pdf_path: str) -> list[dict]:
    """
    Find formula bounding boxes on every page using pdfplumber.

    Algorithm (v2 — fraction-aware):
      1. Identify body font size (most frequent rounded size).
      2. For each body-level baseline, collect ALL words (any size) within
         [baseline - 0.85×body, baseline + 1.8×body].  This single expansion
         reliably captures fraction numerators, denominators, and scripts that
         are attached to a body-level expression.
      3. Merge any two seed regions that share at least one word (shared words
         mean they belong to the same multi-line expression, e.g. σ numerator
         and denominator both pull in the fraction-bar region).
      4. Strip leading label words ("Sigmoid Activation Function:", etc.) from
         the top of each merged region.
      5. Apply MATH_RE filter on the remaining words.

    Returns list of dicts:
        {page_0, page, top, bottom, x0, x1, text, words}
    where page_0 is 0-based index and page is 1-based label.
    """
    import pdfplumber

    EXPAND_UP   = 2.5    # fraction of body_size to expand upward — large enough
                         # to capture radical indices (e.g. the "4" in ∜356) and
                         # tall superscripts that sit far above the baseline
    EXPAND_DOWN = 1.8    # fraction of body_size to expand downward from baseline

    results: list[dict] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_label = page_num + 1
            words = page.extract_words(extra_attrs=["size", "fontname"])
            if not words:
                continue

            # ── Body size ──────────────────────────────────────────────────────
            size_counts = Counter(round(w["size"], 0) for w in words)
            body_size   = size_counts.most_common(1)[0][0]

            body_words = [w for w in words if abs(w["size"] - body_size) < 1.5]
            if not body_words:
                continue
            baselines = sorted(set(round(w["top"], 0) for w in body_words))

            # ── Step 1: Build seed regions ─────────────────────────────────────
            # Each word is assigned to its NEAREST baseline only, using midpoints
            # between adjacent baselines as hard boundaries.  This prevents a
            # sub/superscript that sits between two formulas from being captured
            # by both seeds (which would incorrectly merge them).
            seeds: list[dict] = []
            for bi, bl in enumerate(baselines):
                expand_top = bl - body_size * EXPAND_UP
                expand_bot = bl + body_size * EXPAND_DOWN
                # Boundaries are the midpoints between adjacent baselines.
                # Use the midpoint itself as the hard floor/ceiling — NOT
                # max(midpoint, expand_up_formula).  Using max() would create
                # a dead zone where words sit below the midpoint but above the
                # formula's natural expand_top, leaving them in no-man's land
                # (excluded from both adjacent seeds).  The midpoint alone gives
                # every word a unique owner.
                if bi > 0:
                    expand_top = (baselines[bi - 1] + bl) / 2
                if bi + 1 < len(baselines):
                    expand_bot = min(expand_bot,
                                     (bl + baselines[bi + 1]) / 2)
                region_words = [w for w in words
                                if expand_top <= w["top"] <= expand_bot]
                if not region_words:
                    continue
                seeds.append({
                    "top"  : min(w["top"]    for w in region_words),
                    "bottom": max(w["bottom"] for w in region_words),
                    "wids" : set(id(w) for w in region_words),
                    "wlist": region_words,
                })

            # ── Step 2: Merge seeds sharing any word ───────────────────────────
            # Words shared between two seed expansions mean both seeds belong to
            # the same multi-line formula (e.g. fraction numerator + denominator).
            merged: list[dict] = []
            for seed in seeds:
                placed = False
                for m in merged:
                    if m["wids"] & seed["wids"]:   # non-empty intersection
                        m["wids"] |= seed["wids"]
                        seen: set[int] = set()
                        combined: list = []
                        for w in m["wlist"] + seed["wlist"]:
                            if id(w) not in seen:
                                seen.add(id(w))
                                combined.append(w)
                        m["wlist"]  = combined
                        m["top"]    = min(m["top"],    seed["top"])
                        m["bottom"] = max(m["bottom"], seed["bottom"])
                        placed = True
                        break
                if not placed:
                    merged.append({
                        "top"  : seed["top"],
                        "bottom": seed["bottom"],
                        "wids" : set(seed["wids"]),
                        "wlist": list(seed["wlist"]),
                    })

            # ── Step 3: Label-strip + MATH_RE filter ───────────────────────────
            for region in merged:
                rwords   = region["wlist"]
                sorted_w = sorted(rwords,
                                  key=lambda w: (round(w["top"] / 3) * 3, w["x0"]))

                # Strip label prefix: non-math words sitting on the topmost row
                top_row_y    = min(w["top"] for w in rwords)
                label_cutoff = top_row_y + body_size * 0.6

                math_words: list = []
                for w in sorted_w:
                    if w["top"] <= label_cutoff and _is_label_word(w["text"]):
                        continue   # drop label token
                    math_words.append(w)

                if not math_words:
                    continue

                text = " ".join(w["text"] for w in math_words)
                if not _MATH_RE.search(text):
                    continue

                results.append({
                    "page_0": page_num,
                    "page"  : page_label,
                    "top"   : min(w["top"]    for w in math_words),
                    "bottom": max(w["bottom"] for w in math_words),
                    "x0"    : min(w["x0"]    for w in math_words),
                    "x1"    : max(w["x1"]    for w in math_words),
                    "text"  : text,
                    "words" : math_words,   # kept for reconcile_latex; removed before return
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
                        page_w: float, page_h: float,
                        next_top: float = float("inf")) -> "Image | None":
    """
    Render the PDF page and crop the formula region.
    next_top: top of the next formula on this page (caps bottom padding).
    Tries pypdfium2 first, then fitz (PyMuPDF).
    Returns a PIL Image or None if neither renderer is available.
    """
    from PIL import Image

    pad_side = CROP_PADDING_SIDE
    pad_top  = CROP_PADDING_TOP
    scale    = RENDER_SCALE

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

    px0 = max(0,  int((x0     - pad_side) * sx))
    py0 = max(0,  int((top    - pad_top)  * sy))
    px1 = min(iw, int((x1     + pad_side) * sx))
    # Cap bottom padding: if next formula is closer than pad_side, use zero bottom
    # padding so tightly-stacked formulas don't bleed into each other.
    gap_to_next = next_top - bottom
    bot_pad     = 0.0 if gap_to_next < pad_side else min(pad_side, gap_to_next / 2)
    py1 = min(ih, int((bottom + bot_pad)  * sy))

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

    for fi, formula in enumerate(formulas):
        p0     = formula["page_0"]
        pw, ph = page_sizes.get(p0, (612.0, 792.0))
        idx    = page_counters.get(p0, 0) + 1
        page_counters[p0] = idx

        # next formula top on the same page (for bottom-padding cap)
        next_top = float("inf")
        for nf in formulas[fi + 1:]:
            if nf["page_0"] == p0:
                next_top = nf["top"]
                break

        img = _crop_formula_image(
            pdf_path, p0,
            formula["top"], formula["bottom"],
            formula["x0"],  formula["x1"],
            pw, ph,
            next_top=next_top,
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
            f["latex"] = reconcile_latex(clean_latex(model(img)), f.get("words", []))
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
            formulas[i]["latex"] = reconcile_latex(
                clean_latex(latex.strip()), formulas[i].get("words", [])
            )
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
        f.pop("crop_img", None)
        f.pop("page_0",   None)
        f.pop("words",    None)
        f["kind"] = "FORMULA"

    formulas.sort(key=lambda x: (x["page"], x["top"]))
    return formulas


def formula_extraction_available() -> bool:
    """Always True — pdfplumber fallback is always available."""
    return True


# ── Standalone test entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "report.pdf"
    out = sys.argv[2] if len(sys.argv) > 2 else "test_output"

    print(f"\n🔍 Testing formula extraction on: {pdf}")
    print(f"📁 Output dir: {out}\n")

    formulas = extract_formulas(pdf, output_dir=out)

    print(f"\n{'─'*60}")
    print(f"Found {len(formulas)} formula(s):\n")
    for i, f in enumerate(formulas, 1):
        print(f"[{i}] Page {f['page']} | "
              f"top={f['top']:.1f}  bottom={f['bottom']:.1f}")
        print(f"     text  : {f['text']}")
        print(f"     latex : {f['latex'] or '(none)'}")
        print(f"     crop  : {f['crop_path'] or '(none)'}")
        print()