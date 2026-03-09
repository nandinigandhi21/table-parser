import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("docling").setLevel(logging.ERROR)  # Suppress cleanup noise

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
import re

def extract_formulas(pdf_path: str):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_formula_enrichment = True  # Enable formula detection
    pipeline_options.do_ocr = False                # Set True for scanned PDFs

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"📄 Converting: {pdf_path}")
    result = converter.convert(pdf_path)
    doc = result.document

    # ── Export to Markdown (LaTeX equations preserved as $$...$$) ──
    markdown = doc.export_to_markdown()

    # ── Extract block equations: $$...$$ ──
    block_eqs = re.findall(r'\$\$(.*?)\$\$', markdown, re.DOTALL)

    # ── Extract inline equations: $...$ ──
    inline_eqs = re.findall(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', markdown)

    # ── Extract formula items directly from document structure ──
    formula_items = []
    for item, _ in doc.iterate_items():
        label = getattr(item, "label", "")
        text  = getattr(item, "text",  "")
        orig  = getattr(item, "orig",  "")  # Raw LaTeX if available

        if str(label).lower() in ("formula", "equation", "math"):
            formula_items.append({
                "label": str(label),
                "text":  text,
                "latex": orig or text,
            })

    return {
        "markdown":      markdown,
        "block_eqs":     [eq.strip() for eq in block_eqs],
        "inline_eqs":    [eq.strip() for eq in inline_eqs],
        "formula_items": formula_items,
    }


def print_results(data: dict):
    print("\n" + "=" * 55)
    print("📐 BLOCK EQUATIONS ($$...$$)")
    print("=" * 55)
    if data["block_eqs"]:
        for i, eq in enumerate(data["block_eqs"], 1):
            print(f"[{i}] {eq}\n")
    else:
        print("  None found.\n")

    print("=" * 55)
    print("🔢 INLINE EQUATIONS ($...$)")
    print("=" * 55)
    if data["inline_eqs"]:
        for i, eq in enumerate(data["inline_eqs"], 1):
            print(f"[{i}] {eq}")
    else:
        print("  None found.\n")

    print("\n" + "=" * 55)
    print("🧮 FORMULA ITEMS (from document structure)")
    print("=" * 55)
    if data["formula_items"]:
        for i, item in enumerate(data["formula_items"], 1):
            print(f"[{i}] Label : {item['label']}")
            print(f"     LaTeX : {item['latex']}\n")
    else:
        print("  None found.\n")

    print("=" * 55)
    print("📝 FULL MARKDOWN OUTPUT")
    print("=" * 55)
    print(data["markdown"])


if __name__ == "__main__":
    PDF_PATH = "formula.pdf"   # 🔁 Replace with your PDF path
    data = extract_formulas(PDF_PATH)
    print_results(data)