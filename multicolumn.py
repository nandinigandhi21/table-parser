import fitz  # PyMuPDF
import json

PDF_PATH = "3Columns.pdf"  # use text-based PDF
OUTPUT_JSON = "3Columns.json"

# -----------------------------
# Open PDF
# -----------------------------
doc = fitz.open(PDF_PATH)
parsed_blocks = []

for page_number, page in enumerate(doc, start=1):
    page_width = page.rect.width

    # Define 3 column boundaries
    col1_end = page_width / 3
    col2_end = 2 * page_width / 3

    def get_column(x0):
        if x0 < col1_end:
            return 1
        elif x0 < col2_end:
            return 2
        else:
            return 3

    blocks = page.get_text("dict")["blocks"]

    for block in blocks:
        if "lines" not in block:
            continue

        x0, y0, x1, y1 = block["bbox"]
        column = get_column(x0)

        text = ""
        max_font_size = 0

        for line in block["lines"]:
            for span in line["spans"]:
                text += span["text"] + " "
                max_font_size = max(max_font_size, span["size"])

        text = text.strip()
        if not text:
            continue

        parsed_blocks.append({
            "page": page_number,
            "column": column,
            "y": y0,
            "font_size": round(max_font_size, 2),
            "text": text
        })

# -----------------------------
# Sort blocks into reading order
# -----------------------------
parsed_blocks.sort(key=lambda b: (b["page"], b["column"], b["y"]))

# -----------------------------
# FONT SIZE HIERARCHY (KEY FIX)
# -----------------------------
font_sizes = sorted({b["font_size"] for b in parsed_blocks}, reverse=True)

TITLE_SIZE = font_sizes[0]
HEADING_SIZE = font_sizes[1] if len(font_sizes) > 1 else TITLE_SIZE
SUBHEADING_SIZE = font_sizes[2] if len(font_sizes) > 2 else HEADING_SIZE

# -----------------------------
# Improved classification
# -----------------------------
def classify_block(block):
    text = block["text"]
    size = block["font_size"]

    if size >= TITLE_SIZE - 0.5:
        return "TITLE"

    if size >= HEADING_SIZE - 0.5 and text[0].isdigit():
        return "HEADING"

    if size >= SUBHEADING_SIZE - 0.5:
        return "SUBHEADING"

    return "BODY"

for block in parsed_blocks:
    block["type"] = classify_block(block)

# -----------------------------
# Merge BODY lines into paragraphs
# -----------------------------
merged_blocks = []

for block in parsed_blocks:
    if (
        merged_blocks
        and block["type"] == "BODY"
        and merged_blocks[-1]["type"] == "BODY"
        and block["page"] == merged_blocks[-1]["page"]
        and block["column"] == merged_blocks[-1]["column"]
        and abs(block["y"] - merged_blocks[-1]["y"]) < 25
    ):
        merged_blocks[-1]["text"] += " " + block["text"]
    else:
        merged_blocks.append(block)

parsed_blocks = merged_blocks

# -----------------------------
# Print output
# -----------------------------
for block in parsed_blocks:
    print(
        f"[Page {block['page']} | Col {block['column']}] "
        f"{block['type']}: {block['text']}"
    )

# -----------------------------
# Save JSON
# -----------------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(parsed_blocks, f, indent=2)

print(f"\nSaved structured output to {OUTPUT_JSON}")