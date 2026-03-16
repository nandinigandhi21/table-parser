from docling.document_converter import DocumentConverter

source = "https://arxiv.org/pdf/1512.03385"  # file path or URL
converter = DocumentConverter()
doc = converter.convert(source).document

print(doc.export_to_markdown())  # output: "### Docling Technical Report[...]"