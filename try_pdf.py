import os
import sys

from classbuild.rag.pdf_processor import PDFProcessingError, process_pdf

path = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"
try:
    with open(path, "rb") as file:
        document = process_pdf(os.path.basename(path), file.read())
except FileNotFoundError:
    print(f"[FAILED] File not found: {path}")
except PDFProcessingError as exc:
    print(f"[FAILED] {exc}")
else:
    print(
        f"[OK] {document.filename}: {document.page_count} pages, "
        f"{document.word_count} words, {document.chunk_count} chunks"
    )
    print("Pages without text:", document.empty_pages or "none")
    print("Fingerprint:", document.fingerprint[:12])
    for chunk in document.chunks:
        preview = chunk.text[:70].replace("\n", " ")
        print(f"  {chunk.chunk_id} (page {chunk.page_number}, {chunk.char_count} chars): {preview}...")