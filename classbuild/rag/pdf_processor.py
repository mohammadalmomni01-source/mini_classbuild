import hashlib
import io
import logging
import re
import unicodedata

from pypdf import PdfReader

from classbuild.rag.chunker import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_pages
from classbuild.rag.models import PageText, ProcessedDocument

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


class PDFProcessingError(Exception):
    """Base class. Every message is written for the user."""


class InvalidFileError(PDFProcessingError):
    pass


class FileTooLargeError(PDFProcessingError):
    pass


class EmptyFileError(PDFProcessingError):
    pass


class CorruptedPDFError(PDFProcessingError):
    pass


class EncryptedPDFError(PDFProcessingError):
    pass


class NoExtractableTextError(PDFProcessingError):
    pass


def fingerprint(data):
    """Identifies a PDF by its content, not its name."""
    return hashlib.sha256(data).hexdigest()


def validate_pdf_upload(filename, data):
    if not filename or not filename.lower().endswith(".pdf"):
        raise InvalidFileError("Only PDF files are supported. Please upload a .pdf file.")
    if not data:
        raise EmptyFileError("The uploaded file is empty.")
    if len(data) > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(
            f"The file is larger than {MAX_FILE_SIZE_MB} MB. Please upload a smaller PDF."
        )
    if b"%PDF" not in data[:1024]:
        raise InvalidFileError("This file does not look like a real PDF.")


def clean_text(text):
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")

    lines = []
    for line in text.split("\n"):
        stripped = line.lstrip(" \t")
        indent = line[: len(line) - len(stripped)]  # keep code indentation
        lines.append(indent + re.sub(r"[ \t]{2,}", " ", stripped).rstrip())
    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line in a row
    return text.strip()


def extract_pages(data):
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise CorruptedPDFError(
            "This PDF could not be read. It may be corrupted. Try another copy."
        ) from exc

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as exc:
            raise EncryptedPDFError(
                "This PDF is password-protected. Please upload an unlocked copy."
            ) from exc
        if not unlocked:
            raise EncryptedPDFError(
                "This PDF is password-protected. Please upload an unlocked copy."
            )

    try:
        raw_pages = list(reader.pages)
    except Exception as exc:
        raise CorruptedPDFError("The pages of this PDF could not be read.") from exc

    pages = []
    for number, page in enumerate(raw_pages, start=1):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception:
            logger.warning("Could not extract text from page %d", number)
            text = ""  # one bad page does not fail the whole document
        pages.append(PageText(page_number=number, text=text))
    return pages


def process_pdf(filename, data, chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    validate_pdf_upload(filename, data)
    pages = extract_pages(data)
    if not pages:
        raise CorruptedPDFError("This PDF has no pages.")

    with_text = [page for page in pages if page.text]
    if not with_text:
        raise NoExtractableTextError(
            "No text could be extracted. This looks like a scanned or image-only PDF "
            "(OCR is not supported)."
        )

    full_text = "\n".join(page.text for page in with_text)
    return ProcessedDocument(
        filename=filename,
        fingerprint=fingerprint(data),
        page_count=len(pages),
        pages_with_text=len(with_text),
        empty_pages=[page.page_number for page in pages if not page.text],
        word_count=len(full_text.split()),
        char_count=len(full_text),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunks=chunk_pages(pages, filename, chunk_size, chunk_overlap),
    )