import logging
from dataclasses import dataclass

from classbuild.rag.models import ProcessedDocument
from classbuild.rag.pdf_processor import PDFProcessingError, fingerprint, process_pdf

logger = logging.getLogger(__name__)
COURSE_MATERIAL_KEY = "course_material"
DOC_KEY = "pdf_document"
DOC_STATUS_KEY = "pdf_status"
INDEX_KEY = "pdf_index"  # used in Step 7
SEARCH_KEY = "pdf_search"  # used in Step 7

GENERIC_ERROR = "Something went wrong while reading this PDF. Please try another file."


@dataclass(frozen=True)
class MaterialStatus:
    document: ProcessedDocument | None
    error: str | None
    processed_now: bool


def clear_learning_material(state):
    """Forget the document and everything derived from it."""
    for key in (DOC_KEY, DOC_STATUS_KEY, INDEX_KEY, SEARCH_KEY):
        state.pop(key, None)


def sync_learning_material(state, upload, processor=None):
    """upload is (filename, bytes), or None when no file is selected."""
    if upload is None:
        clear_learning_material(state)
        return MaterialStatus(None, None, False)

    filename, data = upload
    identity = (filename, fingerprint(data))

    # Same file as last time: reuse the stored document (or stored error).
    status = state.get(DOC_STATUS_KEY)
    if status and status["identity"] == identity:
        return MaterialStatus(state.get(DOC_KEY), status["error"], False)

    # A different file: forget everything about the previous one first.
    clear_learning_material(state)

    document, error = None, None
    try:
        document = (processor or process_pdf)(filename, data)
    except PDFProcessingError as exc:
        error = str(exc)
    except Exception:
        logger.exception("Unexpected error while processing the PDF")
        error = GENERIC_ERROR

    if document is not None:
        state[DOC_KEY] = document
        logger.info("Processed PDF %s (%d chunks)", filename, document.chunk_count)
    state[DOC_STATUS_KEY] = {"identity": identity, "error": error}
    return MaterialStatus(document, error, True)

def attach_material_to_course(state):
    """Record which PDF (if any) the course was generated with."""
    document = state.get(DOC_KEY)
    state[COURSE_MATERIAL_KEY] = document.fingerprint if document is not None else None


def material_for_course(state):
    """The loaded PDF, only if it is the same file the current course was made with."""
    document = state.get(DOC_KEY)
    if document is not None and state.get(COURSE_MATERIAL_KEY) == document.fingerprint:
        return document
    return None