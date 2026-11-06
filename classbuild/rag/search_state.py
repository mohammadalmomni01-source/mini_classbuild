import logging
from dataclasses import dataclass

from classbuild.rag.document_state import DOC_KEY, INDEX_KEY, SEARCH_KEY
from classbuild.rag.embeddings import EmbeddingService
from classbuild.rag.retriever import (
    DEFAULT_TOP_K,
    EmptyQueryError,
    NoDocumentError,
    build_index,
    search,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchOutcome:
    query: str
    results: list
    index_built_now: bool


def has_current_index(state, document, service):
    index = state.get(INDEX_KEY)
    return (
        index is not None
        and document is not None
        and index.document_fingerprint == document.fingerprint
        and index.chunk_ids == tuple(chunk.chunk_id for chunk in document.chunks)
        and index.model_name == service.model_name
    )


def get_or_build_index(state, document, service):
    if has_current_index(state, document, service):
        return state[INDEX_KEY]
    index = build_index(document, service)
    state[INDEX_KEY] = index
    logger.info("Built embedding index for %d chunks", index.size)
    return index


def run_search(state, query, service=None, top_k=DEFAULT_TOP_K):
    state.pop(SEARCH_KEY, None)  # never keep old results around

    document = state.get(DOC_KEY)
    if document is None:
        raise NoDocumentError("Upload a PDF first.")
    if not isinstance(query, str) or not query.strip():
        raise EmptyQueryError("Please enter a question first.")

    service = service or EmbeddingService()
    built_now = not has_current_index(state, document, service)
    index = get_or_build_index(state, document, service)
    results = search(index, document, query, service, top_k)

    state[SEARCH_KEY] = {
        "query": query.strip(),
        "fingerprint": document.fingerprint,
        "results": results,
    }
    return SearchOutcome(query.strip(), results, built_now)


def current_results(state):
    """The last search, only if it belongs to the PDF that is loaded now."""
    document = state.get(DOC_KEY)
    saved = state.get(SEARCH_KEY)
    if document is None or not saved or saved["fingerprint"] != document.fingerprint:
        return None
    return saved