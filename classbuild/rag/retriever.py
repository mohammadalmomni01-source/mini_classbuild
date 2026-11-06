from dataclasses import dataclass

import numpy as np

DEFAULT_TOP_K = 3


class RetrievalError(Exception):
    """Base class. Every message is written for the user."""


class EmptyQueryError(RetrievalError):
    pass


class NoDocumentError(RetrievalError):
    pass


class IndexMismatchError(RetrievalError):
    pass


@dataclass(frozen=True)
class DocumentIndex:
    """The vectors of one document's chunks."""

    document_fingerprint: str
    model_name: str
    chunk_ids: tuple
    embeddings: np.ndarray

    def __post_init__(self):
        self.embeddings.setflags(write=False)  # read-only

    @property
    def size(self):
        return len(self.chunk_ids)


@dataclass(frozen=True)
class SearchResult:
    rank: int
    chunk_id: str
    source_filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float


def cosine_similarity(query, matrix):
    """Similarity of one vector against every row: 1 = same direction, 0 = unrelated."""
    query = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    if query.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
        raise ValueError("The vectors have different sizes.")

    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    scores = np.zeros(matrix.shape[0], dtype=np.float32)
    usable = norms > 0  # a zero vector scores 0 instead of dividing by zero
    scores[usable] = (matrix[usable] @ query) / norms[usable]
    return np.nan_to_num(scores)


def build_index(document, service):
    if document is None or not document.chunks:
        raise NoDocumentError("Upload a PDF first.")
    matrix = service.embed_texts([chunk.text for chunk in document.chunks])
    return DocumentIndex(
        document_fingerprint=document.fingerprint,
        model_name=service.model_name,
        chunk_ids=tuple(chunk.chunk_id for chunk in document.chunks),
        embeddings=matrix,
    )


def check_index(index, document):
    ids = tuple(chunk.chunk_id for chunk in document.chunks)
    if index.document_fingerprint != document.fingerprint or index.chunk_ids != ids:
        raise IndexMismatchError("The search index belongs to a different PDF. Please search again.")


def search(index, document, query, service, top_k=DEFAULT_TOP_K):
    if document is None:
        raise NoDocumentError("Upload a PDF first.")
    if not isinstance(query, str) or not query.strip():
        raise EmptyQueryError("Please enter a question first.")
    check_index(index, document)

    scores = cosine_similarity(service.embed_query(query), index.embeddings)
    order = np.argsort(-scores, kind="stable")[:top_k]  # ties keep document order

    results = []
    for rank, position in enumerate(order, start=1):
        chunk = document.chunks[int(position)]
        results.append(
            SearchResult(
                rank=rank,
                chunk_id=chunk.chunk_id,
                source_filename=chunk.source_filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=float(scores[position]),
            )
        )
    return results