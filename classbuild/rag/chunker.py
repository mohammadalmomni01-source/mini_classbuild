from classbuild.rag.models import DocumentChunk

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150

# Preferred places to cut, strongest first.
_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ")


def _validate(chunk_size, chunk_overlap):
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")


def _find_cut(text, start, limit):
    """Where the chunk that begins at `start` should end (exclusive)."""
    end = min(start + limit, len(text))
    if end == len(text):
        return end
    window_start = start + limit // 2  # only cut in the second half of the window
    for separator in _SEPARATORS:
        position = text.rfind(separator, window_start, end)
        if position != -1:
            return position + len(separator)
    return end  # no good separator: cut at the limit


def _next_start(text, cut, start, overlap):
    """Begin the next chunk about `overlap` characters before the cut, at a word start."""
    if overlap == 0:
        return cut
    candidate = max(cut - overlap, start + 1)
    while candidate < cut and not text[candidate - 1].isspace():
        candidate += 1
    return candidate


def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    _validate(chunk_size, chunk_overlap)
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        cut = _find_cut(text, start, chunk_size)
        piece = text[start:cut].strip()
        if piece:
            chunks.append(piece)
        if cut >= len(text):
            break
        start = _next_start(text, cut, start, chunk_overlap)
    return chunks


def chunk_pages(pages, source_filename, chunk_size=DEFAULT_CHUNK_SIZE,
                chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Chunk every page separately, so a chunk never spans two pages."""
    chunks = []
    for page in pages:
        pieces = chunk_text(page.text, chunk_size, chunk_overlap)
        for index, piece in enumerate(pieces, start=1):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"p{page.page_number}-c{index}",
                    source_filename=source_filename,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=piece,
                )
            )
    return chunks