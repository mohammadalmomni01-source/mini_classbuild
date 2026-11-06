import sys

from classbuild.rag.embeddings import EmbeddingError, EmbeddingService
from classbuild.rag.pdf_processor import process_pdf
from classbuild.rag.retriever import RetrievalError, build_index, search

questions = sys.argv[1:] or [
    "What does dynamically typed mean in Python?",
    "What is the difference between lists and tuples?",
    "How do Kubernetes pods work?",
]

with open("sample.pdf", "rb") as file:
    document = process_pdf("sample.pdf", file.read())

service = EmbeddingService()
try:
    index = build_index(document, service)
    print(f"Embedded {index.size} chunks, {index.embeddings.shape[1]} numbers each")
    for question in questions:
        print(f"\nQ: {question}")
        for result in search(index, document, question, service):
            preview = result.text[:70].replace("\n", " ")
            print(
                f"  #{result.rank} {result.chunk_id} (page {result.page_number}) "
                f"score {result.score:.3f}: {preview}..."
            )
except (EmbeddingError, RetrievalError) as exc:
    print(f"[FAILED] {exc}")