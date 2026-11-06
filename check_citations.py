import copy

from pydantic import ValidationError

from classbuild.models.lesson import SourceReference, validate_lesson

source = SourceReference(
    chunk_id="p1-c2",
    source_filename="sample.pdf",
    page_number=1,
    chunk_index=2,
    similarity=0.45,
    text="Python determines the type automatically.",
)

base = {
    "lesson_number": 1,
    "title": "Variables",
    "introduction": "Intro.",
    "sections": [
        {"heading": "A", "explanation": "x", "example": "y", "source_ids": ["p1-c2"]},
        {"heading": "B", "explanation": "x", "example": "y", "source_ids": []},
    ],
    "summary": "Summary.",
    "exercises": [{"question": "q1", "hint": "h1"}, {"question": "q2", "hint": "h2"}],
}
grounded = copy.deepcopy(base)
grounded["grounding"] = {
    "document_fingerprint": "abc123",
    "source_filename": "sample.pdf",
    "retrieval_query": "variables",
    "sources": [source.model_dump()],
}
allowed = {source.chunk_id: source}
context = {"allowed_sources": allowed, "expected_fingerprint": "abc123"}


def check(label, data, should_pass, **ctx):
    try:
        validate_lesson(data, **ctx)
    except ValidationError as exc:
        tag = "BUG" if should_pass else "OK"
        print(f"[{tag}] {label}: rejected -> {exc.errors()[0]['msg']}")
    else:
        tag = "OK" if should_pass else "BUG"
        print(f"[{tag}] {label}: accepted")


ordinary = copy.deepcopy(base)
ordinary["sections"][0]["source_ids"] = []

invented = copy.deepcopy(grounded)
invented["sections"][0]["source_ids"] = ["p9-c9"]

uncited = copy.deepcopy(grounded)
uncited["sections"][0]["source_ids"] = []

tampered = copy.deepcopy(grounded)
tampered["grounding"]["sources"][0]["page_number"] = 5

check("valid grounded lesson", grounded, True, **context)
check("ordinary lesson", ordinary, True)
check("ordinary lesson with a citation", base, False)
check("invented source id", invented, False, **context)
check("grounded lesson without citations", uncited, False, **context)
check("changed page number", tampered, False, **context)
check("source from another PDF", grounded, False, allowed_sources=allowed, expected_fingerprint="other")