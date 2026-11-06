from classbuild.course_config import validate_course_config
from classbuild.llm import GroqClientError
from classbuild.rag.embeddings import EmbeddingError, EmbeddingService
from classbuild.rag.pdf_processor import process_pdf
from classbuild.rag.retriever import RetrievalError, build_index
from classbuild.services.course_generator import CourseGenerator
from classbuild.services.grounded_lesson_generator import (
    GroundedLessonGenerator,
    InsufficientEvidenceError,
)
from classbuild.services.structured_output import CourseGenerationError

with open("sample.pdf", "rb") as file:
    document = process_pdf("sample.pdf", file.read())

embeddings = EmbeddingService()
index = build_index(document, embeddings)
generator = GroundedLessonGenerator(embeddings=embeddings)

for topic in ("Introduction to Python", "Kubernetes for Beginners"):
    print(f"\n=== {topic} ===")
    try:
        config = validate_course_config(topic, "Beginner", 3)
        outline = CourseGenerator().generate_outline(config)
        lesson = generator.generate_lesson(outline, 2, document, index)
    except InsufficientEvidenceError as exc:
        scores = ", ".join(f"{passage.score:.2f}" for passage in exc.passages)
        print(f"[NO EVIDENCE] {exc.reason}: {exc} (scores: {scores})")
    except (CourseGenerationError, GroqClientError, EmbeddingError, RetrievalError) as exc:
        print(f"[FAILED] {exc}")
    else:
        print(f"[OK] Lesson {lesson.lesson_number}: {lesson.title}")
        print("Cited pages:", lesson.cited_pages)
        for section in lesson.sections:
            print(f"  - {section.heading} -> {section.source_ids}")
        for source in lesson.grounding.sources:
            print(f"  source {source.chunk_id}, page {source.page_number}, "
                  f"similarity {source.similarity:.3f}")