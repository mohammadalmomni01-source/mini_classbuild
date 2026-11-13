import unittest
import numpy as np

from classbuild.models.course import CourseOutline
from classbuild.rag.retriever import DocumentIndex
from classbuild.services.grounded_lesson_generator import (
    GroundedLessonGenerator,
    InsufficientEvidenceError,
)
from classbuild.services.structured_output import ContentValidationError
from tests.fixtures import sample_document, valid_outline


class FakeEmbeddingService:
    def __init__(self, score=0.85):
        self.model_name = "fake-model"
        self.score = score

    def embed_query(self, query):
        return np.array([self.score, 0.0], dtype=np.float32)

    def embed_texts(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class FakeGroq:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def generate_json(self, **kwargs):
        self.calls.append(kwargs)
        return self.answers.pop(0)


class GroundedLessonTests(unittest.TestCase):
    def setUp(self):
        self.outline = CourseOutline.model_validate(valid_outline(3))
        self.doc = sample_document()
        self.index = DocumentIndex(
            document_fingerprint=self.doc.fingerprint,
            model_name="fake-model",
            chunk_ids=tuple(c.chunk_id for c in self.doc.chunks),
            embeddings=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        )

    def test_low_similarity_raises_error(self):
        # Query with perpendicular vector -> similarity 0.0 < 0.30
        embeddings = FakeEmbeddingService(score=0.10)
        embeddings.embed_query = lambda q: np.array([0.0, 1.0], dtype=np.float32)
        generator = GroundedLessonGenerator(client=FakeGroq([]), embeddings=embeddings)

        with self.assertRaises(InsufficientEvidenceError) as ctx:
            generator.generate_lesson(self.outline, 1, self.doc, self.index)
        self.assertEqual(ctx.exception.reason, "low_similarity")

    def test_unknown_source_id_rejected(self):
        # Returns draft citing an unknown id
        bad_draft = {
            "evidence_relevant": True,
            "lesson_number": 1,
            "title": self.outline.lessons[0].title,
            "introduction": "Intro text.",
            "sections": [
                {
                    "source_ids": ["unknown-c999"],
                    "heading": "Section 1",
                    "explanation": "Explanation 1",
                    "example": "example 1",
                },
                {
                    "source_ids": ["unknown-c999"],
                    "heading": "Section 2",
                    "explanation": "Explanation 2",
                    "example": "example 2",
                },
            ],
            "summary": "Summary text.",
            "exercises": [
                {"question": "Q1", "hint": "H1"},
                {"question": "Q2", "hint": "H2"},
            ],
        }
        embeddings = FakeEmbeddingService(score=0.90)
        generator = GroundedLessonGenerator(
            client=FakeGroq([bad_draft, bad_draft]), embeddings=embeddings
        )

        with self.assertRaises(ContentValidationError) as ctx:
            generator.generate_lesson(self.outline, 1, self.doc, self.index)
        self.assertTrue(any("unknown source ids" in p for p in ctx.exception.problems))

    def test_sources_from_retrieval(self):
        good_draft = {
            "evidence_relevant": True,
            "lesson_number": 1,
            "title": self.outline.lessons[0].title,
            "introduction": "Intro text.",
            "sections": [
                {
                    "source_ids": ["p1-c1"],
                    "heading": "Section 1",
                    "explanation": "Explanation 1",
                    "example": "example 1",
                },
                {
                    "source_ids": ["p1-c1"],
                    "heading": "Section 2",
                    "explanation": "Explanation 2",
                    "example": "example 2",
                },
            ],
            "summary": "Summary text.",
            "exercises": [
                {"question": "Q1", "hint": "H1"},
                {"question": "Q2", "hint": "H2"},
            ],
        }
        embeddings = FakeEmbeddingService(score=0.95)
        generator = GroundedLessonGenerator(
            client=FakeGroq([good_draft]), embeddings=embeddings
        )

        lesson = generator.generate_lesson(self.outline, 1, self.doc, self.index)
        self.assertTrue(lesson.is_grounded)
        source = lesson.grounding.sources[0]
        self.assertEqual(source.chunk_id, "p1-c1")
        self.assertEqual(source.page_number, 1)
        self.assertEqual(source.source_filename, "sample.pdf")
        self.assertEqual(source.text, self.doc.chunks[0].text)


if __name__ == "__main__":
    unittest.main()
