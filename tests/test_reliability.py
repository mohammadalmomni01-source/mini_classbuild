import os
import unittest
from unittest.mock import patch

import httpx
import numpy as np
import pydantic

from classbuild.config import get_settings
from classbuild.llm.groq_client import GroqClient, redact_secrets
from classbuild.models.course import CourseOutline
from classbuild.rag.retriever import DocumentIndex
from classbuild.services.grounded_lesson_generator import (
    GroundedLessonGenerator,
    LessonValidationError,
)
from tests.fixtures import sample_document, valid_outline


class ReliabilityTests(unittest.TestCase):
    def test_uncited_page_mention_rejected(self):
        outline = CourseOutline.model_validate(valid_outline(3))
        doc = sample_document()
        index = DocumentIndex(
            document_fingerprint=doc.fingerprint,
            model_name="fake-model",
            chunk_ids=tuple(c.chunk_id for c in doc.chunks),
            embeddings=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        )

        class FakeEmbeddingService:
            model_name = "fake-model"
            def embed_query(self, query):
                return np.array([1.0, 0.0], dtype=np.float32)

        # Draft mentions page 3 in explanation, but only page 1 is cited!
        draft = {
            "evidence_relevant": True,
            "lesson_number": 1,
            "title": outline.lessons[0].title,
            "introduction": "Intro mentioning nothing.",
            "sections": [
                {
                    "source_ids": ["p1-c1"],
                    "heading": "Section 1",
                    "explanation": "As seen on page 3, this is an important concept.",
                    "example": "example 1",
                },
                {
                    "source_ids": ["p1-c1"],
                    "heading": "Section 2",
                    "explanation": "Normal explanation.",
                    "example": "example 2",
                },
            ],
            "summary": "Summary text.",
            "exercises": [
                {"question": "Q1", "hint": "H1"},
                {"question": "Q2", "hint": "H2"},
            ],
        }

        class FakeGroq:
            def generate_json(self, **kwargs):
                return draft

        generator = GroundedLessonGenerator(client=FakeGroq(), embeddings=FakeEmbeddingService())
        with self.assertRaises(LessonValidationError) as ctx:
            generator.generate_lesson(outline, 1, doc, index)
        self.assertTrue(any("Page 3 is mentioned" in p for p in ctx.exception.problems))

    def test_temporary_error_retried_once(self):
        attempts = 0

        def handler(request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(503, json={"error": {"message": "Service unavailable"}})

        transport = httpx.MockTransport(handler)
        client = GroqClient(
            api_key="gsk_" + "a" * 24,
            max_retries=1,
            timeout_seconds=5,
        )
        # Patch the underlying httpx client transport
        client._client._client = httpx.Client(transport=transport)

        with self.assertRaises(Exception):
            client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(attempts, 2)  # 1 initial + 1 retry

    def test_client_errors_not_retried(self):
        attempts = 0

        def handler(request):
            nonlocal attempts
            attempts += 1
            return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})

        transport = httpx.MockTransport(handler)
        client = GroqClient(
            api_key="gsk_" + "a" * 24,
            max_retries=2,
            timeout_seconds=5,
        )
        client._client._client = httpx.Client(transport=transport)

        with self.assertRaises(Exception):
            client.chat([{"role": "user", "content": "hi"}])

        self.assertEqual(attempts, 1)  # 401 must NEVER be retried

    def test_api_key_redacted_from_errors(self):
        fake_key = "gsk_" + "1234567890abcdef12345678"
        text = f"Error with key {fake_key} occurred while processing."
        redacted = redact_secrets(text)
        self.assertNotIn(fake_key, redacted)
        self.assertIn("[redacted]", redacted)

    def test_groq_max_retries_clamped(self):
        with patch.dict(os.environ, {"GROQ_MAX_RETRIES": "99"}):
            get_settings.cache_clear()
            settings = get_settings()
            self.assertEqual(settings.max_retries, 3)

        with patch.dict(os.environ, {"GROQ_MAX_RETRIES": "-5"}):
            get_settings.cache_clear()
            settings = get_settings()
            self.assertEqual(settings.max_retries, 0)

        with patch.dict(os.environ, {"GROQ_MAX_RETRIES": "invalid"}):
            get_settings.cache_clear()
            settings = get_settings()
            self.assertEqual(settings.max_retries, 1)

        get_settings.cache_clear()


if __name__ == "__main__":
    unittest.main()
