import tempfile
import unittest

import numpy as np

from classbuild.course_config import validate_course_config
from classbuild.export import course_filename, course_to_markdown
from classbuild.models.chat import ChatMessage
from classbuild.persistence import CourseStore, build_saved_course, new_course_id
from classbuild.rag.chunker import chunk_pages
from classbuild.rag.pdf_processor import process_pdf
from classbuild.rag.retriever import build_index
from classbuild.services.course_generator import CourseGenerator
from classbuild.services.grounded_lesson_generator import GroundedLessonGenerator
from classbuild.services.lesson_chat import (
    LessonChatContext,
    LessonChatService,
    PdfAnswerDraft,
)
from classbuild.services.lesson_generator import LessonGenerator
from classbuild.services.quiz_generator import QuizGenerator
from classbuild.services.quiz_grader import grade_quiz
from tests.fixtures import valid_lesson, valid_outline, valid_quiz


class FakeEmbeddingService:
    def __init__(self):
        self.model_name = "fake-model"

    def embed_query(self, query):
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_texts(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class ScriptedGroq:
    """Replaces Groq with prepared answers. Verifies all are consumed."""
    def __init__(self, json_answers, chat_answers=None):
        self.json_answers = list(json_answers)
        self.chat_answers = list(chat_answers or [])
        self.json_calls = []
        self.chat_calls = []

    def generate_json(self, **kwargs):
        self.json_calls.append(kwargs)
        if not self.json_answers:
            raise RuntimeError("No more prepared JSON answers in ScriptedGroq!")
        return self.json_answers.pop(0)

    def chat(self, messages, **kwargs):
        self.chat_calls.append(messages)
        if not self.chat_answers:
            raise RuntimeError("No more prepared chat answers in ScriptedGroq!")
        return self.chat_answers.pop(0)

    def verify_all_used(self, test_case):
        test_case.assertEqual(len(self.json_answers), 0, "Not all JSON answers were used")
        test_case.assertEqual(len(self.chat_answers), 0, "Not all chat answers were used")


class EndToEndWorkflowTests(unittest.TestCase):
    def test_complete_student_workflow(self):
        outline_raw = valid_outline(3)
        plain_lesson_raw = valid_lesson(1, outline_raw["lessons"][0]["title"], is_grounded=False)
        quiz_raw = valid_quiz(1, outline_raw["lessons"][0]["title"])

        grounded_draft = {
            "evidence_relevant": True,
            "lesson_number": 2,
            "title": outline_raw["lessons"][1]["title"],
            "introduction": "Introduction to grounded lesson.",
            "sections": [
                {
                    "source_ids": ["p1-c1"],
                    "heading": "PDF Concept",
                    "explanation": "Explanation from PDF.",
                    "example": "x = 10",
                },
                {
                    "source_ids": ["p1-c1"],
                    "heading": "PDF Practice",
                    "explanation": "Practice from PDF.",
                    "example": "y = 20",
                },
            ],
            "summary": "Summary of grounded lesson.",
            "exercises": [
                {"question": "Q1?", "hint": "H1"},
                {"question": "Q2?", "hint": "H2"},
            ],
        }

        # Chat answer 1 (PDF mode)
        chat_draft_1 = {
            "evidence_sufficient": True,
            "answer": "This is covered in the PDF [p1-c1].",
            "source_ids": ["p1-c1"],
        }
        # Chat answer 2 (Follow-up)
        chat_draft_2 = {
            "evidence_sufficient": True,
            "answer": "Following up on your question, yes indeed [p1-c1].",
            "source_ids": ["p1-c1"],
        }

        scripted_groq = ScriptedGroq(
            json_answers=[
                outline_raw,       # 1. CourseGenerator
                plain_lesson_raw,  # 2. LessonGenerator
                quiz_raw,          # 3. QuizGenerator
                grounded_draft,    # 4. GroundedLessonGenerator
                chat_draft_1,      # 5. LessonChatService question 1
                chat_draft_2,      # 6. LessonChatService question 2
            ]
        )

        # 1. Configuration & Outline
        config = validate_course_config("Python Basics", "Beginner", 3)
        outline = CourseGenerator(scripted_groq).generate_outline(config)
        self.assertEqual(len(outline.lessons), 3)

        # 2. Plain Lesson
        lesson1 = LessonGenerator(scripted_groq).generate_lesson(outline, 1)
        self.assertEqual(lesson1.lesson_number, 1)
        self.assertFalse(lesson1.is_grounded)

        # 3. Quiz & Grading
        quiz1 = QuizGenerator(scripted_groq).generate_quiz(outline, 1, lesson1)
        self.assertEqual(len(quiz1.questions), 5)
        answers = [q.correct_answer for q in quiz1.questions]
        result = grade_quiz(quiz1, answers)
        self.assertEqual(result.correct_count, 5)
        self.assertEqual(result.percentage, 100.0)

        # 4. PDF Chunks, Index & Retrieval
        from classbuild.rag.models import PageText, ProcessedDocument
        raw_pages = [PageText(page_number=1, text="Python programming fundamentals and syntax.")]
        chunks = chunk_pages(raw_pages, "test.pdf", chunk_size=500, chunk_overlap=50)
        doc = ProcessedDocument(
            filename="test.pdf",
            fingerprint="c" * 32,
            page_count=1,
            pages_with_text=1,
            word_count=5,
            char_count=45,
            empty_pages=[],
            chunks=chunks,
            chunk_size=500,
            chunk_overlap=50,
        )
        embeddings = FakeEmbeddingService()
        index = build_index(doc, embeddings)
        self.assertEqual(index.size, len(chunks))

        # 5. Grounded Lesson
        grounded_gen = GroundedLessonGenerator(client=scripted_groq, embeddings=embeddings)
        lesson2 = grounded_gen.generate_lesson(outline, 2, doc, index)
        self.assertTrue(lesson2.is_grounded)
        self.assertEqual(lesson2.lesson_number, 2)

        # 6. Chat & Follow-up with history
        chat_service = LessonChatService(client=scripted_groq, embeddings=embeddings)
        ctx = LessonChatContext.from_course(outline, 2, lesson2)
        reply1 = chat_service.answer(ctx, "What does the PDF say?", history=(), document=doc, index=index)
        self.assertEqual(reply1.answer.status, "answered")

        history = [reply1.question, reply1.answer]
        reply2 = chat_service.answer(ctx, "Can you clarify that?", history=history, document=doc, index=index)
        self.assertEqual(reply2.answer.status, "answered")

        # 7. Save & Load
        lessons = {1: lesson1, 2: lesson2}
        quizzes = {1: quiz1}
        cid = new_course_id()
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = CourseStore(directory=tmp_dir)
            course = build_saved_course(
                course_id=cid,
                topic="Python Basics",
                outline=outline,
                lessons=lessons,
                quizzes=quizzes,
                selected_lesson=2,
            )
            store.save(course)
            loaded = store.load(cid)

            self.assertEqual(loaded.course_id, cid)
            self.assertEqual(len(loaded.lessons), 2)
            self.assertEqual(len(loaded.quizzes), 1)
            self.assertEqual(loaded.selected_lesson, 2)

            # 8. Markdown Export
            md = course_to_markdown(
                loaded.outline,
                loaded.topic,
                loaded.lessons_by_number(),
                loaded.quizzes_by_number(),
            )
            self.assertIn("# Introduction to Python", md)
            self.assertIn("## Answer Key", md)
            self.assertEqual(course_filename(loaded.outline.title), "introduction-to-python.md")

        # 9. Verify every prepared answer was used!
        scripted_groq.verify_all_used(self)


if __name__ == "__main__":
    unittest.main()
