import json
import tempfile
import unittest
from pathlib import Path

from classbuild.export import course_filename, course_to_markdown
from classbuild.models.course import CourseOutline
from classbuild.models.lesson import LessonContent
from classbuild.models.quiz import Quiz
from classbuild.persistence import (
    CorruptedCourseFileError,
    CourseFileTooLargeError,
    CourseNotFoundError,
    CourseStore,
    InvalidCourseIdError,
    UnsupportedSchemaVersionError,
    build_saved_course,
    new_course_id,
)
from tests.fixtures import valid_lesson, valid_outline, valid_quiz


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.store = CourseStore(directory=self.tmp_dir.name)
        self.outline_data = valid_outline(3)
        self.outline = CourseOutline.model_validate(self.outline_data)
        self.cid = new_course_id()
        self.lesson1 = LessonContent.model_validate(valid_lesson(1, self.outline.lessons[0].title))
        self.quiz1 = Quiz.model_validate(valid_quiz(1, self.outline.lessons[0].title))

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_save_and_load_identical_content(self):
        course = build_saved_course(
            course_id=self.cid,
            topic="Introduction to Python",
            outline=self.outline,
            lessons={1: self.lesson1},
            quizzes={1: self.quiz1},
            selected_lesson=1,
        )
        path = self.store.save(course)
        self.assertTrue(path.exists())

        loaded = self.store.load(self.cid)
        self.assertEqual(loaded.course_id, self.cid)
        self.assertEqual(loaded.outline.title, self.outline.title)
        self.assertEqual(len(loaded.lessons), 1)
        self.assertEqual(loaded.lessons[0].title, self.lesson1.title)
        self.assertEqual(len(loaded.quizzes), 1)
        self.assertEqual(loaded.quizzes[0].lesson_number, 1)
        self.assertEqual(loaded.selected_lesson, 1)

    def test_damaged_file_reported(self):
        # Create a damaged JSON file
        bad_path = Path(self.tmp_dir.name) / f"course-{self.cid}.json"
        bad_path.write_text("{ incomplete json ...", encoding="utf-8")

        listing = self.store.list_courses()
        self.assertEqual(len(listing.courses), 0)
        self.assertEqual(len(listing.errors), 1)
        self.assertIn("not valid JSON", listing.errors[0])

    def test_path_traversal_rejected(self):
        with self.assertRaises(InvalidCourseIdError):
            self.store.path_for("../secret")

        with self.assertRaises(InvalidCourseIdError):
            self.store.path_for("invalid_id_not_32_hex")

    def test_file_not_found(self):
        missing_id = "0" * 32
        with self.assertRaises(CourseNotFoundError):
            self.store.load(missing_id)

    def test_course_file_too_large(self):
        # Store with very small limit (10 bytes)
        small_store = CourseStore(directory=self.tmp_dir.name, max_bytes=10)
        course = build_saved_course(
            course_id=self.cid,
            topic="Intro",
            outline=self.outline,
            lessons={},
            quizzes={},
        )
        with self.assertRaises(CourseFileTooLargeError):
            small_store.save(course)

    def test_unsupported_schema_version(self):
        course = build_saved_course(
            course_id=self.cid,
            topic="Intro",
            outline=self.outline,
            lessons={},
            quizzes={},
        )
        self.store.save(course)
        path = self.store.path_for(self.cid)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["schema_version"] = 999
        path.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(UnsupportedSchemaVersionError):
            self.store.load(self.cid)

    def test_markdown_export(self):
        md = course_to_markdown(
            self.outline,
            topic="Introduction to Python",
            lessons={1: self.lesson1},
            quizzes={1: self.quiz1},
        )
        self.assertIn("# Introduction to Python", md)
        self.assertIn("## Course Objectives", md)
        self.assertIn("## Lesson Outline", md)
        self.assertIn("## Lesson 1: Lesson Title 1", md)
        self.assertIn("This lesson has not been generated yet", md)  # for lesson 2 & 3
        self.assertIn("## Quizzes", md)
        self.assertIn("## Answer Key", md)
        filename = course_filename(self.outline.title)
        self.assertEqual(filename, "introduction-to-python.md")


if __name__ == "__main__":
    unittest.main()
