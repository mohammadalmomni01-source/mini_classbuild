import unittest
from pydantic import ValidationError

from classbuild.models.course import validate_outline
from tests.fixtures import valid_outline


class CourseOutlineTests(unittest.TestCase):
    def test_valid_outline_passes(self):
        data = valid_outline(3, "Beginner")
        outline = validate_outline(data, expected_lesson_count=3, expected_difficulty="Beginner")
        self.assertEqual(len(outline.lessons), 3)
        self.assertEqual(outline.difficulty, "Beginner")

    def test_wrong_lesson_count_rejected(self):
        data = valid_outline(3, "Beginner")
        with self.assertRaises(ValidationError) as ctx:
            validate_outline(data, expected_lesson_count=4)
        self.assertIn("expected 4 lessons but got 3", str(ctx.exception))

    def test_wrong_difficulty_rejected(self):
        data = valid_outline(3, "Beginner")
        with self.assertRaises(ValidationError) as ctx:
            validate_outline(data, expected_difficulty="Advanced")
        self.assertIn("expected difficulty Advanced but got Beginner", str(ctx.exception))

    def test_nonsequential_lesson_numbers_rejected(self):
        data = valid_outline(3)
        data["lessons"][1]["lesson_number"] = 3
        data["lessons"][2]["lesson_number"] = 2
        with self.assertRaises(ValidationError) as ctx:
            validate_outline(data)
        self.assertIn("lesson numbers must be 1, 2, 3 ... in order", str(ctx.exception))

    def test_duplicate_lesson_titles_rejected(self):
        data = valid_outline(3)
        data["lessons"][2]["title"] = "  LESSON title 1  "  # case & spacing variation
        with self.assertRaises(ValidationError) as ctx:
            validate_outline(data)
        self.assertIn("Lesson titles must be different", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
