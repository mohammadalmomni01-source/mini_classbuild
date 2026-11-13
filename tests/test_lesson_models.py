import unittest
from pydantic import ValidationError

from classbuild.models.lesson import validate_lesson
from tests.fixtures import valid_lesson


class LessonContentTests(unittest.TestCase):
    def test_wrong_lesson_number_rejected(self):
        data = valid_lesson(lesson_number=1, title="Introduction")
        with self.assertRaises(ValidationError) as ctx:
            validate_lesson(data, expected_lesson_number=2)
        self.assertIn("expected lesson 2 but got lesson 1", str(ctx.exception))

    def test_wrong_title_rejected(self):
        data = valid_lesson(lesson_number=1, title="Introduction")
        with self.assertRaises(ValidationError) as ctx:
            validate_lesson(data, expected_title="Advanced Topic")
        self.assertIn("expected title 'Advanced Topic' but got 'Introduction'", str(ctx.exception))

    def test_grounded_lesson_requires_citation(self):
        data = valid_lesson(lesson_number=1, title="Introduction", is_grounded=True)
        # Remove citation from all sections
        for section in data["sections"]:
            section["source_ids"] = []
        with self.assertRaises(ValidationError) as ctx:
            validate_lesson(
                data,
                expected_fingerprint=data["grounding"]["document_fingerprint"],
            )
        self.assertIn("a PDF-grounded lesson must cite at least one passage", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
