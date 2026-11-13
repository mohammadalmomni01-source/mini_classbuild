import unittest
from pydantic import ValidationError

from classbuild.models.quiz import validate_quiz
from tests.fixtures import valid_quiz


class QuizModelTests(unittest.TestCase):
    def test_wrong_question_count_rejected(self):
        data = valid_quiz(1, "Lesson Title 1")
        # Remove one question (has 4 instead of 5)
        data["questions"].pop()
        with self.assertRaises(ValidationError) as ctx:
            validate_quiz(data)
        self.assertIn("List should have at least 5 items", str(ctx.exception))

    def test_duplicate_options_rejected(self):
        data = valid_quiz(1, "Lesson Title 1")
        data["questions"][0]["options"][1] = data["questions"][0]["options"][0]
        with self.assertRaises(ValidationError) as ctx:
            validate_quiz(data)
        self.assertIn("options must be different from each other", str(ctx.exception))

    def test_duplicate_questions_rejected(self):
        data = valid_quiz(1, "Lesson Title 1")
        data["questions"][1]["question"] = data["questions"][0]["question"]
        with self.assertRaises(ValidationError) as ctx:
            validate_quiz(data)
        self.assertIn("questions must be different from each other", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
