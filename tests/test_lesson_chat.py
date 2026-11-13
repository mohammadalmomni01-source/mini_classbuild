import unittest

from classbuild.models.course import CourseOutline
from classbuild.models.lesson import LessonContent
from classbuild.services.lesson_chat import (
    MAX_QUESTION_CHARS,
    ChatInputError,
    LessonChatContext,
    LessonChatService,
    validate_question,
)
from tests.fixtures import valid_lesson, valid_outline


class LessonChatTests(unittest.TestCase):
    def setUp(self):
        self.outline = CourseOutline.model_validate(valid_outline(3))
        self.lesson = LessonContent.model_validate(valid_lesson(1, self.outline.lessons[0].title))
        self.ctx = LessonChatContext.from_course(self.outline, 1, self.lesson)
        self.service = LessonChatService()

    def test_empty_question_rejected(self):
        with self.assertRaises(ChatInputError):
            validate_question("")
        with self.assertRaises(ChatInputError):
            validate_question("   \n\t  ")
        with self.assertRaises(ChatInputError):
            self.service.answer(self.ctx, "")

    def test_long_question_rejected(self):
        long_q = "a" * (MAX_QUESTION_CHARS + 1)
        with self.assertRaises(ChatInputError) as ctx:
            validate_question(long_q)
        self.assertIn("too long", str(ctx.exception))
        with self.assertRaises(ChatInputError):
            self.service.answer(self.ctx, long_q)


if __name__ == "__main__":
    unittest.main()
