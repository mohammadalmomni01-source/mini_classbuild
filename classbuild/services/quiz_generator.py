from classbuild.llm import GroqClient
from classbuild.models.quiz import validate_quiz
from classbuild.services.lesson_generator import find_lesson
from classbuild.services.structured_output import CourseGenerationError, generate_validated

MAX_COMPLETION_TOKENS = 3000

SYSTEM_PROMPT = """You write multiple-choice quizzes for one lesson of a course.
Reply with ONE JSON object only, with no extra text, in exactly this shape:
{
  "lesson_number": 1,
  "lesson_title": "string",
  "questions": [
    {
      "question": "string",
      "options": ["option A", "option B", "option C", "option D"],
      "correct_answer": 0,
      "explanation": "string"
    }
  ]
}
Rules: write exactly 5 different questions, give each question exactly 4 distinct
options, make exactly one option correct, set correct_answer to the index (0, 1, 2
or 3) of the correct option, explain why it is correct, ask only about concepts taught
in the lesson text, mix understanding and application, avoid trick questions, and
spread the correct answers across different positions. Use exactly the lesson number
and title you are given."""


class QuizPrerequisiteError(CourseGenerationError):
    """The lesson content needed for the quiz is missing or belongs to another lesson."""


def build_quiz_prompt(outline, lesson, content):
    sections = "\n\n".join(
        f"{section.heading}\n{section.explanation}\nExample:\n{section.example}"
        for section in content.sections
    )
    return (
        f"Course: {outline.title}\n"
        f"Difficulty: {outline.difficulty}\n"
        f"Lesson {lesson.lesson_number}: {lesson.title}\n"
        f"Lesson objectives: {'; '.join(lesson.learning_objectives)}\n\n"
        f"Lesson text:\n{content.introduction}\n\n{sections}\n\n"
        f"Summary:\n{content.summary}\n\n"
        f'Write 5 questions. Use lesson_number {lesson.lesson_number} '
        f'and lesson_title "{lesson.title}".'
    )


class QuizGenerator:
    def __init__(self, client=None):
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = GroqClient()
        return self._client

    def generate_quiz(self, outline, lesson_number, content):
        lesson = find_lesson(outline, lesson_number)  # fails before any API call
        if content is None or content.lesson_number != lesson.lesson_number:
            raise QuizPrerequisiteError(
                "Generate this lesson's content before creating its quiz."
            )
        return generate_validated(
            self.client,
            prompt=build_quiz_prompt(outline, lesson, content),
            system_prompt=SYSTEM_PROMPT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            validate=lambda data: validate_quiz(
                data,
                expected_lesson_number=lesson.lesson_number,
                expected_title=lesson.title,
            ),
            what="quiz",
        )