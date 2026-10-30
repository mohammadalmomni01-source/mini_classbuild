from classbuild.llm import GroqClient
from classbuild.models.lesson import validate_lesson
from classbuild.services.structured_output import CourseGenerationError, generate_validated

MAX_COMPLETION_TOKENS = 4000  # lessons are longer than outlines

SYSTEM_PROMPT = """You are an experienced teacher writing one lesson of a course.
Reply with ONE JSON object only, with no extra text, in exactly this shape:
{
  "lesson_number": 1,
  "title": "string",
  "introduction": "string",
  "sections": [
    {"heading": "string", "explanation": "string", "example": "string"}
  ],
  "summary": "string",
  "exercises": [
    {"question": "string", "hint": "string"}
  ]
}
Rules: use exactly the lesson number and title you are given, write 2 to 4 sections,
write exactly 2 exercises that each have a hint, give every section a concrete example
(code for programming topics, otherwise a real-world scenario), match the difficulty,
aim for about 400 to 700 words, do not repeat the other lessons, and do not include
a quiz."""


class LessonSelectionError(CourseGenerationError):
    """There is no outline, or the chosen lesson is not in it."""


def find_lesson(outline, lesson_number):
    if outline is None:
        raise LessonSelectionError("Generate a course outline first.")
    for lesson in outline.lessons:
        if lesson.lesson_number == lesson_number:
            return lesson
    available = ", ".join(str(item.lesson_number) for item in outline.lessons)
    raise LessonSelectionError(
        f"Lesson {lesson_number} does not exist. Available lessons: {available}."
    )


def build_lesson_prompt(outline, lesson):
    others = [
        f"{item.lesson_number}. {item.title}"
        for item in outline.lessons
        if item.lesson_number != lesson.lesson_number
    ]
    return (
        f"Course: {outline.title}\n"
        f"Difficulty: {outline.difficulty}\n"
        f"Course objectives: {'; '.join(outline.learning_objectives)}\n\n"
        f"Write lesson {lesson.lesson_number}: {lesson.title}\n"
        f"Lesson description: {lesson.description}\n"
        f"Lesson objectives: {'; '.join(lesson.learning_objectives)}\n\n"
        f"Other lessons (do not repeat them): {'; '.join(others) or 'none'}\n\n"
        f'Use lesson_number {lesson.lesson_number} and the exact title "{lesson.title}".'
    )


class LessonGenerator:
    def __init__(self, client=None):
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = GroqClient()
        return self._client

    def generate_lesson(self, outline, lesson_number):
        lesson = find_lesson(outline, lesson_number)  # fails before any API call
        return generate_validated(
            self.client,
            prompt=build_lesson_prompt(outline, lesson),
            system_prompt=SYSTEM_PROMPT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            validate=lambda data: validate_lesson(
                data,
                expected_lesson_number=lesson.lesson_number,
                expected_title=lesson.title,
            ),
            what="lesson",
        )