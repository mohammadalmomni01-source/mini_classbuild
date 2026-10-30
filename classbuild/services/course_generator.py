from classbuild.llm import GroqClient
from classbuild.models.course import validate_outline
from classbuild.services.structured_output import generate_validated

MAX_COMPLETION_TOKENS = 3000

SYSTEM_PROMPT = """You are an instructional designer.
Reply with ONE JSON object only, with no extra text, in exactly this shape:
{
  "title": "string",
  "description": "string",
  "difficulty": "Beginner | Intermediate | Advanced",
  "learning_objectives": ["3 to 5 strings"],
  "lessons": [
    {
      "lesson_number": 1,
      "title": "string",
      "description": "a one or two sentence summary",
      "learning_objectives": ["exactly 2 or 3 strings, never more than 3"]
    }
  ]
}
Rules: stay on the topic, match the requested difficulty, order lessons from
introductory to advanced, never repeat a lesson topic, make objectives practical,
and number lessons 1, 2, 3 ... in order. Each lesson has 2 or 3 objectives,
never 4 or more."""


def build_prompt(config):
    return (
        f"Topic: {config.topic}\n"
        f"Difficulty: {config.difficulty}\n"
        f"Create a course with exactly {config.lesson_count} lessons, "
        f"numbered 1 to {config.lesson_count}."
    )


class CourseGenerator:
    def __init__(self, client=None):
        self._client = client

    @property
    def client(self):
        if self._client is None:
            self._client = GroqClient()
        return self._client

    def generate_outline(self, config):
        return generate_validated(
            self.client,
            prompt=build_prompt(config),
            system_prompt=SYSTEM_PROMPT,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            validate=lambda data: validate_outline(
                data,
                expected_lesson_count=config.lesson_count,
                expected_difficulty=config.difficulty,
            ),
            what="course outline",
        )