from dataclasses import dataclass

from classbuild.llm import GroqClient
from classbuild.models.chat import ChatMessage
from classbuild.services.lesson_generator import find_lesson
from classbuild.services.structured_output import CourseGenerationError

MAX_QUESTION_CHARS = 1000
MAX_HISTORY_TURNS = 6
MAX_HISTORY_MESSAGE_CHARS = 1500
MAX_COMPLETION_TOKENS = 2000

LESSON_SYSTEM_PROMPT = """You are a friendly tutor helping a learner with ONE lesson of a course.
Answer using the lesson below and the conversation so far. Keep answers short and clear.
Use simpler words when asked, and give new examples when asked. If a question goes beyond
the lesson, say so briefly. Never claim the lesson says something it does not."""


class ChatInputError(CourseGenerationError):
    """The question is empty or too long. Nothing is sent to Groq."""


class ChatAnswerError(CourseGenerationError):
    """The lesson is missing, or an answer failed the checks."""


@dataclass(frozen=True)
class LessonChatContext:
    course_title: str
    difficulty: str
    lesson: object  # the lesson's entry in the outline
    content: object  # the generated lesson

    @classmethod
    def from_course(cls, outline, lesson_number, content):
        lesson = find_lesson(outline, lesson_number)
        if content is None or content.lesson_number != lesson.lesson_number:
            raise ChatAnswerError("Generate this lesson before chatting about it.")
        return cls(outline.title, outline.difficulty, lesson, content)


@dataclass(frozen=True)
class ChatReply:
    question: ChatMessage
    answer: ChatMessage
    passages: list


def validate_question(question):
    question = (question or "").strip()
    if not question:
        raise ChatInputError("Please type a question first.")
    if len(question) > MAX_QUESTION_CHARS:
        raise ChatInputError(
            f"Your question is too long. Please keep it under {MAX_QUESTION_CHARS} characters."
        )
    return question


def recent_history(history, max_turns=MAX_HISTORY_TURNS):
    """The last few exchanges as Groq messages, starting with a question."""
    messages = list(history)[-2 * max_turns:]
    while messages and messages[0].role != "user":
        messages.pop(0)
    return [
        {"role": message.role, "content": message.content[:MAX_HISTORY_MESSAGE_CHARS]}
        for message in messages
    ]


def lesson_context_text(ctx):
    content = ctx.content
    sections = "\n\n".join(
        f"## {section.heading}\n{section.explanation}\nExample:\n{section.example}"
        for section in content.sections
    )
    exercises = "\n".join(
        f"{number}. {exercise.question} (hint: {exercise.hint})"
        for number, exercise in enumerate(content.exercises, start=1)
    )
    return (
        f"Course: {ctx.course_title} ({ctx.difficulty})\n"
        f"Lesson {content.lesson_number}: {content.title}\n"
        f"Lesson objectives: {'; '.join(ctx.lesson.learning_objectives)}\n\n"
        f"{content.introduction}\n\n{sections}\n\n"
        f"Summary: {content.summary}\n\nExercises:\n{exercises}"
    )


class LessonChatService:
    def __init__(self, client=None, embeddings=None):
        self._client = client
        self.embeddings = embeddings

    @property
    def client(self):
        if self._client is None:
            self._client = GroqClient()
        return self._client

    def answer(self, ctx, question, history=(), document=None, index=None):
        question = validate_question(question)  # fails before any API call
        if document is not None:
            raise ChatAnswerError("Answering from the PDF is added in the next part.")
        return self._answer_from_lesson(ctx, question, history)

    def _answer_from_lesson(self, ctx, question, history):
        # The whole lesson goes with every question; only recent turns are kept.
        system = LESSON_SYSTEM_PROMPT + "\n\nLESSON:\n" + lesson_context_text(ctx)
        messages = [{"role": "system", "content": system}]
        messages += recent_history(history)
        messages.append({"role": "user", "content": question})

        result = self.client.chat(messages, max_completion_tokens=MAX_COMPLETION_TOKENS)
        return ChatReply(
            question=ChatMessage(role="user", content=question, mode="lesson"),
            answer=ChatMessage(
                role="assistant",
                content=result.text,
                mode="lesson",
                truncated=result.finish_reason == "length",
            ),
            passages=[],
        )