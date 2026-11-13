"""
scripts/evaluation_scenarios.py
The list of evaluation scenarios for Mini ClassBuild Stage 3.
Each scenario names a behaviour and points to the existing unit tests that prove it.
This file contains data only; no test logic is duplicated here.
"""
from dataclasses import dataclass, field

# Prefix for all test ids so they can be loaded by unittest
T = "tests."


@dataclass(frozen=True)
class Scenario:
    """A behaviour the project must have, linked to the tests that prove it."""
    number: int
    name: str
    category: str
    checks: str               # one sentence: what this scenario verifies
    tests: list[str]          # full test ids  (module.Class.method)


SCENARIOS: list[Scenario] = [
    # ── Outlines ───────────────────────────────────────────────────────────
    Scenario(
        1, "Valid outline passes", "Outlines",
        "A well-formed outline with the right lesson count and difficulty is accepted.",
        [T + "test_course_models.CourseOutlineTests.test_valid_outline_passes"],
    ),
    Scenario(
        2, "Wrong lesson count rejected", "Outlines",
        "An outline with fewer or more lessons than requested is rejected.",
        [T + "test_course_models.CourseOutlineTests.test_wrong_lesson_count_rejected"],
    ),
    Scenario(
        3, "Wrong difficulty rejected", "Outlines",
        "An outline whose difficulty does not match the request is rejected.",
        [T + "test_course_models.CourseOutlineTests.test_wrong_difficulty_rejected"],
    ),
    Scenario(
        4, "Lesson numbers must be sequential", "Outlines",
        "Lessons numbered out of order or with gaps are rejected.",
        [T + "test_course_models.CourseOutlineTests.test_nonsequential_lesson_numbers_rejected"],
    ),
    Scenario(
        5, "Duplicate lesson titles rejected", "Outlines",
        "An outline that repeats a lesson title (ignoring case/spacing) is rejected.",
        [T + "test_course_models.CourseOutlineTests.test_duplicate_lesson_titles_rejected"],
    ),
    # ── Structured output ──────────────────────────────────────────────────
    Scenario(
        6, "Invalid JSON triggers retry", "Structured output",
        "When the model returns invalid JSON, one retry is made before failing.",
        [T + "test_structured_output.StructuredOutputTests.test_invalid_json_triggers_retry"],
    ),
    Scenario(
        7, "Valid JSON on retry is accepted", "Structured output",
        "If the retry returns valid JSON that passes validation, it is accepted.",
        [T + "test_structured_output.StructuredOutputTests.test_valid_json_on_retry_accepted"],
    ),
    # ── Lessons ────────────────────────────────────────────────────────────
    Scenario(
        8, "Lesson number and title match the outline", "Lessons",
        "A lesson whose number or title differs from the outline is rejected.",
        [T + "test_lesson_models.LessonContentTests.test_wrong_lesson_number_rejected",
         T + "test_lesson_models.LessonContentTests.test_wrong_title_rejected"],
    ),
    Scenario(
        9, "Grounded lesson cites at least one source", "Lessons",
        "A PDF-based lesson must cite at least one passage; uncited lessons are rejected.",
        [T + "test_lesson_models.LessonContentTests.test_grounded_lesson_requires_citation"],
    ),
    Scenario(
        10, "No uncited page mentions in grounded lessons", "Lessons",
        "A page number mentioned in lesson text must appear among its cited pages.",
        [T + "test_reliability.ReliabilityTests.test_uncited_page_mention_rejected"],
    ),
    # ── Quizzes ────────────────────────────────────────────────────────────
    Scenario(
        11, "Quiz has the correct number of questions", "Quizzes",
        "A quiz with too few or too many questions is rejected.",
        [T + "test_quiz.QuizModelTests.test_wrong_question_count_rejected"],
    ),
    Scenario(
        12, "Duplicate quiz options rejected", "Quizzes",
        "A question whose options are not all distinct is rejected.",
        [T + "test_quiz.QuizModelTests.test_duplicate_options_rejected"],
    ),
    Scenario(
        13, "Duplicate quiz questions rejected", "Quizzes",
        "A quiz that repeats the same question text is rejected.",
        [T + "test_quiz.QuizModelTests.test_duplicate_questions_rejected"],
    ),
    # ── RAG / citations ────────────────────────────────────────────────────
    Scenario(
        14, "Low-similarity passages trigger InsufficientEvidenceError", "RAG citations",
        "When all retrieved passages are below the similarity threshold, no Groq call is made.",
        [T + "test_grounded_lesson.GroundedLessonTests.test_low_similarity_raises_error"],
    ),
    Scenario(
        15, "Unknown source ids rejected", "RAG citations",
        "A draft that cites an id not in the allowed list is rejected.",
        [T + "test_grounded_lesson.GroundedLessonTests.test_unknown_source_id_rejected"],
    ),
    Scenario(
        16, "Source data comes from retrieval, not the AI", "RAG citations",
        "Page numbers and text in sources are taken from retrieval results, never from the model.",
        [T + "test_grounded_lesson.GroundedLessonTests.test_sources_from_retrieval"],
    ),
    # ── Chatbot ────────────────────────────────────────────────────────────
    Scenario(
        17, "Empty question is rejected before any API call", "Chatbot",
        "A blank or whitespace-only question raises ChatInputError without calling Groq.",
        [T + "test_lesson_chat.LessonChatTests.test_empty_question_rejected"],
    ),
    Scenario(
        18, "Question over the length limit is rejected", "Chatbot",
        "A question exceeding MAX_QUESTION_CHARS is rejected before any API call.",
        [T + "test_lesson_chat.LessonChatTests.test_long_question_rejected"],
    ),
    # ── Errors and retries ─────────────────────────────────────────────────
    Scenario(
        19, "Temporary errors are retried within the budget", "Errors and retries",
        "A 503 response is retried at most max_retries times; the total stays within budget.",
        [T + "test_reliability.ReliabilityTests.test_temporary_error_retried_once"],
    ),
    Scenario(
        20, "400/401/403/404 errors are never retried", "Errors and retries",
        "Client errors cause immediate failure; no retry is attempted.",
        [T + "test_reliability.ReliabilityTests.test_client_errors_not_retried"],
    ),
    Scenario(
        21, "API key is never shown in error messages", "Errors and retries",
        "Any error message containing a Groq key is redacted before display.",
        [T + "test_reliability.ReliabilityTests.test_api_key_redacted_from_errors"],
    ),
    Scenario(
        22, "Failed request preserves previous course", "Errors and retries",
        "When a new outline request fails, the previous course remains unchanged on screen.",
        [T + "test_app.AppTests.test_failed_request_preserves_previous_course"],
    ),
    # ── State ──────────────────────────────────────────────────────────────
    Scenario(
        23, "Stale grounded lessons are removed when PDF changes", "State",
        "If the uploaded PDF changes, PDF-based lessons for the previous PDF are removed.",
        [T + "test_app.AppTests.test_stale_grounded_lessons_removed"],
    ),
    # ── Saving and loading ─────────────────────────────────────────────────
    Scenario(
        24, "Course saves and loads with identical content", "Saving and loading",
        "A saved course can be loaded in a new session with the same outline, lessons and quizzes.",
        [T + "test_persistence.PersistenceTests.test_save_and_load_identical_content"],
    ),
    Scenario(
        25, "Damaged files are reported without breaking the app", "Saving and loading",
        "Corrupted or partial JSON files appear in the error list; they do not crash the app.",
        [T + "test_persistence.PersistenceTests.test_damaged_file_reported"],
    ),
]
