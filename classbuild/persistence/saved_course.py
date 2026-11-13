"""
classbuild/persistence/saved_course.py
Defines the saved-course file format using existing Pydantic models.
"""
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from classbuild.models.course import CourseOutline
from classbuild.models.lesson import LessonContent
from classbuild.models.quiz import Quiz

SCHEMA_VERSION = 1
APP_NAME = "Mini ClassBuild"
COURSE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def new_course_id() -> str:
    """A random 32-hex-character storage id, never derived from the title."""
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class SourceDocument(BaseModel):
    """The PDF a course's grounded lessons came from: filename and fingerprint only."""
    model_config = ConfigDict(extra="forbid")

    filename: str
    fingerprint: str


class SavedCourse(BaseModel):
    """
    The on-disk format for a saved course.
    Contains no API key, chat history, submitted answers, PDF content or embeddings.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    app: str = APP_NAME
    course_id: str
    created_at: datetime
    updated_at: datetime
    topic: str
    difficulty: str
    outline: CourseOutline
    lessons: list[LessonContent] = Field(default_factory=list)
    quizzes: list[Quiz] = Field(default_factory=list)
    source_document: Optional[SourceDocument] = None
    selected_lesson: Optional[int] = None

    @model_validator(mode="after")
    def check_consistency(self):
        outline_numbers = {lesson.lesson_number for lesson in self.outline.lessons}
        lesson_numbers = {lesson.lesson_number for lesson in self.lessons}
        quiz_numbers = {quiz.lesson_number for quiz in self.quizzes}

        # 1. Lessons must match the outline
        unknown_lessons = lesson_numbers - outline_numbers
        if unknown_lessons:
            raise ValueError(
                f"Saved lessons {sorted(unknown_lessons)} are not in the outline."
            )

        # 2. A quiz needs its lesson to be generated
        quizzes_without_lessons = quiz_numbers - lesson_numbers
        if quizzes_without_lessons:
            raise ValueError(
                f"Quizzes for lessons {sorted(quizzes_without_lessons)} have no saved lesson."
            )

        # 3. No duplicate lesson numbers
        lesson_list = [lesson.lesson_number for lesson in self.lessons]
        if len(set(lesson_list)) != len(lesson_list):
            raise ValueError("Duplicate lesson numbers found in saved lessons.")

        # 4. No duplicate quiz lesson numbers
        quiz_list = [quiz.lesson_number for quiz in self.quizzes]
        if len(set(quiz_list)) != len(quiz_list):
            raise ValueError("Duplicate lesson numbers found in saved quizzes.")

        # 5. Every grounded lesson must come from source_document
        for lesson in self.lessons:
            if lesson.is_grounded:
                if self.source_document is None:
                    raise ValueError(
                        f"Lesson {lesson.lesson_number} is grounded but no source_document is set."
                    )
                if lesson.grounding.document_fingerprint != self.source_document.fingerprint:
                    raise ValueError(
                        f"Lesson {lesson.lesson_number} grounding fingerprint does not match "
                        f"source_document fingerprint."
                    )

        # 6. selected_lesson must exist in the outline (if set)
        if self.selected_lesson is not None and self.selected_lesson not in outline_numbers:
            raise ValueError(
                f"selected_lesson {self.selected_lesson} is not in the outline."
            )

        # 7. course_id must be 32 hex characters
        if not COURSE_ID_PATTERN.match(self.course_id):
            raise ValueError(
                f"course_id must be 32 hexadecimal characters, got {self.course_id!r}."
            )

        return self

    def lessons_by_number(self) -> dict:
        """The dict the app keeps in session state: {lesson_number: LessonContent}."""
        return {lesson.lesson_number: lesson for lesson in self.lessons}

    def quizzes_by_number(self) -> dict:
        """The dict the app keeps in session state: {lesson_number: Quiz}."""
        return {quiz.lesson_number: quiz for quiz in self.quizzes}

    def content_hash(self) -> str:
        """
        A stable hash of the content (without timestamps).
        The app compares this to show 'unsaved changes'.
        """
        data = self.model_dump(
            exclude={"created_at", "updated_at"},
            mode="json",
        )
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _derive_source_document(lessons: list[LessonContent]) -> Optional[SourceDocument]:
    """Extract SourceDocument from grounded lessons (all must share the same PDF)."""
    for lesson in lessons:
        if lesson.is_grounded:
            return SourceDocument(
                filename=lesson.grounding.source_filename,
                fingerprint=lesson.grounding.document_fingerprint,
            )
    return None


def build_saved_course(
    *,
    course_id: str,
    topic: str,
    outline: CourseOutline,
    lessons: dict,
    quizzes: dict,
    created_at: Optional[datetime] = None,
    selected_lesson: Optional[int] = None,
) -> SavedCourse:
    """
    Build a validated SavedCourse from session-state data.
    source_document is derived automatically from grounded lessons.
    """
    now = datetime.now(timezone.utc)
    lesson_list = list(lessons.values())
    quiz_list = list(quizzes.values())
    source_doc = _derive_source_document(lesson_list)

    return SavedCourse(
        course_id=course_id,
        created_at=created_at or now,
        updated_at=now,
        topic=topic,
        difficulty=outline.difficulty,
        outline=outline,
        lessons=lesson_list,
        quizzes=quiz_list,
        source_document=source_doc,
        selected_lesson=selected_lesson,
    )
