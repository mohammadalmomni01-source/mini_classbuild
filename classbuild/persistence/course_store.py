"""
classbuild/persistence/course_store.py
Reads and writes saved courses as course-<id>.json files in one folder, safely.
Standard library only (plus the saved_course module).
"""
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from classbuild.persistence.saved_course import COURSE_ID_PATTERN, SCHEMA_VERSION, SavedCourse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SAVE_DIR = "saved_courses"
SAVE_DIR_ENV = "MINICLASSBUILD_SAVE_DIR"
MAX_COURSE_FILE_BYTES = 2 * 1024 * 1024  # 2 MB

_FILE_PREFIX = "course-"
_FILE_SUFFIX = ".json"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CourseStorageError(Exception):
    """Base class for all persistence errors."""


class InvalidCourseIdError(CourseStorageError):
    """The course id is not a 32-hex-character string."""


class CourseNotFoundError(CourseStorageError):
    """No file exists for this course id."""


class CourseFileTooLargeError(CourseStorageError):
    """The saved file exceeds the size limit."""


class CorruptedCourseFileError(CourseStorageError):
    """The file cannot be parsed or validated."""


class UnsupportedSchemaVersionError(CourseStorageError):
    """The file was written by a newer version of this app."""


# ---------------------------------------------------------------------------
# Summary / listing
# ---------------------------------------------------------------------------


@dataclass
class SavedCourseSummary:
    """One line in the saved-courses list."""
    course_id: str
    title: str
    topic: str
    difficulty: str
    lesson_count: int
    generated_lesson_count: int
    quiz_count: int
    has_grounded_lessons: bool
    updated_at: datetime


@dataclass
class CourseListing:
    """The complete listing plus a note for every unreadable file."""
    courses: list[SavedCourseSummary]
    errors: list[str]  # one human-readable message per damaged file


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class CourseStore:
    """
    The local JSON file store.
    Ids must be 32 hex characters; paths are always verified to stay inside
    the store directory, so values like ../../secret are rejected.
    """

    def __init__(self, directory: Optional[str] = None,
                 max_bytes: int = MAX_COURSE_FILE_BYTES):
        raw = (
            directory
            or os.environ.get(SAVE_DIR_ENV)
            or DEFAULT_SAVE_DIR
        )
        self._dir = Path(raw)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_bytes

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def path_for(self, course_id: str) -> Path:
        """
        The only place a file name is built.
        Validates the id and resolves the path; raises InvalidCourseIdError
        if the id is bad or the resolved path escapes the store directory.
        """
        if not COURSE_ID_PATTERN.match(str(course_id)):
            raise InvalidCourseIdError(
                f"Course id must be 32 hex characters; got {course_id!r}."
            )
        name = f"{_FILE_PREFIX}{course_id}{_FILE_SUFFIX}"
        resolved = (self._dir / name).resolve()
        if not str(resolved).startswith(str(self._dir.resolve())):
            raise InvalidCourseIdError(
                f"Resolved path {resolved} escapes the store directory."
            )
        return resolved

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, course: SavedCourse) -> Path:
        """
        Validate, enforce the size limit, and write atomically.
        Returns the path that was written.
        """
        path = self.path_for(course.course_id)
        content = course.model_dump_json(indent=2)
        encoded = content.encode("utf-8")
        if len(encoded) > self._max_bytes:
            raise CourseFileTooLargeError(
                f"The saved course is {len(encoded):,} bytes, "
                f"which exceeds the {self._max_bytes:,}-byte limit."
            )
        # Atomic write: write to a temp file in the same directory, then replace.
        fd, tmp_path = tempfile.mkstemp(
            dir=self._dir, prefix=".tmp-course-", suffix=".json"
        )
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(encoded)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        return path

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load(self, course_id: str) -> SavedCourse:
        """
        Load and validate a course.
        Pipeline: size check → JSON → schema version → Pydantic validation → id match.
        """
        path = self.path_for(course_id)
        if not path.exists():
            raise CourseNotFoundError(f"No saved course found for id {course_id!r}.")

        raw = path.read_bytes()
        if len(raw) > self._max_bytes:
            raise CourseFileTooLargeError(
                f"The saved file is {len(raw):,} bytes, "
                f"which exceeds the {self._max_bytes:,}-byte limit."
            )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CorruptedCourseFileError(
                f"The saved file for {course_id!r} is not valid JSON: {exc}"
            ) from exc

        # Schema version check before Pydantic (future-proofing)
        file_version = data.get("schema_version")
        if file_version is None or not isinstance(file_version, int):
            raise CorruptedCourseFileError(
                f"The saved file for {course_id!r} has no valid schema_version."
            )
        if file_version > SCHEMA_VERSION:
            raise UnsupportedSchemaVersionError(
                f"The saved file uses schema version {file_version}, "
                f"but this app only supports version {SCHEMA_VERSION}."
            )

        try:
            course = SavedCourse.model_validate(data)
        except Exception as exc:
            raise CorruptedCourseFileError(
                f"The saved file for {course_id!r} failed validation: {exc}"
            ) from exc

        # File name must match the id inside
        if course.course_id != course_id:
            raise CorruptedCourseFileError(
                f"The file name says {course_id!r} but the content says {course.course_id!r}."
            )

        return course

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_courses(self) -> CourseListing:
        """
        Summaries of all readable files, newest first.
        Temporary and unrelated files are ignored.
        """
        summaries: list[SavedCourseSummary] = []
        errors: list[str] = []

        for entry in self._dir.iterdir():
            name = entry.name
            # Only process course-<32hex>.json files; ignore temps and unrelated files
            if not (
                name.startswith(_FILE_PREFIX)
                and name.endswith(_FILE_SUFFIX)
                and not name.startswith(".tmp-")
            ):
                continue

            course_id = name[len(_FILE_PREFIX):-len(_FILE_SUFFIX)]
            if not COURSE_ID_PATTERN.match(course_id):
                continue  # unrelated file

            try:
                course = self.load(course_id)
            except CourseStorageError as exc:
                errors.append(f"{name}: {exc}")
                continue

            summaries.append(
                SavedCourseSummary(
                    course_id=course.course_id,
                    title=course.outline.title,
                    topic=course.topic,
                    difficulty=course.difficulty,
                    lesson_count=len(course.outline.lessons),
                    generated_lesson_count=len(course.lessons),
                    quiz_count=len(course.quizzes),
                    has_grounded_lessons=course.source_document is not None,
                    updated_at=course.updated_at,
                )
            )

        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return CourseListing(courses=summaries, errors=errors)
