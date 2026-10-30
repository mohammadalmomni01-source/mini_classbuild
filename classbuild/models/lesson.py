import re

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from classbuild.models.course import NonEmptyStr


def _normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


class LessonSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: NonEmptyStr
    explanation: NonEmptyStr
    example: NonEmptyStr


class Exercise(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: NonEmptyStr
    hint: NonEmptyStr


class LessonContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lesson_number: int = Field(ge=1)
    title: NonEmptyStr
    introduction: NonEmptyStr
    sections: list[LessonSection] = Field(min_length=2, max_length=4)
    summary: NonEmptyStr
    exercises: list[Exercise] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def check_matches_selected_lesson(self, info: ValidationInfo):
        context = info.context or {}
        number = context.get("expected_lesson_number")
        title = context.get("expected_title")
        if number is not None and self.lesson_number != number:
            raise ValueError(f"expected lesson {number} but got lesson {self.lesson_number}")
        if title is not None and _normalize(self.title) != _normalize(title):
            raise ValueError(f"expected title '{title}' but got '{self.title}'")
        return self


def validate_lesson(data, expected_lesson_number=None, expected_title=None):
    return LessonContent.model_validate(
        data,
        context={
            "expected_lesson_number": expected_lesson_number,
            "expected_title": expected_title,
        },
    )