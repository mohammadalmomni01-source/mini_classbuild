import re

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

from classbuild.models.course import NonEmptyStr

QUESTION_COUNT = 5
OPTION_COUNT = 4


def _normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: NonEmptyStr
    options: list[NonEmptyStr] = Field(min_length=OPTION_COUNT, max_length=OPTION_COUNT)
    correct_answer: StrictInt = Field(ge=0, le=OPTION_COUNT - 1)
    explanation: NonEmptyStr

    @field_validator("options")
    @classmethod
    def options_must_be_distinct(cls, options):
        normalized = [_normalize(option) for option in options]
        if len(set(normalized)) != len(normalized):
            raise ValueError("options must be different from each other")
        return options


class Quiz(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lesson_number: int = Field(ge=1)
    lesson_title: NonEmptyStr
    questions: list[Question] = Field(min_length=QUESTION_COUNT, max_length=QUESTION_COUNT)

    @model_validator(mode="after")
    def questions_must_be_distinct(self):
        normalized = [_normalize(item.question) for item in self.questions]
        if len(set(normalized)) != len(normalized):
            raise ValueError("questions must be different from each other")
        return self

    @model_validator(mode="after")
    def check_matches_selected_lesson(self, info: ValidationInfo):
        context = info.context or {}
        number = context.get("expected_lesson_number")
        title = context.get("expected_title")
        if number is not None and self.lesson_number != number:
            raise ValueError(f"expected lesson {number} but got lesson {self.lesson_number}")
        if title is not None and _normalize(self.lesson_title) != _normalize(title):
            raise ValueError(f"expected title '{title}' but got '{self.lesson_title}'")
        return self


def validate_quiz(data, expected_lesson_number=None, expected_title=None):
    return Quiz.model_validate(
        data,
        context={
            "expected_lesson_number": expected_lesson_number,
            "expected_title": expected_title,
        },
    )