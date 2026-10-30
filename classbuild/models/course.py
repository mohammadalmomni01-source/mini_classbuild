from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationInfo,
    model_validator,
)

Difficulty = Literal["Beginner", "Intermediate", "Advanced"]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class LessonOutline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lesson_number: int = Field(ge=1)
    title: NonEmptyStr
    description: NonEmptyStr  # the short summary
    learning_objectives: list[NonEmptyStr] = Field(min_length=2, max_length=3)


class CourseOutline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: NonEmptyStr
    description: NonEmptyStr
    difficulty: Difficulty
    learning_objectives: list[NonEmptyStr] = Field(min_length=3, max_length=5)
    lessons: list[LessonOutline] = Field(min_length=1)

    @model_validator(mode="after")
    def check_lesson_numbers(self):
        numbers = [lesson.lesson_number for lesson in self.lessons]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("lesson numbers must be 1, 2, 3 ... in order")
        return self

    @model_validator(mode="after")
    def check_matches_request(self, info: ValidationInfo):
        context = info.context or {}
        count = context.get("expected_lesson_count")
        difficulty = context.get("expected_difficulty")
        if count is not None and len(self.lessons) != count:
            raise ValueError(f"expected {count} lessons but got {len(self.lessons)}")
        if difficulty is not None and self.difficulty != difficulty:
            raise ValueError(f"expected difficulty {difficulty} but got {self.difficulty}")
        return self


def validate_outline(data, expected_lesson_count=None, expected_difficulty=None):
    return CourseOutline.model_validate(
        data,
        context={
            "expected_lesson_count": expected_lesson_count,
            "expected_difficulty": expected_difficulty,
        },
    )