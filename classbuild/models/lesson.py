import re

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from classbuild.models.course import NonEmptyStr


def _normalize(text):
    return re.sub(r"\s+", " ", text).strip().lower()


class SourceReference(BaseModel):
    """A cited PDF passage. Filled from retrieval results, never from the AI."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: NonEmptyStr
    source_filename: NonEmptyStr
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=1)
    similarity: float
    text: NonEmptyStr


class LessonGrounding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_fingerprint: NonEmptyStr
    source_filename: NonEmptyStr
    retrieval_query: NonEmptyStr
    sources: list[SourceReference] = Field(min_length=1)


class LessonSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: NonEmptyStr
    explanation: NonEmptyStr
    example: NonEmptyStr
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("source_ids")
    @classmethod
    def no_duplicate_ids(cls, ids):
        if len(set(ids)) != len(ids):
            raise ValueError("source ids must not repeat within a section")
        return ids


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
    grounding: LessonGrounding | None = None  # None for ordinary lessons

    @property
    def is_grounded(self):
        return self.grounding is not None

    @property
    def cited_pages(self):
        if self.grounding is None:
            return []
        return sorted({source.page_number for source in self.grounding.sources})

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

    @model_validator(mode="after")
    def check_citations(self, info: ValidationInfo):
        context = info.context or {}
        cited = [source_id for section in self.sections for source_id in section.source_ids]

        if self.grounding is None:
            if cited:
                raise ValueError("a lesson without a PDF must not contain citations")
            return self

        sources = self.grounding.sources
        listed = [source.chunk_id for source in sources]
        if len(set(listed)) != len(listed):
            raise ValueError("the same source is listed twice")
        if not cited:
            raise ValueError("a PDF-grounded lesson must cite at least one passage")

        unknown = sorted(set(cited) - set(listed))
        if unknown:
            raise ValueError(f"cited ids that are not listed as sources: {', '.join(unknown)}")
        unused = sorted(set(listed) - set(cited))
        if unused:
            raise ValueError(f"listed sources that no section cites: {', '.join(unused)}")

        fingerprint = context.get("expected_fingerprint")
        if fingerprint is not None and self.grounding.document_fingerprint != fingerprint:
            raise ValueError("the sources do not come from the current PDF")

        allowed = context.get("allowed_sources")
        if allowed is not None:
            for source in sources:
                if allowed.get(source.chunk_id) != source:
                    raise ValueError(
                        f"source {source.chunk_id} does not match the retrieved passage"
                    )
        return self


def validate_lesson(
    data,
    expected_lesson_number=None,
    expected_title=None,
    allowed_sources=None,
    expected_fingerprint=None,
):
    return LessonContent.model_validate(
        data,
        context={
            "expected_lesson_number": expected_lesson_number,
            "expected_title": expected_title,
            "allowed_sources": allowed_sources,
            "expected_fingerprint": expected_fingerprint,
        },
    )