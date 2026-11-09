import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from classbuild.models.course import NonEmptyStr
from classbuild.models.lesson import SourceReference

CITATION_MARKER = re.compile(r"\[(p\d+-c\d+)\]")  # inline markers like [p3-c1]


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["user", "assistant"]
    content: NonEmptyStr
    mode: Literal["lesson", "pdf"] = "lesson"
    status: Literal["answered", "insufficient_evidence"] = "answered"
    sources: list[SourceReference] = Field(default_factory=list)
    document_fingerprint: str | None = None
    truncated: bool = False

    @property
    def cited_pages(self):
        return sorted({source.page_number for source in self.sources})

    @model_validator(mode="after")
    def check_sources(self):
        is_pdf_answer = self.role == "assistant" and self.mode == "pdf"
        if not is_pdf_answer:
            if self.sources or self.document_fingerprint:
                raise ValueError("only PDF answers may carry sources")
            return self

        if not self.document_fingerprint:
            raise ValueError("a PDF answer must record which PDF it used")
        if self.status == "insufficient_evidence":
            if self.sources:
                raise ValueError("an answer without evidence must not cite anything")
            return self

        ids = [source.chunk_id for source in self.sources]
        if not ids:
            raise ValueError("an answered PDF reply must cite at least one passage")
        if len(set(ids)) != len(ids):
            raise ValueError("each source must be listed once")
        stray = sorted(set(CITATION_MARKER.findall(self.content)) - set(ids))
        if stray:
            raise ValueError(f"the answer cites passages it does not list: {', '.join(stray)}")
        return self