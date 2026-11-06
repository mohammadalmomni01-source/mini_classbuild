from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class PageText(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    text: str  # cleaned text, empty for image-only pages


class DocumentChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str  # "p3-c2" = second chunk of page 3
    source_filename: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=1)
    text: str

    @property
    def char_count(self):
        return len(self.text)


class ProcessedDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    filename: str
    fingerprint: str
    page_count: int
    pages_with_text: int
    empty_pages: list[int]
    word_count: int
    char_count: int
    chunk_size: int
    chunk_overlap: int
    chunks: list[DocumentChunk]

    @computed_field
    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @model_validator(mode="after")
    def check_consistency(self):
        if self.page_count != self.pages_with_text + len(self.empty_pages):
            raise ValueError("page counts do not add up")
        ids = [chunk.chunk_id for chunk in self.chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("chunk ids must be unique")
        return self

    def get_chunk(self, chunk_id):
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None