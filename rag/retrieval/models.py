"""Shared request/response models for the ingestion retrieval boundary."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.models import TextSpan

DocumentType = Literal[
    "brand_guidelines", "tone_of_voice", "approved_claims", "product_facts",
    "glossary", "asset_rules", "legal_compliance", "other_reference",
]


class KnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    brand_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    source_types: tuple[str, ...] = ("guideline", "approved_fact")
    document_types: tuple[DocumentType, ...] = ()
    top_k: int = Field(default=8, ge=1, le=20)

    @model_validator(mode="after")
    def query_is_not_whitespace(self) -> "KnowledgeSearchRequest":
        if not self.query.strip():
            raise ValueError("query cannot contain only whitespace")
        return self


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    brand_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    document_type: DocumentType = "other_reference"
    title: str
    version_number: int = Field(ge=1)
    chunk_id: str = Field(min_length=1)
    chunk_text: str = Field(min_length=1)
    source_text: str | None = None
    location: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str | None = None
    rank: int = Field(ge=1)
    score: float | None = None


class KnowledgeSearchResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    brand_id: str = Field(min_length=1)
    retrieval_method: Literal["hybrid", "vector", "keyword"]
    results: tuple[RetrievedChunk, ...] = ()

    @model_validator(mode="after")
    def results_are_brand_scoped(self) -> "KnowledgeSearchResponse":
        mismatched = [item.source_id for item in self.results if item.brand_id != self.brand_id]
        if mismatched:
            raise ValueError("retrieval response contains results for another brand")
        return self


class EvidenceBundle(BaseModel):
    """Retrieved source candidates linked to one exact submitted passage."""

    model_config = ConfigDict(frozen=True)

    query: str
    passage_span: TextSpan
    passage_quote: str
    retrieval_method: Literal["hybrid", "vector", "keyword"]
    sources: tuple[RetrievedChunk, ...] = ()
