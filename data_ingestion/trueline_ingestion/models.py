"""Pydantic models = the contract shapes. RAG code depends on these, not on SQL."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SourceType = Literal["guideline", "approved_fact"]
DocumentType = Literal[
    "brand_guidelines", "tone_of_voice", "approved_claims", "product_facts",
    "glossary", "asset_rules", "legal_compliance", "other_reference",
]
DEFAULT_DOCUMENT_TYPE: DocumentType = "other_reference"
DEFAULT_SOURCE_TYPES: list[str] = ["guideline", "approved_fact"]


class KnowledgeSearchRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    brand_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    source_types: tuple[SourceType, ...] = tuple(DEFAULT_SOURCE_TYPES)
    document_types: tuple[DocumentType, ...] = ()
    top_k: int = Field(default=8, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v

    @field_validator("source_types")
    @classmethod
    def _non_empty(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if not v:
            raise ValueError("source_types must not be empty")
        return v


class RetrievedChunk(BaseModel):
    """Mapping-compatible representation of rag.retrieval.models.RetrievedChunk."""

    model_config = ConfigDict(frozen=True)

    brand_id: str
    source_id: str
    version_id: str
    source_type: SourceType
    document_type: DocumentType = DEFAULT_DOCUMENT_TYPE
    title: str
    version_number: int
    chunk_id: str
    chunk_text: str
    source_text: Optional[str] = None
    location: dict[str, Any] = Field(default_factory=dict)
    embedding_model: Optional[str] = None
    rank: int
    score: Optional[float] = None


class KnowledgeSearchResponse(BaseModel):
    """Mapping-compatible representation of the canonical RAG response."""

    model_config = ConfigDict(frozen=True)

    brand_id: str
    retrieval_method: Literal["hybrid", "vector", "keyword"]
    results: tuple[RetrievedChunk, ...] = ()


# ---- ingestion-side models ----

class SourceCreate(BaseModel):
    brand_id: str = Field(min_length=1)
    source_type: SourceType
    document_type: DocumentType = DEFAULT_DOCUMENT_TYPE
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    rule_data: list[dict[str, Any]] = Field(default_factory=list)
    extracted_rules: list[dict[str, Any]] = Field(default_factory=list)
    approved: bool = True  # admin confirmation; False => stays pending, not searchable

    @field_validator("title", "text")
    @classmethod
    def _strip_check(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be blank")
        return v


class SourceUpdate(BaseModel):
    brand_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    document_type: Optional[DocumentType] = None
    text: str = Field(min_length=1)
    title: Optional[str] = None
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    rule_data: Optional[list[dict[str, Any]]] = None
    extracted_rules: Optional[list[dict[str, Any]]] = None
    approved: bool = True


class SourceVersionRecord(BaseModel):
    """Logical knowledge record (contract section 'Knowledge record shape')."""
    source_id: str
    brand_id: str
    source_type: SourceType
    document_type: DocumentType = DEFAULT_DOCUMENT_TYPE
    title: str
    status: str
    approval_status: str
    version_id: str
    version: int
    text: str
    rule_data: list[dict[str, Any]] = Field(default_factory=list)
    extracted_rules: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
