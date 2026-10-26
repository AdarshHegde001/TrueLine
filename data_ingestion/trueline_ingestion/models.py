"""Pydantic models = the contract shapes. RAG code depends on these, not on SQL."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

SourceType = Literal["guideline", "approved_fact"]
DEFAULT_SOURCE_TYPES: list[str] = ["guideline", "approved_fact"]


class KnowledgeSearchRequest(BaseModel):
    brand_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    source_types: list[SourceType] = Field(default_factory=lambda: list(DEFAULT_SOURCE_TYPES))
    top_k: int = Field(default=8, ge=1, le=20)
    filters: Optional[dict[str, Any]] = None  # only {"status": "active"} is accepted

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v

    @field_validator("source_types")
    @classmethod
    def _non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("source_types must not be empty")
        return v

    @field_validator("filters")
    @classmethod
    def _only_active(cls, v: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        if v and (set(v) - {"status"} or v.get("status", "active") != "active"):
            raise ValueError("only filters={'status': 'active'} is supported")
        return v


class RetrievalInfo(BaseModel):
    method: Literal["hybrid", "vector", "keyword"]
    rank: int
    # Retrieval signal only. NOT confidence and NOT proof a source supports a finding.
    score: Optional[float] = None


class KnowledgeSearchResult(BaseModel):
    source_id: str
    version_id: str
    brand_id: str
    source_type: SourceType
    title: str
    version: int
    chunk_id: str
    chunk_text: str
    text: str  # authoritative full source text for this version
    location: dict[str, Any] = Field(default_factory=dict)
    embedding_model: str
    retrieval: RetrievalInfo


class KnowledgeSearchResponse(BaseModel):
    brand_id: str
    retrieval_method: Literal["hybrid", "vector", "keyword"]
    results: list[KnowledgeSearchResult]


# ---- ingestion-side models ----

class SourceCreate(BaseModel):
    brand_id: str = Field(min_length=1)
    source_type: SourceType
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
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
    text: str = Field(min_length=1)
    title: Optional[str] = None
    document_id: Optional[str] = None
    document_name: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    approved: bool = True


class SourceVersionRecord(BaseModel):
    """Logical knowledge record (contract section 'Knowledge record shape')."""
    source_id: str
    brand_id: str
    source_type: SourceType
    title: str
    status: str
    approval_status: str
    version_id: str
    version: int
    text: str
    metadata: dict[str, Any]
    created_at: str
    updated_at: str
