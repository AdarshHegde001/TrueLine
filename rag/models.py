"""Shared typed data models for parsing and deterministic review."""

from __future__ import annotations

from enum import Enum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FindingLabel(str, Enum):
    VIOLATION = "violation"
    PASSED = "passed"
    UNVERIFIED = "unverified"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TextSpan(BaseModel):
    """Half-open character range into ``ParsedDocument.text`` (0-based)."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end: int = Field(ge=0)

    @model_validator(mode="after")
    def end_is_not_before_start(self) -> "TextSpan":
        if self.end < self.start:
            raise ValueError("span end must be greater than or equal to start")
        return self


class ContentSegment(BaseModel):
    """A source-ordered block whose offsets point into canonical parsed text."""

    model_config = ConfigDict(frozen=True)

    text: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    kind: str = "text"
    page: int | None = Field(default=None, ge=1)
    paragraph: int | None = Field(default=None, ge=0)
    table: int | None = Field(default=None, ge=0)
    row: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def offsets_match_text(self) -> "ContentSegment":
        if self.end - self.start != len(self.text):
            raise ValueError("segment offsets must cover exactly the segment text")
        return self


class ParsedDocument(BaseModel):
    """Extracted text and location metadata; offsets refer to ``text``."""

    model_config = ConfigDict(frozen=True)

    document_name: str
    media_type: str
    text: str
    segments: tuple[ContentSegment, ...]
    parser: str
    parser_version: str
    offset_unit: Literal["unicode_code_points"] = "unicode_code_points"
    warnings: tuple[str, ...] = ()

    @model_validator(mode="after")
    def segments_match_document(self) -> "ParsedDocument":
        previous_end = 0
        for segment in self.segments:
            if segment.end > len(self.text):
                raise ValueError("segment extends beyond parsed document text")
            if self.text[segment.start : segment.end] != segment.text:
                raise ValueError("segment text does not match parsed document offsets")
            if segment.start < previous_end:
                raise ValueError("segments must be in source order and cannot overlap")
            previous_end = segment.end
        return self


class SourceReference(BaseModel):
    """Approved source identity attached to a deterministic check result."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(min_length=1)
    version_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    title: str | None = None


class Finding(BaseModel):
    """A deterministic finding; AI-produced findings will use a later schema."""

    model_config = ConfigDict(frozen=True)

    finding_id: str = Field(default_factory=lambda: f"f_{uuid4().hex}")
    detector: str
    rule_id: str = Field(min_length=1)
    label: FindingLabel
    severity: Severity | None = None
    span: TextSpan | None = None
    quote: str | None = None
    source: SourceReference
    explanation: str
    replacement: str | None = None
    validation_status: str = "structurally_validated"

    @model_validator(mode="after")
    def quote_requires_span(self) -> "Finding":
        if (self.quote is None) != (self.span is None):
            raise ValueError("quote and span must either both be set or both be absent")
        if self.label == FindingLabel.VIOLATION and self.source is None:
            raise ValueError("violation findings require an approved source")
        return self


class RuleCheckResult(BaseModel):
    """Output from deterministic rule checks; no model calls are involved."""

    model_config = ConfigDict(frozen=True)

    findings: tuple[Finding, ...] = ()
    checks_run: tuple[str, ...] = ()
    skipped_checks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
