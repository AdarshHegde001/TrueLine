"""Shared parser protocol and helpers."""

from __future__ import annotations

from typing import Protocol

from rag.models import ContentSegment, ParsedDocument


class DocumentParseError(ValueError):
    """Raised when the input is unsupported or cannot be parsed safely."""


class DocumentParser(Protocol):
    parser_name: str
    parser_version: str
    media_type: str

    def parse(self, data: bytes, *, document_name: str) -> ParsedDocument:
        """Parse bytes into canonical text and location-aware segments."""


def assemble_document(
    *,
    document_name: str,
    media_type: str,
    parser: str,
    parser_version: str,
    blocks: list[tuple[str, dict[str, int | str | None]]],
    warnings: tuple[str, ...] = (),
) -> ParsedDocument:
    """Join ordered blocks with newlines and calculate canonical offsets."""

    text_parts: list[str] = []
    segments: list[ContentSegment] = []
    cursor = 0

    for block_text, location in blocks:
        normalized = block_text.strip()
        if not normalized:
            continue
        if text_parts:
            text_parts.append("\n")
            cursor += 1
        start = cursor
        text_parts.append(normalized)
        cursor += len(normalized)
        segments.append(
            ContentSegment(
                text=normalized,
                start=start,
                end=cursor,
                kind=str(location.get("kind") or "text"),
                page=location.get("page"),
                paragraph=location.get("paragraph"),
                table=location.get("table"),
                row=location.get("row"),
            )
        )

    canonical_text = "\n".join(segment.text for segment in segments)
    if not canonical_text:
        raise DocumentParseError("No extractable text was found in the document")

    return ParsedDocument(
        document_name=document_name,
        media_type=media_type,
        text=canonical_text,
        segments=tuple(segments),
        parser=parser,
        parser_version=parser_version,
        warnings=warnings,
    )
