"""UTF-8 plain-text parser."""

from __future__ import annotations

from rag.models import ContentSegment, ParsedDocument
from rag.parsing.base import DocumentParseError


class PlainTextParser:
    parser_name = "utf8-text"
    parser_version = "1"
    media_type = "text/plain"

    def parse(self, data: bytes, *, document_name: str) -> ParsedDocument:
        try:
            text = data.decode("utf-8-sig", errors="strict")
        except UnicodeDecodeError as exc:
            raise DocumentParseError("Text file is not valid UTF-8") from exc

        if not text.strip():
            raise DocumentParseError("No extractable text was found in the document")

        segments: list[ContentSegment] = []
        offset = 0
        for paragraph_index, line in enumerate(text.splitlines(keepends=True)):
            line_text = line.rstrip("\r\n")
            if line_text.strip():
                segments.append(
                    ContentSegment(
                        text=line_text,
                        start=offset,
                        end=offset + len(line_text),
                        kind="paragraph",
                        paragraph=paragraph_index,
                    )
                )
            offset += len(line)

        return ParsedDocument(
            document_name=document_name,
            media_type=self.media_type,
            text=text,
            segments=tuple(segments),
            parser=self.parser_name,
            parser_version=self.parser_version,
        )
