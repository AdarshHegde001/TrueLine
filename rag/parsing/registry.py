"""Content-type dispatch for supported submission formats."""

from __future__ import annotations

from pathlib import PurePath

from rag.models import ParsedDocument
from rag.parsing.base import DocumentParseError
from rag.parsing.docx import DocxParser
from rag.parsing.pdf import PdfParser
from rag.parsing.plain_text import PlainTextParser


def parse_document(
    data: bytes,
    *,
    document_name: str,
    content_type: str | None = None,
) -> ParsedDocument:
    """Parse UTF-8 text, text-based PDF, or DOCX based on type/extension."""

    extension = PurePath(document_name).suffix.lower()
    normalized_type = (content_type or "").split(";", maxsplit=1)[0].strip().lower()
    extension_parsers = {
        ".txt": PlainTextParser,
        ".md": PlainTextParser,
        ".pdf": PdfParser,
        ".docx": DocxParser,
    }
    media_type_parsers = {
        "text/plain": PlainTextParser,
        "text/markdown": PlainTextParser,
        "application/pdf": PdfParser,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxParser,
    }
    parser_type = extension_parsers.get(extension)
    media_parser_type = media_type_parsers.get(normalized_type)
    if parser_type and media_parser_type and parser_type is not media_parser_type:
        raise DocumentParseError("Filename extension conflicts with the declared content type")
    parser_type = media_parser_type or parser_type
    if parser_type:
        return parser_type().parse(data, document_name=document_name)
    raise DocumentParseError(f"Unsupported document type: {content_type or extension or 'unknown'}")
