"""Document parsing adapters that preserve canonical text offsets."""

from rag.parsing.base import DocumentParseError, DocumentParser
from rag.parsing.registry import parse_document

__all__ = ["DocumentParseError", "DocumentParser", "parse_document"]
