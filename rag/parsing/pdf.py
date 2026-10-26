"""Text-based PDF parser backed by PyMuPDF (loaded only when used)."""

from __future__ import annotations

from rag.models import ParsedDocument
from rag.parsing.base import DocumentParseError, assemble_document


class PdfParser:
    parser_name = "pymupdf"
    parser_version = "1"
    media_type = "application/pdf"

    def parse(self, data: bytes, *, document_name: str) -> ParsedDocument:
        try:
            import fitz
        except ImportError as exc:
            raise DocumentParseError(
                "PDF support requires PyMuPDF; install the RAG dependencies"
            ) from exc

        try:
            with fitz.open(stream=data, filetype="pdf") as pdf:
                blocks = []
                empty_pages = 0
                for page_number, page in enumerate(pdf, start=1):
                    page_text = page.get_text("text")
                    if not page_text.strip():
                        empty_pages += 1
                    blocks.append((page_text, {"kind": "page", "page": page_number}))
        except Exception as exc:
            raise DocumentParseError("The PDF could not be opened or parsed") from exc

        warnings = (
            (
                f"{empty_pages} page(s) yielded no text; they may be blank or scanned, "
                "and OCR is not implemented.",
            )
            if empty_pages
            else ()
        )
        return assemble_document(
            document_name=document_name,
            media_type=self.media_type,
            parser=self.parser_name,
            parser_version=self.parser_version,
            blocks=blocks,
            warnings=warnings,
        )
