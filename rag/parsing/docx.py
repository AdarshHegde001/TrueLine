"""DOCX parser backed by python-docx, retaining body order and table rows."""

from __future__ import annotations

from rag.models import ParsedDocument
from rag.parsing.base import DocumentParseError, assemble_document


class DocxParser:
    parser_name = "python-docx"
    parser_version = "1"
    media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    def parse(self, data: bytes, *, document_name: str) -> ParsedDocument:
        try:
            from docx import Document
            from docx.oxml.table import CT_Tbl
            from docx.oxml.text.paragraph import CT_P
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise DocumentParseError(
                "DOCX support requires python-docx; install the RAG dependencies"
            ) from exc

        from io import BytesIO

        try:
            document = Document(BytesIO(data))
            blocks: list[tuple[str, dict[str, int | str | None]]] = []
            paragraph_index = 0
            table_index = 0
            for child in document.element.body.iterchildren():
                if isinstance(child, CT_P):
                    paragraph = Paragraph(child, document)
                    kind = "heading" if paragraph.style.name.startswith("Heading") else "paragraph"
                    blocks.append(
                        (paragraph.text, {"kind": kind, "paragraph": paragraph_index})
                    )
                    paragraph_index += 1
                elif isinstance(child, CT_Tbl):
                    table = Table(child, document)
                    for row_index, row in enumerate(table.rows):
                        cell_text = "\t".join(cell.text for cell in row.cells)
                        blocks.append(
                            (
                                cell_text,
                                {
                                    "kind": "table_row",
                                    "table": table_index,
                                    "row": row_index,
                                },
                            )
                        )
                    table_index += 1
        except Exception as exc:
            raise DocumentParseError("The DOCX file could not be opened or parsed") from exc

        return assemble_document(
            document_name=document_name,
            media_type=self.media_type,
            parser=self.parser_name,
            parser_version=self.parser_version,
            blocks=blocks,
        )
