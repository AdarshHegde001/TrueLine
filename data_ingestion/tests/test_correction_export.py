"""Document-format patching keeps the uploaded DOCX structure intact."""

from io import BytesIO

from docx import Document

from rag.agentic.correction import _build_docx
from rag.parsing.docx import DocxParser


def _source_docx():
    source = Document()
    source.add_heading("Campaign", level=1)
    source.add_paragraph("The draft says the benefit doubles growth.")
    table = source.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "Keep this table"
    output = BytesIO()
    source.save(output)
    return output.getvalue()


def test_targeted_docx_patch_preserves_headings_and_tables():
    data = _source_docx()
    parsed = DocxParser().parse(data, document_name="draft.docx")
    segment_index = next(
        index for index, segment in enumerate(parsed.segments)
        if segment.text == "The draft says the benefit doubles growth."
    )

    corrected_data, skipped = _build_docx(parsed, data, [{
        "segment_index": segment_index,
        "exact_quote": "doubles growth",
        "corrected_text": "may improve growth",
    }])

    corrected = Document(BytesIO(corrected_data))
    assert skipped == []
    assert corrected.paragraphs[0].style.name.startswith("Heading")
    assert corrected.paragraphs[1].text == "The draft says the benefit may improve growth."
    assert len(corrected.tables) == 1
    assert corrected.tables[0].cell(0, 0).text == "Keep this table"


def test_targeted_docx_patch_skips_a_quote_not_in_the_summary_target():
    data = _source_docx()
    parsed = DocxParser().parse(data, document_name="draft.docx")
    segment_index = next(
        index for index, segment in enumerate(parsed.segments)
        if segment.text == "The draft says the benefit doubles growth."
    )

    corrected_data, skipped = _build_docx(parsed, data, [{
        "segment_index": segment_index,
        "exact_quote": "unflagged phrase",
        "corrected_text": "new wording",
    }])

    corrected = Document(BytesIO(corrected_data))
    assert corrected.paragraphs[1].text == "The draft says the benefit doubles growth."
    assert len(skipped) == 1
