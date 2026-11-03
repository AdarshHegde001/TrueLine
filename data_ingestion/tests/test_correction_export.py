"""DOCX correction export tests use in-memory fixtures and need no database."""
from io import BytesIO

from docx import Document

from rag.agentic.correction import _tracked_replace, apply_approved_corrections
from rag.parsing.docx import DocxParser


def test_tracked_replace_marks_original_and_replacement():
    doc = Document()
    paragraph = doc.add_paragraph()
    paragraph.add_run("Intro ")
    target = paragraph.add_run("old claim")
    target.bold = True
    paragraph.add_run(" ending")

    assert _tracked_replace(paragraph, "old claim", "revised claim", 1)
    xml = paragraph._p.xml
    assert "<w:del" in xml and "<w:ins" in xml
    assert "old claim" in xml and "revised claim" in xml
    assert "<w:b" in xml  # the original formatting is carried to the tracked runs


def test_tracked_replace_skips_a_quote_spanning_runs():
    doc = Document()
    paragraph = doc.add_paragraph()
    paragraph.add_run("across ")
    paragraph.add_run("runs")
    assert not _tracked_replace(paragraph, "across runs", "replacement", 1)


def test_approved_correction_retains_original_docx_structure():
    source = Document()
    source.add_heading("Campaign", level=1)
    source.add_paragraph("The draft says the benefit doubles growth.")
    table = source.add_table(rows=1, cols=1)
    table.cell(0, 0).text = "Keep this table"
    original = BytesIO()
    source.save(original)
    data = original.getvalue()
    parsed = DocxParser().parse(
        data, document_name="draft.docx",
    )
    paragraph_segment = next(
        index for index, segment in enumerate(parsed.segments)
        if segment.text == "The draft says the benefit doubles growth."
    )
    result = apply_approved_corrections(
        parsed, data,
        [{"edit_id": "edit-1", "segment_index": paragraph_segment,
          "exact_quote": "doubles growth", "corrected_text": "may improve growth"}],
        ["edit-1"],
    )
    corrected = Document(BytesIO(__import__("base64").b64decode(result["fileBase64"])))
    assert result["editsApplied"] == 1
    assert len(corrected.tables) == 1
    assert corrected.tables[0].cell(0, 0).text == "Keep this table"
    assert corrected.paragraphs[0].style.name.startswith("Heading")
    assert "<w:del" in corrected.paragraphs[1]._p.xml
    assert "<w:ins" in corrected.paragraphs[1]._p.xml


def test_unapproved_correction_is_not_applied():
    source = Document()
    source.add_paragraph("Leave this sentence alone.")
    data_buffer = BytesIO()
    source.save(data_buffer)
    data = data_buffer.getvalue()
    parsed = DocxParser().parse(data, document_name="draft.docx")
    result = apply_approved_corrections(
        parsed, data,
        [{"edit_id": "edit-1", "segment_index": 0,
          "exact_quote": "Leave this", "corrected_text": "Change this"}],
        [],
    )
    corrected = Document(BytesIO(__import__("base64").b64decode(result["fileBase64"])))
    assert result["editsApplied"] == 0
    assert "<w:del" not in corrected.paragraphs[0]._p.xml
