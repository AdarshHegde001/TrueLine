"""Focused PDF parser regression tests; OCR itself is stubbed where appropriate."""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")

from rag.parsing.pdf import PdfParser, _merge_native_and_ocr


def _pdf_with_page(configure):
    document = fitz.open()
    page = document.new_page()
    configure(page)
    data = document.tobytes()
    document.close()
    return data


def test_digital_pdf_uses_native_text_without_ocr(monkeypatch):
    data = _pdf_with_page(lambda page: page.insert_text(
        (72, 72), "This sufficiently long digital PDF text should not need OCR."
    ))

    def unexpected_ocr(*_args, **_kwargs):
        raise AssertionError("OCR must not run on a text-rich page")

    monkeypatch.setattr(fitz.Page, "get_textpage_ocr", unexpected_ocr)
    parsed = PdfParser().parse(data, document_name="digital.pdf")

    assert "sufficiently long digital PDF text" in parsed.text
    assert parsed.pages[0].page == 1
    assert parsed.pages[0].method == "text"
    assert parsed.pages[0].ocr_used is False


def test_image_heavy_sparse_page_reports_ocr_failure_and_keeps_native_text(monkeypatch):
    def configure(page):
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), 0)
        pixmap.clear_with(255)
        page.insert_image(page.rect, stream=pixmap.tobytes("png"))
        page.insert_text((72, 72), "Title")

    data = _pdf_with_page(configure)

    def failed_ocr(*_args, **_kwargs):
        raise RuntimeError("simulated OCR failure")

    monkeypatch.setattr(fitz.Page, "get_textpage_ocr", failed_ocr)
    parsed = PdfParser().parse(data, document_name="mixed.pdf")

    assert "Title" in parsed.text
    assert parsed.pages[0].ocr_used is True
    assert parsed.pages[0].method == "text"
    assert any("OCR could not process PDF page 1" in warning for warning in parsed.warnings)


def test_reference_pdf_image_is_saved_with_page_and_hash_metadata(tmp_path, monkeypatch):
    def configure(page):
        pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 20, 20), 0)
        pixmap.clear_with(255)
        page.insert_image(fitz.Rect(50, 50, 150, 150), stream=pixmap.tobytes("png"))
        page.insert_text((72, 200), "A useful digital caption for this embedded image.")

    data = _pdf_with_page(configure)
    monkeypatch.setenv("TRUELINE_ASSET_DIR", str(tmp_path))
    parsed = PdfParser().parse(data, document_name="reference.pdf", store_images=True)

    assert parsed.images
    asset = parsed.images[0]
    assert asset.page == 1 and asset.sha256
    assert asset.storage_ref
    assert (tmp_path / asset.storage_ref).is_file()


def test_vector_visual_page_is_saved_as_page_render(tmp_path):
    def configure(page):
        page.insert_text((72, 55), "A digital page with enough text to avoid OCR during extraction.")
        for rect in (
            fitz.Rect(50, 120, 230, 260), fitz.Rect(260, 120, 440, 260),
            fitz.Rect(50, 300, 230, 440), fitz.Rect(260, 300, 440, 440),
        ):
            page.draw_rect(rect, color=(0, 0, 1), width=2)

    parsed = PdfParser().parse(
        _pdf_with_page(configure), document_name="vector-chart.pdf",
        store_images=True, asset_storage_dir=tmp_path,
    )
    render = next(image for image in parsed.images if image.extraction_method == "page_render")
    assert render.page == 1 and render.format == "png"
    assert (tmp_path / render.storage_ref).is_file()


def test_native_and_ocr_lines_are_deduplicated():
    merged = _merge_native_and_ocr("Approved claims\nUse the product name", "Approved claims\nUse the product name\nOCR-only line")
    assert merged.splitlines() == ["Approved claims", "Use the product name", "OCR-only line"]


def test_reference_chunk_location_keeps_all_overlapping_pdf_pages():
    from types import SimpleNamespace

    from trueline_ingestion.ingest import _insert_version

    text = "Rules from the first page.\nRules from the second page."
    split = text.index("Rules from the second")

    class FakeConnection:
        def __init__(self):
            self.statements = []

        def execute(self, statement, parameters):
            self.statements.append((statement, parameters))

    settings = SimpleNamespace(chunk_max_chars=500, chunking_version="test")
    embedder = SimpleNamespace(tag="fake")
    conn = FakeConnection()
    _insert_version(
        conn, embedder, settings, source_id="src_test", brand_id="brand_test",
        number=1, text=text, document_id=None, document_name="guide.pdf",
        metadata={"page_spans": [
            {"page": 1, "start": 0, "end": split - 1},
            {"page": 2, "start": split, "end": len(text)},
        ]}, document_type="brand_guidelines", rule_data=[], approved=True, title="Guide",
    )

    from psycopg.types.json import Jsonb

    chunk_insert = next(item for item in conn.statements if "INSERT INTO knowledge_chunks" in item[0])
    location = next(value.obj for value in chunk_insert[1] if isinstance(value, Jsonb))
    assert location["pages"] == [1, 2]


def test_tesseract_is_discovered_from_user_install_when_parent_path_is_stale(tmp_path, monkeypatch):
    import os
    import shutil

    if os.name != "nt":
        pytest.skip("Windows user-install discovery")
    install_dir = tmp_path / "LocalAppData" / "Programs" / "Tesseract-OCR"
    install_dir.mkdir(parents=True)
    (install_dir / "tesseract.exe").write_bytes(b"stub")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(shutil, "which", lambda _command: None)

    from rag.parsing.pdf import _ensure_tesseract_path

    assert _ensure_tesseract_path() is True
    assert str(install_dir) in os.environ["PATH"]


def test_submitted_image_is_staged_for_visual_review_without_approving_it(tmp_path, monkeypatch):
    from rag.parsing.image import ImageParser

    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 32, 24), 0)
    pixmap.clear_with(255)
    data = pixmap.tobytes("png")
    monkeypatch.setattr("rag.parsing.image._ensure_tesseract_path", lambda: False)
    monkeypatch.setattr(fitz.Page, "get_textpage_ocr", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError()))

    parsed = ImageParser().parse(
        data, document_name="submitted.png", store_images=True, asset_storage_dir=tmp_path,
    )

    assert parsed.text == ""
    assert parsed.pages[0].ocr_used
    assert parsed.images[0].extraction_method == "submitted"
    assert (tmp_path / parsed.images[0].storage_ref).is_file()
    assert parsed.warnings


def test_approved_logo_storage_is_session_scoped_and_keeps_private_file(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    from trueline_ingestion.logos import add_logo

    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 32, 24), 0)
    pixmap.clear_with(255)
    data = pixmap.tobytes("png")

    class FakeConnection:
        def execute(self, statement, parameters):
            if "SELECT logo_id" in statement and "sha256=%s" in statement:
                return FakeResult(None)
            if "SELECT count(*)" in statement:
                return FakeResult({"n": 0})
            if "INSERT INTO knowledge_logos" in statement:
                logo_id, _brand, title, name, mime, _ref, _digest, _visual_hash, _ocr_text, width, height = parameters
                return FakeResult({
                    "logo_id": logo_id, "title": title, "original_name": name,
                    "content_type": mime, "width": width, "height": height,
                    "created_at": datetime.now(timezone.utc),
                })
            raise AssertionError(statement)

    class FakeResult:
        def __init__(self, row):
            self.row = row

        def fetchone(self):
            return self.row

    monkeypatch.setenv("TRUELINE_ASSET_DIR", str(tmp_path))
    logo = add_logo(
        FakeConnection(), brand_id="session_brand", title="Primary logo",
        filename="primary.png", content_type="image/png", data=data,
    )
    assert logo["title"] == "Primary logo"
    assert not any(key in logo for key in ("storage_ref", "sha256"))
    assert list(tmp_path.glob("logo_*/*"))
