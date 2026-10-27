"""OCR and provenance for an image submitted directly for document review."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import uuid

from rag.models import ContentSegment, ExtractedImage, PageExtraction, ParsedDocument
from rag.parsing.base import DocumentParseError
from rag.parsing.pdf import _ensure_tesseract_path, _env_int


class ImageParser:
    parser_name = "pymupdf-image-ocr"
    parser_version = "1"
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

    def parse(self, data: bytes, *, document_name: str, store_images: bool = False,
              asset_storage_dir: str | Path | None = None) -> ParsedDocument:
        try:
            import fitz
        except ImportError as exc:
            raise DocumentParseError("Image OCR requires PyMuPDF; install the RAG dependencies") from exc

        extension = Path(document_name).suffix.lower()
        filetype = "jpeg" if extension in {".jpg", ".jpeg"} else extension.lstrip(".")
        try:
            image_document = fitz.open(stream=data, filetype=filetype)
            if image_document.page_count != 1:
                image_document.close()
                raise ValueError("image page count")
            page = image_document[0]
            width, height = int(page.rect.width), int(page.rect.height)
            if width < 1 or height < 1 or width * height > 100_000_000:
                image_document.close()
                raise ValueError("image dimensions")
        except Exception as exc:
            raise DocumentParseError("The image could not be opened or is too large") from exc

        warnings: list[str] = []
        text = ""
        dpi = _env_int("TRUELINE_OCR_DPI", 200, minimum=120, maximum=300)
        _ensure_tesseract_path()
        try:
            text_page = page.get_textpage_ocr(language="eng", dpi=dpi, full=True)
            text = (page.get_text("text", textpage=text_page) or "").strip()
        except Exception:
            warnings.append("OCR could not read this image; check that Tesseract and English language data are available.")
        finally:
            image_document.close()

        digest = hashlib.sha256(data).hexdigest()
        storage_ref = None
        if store_images:
            root = Path(asset_storage_dir or os.environ.get(
                "TRUELINE_ASSET_DIR",
                Path(__file__).resolve().parents[2] / "data_ingestion" / "extracted_assets",
            )).resolve()
            directory = root / f"review_{uuid.uuid4().hex}"
            try:
                directory.mkdir(parents=True, exist_ok=False)
                safe_ext = extension.lstrip(".")
                (directory / f"{digest}.{safe_ext}").write_bytes(data)
                storage_ref = str((directory / f"{digest}.{safe_ext}").relative_to(root))
            except OSError:
                shutil.rmtree(directory, ignore_errors=True)
                warnings.append("The submitted image could not be staged for visual review.")

        segment = (ContentSegment(text=text, start=0, end=len(text), kind="image_ocr", page=1),) if text else ()
        return ParsedDocument(
            document_name=document_name,
            media_type=self.media_types[extension],
            text=text,
            segments=segment,
            parser=self.parser_name,
            parser_version=self.parser_version,
            warnings=tuple(warnings),
            pages=(PageExtraction(page=1, method="ocr" if text else "empty", ocr_used=True,
                                  character_count=len(text)),),
            images=(ExtractedImage(
                image_id=f"img_{uuid.uuid4().hex[:12]}", page=1,
                format="jpg" if extension in {".jpg", ".jpeg"} else extension.lstrip("."),
                width=width, height=height, extraction_method="submitted",
                storage_ref=storage_ref, sha256=digest,
            ),),
        )
