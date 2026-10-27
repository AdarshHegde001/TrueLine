"""Adaptive PDF text/OCR extraction and optional private image preservation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import uuid

from rag.models import ExtractedImage, PageExtraction, ParsedDocument
from rag.parsing.base import DocumentParseError, assemble_document


class PdfParser:
    parser_name = "pymupdf"
    parser_version = "2"
    media_type = "application/pdf"

    def parse(self, data: bytes, *, document_name: str, store_images: bool = False,
              asset_storage_dir: str | Path | None = None) -> ParsedDocument:
        try:
            import fitz
        except ImportError as exc:
            raise DocumentParseError(
                "PDF support requires PyMuPDF; install the RAG dependencies"
            ) from exc

        try:
            pdf = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            raise DocumentParseError("The PDF could not be opened or parsed") from exc

        if pdf.is_encrypted and not pdf.authenticate(""):
            pdf.close()
            raise DocumentParseError("The PDF is encrypted and cannot be opened without a password")

        # Thresholds are conservative and overridable for local/deployment tuning.
        min_chars = _env_int("TRUELINE_OCR_MIN_CHARS", 40, minimum=1, maximum=1000)
        min_image_coverage = _env_float("TRUELINE_OCR_IMAGE_COVERAGE", 0.25, minimum=0.05, maximum=0.95)
        dpi = _env_int("TRUELINE_OCR_DPI", 200, minimum=120, maximum=300)
        warnings: list[str] = []
        blocks: list[tuple[str, dict[str, int | str | None]]] = []
        page_results: list[PageExtraction] = []
        images: list[ExtractedImage] = []
        asset_dir: Path | None = None
        stored_by_hash: dict[str, str] = {}

        if store_images:
            try:
                root = Path(asset_storage_dir or os.environ.get(
                    "TRUELINE_ASSET_DIR",
                    Path(__file__).resolve().parents[2] / "data_ingestion" / "extracted_assets",
                )).resolve()
                asset_dir = root / f"doc_{uuid.uuid4().hex}"
                asset_dir.mkdir(parents=True, exist_ok=False)
            except OSError:
                asset_dir = None
                warnings.append("PDF images could not be saved; extracted text remains available.")

        try:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text") or ""
                normalized = text.strip()
                image_coverage = _image_coverage(page)
                drawing_count, drawing_coverage = _drawing_coverage(page)
                image_refs = page.get_images(full=True)
                needs_ocr = len(normalized) < min_chars and max(image_coverage, drawing_coverage) >= min_image_coverage
                method = "text" if normalized else "empty"
                ocr_used = False

                if needs_ocr:
                    ocr_used = True
                    _ensure_tesseract_path()
                    try:
                        text_page = page.get_textpage_ocr(language="eng", dpi=dpi, full=True)
                        recognized = (page.get_text("text", textpage=text_page) or "").strip()
                        if recognized:
                            native = (page.get_text("text") or "").strip()
                            if native:
                                text = _merge_native_and_ocr(native, recognized)
                                normalized = text.strip()
                                method = "text+ocr"
                            else:
                                text = recognized
                                normalized = recognized
                                method = "ocr"
                        elif normalized:
                            method = "text"
                            warnings.append(f"OCR returned no text for PDF page {page_number}; native text was retained.")
                        else:
                            method = "empty"
                            warnings.append(f"OCR returned no text for PDF page {page_number}.")
                    except Exception:
                        method = "text" if normalized else "empty"
                        warnings.append(
                            f"OCR could not process PDF page {page_number}; "
                            "check that Tesseract and English language data are installed."
                        )

                if not normalized and image_refs:
                    warnings.append(f"PDF page {page_number} contains images but yielded no text.")
                page_results.append(PageExtraction(
                    page=page_number, method=method, ocr_used=ocr_used,
                    character_count=len(normalized),
                ))
                blocks.append((text, {"kind": "page", "page": page_number}))

                # Image recovery is independent from text extraction and is always
                # best-effort. Review parsing deliberately does not persist assets.
                for image_ref in image_refs:
                    xref = int(image_ref[0])
                    try:
                        extracted = pdf.extract_image(xref)
                        payload = extracted["image"]
                        digest = hashlib.sha256(payload).hexdigest()
                        storage_ref = None
                        if asset_dir is not None:
                            storage_ref = stored_by_hash.get(digest)
                            if storage_ref is None:
                                extension = _safe_extension(extracted.get("ext"))
                                asset_path = asset_dir / f"{digest}.{extension}"
                                if not asset_path.exists():
                                    asset_path.write_bytes(payload)
                                storage_ref = str(asset_path.relative_to(asset_dir.parent))
                                stored_by_hash[digest] = storage_ref
                        images.append(ExtractedImage(
                            image_id=f"img_{xref}", page=page_number,
                            format=_safe_extension(extracted.get("ext")),
                            width=int(extracted.get("width") or 0),
                            height=int(extracted.get("height") or 0),
                            storage_ref=storage_ref, sha256=digest,
                        ))
                    except Exception:
                        warnings.append(f"An embedded image on PDF page {page_number} could not be extracted.")

                # Preserve pages whose useful visuals are vector paths (for example,
                # vector logos or charts) only when there is substantial drawing content.
                if store_images and not image_refs and drawing_count >= 4 and drawing_coverage >= 0.08:
                    try:
                        longest_edge = max(float(page.rect.width), float(page.rect.height), 1.0)
                        render_scale = min(1.5, 1800.0 / longest_edge)
                        rendered = page.get_pixmap(matrix=fitz.Matrix(render_scale, render_scale), alpha=False)
                        payload = rendered.tobytes("png")
                        digest = hashlib.sha256(payload).hexdigest()
                        storage_ref = None
                        if asset_dir is not None:
                            storage_ref = stored_by_hash.get(digest)
                            if storage_ref is None:
                                asset_path = asset_dir / f"{digest}.png"
                                if not asset_path.exists():
                                    asset_path.write_bytes(payload)
                                storage_ref = str(asset_path.relative_to(asset_dir.parent))
                                stored_by_hash[digest] = storage_ref
                        images.append(ExtractedImage(
                            image_id=f"page_{page_number}", page=page_number,
                            format="png", width=rendered.width, height=rendered.height,
                            extraction_method="page_render", storage_ref=storage_ref,
                            sha256=digest,
                        ))
                    except Exception:
                        warnings.append(f"PDF page {page_number} visuals could not be rendered for review.")
        except Exception as exc:
            if asset_dir is not None:
                shutil.rmtree(asset_dir, ignore_errors=True)
            raise DocumentParseError("The PDF could not be opened or parsed") from exc
        finally:
            pdf.close()

        if not any(page.character_count for page in page_results):
            if asset_dir is not None:
                shutil.rmtree(asset_dir, ignore_errors=True)
            if any("OCR could not process" in warning for warning in warnings):
                raise DocumentParseError(
                    "No extractable text was found. OCR could not process the scanned page; "
                    "check that Tesseract is installed and available to the TrueLine worker."
                )
            raise DocumentParseError("No extractable text was found in the document")

        try:
            document = assemble_document(
                document_name=document_name,
                media_type=self.media_type,
                parser=self.parser_name,
                parser_version=self.parser_version,
                blocks=blocks,
                warnings=tuple(dict.fromkeys(warnings)),
            )
        except Exception:
            if asset_dir is not None:
                shutil.rmtree(asset_dir, ignore_errors=True)
            raise
        return document.model_copy(update={"pages": tuple(page_results), "images": tuple(images)})


def _image_coverage(page) -> float:
    """Estimate page coverage from placed image rectangles, bounded to [0, 1]."""
    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    areas = []
    try:
        for info in page.get_image_info(xrefs=True):
            x0, y0, x1, y1 = info.get("bbox", (0, 0, 0, 0))
            areas.append(max(0.0, float(x1 - x0)) * max(0.0, float(y1 - y0)))
        if not areas:
            for image in page.get_images(full=True):
                for rect in page.get_image_rects(image[0]):
                    areas.append(max(0.0, float(rect.width)) * max(0.0, float(rect.height)))
    except Exception:
        return 0.0
    # Sum is intentionally capped: overlapping image placements may overcount.
    return min(sum(areas) / page_area, 1.0)


def _drawing_coverage(page) -> tuple[int, float]:
    page_area = max(float(page.rect.width * page.rect.height), 1.0)
    try:
        drawings = page.get_drawings()
        areas = []
        for drawing in drawings:
            rect = drawing.get("rect")
            if rect:
                areas.append(max(0.0, float(rect.width)) * max(0.0, float(rect.height)))
        return len(drawings), min(sum(areas) / page_area, 1.0)
    except Exception:
        return 0, 0.0


def _ensure_tesseract_path() -> bool:
    """Find a standard per-user Windows install even when the parent PATH is stale."""
    if shutil.which("tesseract"):
        return True
    if os.name == "nt":
        roots = [Path(os.environ[name]) for name in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)")
                 if os.environ.get(name)]
        candidates = [root / "Programs" / "Tesseract-OCR" for root in roots[:1]]
        candidates.extend(root / "Tesseract-OCR" for root in roots[1:])
        for directory in candidates:
            executable = directory / "tesseract.exe"
            try:
                if executable.is_file():
                    os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
                    return True
            except OSError:
                # A restricted process may not be allowed to probe user install paths.
                continue
    return False


def _safe_extension(value: object) -> str:
    ext = str(value or "bin").lower()
    return ext if ext.isalnum() and len(ext) <= 8 else "bin"


def _merge_native_and_ocr(native: str, recognized: str) -> str:
    """Keep native lines and add OCR lines that are not already represented."""
    def key(line: str) -> str:
        return " ".join("".join(char.lower() for char in line if char.isalnum()).split())

    native_lines = [line.strip() for line in native.splitlines() if line.strip()]
    seen = {key(line) for line in native_lines if key(line)}
    additions = [
        line.strip() for line in recognized.splitlines()
        if line.strip() and key(line) and key(line) not in seen
    ]
    return "\n".join(native_lines + additions)


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return min(max(value, minimum), maximum)


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return min(max(value, minimum), maximum)
