"""Line-oriented worker that connects Express to the existing Python RAG stack."""

from __future__ import annotations

import contextlib
import base64
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data_ingestion"))


def emit(value: dict) -> None:
    # The Node bridge consumes a UTF-8 JSON-lines protocol. Escaping non-ASCII
    # also makes each response safe if a platform pipe ignores the encoding hint.
    sys.stdout.write(json.dumps(value, ensure_ascii=True, default=str) + "\n")
    sys.stdout.flush()


def main() -> None:
    # Do not inherit a host-specific console code page for the worker protocol.
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        # Keep library/model diagnostics away from the JSON-lines protocol.
        with contextlib.redirect_stdout(sys.stderr):
            from rag.parsing.registry import parse_document
            from rag.agentic.workflow import model_configuration, run_agentic_review
            from rag.agentic.rulebook import extract_reference_rules
            from rag.agentic.correction import generate_corrected_docx
            from rag.review import MAX_REVIEW_TEXT_CHARS, ReviewDocumentTooLargeError, prepare_review
            from rag.retrieval.client import IngestionKnowledgeAdapter
            from trueline_ingestion import db, ingest
            from trueline_ingestion import logos
            from trueline_ingestion.config import get_settings
            from trueline_ingestion.embeddings import BGEM3Embedder, FakeEmbedder
            from trueline_ingestion.models import SourceCreate
            from trueline_ingestion.search import KnowledgeService
            from trueline_ingestion import sessions

            settings = get_settings()
            db_connection = db.connect(settings)
            try:
                db.migrate(db_connection)
            finally:
                db_connection.close()

            embedder_mode = os.environ.get("INGESTION_EMBEDDER", "bge-m3").lower()
            embedder = FakeEmbedder() if embedder_mode == "fake" else BGEM3Embedder(settings)
            knowledge = KnowledgeService(embedder, settings)

        emit({"type": "ready", "embedder": embedder_mode, "agentModels": model_configuration()})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                operation = request.get("operation")
                if operation == "ingest":
                    result = _ingest(request, parse_document, extract_reference_rules, ingest, SourceCreate,
                                     db, settings, embedder, sessions)
                elif operation == "review":
                    result = _review(request, parse_document, prepare_review,
                                     IngestionKnowledgeAdapter, knowledge, db, ingest, settings,
                                     run_agentic_review, sessions, logos,
                                     MAX_REVIEW_TEXT_CHARS, ReviewDocumentTooLargeError)
                elif operation == "correct":
                    result = _correct(request, parse_document, generate_corrected_docx,
                                      sessions, db, settings)
                elif operation == "start_session":
                    with db.connect(settings) as conn:
                        result = sessions.start_session(conn, request["organizationId"])
                elif operation == "resume_session":
                    with db.connect(settings) as conn:
                        result = sessions.get_session(
                            conn, request["sessionId"], request["organizationId"]
                        )
                elif operation == "end_session":
                    with db.connect(settings) as conn:
                        result = sessions.end_session(
                            conn, request["sessionId"], request["organizationId"]
                        )
                elif operation == "list_knowledge":
                    brand_id = _session_brand_id(request, db, settings, sessions)
                    with db.connect(settings) as conn:
                        result = sessions.list_sources(conn, brand_id)
                elif operation == "list_logos":
                    brand_id = _session_brand_id(request, db, settings, sessions)
                    with db.connect(settings) as conn:
                        result = logos.list_logos(conn, brand_id)
                elif operation == "upload_logo":
                    brand_id = _session_brand_id(request, db, settings, sessions)
                    logo_data = base64.b64decode(request["fileBase64"], validate=True)
                    with db.connect(settings) as conn:
                        result = logos.add_logo(
                            conn, brand_id=brand_id, title=request.get("title", ""),
                            filename=request["fileName"], content_type=request["contentType"],
                            data=logo_data,
                        )
                elif operation == "delete_logo":
                    brand_id = _session_brand_id(request, db, settings, sessions)
                    with db.connect(settings) as conn:
                        result = logos.delete_logo(conn, brand_id, request["logoId"])
                elif operation == "get_logo_image":
                    brand_id = _session_brand_id(request, db, settings, sessions)
                    with db.connect(settings) as conn:
                        logo_record, image_data = logos.read_logo(conn, brand_id, request["logoId"])
                    result = {
                        "contentType": logo_record["content_type"],
                        "fileBase64": base64.b64encode(image_data).decode("ascii"),
                    }
                else:
                    raise ValueError(f"Unsupported worker operation: {operation}")
                emit({"id": request.get("id"), "ok": True, "result": result})
            except Exception as exc:  # return a bounded, user-safe error per request
                emit({"id": request.get("id"), "ok": False, "error": str(exc),
                      "status": getattr(exc, "http_status", 500)})
    except Exception as exc:
        emit({"type": "fatal", "error": str(exc)})
        traceback.print_exc(file=sys.stderr)
        raise


def _session_brand_id(request, db, settings, sessions):
    with db.connect(settings) as conn:
        session = sessions.get_session(
            conn, request.get("sessionId", ""), request["organizationId"]
        )
    return session["brandId"]


def _ingest(request, parse_document, extract_reference_rules, ingest, SourceCreate, db, settings, embedder, sessions):
    brand_id = _session_brand_id(request, db, settings, sessions)
    data = base64.b64decode(request["fileBase64"], validate=True)
    document = parse_document(data, document_name=request["fileName"],
                              content_type=request.get("contentType"), store_images=True)
    document_type = request["documentType"]
    source_type = "approved_fact" if document_type in {"approved_claims", "product_facts"} else "guideline"
    approved = bool(request.get("approved", True))
    extraction = {"status": "not_run", "rules": [], "notes": ""}
    if approved:
        try:
            extraction = extract_reference_rules(document)
        except Exception as exc:
            extraction = {"status": "failed", "rules": [],
                          "notes": f"Rule extraction failed ({type(exc).__name__})."}
    record = SourceCreate(
        brand_id=brand_id,
        source_type=source_type,
        document_type=document_type,
        title=request["title"],
        text=document.text,
        document_name=document.document_name,
        metadata={
            "parser": document.parser,
            "parser_version": document.parser_version,
            "offset_unit": document.offset_unit,
            "parse_warnings": list(document.warnings),
            "page_extraction": [page.model_dump() for page in document.pages],
            "images": [image.model_dump() for image in document.images],
            "page_spans": [
                {"page": segment.page, "start": segment.start, "end": segment.end}
                for segment in document.segments if segment.page is not None
            ],
            "document_type": document_type,
            "rule_extraction_status": extraction["status"],
            "rule_extraction_notes": extraction.get("notes", ""),
        },
        rule_data=request.get("ruleData") or [],
        extracted_rules=extraction["rules"],
        approved=approved,
    )
    with db.connect(settings) as conn:
        result = ingest.create_source(conn, embedder, record, settings)
    return {
        "source_id": result.source_id,
        "version_id": result.version_id,
        "version": result.version,
        "title": result.title,
        "source_type": result.source_type,
        "document_type": document_type,
        "status": result.status,
        "approval_status": result.approval_status,
        "chunk_count": len(ingest.chunk_text(document.text, settings.chunk_max_chars)),
        "rule_count": len(result.rule_data),
        "extracted_rule_count": len(result.extracted_rules),
        "rule_extraction_status": extraction["status"],
        "parse_warnings": list(document.warnings),
        "pages_processed": len(document.pages),
        "ocr_pages": sum(page.ocr_used for page in document.pages),
        "image_count": len(document.images),
    }


def _review(request, parse_document, prepare_review, Adapter, knowledge, db, ingest, settings,
            run_agentic_review, sessions, logos, max_review_text_chars, too_large_error):
    brand_id = _session_brand_id(request, db, settings, sessions)
    data = base64.b64decode(request["fileBase64"], validate=True)
    with tempfile.TemporaryDirectory(prefix="trueline-review-") as image_root:
        document = parse_document(
            data, document_name=request["fileName"], content_type=request.get("contentType"),
            store_images=True, asset_storage_dir=image_root,
        )
        if len(document.text) > max_review_text_chars:
            raise too_large_error(
                f"Extracted review text is {len(document.text):,} characters; "
                f"the limit is {max_review_text_chars:,}."
            )
        with db.connect(settings) as conn:
            logo_records = logos.get_logo_assets(conn, brand_id)
            deterministic_rules = ingest.get_deterministic_rules(conn, brand_id, request.get("channel") or None)
            rulebook = ingest.get_rulebook(conn, brand_id, request.get("channel") or None)
        approved_logos = []
        for logo in logo_records:
            try:
                approved_logos.append({
                    **logo,
                    "data": logos.read_storage_ref(logo["storage_ref"]),
                })
            except Exception:
                continue
        visual_assets = _select_visual_assets(document.images, image_root)
        visual_images = _select_visual_images(document.images, image_root)
        # Temporary paths are internal only; return source/page/hash metadata, not paths.
        safe_images = tuple(image.model_copy(update={"storage_ref": None}) for image in document.images)
        document = document.model_copy(update={"images": safe_images})

        adapter = Adapter(
            search_knowledge=lambda search_request: knowledge.search(search_request).model_dump(),
            get_deterministic_rules=lambda **kwargs: deterministic_rules,
        )
        prepared = prepare_review(
            document,
            brand_id=brand_id,
            channel=request.get("channel") or None,
            retriever=adapter,
        )
        result = run_agentic_review(
            prepared, adapter, rulebook=rulebook,
            approved_logos=approved_logos, visual_assets=visual_assets,
            visual_images=visual_images,
        )
        result["visualAssets"] = [image.model_dump(mode="json") for image in safe_images]
        result["logoReferences"] = [
            {key: value for key, value in logo.items()
             if key not in {"storage_ref", "sha256", "visual_hash", "ocr_text", "data"}}
            for logo in logo_records
        ]
        return result


def _correct(request, parse_document, generate_corrected_docx, sessions, db, settings):
    _session_brand_id(request, db, settings, sessions)
    data = base64.b64decode(request["fileBase64"], validate=True)
    document = parse_document(
        data, document_name=request["fileName"], content_type=request.get("contentType"),
        store_images=False,
    )
    return generate_corrected_docx(document, request.get("reviewData"))


def _select_visual_assets(images, image_root, *, max_assets=8, max_item_bytes=4 * 1024 * 1024,
                          max_total_bytes=8 * 1024 * 1024):
    """Load bounded embedded images so the visual model can identify logo candidates itself."""
    selected = []
    seen = set()
    used = 0
    root = Path(image_root).resolve()
    for image in images:
        if image.extraction_method != "embedded" or not image.storage_ref or image.image_id in seen:
            continue
        if len(selected) >= max_assets:
            break
        image_path = (root / image.storage_ref).resolve()
        if root not in image_path.parents:
            continue
        try:
            if image_path.stat().st_size > max_item_bytes:
                continue
            image_data = image_path.read_bytes()
        except OSError:
            continue
        if used + len(image_data) > max_total_bytes:
            continue
        extension = str(image.format or "").lower()
        content_type = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp",
        }.get(extension)
        if not content_type:
            continue
        selected.append({
            "image_id": image.image_id, "page": image.page or 1,
            "content_type": content_type, "data": image_data,
            "extraction_method": "embedded",
        })
        seen.add(image.image_id)
        used += len(image_data)
    return selected


def _select_visual_images(images, image_root, *, max_item_bytes=4 * 1024 * 1024):
    """Load bounded full-page previews for layout analysis; never expose paths."""
    selected = []
    seen_pages = set()
    for image in images:
        if image.extraction_method not in {"page_render", "submitted"} or not image.storage_ref:
            continue
        if image.page in seen_pages:
            continue
        image_path = (Path(image_root) / image.storage_ref).resolve()
        if Path(image_root).resolve() not in image_path.parents:
            continue
        try:
            data = image_path.read_bytes()
        except OSError:
            continue
        if len(data) > max_item_bytes:
            continue
        extension = str(image.format or "").lower()
        content_type = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp",
        }.get(extension)
        if not content_type:
            continue
        selected.append({"image_id": image.image_id, "page": image.page,
                         "content_type": content_type, "data": data})
        seen_pages.add(image.page)
    return selected


if __name__ == "__main__":
    main()
