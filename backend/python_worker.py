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
    sys.stdout.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def main() -> None:
    try:
        # Keep library/model diagnostics away from the JSON-lines protocol.
        with contextlib.redirect_stdout(sys.stderr):
            from rag.parsing.registry import parse_document
            from rag.agentic.workflow import model_configuration, run_agentic_review
            from rag.agentic.rulebook import extract_reference_rules
            from rag.agentic.correction import generate_corrected_docx
            from rag.review import prepare_review
            from rag.retrieval.client import IngestionKnowledgeAdapter
            from trueline_ingestion import db, ingest
            from trueline_ingestion import logos
            from trueline_ingestion.config import get_settings
            from trueline_ingestion.embeddings import create_embedder
            from trueline_ingestion.models import SourceCreate
            from trueline_ingestion.search import KnowledgeService
            from trueline_ingestion import sessions

            settings = get_settings()
            db_connection = db.connect(settings)
            try:
                db.migrate(db_connection)
            finally:
                db_connection.close()

            embedder_mode = os.environ.get("INGESTION_EMBEDDER", "gemini").lower()
            embedder = create_embedder(embedder_mode, settings)
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
                                     run_agentic_review, sessions, logos)
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
            run_agentic_review, sessions, logos):
    brand_id = _session_brand_id(request, db, settings, sessions)
    data = base64.b64decode(request["fileBase64"], validate=True)
    with tempfile.TemporaryDirectory(prefix="trueline-review-") as image_root:
        document = parse_document(
            data, document_name=request["fileName"], content_type=request.get("contentType"),
            store_images=True, asset_storage_dir=image_root,
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
        logo_candidates = _select_logo_candidates(document.images, image_root, approved_logos)
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
            logo_candidates=logo_candidates, approved_logos=approved_logos,
        )
        result["visualAssets"] = [image.model_dump(mode="json") for image in safe_images]
        result["logoReferences"] = [
            {key: value for key, value in logo.items()
             if key not in {"storage_ref", "sha256", "visual_hash", "ocr_text", "data"}}
            for logo in logo_records
        ]
        result["logoReview"] = result["agentResults"].get("logo", {})
        return result


def _correct(request, parse_document, generate_corrected_docx, sessions, db, settings):
    _session_brand_id(request, db, settings, sessions)
    data = base64.b64decode(request["fileBase64"], validate=True)
    document = parse_document(
        data, document_name=request["fileName"], content_type=request.get("contentType"),
        store_images=False,
    )
    return generate_corrected_docx(document, request.get("reviewData"))


def _select_logo_candidates(images, image_root, approved_logos):
    """OCR/hash-filter locally; only likely matches reach the external logo agent."""
    if not approved_logos:
        return []
    from rag.parsing.image import ImageParser
    from rag.parsing.vision import hash_similarity, image_dhash, normalized_image_text

    reference_tokens = {
        logo["logo_id"]: normalized_image_text(logo.get("ocr_text", ""))
        for logo in approved_logos
    }
    references_by_hash = []
    for logo in approved_logos:
        try:
            visual_hash = logo.get("visual_hash") or image_dhash(logo["data"])
        except Exception:
            visual_hash = ""
        references_by_hash.append((logo, visual_hash))

    candidates = []
    for image in images:
        # A rendered PDF page contains charts and text around vector graphics, so it
        # is never sent as a logo candidate. Only standalone image assets are screened.
        if image.extraction_method == "page_render" or not image.storage_ref:
            continue
        image_path = (Path(image_root) / image.storage_ref).resolve()
        if Path(image_root).resolve() not in image_path.parents:
            continue
        try:
            image_data = image_path.read_bytes()
            candidate_hash = image_dhash(image_data)
        except Exception:
            continue
        extension = str(image.format or "").lower()
        content_type = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp",
        }.get(extension)
        if not content_type:
            continue
        similarities = [
            (hash_similarity(candidate_hash, reference_hash), logo)
            for logo, reference_hash in references_by_hash
        ]
        # Skip OCR for images that are visually far from every approved logo.
        # Text such as a chart title or brand name alone must never make an image
        # eligible for upload to Gemini.
        if not similarities or max(score for score, _logo in similarities) < 0.70:
            continue
        try:
            candidate_ocr = ImageParser().parse(
                image_data, document_name=f"candidate.{extension}", store_images=False,
            ).text
        except Exception:
            candidate_ocr = ""
        candidate_tokens = normalized_image_text(candidate_ocr)
        best = (0.0, None)
        for similarity, logo in similarities:
            shared_tokens = candidate_tokens & reference_tokens[logo["logo_id"]]
            text_match = bool(shared_tokens)
            # Exact/near visual matches pass directly. Lower similarity needs
            # matching logo text as a second signal, reducing chart false positives.
            score = similarity if similarity >= 0.82 or (similarity >= 0.70 and text_match) else 0.0
            if score > best[0]:
                best = (score, logo)
        if best[1] is None:
            continue
        candidates.append({
            "image_id": image.image_id,
            "page": image.page or 1,
            "content_type": content_type,
            "data": image_data,
            "matched_reference_id": best[1]["logo_id"],
            "local_match_score": round(best[0], 3),
        })
    return candidates


if __name__ == "__main__":
    main()
