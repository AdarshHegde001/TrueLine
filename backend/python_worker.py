"""Line-oriented worker that connects Express to the existing Python RAG stack."""

from __future__ import annotations

import contextlib
import base64
import json
import os
import sys
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
            from rag.review import prepare_review
            from rag.retrieval.client import IngestionKnowledgeAdapter
            from trueline_ingestion import db, ingest
            from trueline_ingestion.config import get_settings
            from trueline_ingestion.embeddings import BGEM3Embedder, FakeEmbedder
            from trueline_ingestion.models import SourceCreate
            from trueline_ingestion.search import KnowledgeService

            settings = get_settings()
            db_connection = db.connect(settings)
            try:
                db.migrate(db_connection)
            finally:
                db_connection.close()

            embedder_mode = os.environ.get("INGESTION_EMBEDDER", "bge-m3").lower()
            embedder = FakeEmbedder() if embedder_mode == "fake" else BGEM3Embedder(settings)
            knowledge = KnowledgeService(embedder, settings)

        emit({"type": "ready", "embedder": embedder_mode})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                operation = request.get("operation")
                if operation == "ingest":
                    result = _ingest(request, parse_document, ingest, SourceCreate,
                                     db, settings, embedder)
                elif operation == "review":
                    result = _review(request, parse_document, prepare_review,
                                     IngestionKnowledgeAdapter, knowledge, db, ingest, settings)
                else:
                    raise ValueError(f"Unsupported worker operation: {operation}")
                emit({"id": request.get("id"), "ok": True, "result": result})
            except Exception as exc:  # return a bounded, user-safe error per request
                emit({"id": request.get("id"), "ok": False, "error": str(exc)})
    except Exception as exc:
        emit({"type": "fatal", "error": str(exc)})
        traceback.print_exc(file=sys.stderr)
        raise


def _ingest(request, parse_document, ingest, SourceCreate, db, settings, embedder):
    data = base64.b64decode(request["fileBase64"], validate=True)
    document = parse_document(data, document_name=request["fileName"],
                              content_type=request.get("contentType"))
    document_type = request["documentType"]
    source_type = "approved_fact" if document_type in {"approved_claims", "product_facts"} else "guideline"
    record = SourceCreate(
        brand_id=request["brandId"],
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
            "document_type": document_type,
        },
        rule_data=request.get("ruleData") or [],
        approved=bool(request.get("approved", True)),
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
        "parse_warnings": list(document.warnings),
    }


def _review(request, parse_document, prepare_review, Adapter, knowledge, db, ingest, settings):
    data = base64.b64decode(request["fileBase64"], validate=True)
    document = parse_document(data, document_name=request["fileName"],
                              content_type=request.get("contentType"))

    def load_rules(*, brand_id: str, channel: str | None):
        with db.connect(settings) as conn:
            return ingest.get_deterministic_rules(conn, brand_id, channel)

    adapter = Adapter(
        search_knowledge=lambda search_request: knowledge.search(search_request).model_dump(),
        get_deterministic_rules=load_rules,
    )
    prepared = prepare_review(
        document,
        brand_id=request["brandId"],
        channel=request.get("channel") or None,
        retriever=adapter,
    )
    return prepared.model_dump(mode="json")


if __name__ == "__main__":
    main()
