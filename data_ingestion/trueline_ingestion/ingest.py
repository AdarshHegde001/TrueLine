"""Write path: create / update / approve / deactivate sources, chunk, embed, index.

Rules enforced here (contract):
  * versions are immutable; an update inserts a NEW version row
  * a version becomes searchable only after ALL its chunks are embedded ('ready')
  * embedding failure => chunks 'failed', version stays 'pending', IndexingError raised
"""
from __future__ import annotations

import uuid
from typing import Optional

import psycopg
from psycopg.types.json import Jsonb

from .chunking import chunk_text
from .config import Settings, get_settings
from .db import vec_literal
from .embeddings import Embedder
from .errors import IndexingError, NotFound, ValidationFailed
from .models import SourceCreate, SourceUpdate, SourceVersionRecord


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------- create / update

def create_source(conn, embedder: Embedder, payload: SourceCreate,
                  settings: Settings | None = None) -> SourceVersionRecord:
    s = settings or get_settings()
    source_id = _id("src")
    with conn.transaction():
        conn.execute(
            "INSERT INTO knowledge_sources (source_id, brand_id, source_type, title) "
            "VALUES (%s,%s,%s,%s)",
            (source_id, payload.brand_id, payload.source_type, payload.title),
        )
        version_id = _insert_version(
            conn, embedder, s, source_id=source_id, brand_id=payload.brand_id, number=1,
            text=payload.text, document_id=payload.document_id,
            document_name=payload.document_name, metadata=payload.metadata,
            approved=payload.approved, title=payload.title,
        )
    conn.commit()
    if payload.approved:
        index_version(conn, embedder, version_id, s)
    return get_version(conn, payload.brand_id, source_id, version_id)


def update_source(conn, embedder: Embedder, payload: SourceUpdate,
                  settings: Settings | None = None) -> SourceVersionRecord:
    """New immutable version. The previous version stays resolvable but leaves
    ordinary retrieval once the new version is indexed successfully."""
    s = settings or get_settings()
    with conn.transaction():
        row = conn.execute(
            "SELECT title FROM knowledge_sources WHERE source_id=%s AND brand_id=%s FOR UPDATE",
            (payload.source_id, payload.brand_id),
        ).fetchone()
        if row is None:
            raise NotFound("source not found")
        number = conn.execute(
            "SELECT COALESCE(MAX(version_number),0)+1 AS n FROM knowledge_source_versions "
            "WHERE source_id=%s", (payload.source_id,)
        ).fetchone()["n"]
        version_id = _insert_version(
            conn, embedder, s, source_id=payload.source_id, brand_id=payload.brand_id,
            number=number, text=payload.text, document_id=payload.document_id,
            document_name=payload.document_name, metadata=payload.metadata,
            approved=payload.approved, title=payload.title or row["title"],
        )
    conn.commit()
    if payload.approved:
        index_version(conn, embedder, version_id, s)
    return get_version(conn, payload.brand_id, payload.source_id, version_id)


def _insert_version(conn, embedder, s: Settings, *, source_id, brand_id, number, text,
                    document_id, document_name, metadata, approved, title) -> str:
    version_id = _id("ver")
    conn.execute(
        "INSERT INTO knowledge_source_versions (version_id, source_id, brand_id, version_number,"
        " status, approval_status, text, document_id, document_name, metadata) "
        "VALUES (%s,%s,%s,%s,'pending',%s,%s,%s,%s,%s)",
        (version_id, source_id, brand_id, number, "approved" if approved else "pending",
         text, document_id, document_name, Jsonb(metadata)),
    )
    chunks = chunk_text(text, s.chunk_max_chars)
    if not chunks:
        raise ValidationFailed("text produced no chunks")
    for c in chunks:
        loc = {k: v for k, v in metadata.items()
               if k in ("document_id", "document_name", "section", "page")}
        if document_id:
            loc.setdefault("document_id", document_id)
        if document_name:
            loc.setdefault("document_name", document_name)
        loc["title"] = title
        # offsets are into knowledge_source_versions.text (Python str indices)
        loc.update(char_start=c.char_start, char_end=c.char_end, offset_base="version_text")
        conn.execute(
            "INSERT INTO knowledge_chunks (chunk_id, brand_id, source_id, version_id, chunk_index,"
            " chunk_text, location, embedding_model, chunking_version, index_status) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,'pending')",
            (_id("chunk"), brand_id, source_id, version_id, c.index, c.text, Jsonb(loc),
             embedder.tag, s.chunking_version),
        )
    return version_id


# ---------------------------------------------------------------- indexing

def index_version(conn, embedder: Embedder, version_id: str,
                  settings: Settings | None = None) -> None:
    """Embed all non-ready chunks of an APPROVED version, then activate it atomically."""
    s = settings or get_settings()
    v = conn.execute(
        "SELECT source_id, brand_id, approval_status FROM knowledge_source_versions "
        "WHERE version_id=%s", (version_id,)
    ).fetchone()
    if v is None:
        raise NotFound("version not found")
    if v["approval_status"] != "approved":
        raise ValidationFailed("only approved versions can be indexed")

    rows = conn.execute(
        "SELECT chunk_id, chunk_text FROM knowledge_chunks WHERE version_id=%s ORDER BY chunk_index",
        (version_id,),
    ).fetchall()
    ids = [r["chunk_id"] for r in rows]
    conn.commit()  # close the read transaction; writes below are their own transaction
    try:
        vecs = embedder.embed([r["chunk_text"] for r in rows])
        if len(vecs) != len(ids) or any(len(x) != s.embedding_dim for x in vecs):
            raise IndexingError(f"embedder returned wrong count/dimension (expected {s.embedding_dim})")
    except Exception as e:  # model missing/OOM/etc. => visible failed state, not searchable
        conn.execute(
            "UPDATE knowledge_chunks SET index_status='failed', updated_at=now() WHERE version_id=%s",
            (version_id,),
        )
        conn.commit()
        raise IndexingError(f"embedding failed: {e}") from e

    with conn.transaction():
        for cid, vec in zip(ids, vecs):
            conn.execute(
                "UPDATE knowledge_chunks SET embedding=%s::vector, embedding_model=%s,"
                " index_status='ready', updated_at=now() WHERE chunk_id=%s",
                (vec_literal(vec), embedder.tag, cid),
            )
        # retire previous current version (kept for historical citation resolution)
        conn.execute(
            "UPDATE knowledge_source_versions SET status='inactive' "
            "WHERE source_id=%s AND status='active' AND version_id<>%s",
            (v["source_id"], version_id),
        )
        conn.execute("UPDATE knowledge_source_versions SET status='active' WHERE version_id=%s",
                     (version_id,))
        conn.execute(
            "UPDATE knowledge_sources SET current_version_id=%s, updated_at=now() WHERE source_id=%s",
            (version_id, v["source_id"]),
        )
    conn.commit()


def approve_version(conn, embedder: Embedder, brand_id: str, version_id: str,
                    settings: Settings | None = None) -> SourceVersionRecord:
    """Admin confirmation of a pending version; then it is indexed and activated."""
    row = conn.execute(
        "SELECT source_id FROM knowledge_source_versions WHERE version_id=%s AND brand_id=%s",
        (version_id, brand_id),
    ).fetchone()
    if row is None:
        raise NotFound("version not found")
    conn.execute("UPDATE knowledge_source_versions SET approval_status='approved' "
                 "WHERE version_id=%s", (version_id,))
    conn.commit()
    index_version(conn, embedder, version_id, settings)
    return get_version(conn, brand_id, row["source_id"], version_id)


def reindex_stale(conn, embedder: Embedder, settings: Settings | None = None) -> list[str]:
    """Re-embed approved versions that have non-ready chunks or chunks built with another
    embedding tag (model/revision change). Chunk TEXT is never changed here."""
    rows = conn.execute(
        "SELECT DISTINCT v.version_id FROM knowledge_source_versions v "
        "JOIN knowledge_chunks c ON c.version_id=v.version_id "
        "WHERE v.approval_status='approved' AND v.status IN ('pending','active') "
        "AND (c.index_status<>'ready' OR c.embedding_model<>%s)", (embedder.tag,)
    ).fetchall()
    done = []
    for r in rows:
        index_version(conn, embedder, r["version_id"], settings)
        done.append(r["version_id"])
    return done


def deactivate_source(conn, brand_id: str, source_id: str) -> None:
    """Remove from ordinary retrieval. Versions/chunks are RETAINED (not mutated)."""
    with conn.transaction():
        n = conn.execute(
            "UPDATE knowledge_sources SET current_version_id=NULL, updated_at=now() "
            "WHERE source_id=%s AND brand_id=%s", (source_id, brand_id)
        ).rowcount
        if n == 0:
            raise NotFound("source not found")
        conn.execute("UPDATE knowledge_source_versions SET status='inactive' "
                     "WHERE source_id=%s AND status='active'", (source_id,))
    conn.commit()


# ---------------------------------------------------------------- citation resolution (for RAG)

def get_version(conn, brand_id: str, source_id: str, version_id: str) -> SourceVersionRecord:
    """Resolve an exact (source_id, version_id) for citation validation. Works for
    superseded/inactive versions too (historical findings), but is brand-scoped."""
    r = conn.execute(
        "SELECT s.source_id, s.brand_id, s.source_type, s.title, s.updated_at, v.version_id,"
        " v.version_number, v.status, v.approval_status, v.text, v.metadata, v.created_at,"
        " v.document_id, v.document_name "
        "FROM knowledge_source_versions v JOIN knowledge_sources s ON s.source_id=v.source_id "
        "WHERE v.version_id=%s AND v.source_id=%s AND v.brand_id=%s",
        (version_id, source_id, brand_id),
    ).fetchone()
    if r is None:
        raise NotFound("source version not found")  # do not reveal other brands' data
    meta = dict(r["metadata"] or {})
    if r["document_id"]:
        meta.setdefault("document_id", r["document_id"])
    if r["document_name"]:
        meta.setdefault("document_name", r["document_name"])
    return SourceVersionRecord(
        source_id=r["source_id"], brand_id=r["brand_id"], source_type=r["source_type"],
        title=r["title"], status=r["status"], approval_status=r["approval_status"],
        version_id=r["version_id"], version=r["version_number"], text=r["text"], metadata=meta,
        created_at=r["created_at"].isoformat(), updated_at=r["updated_at"].isoformat(),
    )


def quote_exists(conn, brand_id: str, source_id: str, version_id: str, quote: str) -> bool:
    """True iff `quote` is an exact substring of that version's canonical text."""
    if not quote:
        return False
    return quote in get_version(conn, brand_id, source_id, version_id).text
