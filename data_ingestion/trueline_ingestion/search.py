"""Read path: brand-scoped hybrid retrieval (PostgreSQL FTS + pgvector, fused by RRF)."""
from __future__ import annotations

import re
from typing import Optional

import psycopg

from .config import Settings, get_settings
from .db import connect, vec_literal
from .embeddings import Embedder
from .errors import SearchServiceError
from .ingest import get_deterministic_rules as _get_deterministic_rules
from .models import KnowledgeSearchRequest, KnowledgeSearchResponse, RetrievedChunk

RRF_K = 60

# Brand scoping + approval/active/ready filters are in SQL for BOTH candidate queries.
_FROM_WHERE = """
FROM knowledge_chunks c
JOIN knowledge_source_versions v ON v.version_id = c.version_id
JOIN knowledge_sources s         ON s.source_id  = c.source_id
WHERE c.brand_id = %(brand)s AND v.brand_id = %(brand)s AND s.brand_id = %(brand)s
  AND c.index_status = 'ready'
  AND c.embedding_model = %(tag)s
  AND v.status = 'active' AND v.approval_status = 'approved'
  AND s.current_version_id = v.version_id
  AND s.source_type = ANY(%(types)s)
  AND (cardinality(%(document_types)s::text[]) = 0
       OR c.document_type = ANY(%(document_types)s::text[]))
"""
_COLS = ("c.chunk_id, c.chunk_text, c.location, c.embedding_model, c.source_id, c.version_id, "
         "c.brand_id, c.document_type, s.source_type, s.title, v.version_number, v.text AS source_text")


def _tsquery(q: str) -> Optional[str]:
    toks = list(dict.fromkeys(re.findall(r"[A-Za-z0-9]+", q.lower())))
    return " | ".join(toks) if toks else None


class KnowledgeService:
    def __init__(self, embedder: Embedder, settings: Settings | None = None, conn_factory=None):
        self.embedder = embedder
        self.s = settings or get_settings()
        self._connect = conn_factory or (lambda: connect(self.s))

    def search(self, req: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
        params = {
            "brand": req.brand_id,
            "tag": self.embedder.tag,
            "types": list(req.source_types),
            "document_types": list(req.document_types),
        }
        pool = self.s.candidate_pool

        qvec, vec_err = None, None
        try:
            query_text = req.query
            if len(req.document_types) == 1:
                query_type = req.document_types[0].replace("_", " ")
                query_text = f"Document type: {query_type}\n{req.query}"
            embed_query = getattr(self.embedder, "embed_query", None)
            qvec = embed_query(query_text) if embed_query else self.embedder.embed([query_text])[0]
            if len(qvec) != self.s.embedding_dim:
                raise ValueError(f"query embedding dim {len(qvec)} != {self.s.embedding_dim}")
        except Exception as e:
            vec_err = e
            if not self.s.allow_keyword_fallback:
                raise SearchServiceError(f"query embedding failed: {e}") from e

        tsq = _tsquery(req.query)
        try:
            with self._connect() as conn:
                kw_rows, vec_rows = [], []
                if tsq:
                    kw_rows = conn.execute(
                        f"SELECT {_COLS}, "
                        f"ts_rank_cd(c.search_tsv, to_tsquery('english', %(tsq)s)) AS sc "
                        + _FROM_WHERE
                        + " AND c.search_tsv @@ to_tsquery('english', %(tsq)s)"
                        + " ORDER BY sc DESC, c.chunk_id LIMIT %(pool)s",
                        {**params, "tsq": tsq, "pool": pool},
                    ).fetchall()
                if qvec is not None:
                    vec_rows = conn.execute(
                        f"SELECT {_COLS}, (c.embedding <=> %(qv)s::vector) AS dist "
                        + _FROM_WHERE
                        + " ORDER BY dist ASC, c.chunk_id LIMIT %(pool)s",
                        {**params, "qv": vec_literal(qvec), "pool": pool},
                    ).fetchall()
        except psycopg.Error as e:
            raise SearchServiceError(f"database/index failure: {e}") from e

        if qvec is not None and tsq:
            method = "hybrid"
        elif qvec is not None:
            method = "vector"
        else:
            method = "keyword"  # only reachable with allow_keyword_fallback=True; labelled honestly

        fused: dict[str, dict] = {}
        for rank, r in enumerate(kw_rows, 1):
            e = fused.setdefault(r["chunk_id"], {"row": r, "score": 0.0})
            e["score"] += 1.0 / (RRF_K + rank)
        for rank, r in enumerate(vec_rows, 1):
            e = fused.setdefault(r["chunk_id"], {"row": r, "score": 0.0})
            e["score"] += 1.0 / (RRF_K + rank)

        ordered = sorted(fused.values(), key=lambda e: (-e["score"], e["row"]["chunk_id"]))[: req.top_k]
        results = [
            RetrievedChunk(
                source_id=e["row"]["source_id"], version_id=e["row"]["version_id"],
                brand_id=e["row"]["brand_id"], source_type=e["row"]["source_type"],
                document_type=e["row"]["document_type"],
                title=e["row"]["title"], version_number=e["row"]["version_number"],
                chunk_id=e["row"]["chunk_id"], chunk_text=e["row"]["chunk_text"],
                source_text=e["row"]["source_text"], location=e["row"]["location"] or {},
                embedding_model=e["row"]["embedding_model"], rank=i,
                score=round(e["score"], 6),
            )
            for i, e in enumerate(ordered, 1)
        ]
        return KnowledgeSearchResponse(
            brand_id=req.brand_id, retrieval_method=method, results=tuple(results)
        )


# ---- the contract's module-level operation: search_knowledge(request) ----
_default: KnowledgeService | None = None


def search_knowledge(request: KnowledgeSearchRequest) -> dict:
    """Return approved, active brand knowledge relevant to a focused query.
    Caller (backend) must already have authorised access to request.brand_id."""
    global _default
    if _default is None:
        from .embeddings import create_embedder
        _default = KnowledgeService(create_embedder())
    return _default.search(request).model_dump()


def get_deterministic_rules(*, brand_id: str, channel: str | None) -> list[dict]:
    """Load all active structured rules for an already-authorized brand."""
    settings = get_settings()
    try:
        with connect(settings) as conn:
            return _get_deterministic_rules(conn, brand_id, channel)
    except psycopg.Error as exc:
        raise SearchServiceError(f"database/index failure while loading rules: {exc}") from exc
