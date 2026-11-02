"""Read path: brand-scoped hybrid retrieval (PostgreSQL FTS + pgvector, fused by RRF)."""
from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Optional

import psycopg

from .config import Settings, get_settings
from .db import connect, vec_literal
from .embeddings import Embedder
from .errors import SearchServiceError
from .ingest import get_deterministic_rules as _get_deterministic_rules
from .models import KnowledgeSearchRequest, KnowledgeSearchResponse, RetrievedChunk

RRF_K = 60
PARENT_CONTEXT_MAX_CHARS = 2_400


@lru_cache(maxsize=2)
def _load_reranker(model_name: str):
    """Load a scoring model separately from the BGE-M3 embedding model."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _parent_context(source_text: str, location: dict, max_chars: int = PARENT_CONTEXT_MAX_CHARS) -> str | None:
    """Return the paragraph containing a child chunk, bounded for model context.

    The child chunk and all persisted offsets remain unchanged. This is derived
    from the immutable source text, so it needs no schema migration or reindex.
    """
    start = location.get("char_start")
    end = location.get("char_end")
    if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(source_text)):
        return None

    paragraph_start = source_text.rfind("\n\n", 0, start)
    paragraph_start = 0 if paragraph_start < 0 else paragraph_start + 2
    paragraph_end = source_text.find("\n\n", end)
    paragraph_end = len(source_text) if paragraph_end < 0 else paragraph_end
    if paragraph_end - paragraph_start <= max_chars:
        context = source_text[paragraph_start:paragraph_end].strip()
        return context or None

    # For unusually long paragraphs, retain the child and balanced nearby text.
    max_chars = max(max_chars, end - start)
    window_start = max(paragraph_start, start - (max_chars - (end - start)) // 2)
    window_start = min(window_start, max(paragraph_start, paragraph_end - max_chars))
    window_end = min(paragraph_end, window_start + max_chars)
    return source_text[window_start:window_end].strip() or None


def _rerank_candidates(query: str, candidates: list[dict], model_name: str) -> tuple[list[dict], list[float]]:
    """Cross-encode the hybrid candidate pool; return ordered rows and scores."""
    scorer = _load_reranker(model_name)
    pairs = [(query, item["row"]["chunk_text"]) for item in candidates]
    raw_scores = scorer.predict(pairs)
    if hasattr(raw_scores, "reshape"):
        raw_scores = raw_scores.reshape(-1).tolist()
    elif not isinstance(raw_scores, (list, tuple)):
        raw_scores = [raw_scores]
    scores = [float(score) for score in raw_scores]
    if len(scores) != len(candidates):
        raise ValueError("reranker returned an unexpected number of scores")
    ranked = sorted(zip(candidates, scores), key=lambda pair: (-pair[1], pair[0]["row"]["chunk_id"]))
    return [item for item, _ in ranked], [score for _, score in ranked]

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
            qvec = self.embedder.embed([query_text])[0]
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

        ordered = sorted(fused.values(), key=lambda e: (-e["score"], e["row"]["chunk_id"]))
        reranker_status = "disabled"
        reranker_scores: dict[str, float] = {}
        if os.environ.get("TRUELINE_RERANKER_ENABLED", "0").strip().lower() in {"1", "true", "yes", "on"} and ordered:
            model_name = os.environ.get("TRUELINE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3").strip()
            try:
                ordered, scores = _rerank_candidates(req.query, ordered, model_name)
                reranker_scores = {item["row"]["chunk_id"]: score for item, score in zip(ordered, scores)}
                reranker_status = "applied"
            except Exception:
                # The hybrid RRF order remains a valid and safe fallback.
                reranker_status = "fallback"
        ordered = ordered[: req.top_k]
        results = [
            RetrievedChunk(
                source_id=e["row"]["source_id"], version_id=e["row"]["version_id"],
                brand_id=e["row"]["brand_id"], source_type=e["row"]["source_type"],
                document_type=e["row"]["document_type"],
                title=e["row"]["title"], version_number=e["row"]["version_number"],
                chunk_id=e["row"]["chunk_id"], chunk_text=e["row"]["chunk_text"],
                source_text=e["row"]["source_text"], location=e["row"]["location"] or {},
                parent_context=_parent_context(e["row"]["source_text"], e["row"]["location"] or {}),
                embedding_model=e["row"]["embedding_model"], rank=i,
                score=round(e["score"], 6),
                reranker_score=reranker_scores.get(e["row"]["chunk_id"]),
            )
            for i, e in enumerate(ordered, 1)
        ]
        return KnowledgeSearchResponse(
            brand_id=req.brand_id, retrieval_method=method,
            reranker_status=reranker_status, results=tuple(results)
        )


# ---- the contract's module-level operation: search_knowledge(request) ----
_default: KnowledgeService | None = None


def search_knowledge(request: KnowledgeSearchRequest) -> dict:
    """Return approved, active brand knowledge relevant to a focused query.
    Caller (backend) must already have authorised access to request.brand_id."""
    global _default
    if _default is None:
        from .embeddings import BGEM3Embedder
        _default = KnowledgeService(BGEM3Embedder())
    return _default.search(request).model_dump()


def get_deterministic_rules(*, brand_id: str, channel: str | None) -> list[dict]:
    """Load all active structured rules for an already-authorized brand."""
    settings = get_settings()
    try:
        with connect(settings) as conn:
            return _get_deterministic_rules(conn, brand_id, channel)
    except psycopg.Error as exc:
        raise SearchServiceError(f"database/index failure while loading rules: {exc}") from exc
