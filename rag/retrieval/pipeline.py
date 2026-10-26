"""Focused, passage-aware retrieval over an injected ingestion service."""

from __future__ import annotations

from collections.abc import Iterator
import re

from rag.models import ParsedDocument, TextSpan
from rag.retrieval.client import KnowledgeRetriever, RetrievalContractError
from rag.retrieval.models import (
    EvidenceBundle,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
)

MAX_QUERY_CHARS = 2_000


class PartialRetrievalError(RetrievalContractError):
    """A retrieval failure after some passage queries already succeeded."""

    def __init__(self, message: str, completed: tuple[EvidenceBundle, ...]) -> None:
        super().__init__(message)
        self.completed = completed


def _query_passages(document: ParsedDocument) -> Iterator[tuple[str, TextSpan]]:
    """Yield bounded text passages with exact document offsets."""

    for segment in document.segments:
        text = segment.text
        cursor = 0
        while cursor < len(text):
            end = min(cursor + MAX_QUERY_CHARS, len(text))
            if end < len(text):
                whitespace_positions = [
                    cursor + match.start()
                    for match in re.finditer(r"\s+", text[cursor:end])
                ]
                if whitespace_positions and whitespace_positions[-1] > cursor + MAX_QUERY_CHARS // 2:
                    end = whitespace_positions[-1]
            raw = text[cursor:end]
            leading_trim = len(raw) - len(raw.lstrip())
            passage = raw.strip()
            if passage:
                start_offset = segment.start + cursor + leading_trim
                yield passage, TextSpan(
                    start=start_offset,
                    end=start_offset + len(passage),
                )
            cursor = end
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1


def retrieve_document_evidence(
    document: ParsedDocument,
    *,
    brand_id: str,
    retriever: KnowledgeRetriever,
    top_k: int = 8,
    source_types: tuple[str, ...] = ("guideline", "approved_fact"),
    document_types: tuple[str, ...] = (),
) -> tuple[EvidenceBundle, ...]:
    """Search once per non-empty parsed block, deduplicating identical queries.

    Returned evidence keeps the exact originating passage and span so later AI
    nodes can evaluate each passage with its own sources instead of one unrelated
    document-wide context.
    """

    bundles: list[EvidenceBundle] = []
    response_cache: dict[str, KnowledgeSearchResponse] = {}
    for query, passage_span in _query_passages(document):
        normalized_query = " ".join(query.casefold().split())
        if not normalized_query:
            continue
        response = response_cache.get(normalized_query)
        if response is None:
            request = KnowledgeSearchRequest(
                brand_id=brand_id,
                query=query,
                source_types=source_types,
                document_types=document_types,
                top_k=top_k,
            )
            try:
                response = retriever.search_knowledge(request)
                if response.brand_id != brand_id:
                    raise RetrievalContractError(
                        "retrieval response brand does not match requested brand"
                    )
            except RetrievalContractError as exc:
                raise PartialRetrievalError(
                    "Evidence retrieval failed for a passage query", tuple(bundles)
                ) from exc
            response_cache[normalized_query] = response
        bundles.append(
            EvidenceBundle(
                query=query,
                passage_span=passage_span,
                passage_quote=document.text[passage_span.start : passage_span.end],
                retrieval_method=response.retrieval_method,
                sources=response.results,
            )
        )
    return tuple(bundles)
