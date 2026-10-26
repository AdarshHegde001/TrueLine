"""Adapter to the teammate-owned ingestion retrieval implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from rag.retrieval.models import KnowledgeSearchRequest, KnowledgeSearchResponse


class RetrievalContractError(RuntimeError):
    """Raised when the ingestion service violates the shared retrieval contract."""


class KnowledgeRetriever(Protocol):
    def search_knowledge(
        self, request: KnowledgeSearchRequest
    ) -> KnowledgeSearchResponse: ...

    def get_deterministic_rules(
        self, *, brand_id: str, channel: str | None
    ) -> list[dict[str, Any]]: ...


class IngestionKnowledgeAdapter:
    """Validate responses from injected ingestion callables.

    The ingestion implementation owns PostgreSQL/pgvector SQL. This adapter keeps
    the RAG pipeline independent of whether that implementation is in-process or
    exposed through an HTTP adapter later.
    """

    def __init__(
        self,
        *,
        search_knowledge: Callable[[KnowledgeSearchRequest], Any],
        get_deterministic_rules: Callable[..., list[dict[str, Any]]],
    ) -> None:
        self._search_knowledge = search_knowledge
        self._get_deterministic_rules = get_deterministic_rules

    def search_knowledge(
        self, request: KnowledgeSearchRequest
    ) -> KnowledgeSearchResponse:
        try:
            raw_response = self._search_knowledge(request)
            response = (
                raw_response
                if isinstance(raw_response, KnowledgeSearchResponse)
                else KnowledgeSearchResponse.model_validate(raw_response)
            )
        except Exception as exc:
            if isinstance(exc, RetrievalContractError):
                raise
            raise RetrievalContractError("Knowledge retrieval failed") from exc

        if response.brand_id != request.brand_id:
            raise RetrievalContractError("Retrieval response brand does not match request")
        if len(response.results) > request.top_k:
            raise RetrievalContractError("Retrieval response exceeded requested top_k")
        return response

    def get_deterministic_rules(
        self, *, brand_id: str, channel: str | None
    ) -> list[dict[str, Any]]:
        try:
            records = self._get_deterministic_rules(brand_id=brand_id, channel=channel)
        except Exception as exc:
            raise RetrievalContractError("Loading deterministic rules failed") from exc
        if not isinstance(records, list):
            raise RetrievalContractError("Deterministic rule response must be a list")
        if any(not isinstance(record, Mapping) for record in records):
            raise RetrievalContractError("Deterministic rule response contains a non-object")
        return [dict(record) for record in records]
