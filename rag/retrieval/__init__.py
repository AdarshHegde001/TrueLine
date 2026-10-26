"""Contract-first retrieval adapters and query orchestration."""

from rag.retrieval.client import IngestionKnowledgeAdapter, RetrievalContractError
from rag.retrieval.models import (
    EvidenceBundle,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    RetrievedChunk,
)
from rag.retrieval.pipeline import PartialRetrievalError, retrieve_document_evidence

__all__ = [
    "EvidenceBundle",
    "IngestionKnowledgeAdapter",
    "KnowledgeSearchRequest",
    "KnowledgeSearchResponse",
    "PartialRetrievalError",
    "RetrievedChunk",
    "RetrievalContractError",
    "retrieve_document_evidence",
]
