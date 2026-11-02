"""Prepare deterministic results and retrieved context for a later AI node."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from rag.context import ReviewContext, build_review_context
from rag.models import ParsedDocument, RuleCheckResult
from rag.retrieval.client import KnowledgeRetriever, RetrievalContractError
from rag.retrieval.models import EvidenceBundle
from rag.retrieval.pipeline import PartialRetrievalError, retrieve_document_evidence
from rag.rules import DeterministicRuleEngine, rules_from_ingestion_records


MAX_REVIEW_TEXT_CHARS = 300_000


class ReviewDocumentTooLargeError(ValueError):
    """Raised when parsed review text exceeds the bounded agent workflow limit."""

    http_status = 413


class PreparedReview(BaseModel):
    """Complete non-AI review input ready for a future LangGraph node."""

    model_config = ConfigDict(frozen=True)

    brand_id: str
    document: ParsedDocument
    deterministic: RuleCheckResult
    evidence: tuple[EvidenceBundle, ...]
    retrieval_status: Literal["complete", "failed"]
    retrieval_error: str | None
    ai_context: ReviewContext


def prepare_review(
    document: ParsedDocument,
    *,
    brand_id: str,
    channel: str | None,
    retriever: KnowledgeRetriever,
    top_k: int = 8,
    max_context_chars: int = 1_500_000,
) -> PreparedReview:
    """Run parsing-adjacent deterministic checks and retrieval, then assemble AI context.

    This function does not call a model. Missing deterministic rules are an empty
    rule set; retrieval failures are kept explicit in the returned context so a
    future graph can block evidence-dependent agents rather than claiming success.
    """

    if len(document.text) > MAX_REVIEW_TEXT_CHARS:
        raise ReviewDocumentTooLargeError(
            f"Extracted review text is {len(document.text):,} characters; "
            f"the limit is {MAX_REVIEW_TEXT_CHARS:,}."
        )

    rule_records = retriever.get_deterministic_rules(brand_id=brand_id, channel=channel)
    rules = rules_from_ingestion_records(rule_records)
    deterministic = DeterministicRuleEngine().run(document, rules, channel=channel)

    evidence: tuple[EvidenceBundle, ...] = ()
    retrieval_status: Literal["complete", "failed"] = "complete"
    retrieval_error: str | None = None
    try:
        evidence = retrieve_document_evidence(
            document,
            brand_id=brand_id,
            retriever=retriever,
            top_k=top_k,
            max_queries=24,
        )
    except PartialRetrievalError as exc:
        evidence = exc.completed
        retrieval_status = "failed"
        retrieval_error = str(exc)
    except RetrievalContractError as exc:
        retrieval_status = "failed"
        retrieval_error = str(exc)

    ai_context = build_review_context(
        document,
        deterministic,
        evidence,
        brand_id=brand_id,
        channel=channel,
        max_chars=max_context_chars,
        retrieval_status=retrieval_status,
        retrieval_error=retrieval_error,
    )
    return PreparedReview(
        brand_id=brand_id,
        document=document,
        deterministic=deterministic,
        evidence=evidence,
        retrieval_status=retrieval_status,
        retrieval_error=retrieval_error,
        ai_context=ai_context,
    )
