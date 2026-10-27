from __future__ import annotations

from types import SimpleNamespace

from rag.agentic import workflow
from rag.models import ContentSegment, ParsedDocument
from rag.review import prepare_review
from rag.retrieval.models import KnowledgeSearchResponse


BRAND = "brand_test"
SOURCE = {
    "brand_id": BRAND,
    "source_id": "source_facts",
    "version_id": "version_2",
    "chunk_id": "chunk_4",
    "source_type": "approved_fact",
    "document_type": "product_facts",
    "title": "Approved Product Facts",
    "version_number": 2,
    "chunk_text": "Nova increases customer retention by 20 percent.",
    "location": {"page": 4, "char_start": 0, "char_end": 50},
}
EVIDENCE = [{"sources": [SOURCE]}]


def _citation(source=SOURCE, quote="Nova increases customer retention by 20 percent."):
    return {
        "source_id": source["source_id"], "version_id": source["version_id"],
        "chunk_id": source["chunk_id"], "quote": quote,
    }


def test_claim_conflict_with_exact_relevant_approved_citation_is_violation_and_enriched():
    result = workflow._normalize_claims_result({
        "status": "complete", "findings": [{
            "document_quote": "Nova decreases customer retention by 20 percent.",
            "claim": "Nova decreases customer retention by 20 percent.",
            "status": "contradicted", "explanation": "This conflicts with the approved fact.",
            "citations": [_citation()],
        }],
    }, EVIDENCE, brand_id=BRAND,
        document_text="Nova decreases customer retention by 20 percent.")

    finding = result["findings"][0]
    assert finding["classification"] == "VIOLATION"
    assert finding["status"] == "contradicted"
    citation = finding["citations"][0]
    assert citation["source_title"] == "Approved Product Facts"
    assert citation["version_number"] == 2
    assert citation["page"] == 4
    assert citation["quote"] == SOURCE["chunk_text"]


def test_wrong_or_irrelevant_citation_downgrades_claim_instead_of_asserting_violation():
    irrelevant = {**SOURCE, "chunk_id": "chunk_elsewhere", "chunk_text": "The blue package uses recycled paper."}
    evidence = [{"sources": [irrelevant]}]
    result = workflow._normalize_claims_result({
        "status": "complete", "findings": [{
            "document_quote": "Nova decreases customer retention by 20 percent.",
            "claim": "Nova decreases customer retention by 20 percent.",
            "status": "contradicted", "explanation": "Model assertion.",
            "citations": [_citation(irrelevant, irrelevant["chunk_text"])],
        }],
    }, evidence, brand_id=BRAND,
        document_text="Nova decreases customer retention by 20 percent.")

    finding = result["findings"][0]
    assert finding["classification"] == "UNVERIFIED"
    assert finding["status"] == "unverified"
    assert finding["citations"] == []


def test_unretrieved_source_and_nonverbatim_excerpt_are_never_returned_as_citations():
    invented = {**_citation(), "source_id": "source_not_retrieved", "quote": "Nova decreases retention by 20 percent."}
    result = workflow._normalize_claims_result({
        "status": "complete", "findings": [{
            "document_quote": "Nova decreases customer retention by 20 percent.",
            "claim": "Nova decreases customer retention by 20 percent.",
            "status": "contradicted", "explanation": "Claim conflicts with evidence.",
            "citations": [invented],
        }],
    }, EVIDENCE, brand_id=BRAND,
        document_text="Nova decreases customer retention by 20 percent.")
    finding = result["findings"][0]
    assert finding["classification"] == "UNVERIFIED"
    assert finding["citations"] == []


def test_supported_claim_with_cited_approved_fact_is_passed():
    result = workflow._normalize_claims_result({
        "status": "complete", "findings": [{
            "document_quote": "Nova increases customer retention by 20 percent.",
            "claim": "Nova increases customer retention by 20 percent.",
            "status": "supported", "explanation": "Matches the approved fact.",
            "citations": [_citation()],
        }],
    }, EVIDENCE, brand_id=BRAND,
        document_text="Nova increases customer retention by 20 percent.")
    assert result["findings"][0]["classification"] == "PASSED"


def test_claim_without_any_approved_evidence_is_unsupported():
    result = workflow._normalize_claims_result({
        "status": "insufficient_evidence", "findings": [{
            "document_quote": "Nova doubles retention.", "claim": "Nova doubles retention.",
            "status": "supported", "explanation": "No source was provided.", "citations": [],
        }],
    }, [], brand_id=BRAND, document_text="Nova doubles retention.")
    assert result["findings"][0]["classification"] == "UNSUPPORTED"
    assert result["findings"][0]["status"] == "unsupported"


def test_claim_quote_not_present_in_reviewed_document_is_unverified():
    result = workflow._normalize_claims_result({
        "status": "complete", "findings": [{
            "document_quote": "Invented review text.", "claim": "Invented review text.",
            "status": "contradicted", "explanation": "Model supplied a quote absent from the draft.",
            "citations": [_citation()],
        }],
    }, EVIDENCE, brand_id=BRAND, document_text="Actual reviewed text.")
    assert result["findings"][0]["classification"] == "UNVERIFIED"
    assert result["findings"][0]["citations"] == []


def test_tone_violation_requires_explicit_rule_quote_inside_valid_citation():
    source = {**SOURCE, "source_type": "guideline", "chunk_text": "Avoid the phrase 'guaranteed results' in marketing copy."}
    evidence = [{"sources": [source]}]
    result = workflow._normalize_tone_result({
        "status": "complete", "findings": [{
            "document_quote": "Guaranteed results for everyone.",
            "assessment": "The wording conflicts with the explicit guidance.",
            "severity": "high", "classification": "VIOLATION",
            "rule_quote": "Avoid the phrase 'guaranteed results' in marketing copy.",
            "citations": [_citation(source, source["chunk_text"])],
        }],
    }, evidence, brand_id=BRAND, document_text="Guaranteed results for everyone.")
    assert result["findings"][0]["classification"] == "VIOLATION"

    result["findings"][0]["rule_quote"] = "Always use cheerful language."
    downgraded = workflow._normalize_tone_result(
        {"status": "complete", "findings": result["findings"]}, evidence,
        brand_id=BRAND, document_text="Guaranteed results for everyone.",
    )
    assert downgraded["findings"][0]["classification"] == "UNVERIFIED"

    ungrounded_pass = workflow._normalize_tone_result({
        "status": "complete", "findings": [{
            "document_quote": "Guaranteed results for everyone.",
            "assessment": "Tone is acceptable.", "classification": "PASSED",
            "citations": [],
        }],
    }, [], brand_id=BRAND, document_text="Guaranteed results for everyone.")
    assert ungrounded_pass["findings"][0]["classification"] == "UNVERIFIED"


def test_logo_result_uses_only_supplied_ids_and_cannot_invent_page_or_violation():
    output = {
        "status": "complete", "findings": [
            {"review_image_id": "img_1", "page": 999, "detection": "logo",
             "reference_logo_id": "logo_1", "comparison": "differences_observed",
             "observation": "The mark appears to use a different color."},
            {"review_image_id": "invented_image", "page": 1, "detection": "logo",
             "reference_logo_id": "logo_1", "comparison": "differences_observed",
             "observation": "Invented candidate."},
        ],
    }
    validated = workflow._validate_logo_findings(
        output,
        [{"image_id": "img_1", "page": 3}],
        [{"logo_id": "logo_1", "title": "Approved primary mark"}],
    )
    assert len(validated["findings"]) == 1
    finding = validated["findings"][0]
    assert finding["page"] == 3
    assert finding["reference_logo_title"] == "Approved primary mark"
    assert finding["classification"] == "UNVERIFIED"
    assert validated["classification"] == "UNVERIFIED"


def test_duplicate_summary_items_collapse_but_distinct_claims_in_same_sentence_remain():
    prepared = SimpleNamespace(deterministic=SimpleNamespace(findings=()))
    finding = {
        "document_quote": "Nova is fast and doubles retention.",
        "claim": "Nova doubles retention.", "classification": "VIOLATION",
        "severity": "high", "explanation": "Conflicts with approved facts.",
        "citations": [{**_citation(), "source_title": "Approved Product Facts",
                       "version_number": 2, "page": 4}],
    }
    claims = {"findings": [finding, {**finding}, {
        **finding, "claim": "Nova is fast.", "explanation": "Speed claim is also contradicted.",
    }]}
    candidates = workflow._summary_candidates(prepared, {"findings": []}, claims)
    assert len(candidates) == 2
    summary, key_findings = workflow._render_summary(candidates, selected_ids=["invented-id"])
    assert summary.count("### Claims · VIOLATION") == 2
    assert "Approved Product Facts, v2, page 4" in summary
    assert "Nova doubles retention." in summary and "Nova is fast." in summary
    assert "invented-id" not in summary
    assert len(key_findings) == 2


def test_overall_classification_includes_deterministic_and_all_agent_results():
    prepared = SimpleNamespace(
        retrieval_status="complete",
        deterministic=SimpleNamespace(findings=[SimpleNamespace(label="violation")]),
    )
    agents = {"classification": "PASSED"}
    assert workflow._overall_classification(
        prepared, agents, agents, agents,
    ) == "VIOLATION"

    prepared.deterministic.findings = []
    assert workflow._overall_classification(
        prepared, agents, {"classification": "UNSUPPORTED"}, agents,
    ) == "UNSUPPORTED"

    prepared.retrieval_status = "failed"
    assert workflow._overall_classification(
        prepared, agents, agents, agents,
    ) == "CHECK_FAILED"


def test_agent_failures_are_classified_and_summary_stays_within_validated_results(monkeypatch):
    document_text = "Nova claims a result."
    document = ParsedDocument(
        document_name="draft.txt", media_type="text/plain", text=document_text,
        segments=(ContentSegment(text=document_text, start=0, end=len(document_text)),),
        parser="plain-text", parser_version="1",
    )

    class EmptyRetriever:
        def search_knowledge(self, request):
            return KnowledgeSearchResponse(
                brand_id=request.brand_id, retrieval_method="vector", results=(),
            )

        def get_deterministic_rules(self, *, brand_id, channel):
            return []

    prepared = prepare_review(
        document, brand_id=BRAND, channel=None, retriever=EmptyRetriever(),
    )

    def fail_model(_agent):
        raise RuntimeError("simulated model outage")

    monkeypatch.setattr(workflow, "_chat", fail_model)
    result = workflow.run_agentic_review(prepared, EmptyRetriever())

    assert result["agentResults"]["tone"]["classification"] == "CHECK_FAILED"
    assert result["agentResults"]["claims"]["classification"] == "CHECK_FAILED"
    assert result["agentResults"]["summary"]["classification"] == "CHECK_FAILED"
    assert result["agentResults"]["summary"]["status"] == "error"
    assert "No unique, evidence-backed issues" in result["agentResults"]["summary"]["summary"]
    assert "CHECK_FAILED" in result["agentResults"]["summary"]["summary"]
