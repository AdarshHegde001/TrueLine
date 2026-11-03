"""The correction agent may act only on summary findings and their evidence."""

from rag.agentic import correction, workflow
from rag.models import ContentSegment, ParsedDocument


def _document(text="Our product eliminates 100% of defects.") -> ParsedDocument:
    return ParsedDocument(
        document_name="draft.txt",
        media_type="text/plain",
        text=text,
        segments=(ContentSegment(text=text, start=0, end=len(text)),),
        parser="test",
        parser_version="1",
    )


def test_correction_uses_finding_scoped_rag_evidence_before_patching(monkeypatch):
    finding = {
        "correction_finding_id": "claims:0",
        "id": "claims:0",
        "category": "Claims",
        "classification": "UNSUPPORTED",
        "severity": "high",
        "subject": "The defect elimination claim lacks support",
        "document_quote": "eliminates 100% of defects",
        "citations": [],
    }
    source = {
        "brand_id": "brand-test",
        "source_id": "facts-1",
        "version_id": "version-1",
        "source_type": "approved_fact",
        "document_type": "product_facts",
        "title": "Approved product facts",
        "version_number": 1,
        "chunk_id": "chunk-1",
        "chunk_text": "In monitored deployments, the product reduced defects by up to 20%.",
        "location": {},
        "score": 0.91,
    }
    citation = {
        "source_id": source["source_id"],
        "version_id": source["version_id"],
        "chunk_id": source["chunk_id"],
        "quote": source["chunk_text"],
    }
    turns = [
        {"tool_calls": [
            {"tool": "search_approved_sources", "finding_id": "claims:0", "query": "defect reduction for this product"},
            {"tool": "inspect_target", "finding_id": "claims:0"},
        ]},
        {"tool_calls": [{
            "tool": "apply_patch", "finding_id": "claims:0", "segment_index": 0,
            "exact_quote": finding["document_quote"],
            "corrected_text": "reduced defects by up to 20% in monitored deployments",
            "evidence_citations": [citation],
        }]},
        {"tool_calls": [{"tool": "verify_patch", "finding_id": "claims:0"}]},
        {"tool_calls": [], "done": True},
    ]
    monkeypatch.setattr(workflow, "_chat", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(workflow, "_json_result", lambda *_args, **_kwargs: turns.pop(0))
    searched_requests = []

    def search(request):
        searched_requests.append(request)
        return {"brand_id": "brand-test", "retrieval_method": "hybrid", "results": [source]}

    edits, _styles, _logos, unresolved, tool_calls = correction._run_document_tool_loop(
        _document(), [finding], b"Our product eliminates 100% of defects.", [],
        brand_id="brand-test", search_knowledge=search,
    )

    assert len(searched_requests) == 1
    assert searched_requests[0].brand_id == "brand-test"
    assert searched_requests[0].document_types == (
        "approved_claims", "product_facts", "brand_guidelines", "legal_compliance", "other_reference",
    )
    assert "eliminates 100% of defects" in searched_requests[0].query
    assert len(edits) == 1
    assert edits[0]["corrected_text"] == "reduced defects by up to 20% in monitored deployments"
    assert unresolved == []
    assert tool_calls == 4


def test_correction_rejects_patch_without_citation_for_summary_finding(monkeypatch):
    finding = {
        "correction_finding_id": "tone:0",
        "id": "tone:0",
        "category": "Tone",
        "classification": "VIOLATION",
        "document_quote": "Zero-defect guaranteed",
        "citations": [],
    }
    turns = [
        {"tool_calls": [{"tool": "inspect_target", "finding_id": "tone:0"}]},
        {"tool_calls": [{
            "tool": "apply_patch", "finding_id": "tone:0", "segment_index": 0,
            "exact_quote": finding["document_quote"], "corrected_text": "Designed to reduce defects",
        }]},
        {"tool_calls": [], "done": True,
         "unresolved_findings": [{"finding_id": "tone:0", "reason": "No direct supporting evidence was available."}]},
    ]
    monkeypatch.setattr(workflow, "_chat", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(workflow, "_json_result", lambda *_args, **_kwargs: turns.pop(0))

    edits, _styles, _logos, unresolved, _calls = correction._run_document_tool_loop(
        _document("Zero-defect guaranteed"), [finding], b"Zero-defect guaranteed", [],
        brand_id="brand-test", search_knowledge=lambda _request: {"results": []},
    )

    assert edits == []
    assert len(unresolved) == 1
    assert "citation" in unresolved[0].lower() or "evidence" in unresolved[0].lower()
