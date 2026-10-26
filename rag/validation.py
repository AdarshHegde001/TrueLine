"""Pure validation helpers shared by deterministic and future AI findings."""

from __future__ import annotations

from rag.models import Finding, FindingLabel, ParsedDocument


class FindingValidationError(ValueError):
    """Raised when a finding cannot be grounded in the parsed submission."""


def validate_finding_span(finding: Finding, document: ParsedDocument) -> None:
    """Ensure any quote and character span exactly match canonical parsed text."""

    if finding.label == FindingLabel.VIOLATION and not finding.source.source_id:
        raise FindingValidationError("Violation findings require a source ID")
    if finding.label == FindingLabel.VIOLATION and not finding.source.version_id:
        raise FindingValidationError("Violation findings require a source version")

    if finding.span is None:
        if finding.quote is not None:
            raise FindingValidationError("A quote without a span is invalid")
        return

    if finding.span.end > len(document.text):
        raise FindingValidationError("Finding span extends beyond parsed document text")
    actual = document.text[finding.span.start : finding.span.end]
    if actual != finding.quote:
        raise FindingValidationError("Finding quote does not match the text at its span")
