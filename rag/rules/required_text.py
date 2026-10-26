"""Deterministic checks for required disclaimers or phrases."""

from __future__ import annotations

import re

from rag.models import Finding, FindingLabel, ParsedDocument
from rag.rules.models import RequiredTextRule


def _phrase_pattern(phrase: str, *, case_sensitive: bool) -> re.Pattern[str]:
    pieces = re.split(r"\s+", phrase.strip())
    literal = r"\s+".join(re.escape(piece) for piece in pieces)
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(literal, flags=flags)


def check_required_text(
    document: ParsedDocument,
    rules: tuple[RequiredTextRule, ...],
    *,
    channel: str | None = None,
) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    for rule in rules:
        if rule.applies_to_channels and channel not in rule.applies_to_channels:
            continue

        pattern = _phrase_pattern(rule.phrase, case_sensitive=rule.case_sensitive)
        match = pattern.search(document.text)
        if match:
            findings.append(
                Finding(
                    detector="required_text_rule",
                    rule_id=rule.rule_id,
                    label=FindingLabel.PASSED,
                    source=rule.source,
                    explanation="The required text is present.",
                )
            )
        else:
            findings.append(
                Finding(
                    detector="required_text_rule",
                    rule_id=rule.rule_id,
                    label=FindingLabel.VIOLATION,
                    source=rule.source,
                    explanation=f"Required text is missing: {rule.phrase}",
                )
            )
    return tuple(findings)
