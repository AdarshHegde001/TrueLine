"""Deterministic terminology checks over parsed source segments."""

from __future__ import annotations

import re

from rag.models import Finding, FindingLabel, ParsedDocument, TextSpan
from rag.rules.models import TermRule, TermRuleMode


def check_terminology(
    document: ParsedDocument,
    rules: tuple[TermRule, ...],
    *,
    channel: str | None = None,
) -> tuple[Finding, ...]:
    findings: list[Finding] = []
    for rule in rules:
        if rule.applies_to_channels and channel not in rule.applies_to_channels:
            continue

        flags = 0 if rule.case_sensitive else re.IGNORECASE
        literal = re.escape(rule.term)
        pattern = rf"(?<!\w){literal}(?!\w)" if rule.whole_word else literal
        matches = [
            (segment.start + match.start(), segment.start + match.end(), match.group(0))
            for segment in document.segments
            for match in re.finditer(pattern, segment.text, flags=flags)
        ]

        if not matches:
            findings.append(
                Finding(
                    detector="terminology_rule",
                    rule_id=rule.rule_id,
                    label=FindingLabel.PASSED,
                    source=rule.source,
                    explanation=f"The prohibited variant '{rule.term}' was not found.",
                )
            )
            continue

        for start, end, quote in matches:
            replacement = (
                rule.replacement
                if rule.mode == TermRuleMode.REPLACE_WITH_PREFERRED
                else None
            )
            action = (
                f"Use '{replacement}' instead."
                if replacement
                else "Remove or replace this banned term."
            )
            findings.append(
                Finding(
                    detector="terminology_rule",
                    rule_id=rule.rule_id,
                    label=FindingLabel.VIOLATION,
                    severity=rule.severity,
                    span=TextSpan(start=start, end=end),
                    quote=quote,
                    source=rule.source,
                    explanation=f"'{rule.term}' violates the approved terminology rule. {action}",
                    replacement=replacement,
                )
            )
    return tuple(findings)
