"""Composable deterministic rule engine; no LLM or database dependencies."""

from __future__ import annotations

from rag.models import ParsedDocument, RuleCheckResult
from rag.rules.models import DeterministicRuleSet
from rag.rules.required_text import check_required_text
from rag.rules.terminology import check_terminology
from rag.validation import validate_finding_span


class DeterministicRuleEngine:
    """Run registered deterministic checks against a parsed document."""

    def run(
        self,
        document: ParsedDocument,
        rules: DeterministicRuleSet,
        *,
        channel: str | None = None,
    ) -> RuleCheckResult:
        terminology_rules = tuple(
            rule
            for rule in rules.terminology
            if not rule.applies_to_channels or channel in rule.applies_to_channels
        )
        required_text_rules = tuple(
            rule
            for rule in rules.required_text
            if not rule.applies_to_channels or channel in rule.applies_to_channels
        )
        findings = (
            *check_terminology(document, terminology_rules, channel=channel),
            *check_required_text(document, required_text_rules, channel=channel),
        )
        for finding in findings:
            validate_finding_span(finding, document)
        checks_run = tuple(
            check_name
            for check_name, configured in (
                ("terminology", bool(terminology_rules)),
                ("required_text", bool(required_text_rules)),
            )
            if configured
        )
        skipped_names: list[str] = []
        if len(terminology_rules) != len(rules.terminology):
            skipped_names.append("channel_scoped_terminology")
        if len(required_text_rules) != len(rules.required_text):
            skipped_names.append("channel_scoped_required_text")
        skipped_checks = tuple(skipped_names)
        warnings = (
            ("Some channel-scoped rules were skipped because their channel did not match the review request.",)
            if skipped_checks
            else ()
        )
        return RuleCheckResult(
            findings=findings,
            checks_run=checks_run,
            skipped_checks=skipped_checks,
            warnings=warnings,
        )
