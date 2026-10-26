"""Map approved structured ingestion records into typed check rules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rag.models import SourceReference
from rag.rules.models import (
    DeterministicRuleSet,
    RequiredTextRule,
    TermRule,
    TermRuleMode,
)


def rules_from_ingestion_records(
    records: Iterable[Mapping[str, Any]],
) -> DeterministicRuleSet:
    """Build check rules from approved records returned by ingestion.

    Each record carries structured rule fields plus the approved source/version
    fields required by the ingestion contract. Invalid or unknown rule types fail
    closed by raising a validation/value error; they are never silently ignored.
    """

    terminology: list[TermRule] = []
    required_text: list[RequiredTextRule] = []

    for record in records:
        kind = record.get("kind")
        source = SourceReference(
            source_id=record["source_id"],
            version_id=record["version_id"],
            source_type=record["source_type"],
            excerpt=record["source_excerpt"],
            title=record.get("source_title"),
        )
        shared = {
            "rule_id": record["rule_id"],
            "applies_to_channels": tuple(record.get("applies_to_channels", ())),
            "source": source,
        }
        if kind == "terminology":
            terminology.append(
                TermRule(
                    **shared,
                    mode=TermRuleMode(record["mode"]),
                    term=record["term"],
                    replacement=record.get("replacement"),
                    case_sensitive=record.get("case_sensitive", False),
                    whole_word=record.get("whole_word", True),
                    severity=record.get("severity", "medium"),
                )
            )
        elif kind == "required_text":
            required_text.append(
                RequiredTextRule(
                    **shared,
                    phrase=record["phrase"],
                    case_sensitive=record.get("case_sensitive", False),
                )
            )
        else:
            raise ValueError(f"Unsupported deterministic rule kind: {kind!r}")

    return DeterministicRuleSet(
        terminology=tuple(terminology),
        required_text=tuple(required_text),
    )
