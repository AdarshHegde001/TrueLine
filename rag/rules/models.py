"""Typed deterministic rule inputs, mapped from approved ingestion records."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rag.models import Severity, SourceReference


class TermRuleMode(str, Enum):
    BANNED = "banned"
    REPLACE_WITH_PREFERRED = "replace_with_preferred"


class TermRule(BaseModel):
    """A literal term rule; prose guidelines are not auto-converted to this type."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    mode: TermRuleMode
    term: str = Field(min_length=1)
    replacement: str | None = None
    case_sensitive: bool = False
    whole_word: bool = True
    severity: Severity = Severity.MEDIUM
    applies_to_channels: tuple[str, ...] = ()
    source: SourceReference

    @model_validator(mode="after")
    def term_is_not_whitespace(self) -> "TermRule":
        if not self.term.strip():
            raise ValueError("term cannot contain only whitespace")
        return self

    @model_validator(mode="after")
    def replacement_required_for_preferred_rule(self) -> "TermRule":
        if self.mode == TermRuleMode.REPLACE_WITH_PREFERRED and not self.replacement:
            raise ValueError("replace_with_preferred rules require a replacement")
        return self


class RequiredTextRule(BaseModel):
    """Exact required phrase/disclaimer, with whitespace-insensitive matching."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    phrase: str = Field(min_length=1)
    case_sensitive: bool = False
    applies_to_channels: tuple[str, ...] = ()
    source: SourceReference

    @model_validator(mode="after")
    def phrase_is_not_whitespace(self) -> "RequiredTextRule":
        if not self.phrase.strip():
            raise ValueError("required phrase cannot contain only whitespace")
        return self


class DeterministicRuleSet(BaseModel):
    model_config = ConfigDict(frozen=True)

    terminology: tuple[TermRule, ...] = ()
    required_text: tuple[RequiredTextRule, ...] = ()
