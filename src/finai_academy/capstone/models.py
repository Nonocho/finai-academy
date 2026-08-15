"""Typed domain contracts for the Financial Analyst Copilot."""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceType(StrEnum):
    """How a statement relates to its supporting information."""

    REPORTED_FACT = "reported_fact"
    CALCULATION = "calculation"
    MANAGEMENT_CLAIM = "management_claim"
    EXTERNAL_FACT = "external_fact"
    INTERPRETATION = "interpretation"


class FindingCategory(StrEnum):
    """The role of a finding inside an analyst brief."""

    KEY_RESULT = "key_result"
    CATALYST = "catalyst"
    RISK = "risk"


class AnalystFinding(BaseModel):
    """One material statement and the evidence classification assigned to it."""

    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1)
    category: FindingCategory
    evidence_type: EvidenceType
    source_excerpt: str | None = Field(
        default=None,
        description="A short exact excerpt from the supplied source when available.",
    )
    rationale: str | None = Field(
        default=None,
        description="Why the item matters; required in spirit for interpretations.",
    )

    @model_validator(mode="after")
    def enforce_evidence_requirements(self) -> Self:
        if (
            self.evidence_type in {EvidenceType.REPORTED_FACT, EvidenceType.MANAGEMENT_CLAIM}
            and not (self.source_excerpt or "").strip()
        ):
            raise ValueError(
                "source_excerpt is required for reported facts and management claims"
            )
        if self.evidence_type == EvidenceType.INTERPRETATION and not (
            self.rationale or ""
        ).strip():
            raise ValueError("rationale is required for interpretations")
        return self


class AnalystBrief(BaseModel):
    """Validated output of the first Financial Analyst Copilot vertical slice."""

    model_config = ConfigDict(extra="forbid")

    company: str = Field(min_length=1)
    reporting_period: str = Field(min_length=1)
    executive_summary: str = Field(min_length=1)
    findings: list[AnalystFinding] = Field(min_length=1)
    open_questions: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
