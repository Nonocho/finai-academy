"""Typed domain contracts for the Financial Analyst Copilot."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from finai_academy.research_planning import PlanStep, ResearchObservation

_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer\s+[a-z0-9._-]+|sk-[a-z0-9]{12,})"
)
_PERSONAL_PATH_PATTERN = re.compile(r"(?i)(?:^|[^A-Za-z0-9])/(?:Users|home)/")
_METRIC_NAMES = (
    "tool_call_correctness",
    "tool_call_efficiency",
    "answer_relevance",
    "answer_completeness",
    "citation_integrity",
)


def _clean_public_value(value: Any) -> Any:
    """Normalize public strings and reject unsafe or non-serializable state."""

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("public text values must not be blank")
        if _SECRET_PATTERN.search(cleaned):
            raise ValueError("public fields must not contain credential-shaped text")
        if _PERSONAL_PATH_PATTERN.search(cleaned):
            raise ValueError("public fields must not contain personal filesystem paths")
        return cleaned
    if isinstance(value, BaseModel):
        _clean_public_value(value.model_dump(mode="python"))
        return value
    if isinstance(value, dict):
        return {_clean_public_value(key): _clean_public_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clean_public_value(item) for item in value)
    if isinstance(value, list):
        return [_clean_public_value(item) for item in value]
    if isinstance(value, set):
        return {_clean_public_value(item) for item in value}
    if value is None or isinstance(value, (bool, int, float, StrEnum)):
        return value
    raise ValueError("public fields must contain JSON-compatible values")


class _FrozenPublicModel(BaseModel):
    """Shared policy for immutable, serializable, safe capstone state."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    @model_validator(mode="before")
    @classmethod
    def clean_public_state(cls, value: Any) -> Any:
        return _clean_public_value(value)


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
            raise ValueError("source_excerpt is required for reported facts and management claims")
        if self.evidence_type == EvidenceType.INTERPRETATION and not (self.rationale or "").strip():
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


class ResearchMode(StrEnum):
    """Whether a run uses the certified classroom mission or a custom question."""

    REFERENCE = "reference"
    CUSTOM = "custom"


class CapstoneProvider(StrEnum):
    """Explicit model-provider choices exposed by the capstone."""

    RECORDED = "recorded"
    OLLAMA = "ollama"
    OPENAI = "openai"


class DataMode(StrEnum):
    """Whether evidence is certified or includes clearly labeled live enrichment."""

    CERTIFIED = "certified"
    LIVE_ENRICHMENT = "live_enrichment"


class RunStatus(StrEnum):
    """Terminal states for a bounded capstone research run."""

    COMPLETED = "completed"
    PLAN_BLOCKED = "plan_blocked"
    EXECUTION_STOPPED = "execution_stopped"
    REPLAN_BUDGET_EXHAUSTED = "replan_budget_exhausted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROVIDER_ERROR = "provider_error"


class ResearchRequest(_FrozenPublicModel):
    """The complete, safe input boundary for one capstone research run."""

    mode: ResearchMode
    question: str = Field(min_length=1)
    companies: tuple[str, ...] = Field(min_length=1)
    provider: CapstoneProvider
    model: str = Field(min_length=1)
    data_mode: DataMode
    include_news: bool = False
    max_steps: int = Field(default=6, ge=1, le=6)
    max_replans: int = Field(default=1, ge=0, le=1)

    @classmethod
    def reference(cls, **overrides: Any) -> ResearchRequest:
        """Load the versioned recorded mission rather than duplicating it in callers."""

        fixture_path = (
            Path(__file__).resolve().parents[3] / "final-project/shared/reference_mission.json"
        )
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        payload = {
            "mode": ResearchMode.REFERENCE,
            "question": fixture["mission"],
            "companies": tuple(fixture["companies"]),
            "provider": fixture["provider"],
            "model": "recorded-capstone-v1",
            "data_mode": fixture["data_mode"],
            "include_news": False,
            "max_steps": fixture["max_steps"],
            "max_replans": fixture["max_replans"],
        }
        payload.update(overrides)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_request_limits(self) -> ResearchRequest:
        if self.mode == ResearchMode.REFERENCE and self.companies != (
            "NVIDIA",
            "Schneider Electric",
        ):
            raise ValueError("reference mode requires exactly NVIDIA and Schneider Electric")
        if self.include_news and self.data_mode != DataMode.LIVE_ENRICHMENT:
            raise ValueError("include_news requires data_mode='live_enrichment'")
        return self


class CapstoneEvidenceHit(_FrozenPublicModel):
    """A company-filtered, source-addressable public retrieval hit."""

    company: str = Field(min_length=1)
    text: str = Field(min_length=1)
    evidence_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    section: str = Field(min_length=1)
    period: str = Field(min_length=1)
    source_reference: str = Field(min_length=1)


class CitedFact(_FrozenPublicModel):
    """A report fact with one metric or document provenance reference."""

    claim: str = Field(min_length=1)
    company: str = Field(min_length=1)
    provenance_kind: Literal["metric", "document"]
    source_reference: str = Field(min_length=1)
    evidence_id: str | None = None

    @model_validator(mode="after")
    def validate_provenance(self) -> CitedFact:
        if self.provenance_kind == "document" and self.evidence_id is None:
            raise ValueError("document fact requires evidence_id")
        if self.provenance_kind == "metric" and self.evidence_id is not None:
            raise ValueError("metric fact forbids evidence_id")
        return self


class CapstoneBriefing(_FrozenPublicModel):
    """Cited report sections kept separate from interpretation and limitations."""

    executive_summary: str = Field(min_length=1)
    cited_facts: tuple[CitedFact, ...] = Field(min_length=1)
    company_evidence: dict[str, tuple[CapstoneEvidenceHit, ...]] = Field(min_length=1)
    cross_company_observations: tuple[str, ...] = Field(min_length=1)
    interpretation: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    open_questions: tuple[str, ...] = Field(default_factory=tuple)
    aggregate_sources: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_aggregate_sources(self) -> CapstoneBriefing:
        expected_sources = tuple(dict.fromkeys(fact.source_reference for fact in self.cited_facts))
        if self.aggregate_sources != expected_sources:
            raise ValueError("aggregate_sources must match cited facts in first-use order")
        return self


class EvidenceGateDecision(_FrozenPublicModel):
    """Public evidence-coverage decision made before any briefing is exposed."""

    passed: bool
    coverage: dict[str, tuple[Literal["document", "metric"], ...]]
    missing_requirements: tuple[str, ...] = ()


class PublicTraceEvent(_FrozenPublicModel):
    """One sanitized trajectory event safe for logs, persistence, and the UI."""

    index: int = Field(ge=1)
    phase: Literal[
        "planning",
        "policy",
        "execution",
        "replanning",
        "evidence_gate",
        "report",
        "guardrail",
    ]
    status: Literal["ok", "error", "blocked"]
    summary: str = Field(min_length=1)
    step_id: int | None = Field(default=None, ge=1)
    attempt_id: int | None = Field(default=None, ge=1)
    plan_revision: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0, ge=0)
    error_code: str | None = None
    failure_owner: str | None = None


class MetricEvaluation(_FrozenPublicModel):
    """One deterministic release metric and a public rationale."""

    name: Literal[
        "tool_call_correctness",
        "tool_call_efficiency",
        "answer_relevance",
        "answer_completeness",
        "citation_integrity",
    ]
    value: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class DeterministicEvaluation(_FrozenPublicModel):
    """The fixed five-metric release scorecard."""

    metrics: tuple[MetricEvaluation, ...] = Field(min_length=5, max_length=5)
    release_passed: bool

    @model_validator(mode="after")
    def validate_metric_set(self) -> DeterministicEvaluation:
        if tuple(metric.name for metric in self.metrics) != _METRIC_NAMES:
            raise ValueError("metrics must contain the five deterministic metrics in fixed order")
        return self


class JudgeEvaluation(_FrozenPublicModel):
    """Optional judge result, deliberately separate from deterministic release."""

    status: Literal["not_run", "passed", "failed", "unavailable"] = "not_run"
    summary: str = Field(default="No judge evaluation was run.", min_length=1)
    score: float | None = Field(default=None, ge=0, le=1)


class ResearchRunResult(_FrozenPublicModel):
    """The complete public result of one bounded Financial Analyst Copilot run."""

    run_id: str = Field(min_length=1)
    request: ResearchRequest
    provider: CapstoneProvider
    model: str = Field(min_length=1)
    data_mode: DataMode
    status: RunStatus
    initial_plan: tuple[PlanStep, ...]
    final_plan: tuple[PlanStep, ...]
    observations: tuple[ResearchObservation, ...]
    trajectory: tuple[PublicTraceEvent, ...]
    evidence_gate: EvidenceGateDecision
    briefing: CapstoneBriefing | None = None
    deterministic_evaluation: DeterministicEvaluation
    judge_evaluation: JudgeEvaluation | None = None
    mlflow_run_id: str | None = None
    mlflow_trace_id: str | None = None
    replan_count: int = Field(ge=0, le=1)
    total_duration_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_run_state(self) -> ResearchRunResult:
        if self.status == RunStatus.COMPLETED and (
            not self.evidence_gate.passed or self.briefing is None
        ):
            raise ValueError("completed run requires a passing evidence gate and briefing")
        if not self.evidence_gate.passed and self.briefing is not None:
            raise ValueError("failed evidence gate cannot expose a briefing")
        if self.deterministic_evaluation.release_passed != all(
            metric.value == 1.0 for metric in self.deterministic_evaluation.metrics
        ):
            raise ValueError("release decision must match deterministic metrics")
        return self
