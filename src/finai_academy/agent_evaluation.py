"""Pure contracts for evaluating the public Lesson 11 agent boundary."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from finai_academy.plan_execute_graph import PlanExecuteResult
from finai_academy.research_planning import (
    EvidenceGateResult,
    PlanStep,
    ResearchObservation,
    ResearchPlan,
    TrajectoryEvent,
)

FinalStatus = Literal[
    "completed",
    "plan_blocked",
    "execution_stopped",
    "replan_budget_exhausted",
    "insufficient_evidence",
    "provider_error",
]
FailureStage = Literal[
    "none",
    "planner",
    "tool_boundary",
    "replanner",
    "evidence_gate",
    "report_writer",
    "dataset",
    "judge",
]
MetricName = Literal[
    "tool_call_correctness",
    "tool_call_efficiency",
    "answer_relevance",
    "answer_completeness",
    "citation_integrity",
]

_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer\s+[a-z0-9._-]+|sk-[a-z0-9]{12,})"
)


def _clean_strings(value: Any) -> Any:
    """Strip and validate strings recursively without changing container order."""

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text values must not be blank")
        return cleaned
    if isinstance(value, Mapping):
        return {_clean_strings(key): _clean_strings(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clean_strings(item) for item in value)
    if isinstance(value, list):
        return [_clean_strings(item) for item in value]
    if isinstance(value, set):
        return {_clean_strings(item) for item in value}
    return value


def _reject_secret_shaped_strings(value: Any) -> None:
    """Reject credential-shaped strings recursively in candidate output fields."""

    if isinstance(value, str):
        if _SECRET_PATTERN.search(value):
            raise ValueError("candidate fields must not contain secret-shaped text")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_secret_shaped_strings(key)
            _reject_secret_shaped_strings(item)
        return
    if isinstance(value, (tuple, list, set)):
        for item in value:
            _reject_secret_shaped_strings(item)


class _StrictFrozenModel(BaseModel):
    """Shared validation policy for evaluation data contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    @model_validator(mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> Any:
        return _clean_strings(value)


class ExpectedToolCall(_StrictFrozenModel):
    """One expected tool call and its declared ordering dependencies."""

    call_id: str
    capability: str
    arguments: dict[str, Any]
    prerequisite_call_ids: tuple[str, ...] = ()


class AgentEvaluationCase(_StrictFrozenModel):
    """Versioned expectations for one agent regression case."""

    case_id: str
    mission: str
    expected_final_status: FinalStatus
    expected_tool_calls: tuple[ExpectedToolCall, ...]
    expected_error_codes: tuple[str, ...]
    expected_replan_count: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)
    required_companies: tuple[str, ...]
    required_evidence_ids: tuple[str, ...]
    required_fact_kinds: tuple[Literal["metric", "document"], ...]
    required_limitations: tuple[str, ...]
    allow_briefing: bool


class CandidateFact(_StrictFrozenModel):
    """A candidate claim that may retain invalid provenance for later scoring."""

    claim: str
    provenance_kind: Literal["metric", "document"] | None = None
    source_references: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def reject_secret_shaped_text(cls, value: Any) -> Any:
        _reject_secret_shaped_strings(value)
        return value


class CandidateBriefing(_StrictFrozenModel):
    """A scoreable report that does not claim certified Lesson 11 provenance."""

    reported_facts: tuple[CandidateFact, ...]
    cross_company_observations: tuple[str, ...]
    interpretation: tuple[str, ...]
    limitations: tuple[str, ...]
    source_references: tuple[str, ...]

    @model_validator(mode="before")
    @classmethod
    def reject_secret_shaped_text(cls, value: Any) -> Any:
        _reject_secret_shaped_strings(value)
        return value


class AgentEvaluationPrediction(_StrictFrozenModel):
    """Public, serializable agent output aligned to one evaluation case."""

    case_id: str
    dataset_version: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_id: str
    agent_version: str
    provider: Literal["recorded", "openai", "ollama"]
    agent_model: str
    prompt_version: str
    max_steps: int = Field(ge=1)
    max_replans: int = Field(ge=0)
    status: FinalStatus
    initial_plan: ResearchPlan
    final_steps: tuple[PlanStep, ...]
    observations: tuple[ResearchObservation, ...]
    trajectory: tuple[TrajectoryEvent, ...]
    replan_count: int = Field(ge=0)
    evidence_gate: EvidenceGateResult
    briefing: CandidateBriefing | None


class MetricScore(_StrictFrozenModel):
    """One normalized deterministic score and public rationale."""

    value: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)


class AgentCaseScores(_StrictFrozenModel):
    """All deterministic scores and diagnostics for one case."""

    case_id: str
    configuration_id: str
    tool_call_correctness: MetricScore
    tool_call_efficiency: MetricScore
    answer_relevance: MetricScore
    answer_completeness: MetricScore
    citation_integrity: MetricScore
    failure_stage: FailureStage
    release_passed: bool
    total_tool_calls: int = Field(ge=0)
    redundant_tool_calls: int = Field(ge=0)
    latency_ms: float = Field(ge=0)


class AgentEvaluationSummary(_StrictFrozenModel):
    """Aggregate deterministic results for one aligned configuration."""

    configuration_id: str
    dataset_version: str
    dataset_sha256: str
    case_count: int = Field(ge=1)
    metric_means: dict[MetricName, float]
    metric_pass_counts: dict[MetricName, int]
    mean_tool_calls: float = Field(ge=0)
    mean_latency_ms: float = Field(ge=0)
    max_latency_ms: float = Field(ge=0)
    release_passed: bool


def canonical_call_signature(capability: str, arguments: Mapping[str, Any]) -> str:
    """Return a stable public signature for one capability invocation."""

    cleaned_capability = _clean_strings(capability)
    return (
        f"{cleaned_capability}:"
        f"{json.dumps(arguments, sort_keys=True, separators=(',', ':'))}"
    )


def prediction_from_plan_execute_result(
    result: PlanExecuteResult,
    *,
    case_id: str,
    dataset_version: str,
    dataset_sha256: str,
    configuration_id: str,
    agent_version: str,
    provider: Literal["recorded", "openai", "ollama"],
    agent_model: str,
    prompt_version: str,
    max_steps: int,
    max_replans: int,
) -> AgentEvaluationPrediction:
    """Convert a certified Lesson 11 result into the permissive evaluation boundary."""

    briefing = None
    if result.briefing is not None:
        briefing = CandidateBriefing(
            reported_facts=tuple(
                CandidateFact(
                    claim=fact.claim,
                    provenance_kind=fact.provenance_kind,
                    source_references=fact.source_references,
                    evidence_ids=fact.evidence_ids,
                )
                for fact in result.briefing.reported_facts
            ),
            cross_company_observations=result.briefing.cross_company_observations,
            interpretation=result.briefing.interpretation,
            limitations=result.briefing.limitations,
            source_references=result.briefing.source_references,
        )
    return AgentEvaluationPrediction(
        case_id=case_id,
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        configuration_id=configuration_id,
        agent_version=agent_version,
        provider=provider,
        agent_model=agent_model,
        prompt_version=prompt_version,
        max_steps=max_steps,
        max_replans=max_replans,
        status=result.status,
        initial_plan=result.initial_plan,
        final_steps=result.final_steps,
        observations=result.observations,
        trajectory=result.trajectory,
        replan_count=result.replan_count,
        evidence_gate=result.evidence_gate,
        briefing=briefing,
    )
