"""Typed, deterministic contracts for Lesson 11 financial research planning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class PlannerToolSpec(BaseModel):
    """Planner-safe metadata for one discovered and allowlisted tool."""

    name: str
    description: str
    input_schema: dict[str, Any]


class PlanStep(BaseModel):
    """One validated unit of financial research work."""

    step_id: int = Field(ge=1)
    capability: str = Field(min_length=1)
    arguments: dict[str, Any]
    purpose: str = Field(min_length=1)
    expected_evidence: tuple[str, ...]
    depends_on: tuple[int, ...] = ()


class ResearchPlan(BaseModel):
    """A proposed ordered research plan before host policy validation."""

    goal: str = Field(min_length=1)
    steps: tuple[PlanStep, ...]


class ReplanDecision(BaseModel):
    """A typed decision about the remaining research work."""

    action: Literal["continue", "replace_remaining", "finish", "stop"]
    reasoning: str = Field(min_length=1)
    replacement_steps: tuple[PlanStep, ...] = ()
    limitations: tuple[str, ...] = ()


class ResearchObservation(BaseModel):
    """Immutable result of one attempted plan step."""

    attempt_id: int = Field(ge=1)
    step_id: int = Field(ge=1)
    plan_revision: int = Field(ge=0)
    capability: str
    arguments: dict[str, Any]
    status: Literal["ok", "error", "blocked"]
    result: dict[str, Any] | None = None
    error_code: str | None = None
    evidence_ids: tuple[str, ...] = ()
    source_references: tuple[str, ...] = ()
    duration_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_outcome(self) -> ResearchObservation:
        if self.status == "ok" and self.result is None:
            raise ValueError("ok observation requires result")
        if self.status == "error" and (not self.error_code or not self.error_code.strip()):
            raise ValueError("error observation requires error_code")
        return self


class TrajectoryEvent(BaseModel):
    """Safe, displayable record of a state transition in the research run."""

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
    step_id: int | None = None
    attempt_id: int | None = None
    duration_ms: float = Field(default=0, ge=0)


class AnalystBriefing(BaseModel):
    """Report sections that keep facts, comparison, interpretation, and limits separate."""

    reported_facts: tuple[str, ...]
    cross_company_observations: tuple[str, ...]
    interpretation: tuple[str, ...]
    limitations: tuple[str, ...]
    source_references: tuple[str, ...]

    @model_validator(mode="after")
    def validate_report_support(self) -> AnalystBriefing:
        if not self.reported_facts:
            raise ValueError("reported_facts must not be empty")
        if not self.limitations:
            raise ValueError("limitations must not be empty")
        if not self.source_references:
            raise ValueError("source_references must not be empty")
        return self


class EvidenceGateResult(BaseModel):
    """Evidence coverage and the requirements that prevent reporting."""

    passed: bool
    coverage: dict[str, tuple[str, ...]]
    missing_requirements: tuple[str, ...] = ()


def validate_plan(
    plan: ResearchPlan,
    catalog: Sequence[PlannerToolSpec],
    max_steps: int,
) -> ResearchPlan:
    """Validate an initial plan against the discovered, permitted tool catalog."""

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not plan.steps or tuple(step.step_id for step in plan.steps) != tuple(
        range(1, len(plan.steps) + 1)
    ):
        raise ValueError("initial_step_ids must begin at 1 and increase sequentially")
    if len(plan.steps) > max_steps:
        raise ValueError(f"step_budget_exceeded: maximum is {max_steps}")

    catalog_by_name = _catalog_by_name(catalog)
    known_step_ids = {step.step_id for step in plan.steps}
    for step in plan.steps:
        _validate_step(step, catalog_by_name)
        _validate_dependencies(step, known_step_ids, prior_ids={item for item in known_step_ids if item < step.step_id})
    return plan


def validate_replacement(
    replacement: Sequence[PlanStep],
    *,
    catalog: Sequence[PlannerToolSpec],
    prior_step_ids: Sequence[int],
    successful_step_ids: Sequence[int],
    max_total_steps: int,
) -> tuple[PlanStep, ...]:
    """Validate a replacement tail while preserving the executed plan prefix."""

    if max_total_steps < 1:
        raise ValueError("max_total_steps must be positive")
    prior_ids = tuple(prior_step_ids)
    if len(prior_ids) + len(replacement) > max_total_steps:
        raise ValueError(f"step_budget_exceeded: maximum is {max_total_steps}")

    previous_max = max(prior_ids, default=0)
    replacement_ids = tuple(step.step_id for step in replacement)
    if any(step_id <= previous_max for step_id in replacement_ids) or replacement_ids != tuple(
        sorted(set(replacement_ids))
    ):
        raise ValueError("replacement_step_ids must be new and strictly increasing")

    catalog_by_name = _catalog_by_name(catalog)
    allowed_dependencies = set(successful_step_ids)
    for step in replacement:
        _validate_step(step, catalog_by_name)
        _validate_dependencies(
            step,
            set(prior_ids) | set(replacement_ids),
            prior_ids=allowed_dependencies,
        )
        allowed_dependencies.add(step.step_id)
    return tuple(replacement)


def evaluate_evidence_gate(
    observations: Sequence[ResearchObservation],
) -> EvidenceGateResult:
    """Require successful metric and document evidence for both maintained companies."""

    required_companies = ("NVIDIA", "Schneider Electric")
    evidence: dict[str, set[str]] = {company: set() for company in required_companies}
    for observation in observations:
        if observation.status != "ok" or observation.result is None:
            continue
        company = observation.result.get("company")
        if company not in evidence:
            continue
        if observation.capability == "get_company_metric":
            evidence[company].add("metric")
        elif observation.capability == "search_financial_documents" and observation.result.get("hits"):
            evidence[company].add("document")

    coverage = {
        company: tuple(kind for kind in ("document", "metric") if kind in evidence[company])
        for company in required_companies
    }
    missing = tuple(
        f"{company} {kind} evidence"
        for company in required_companies
        for kind in ("metric", "document")
        if kind not in evidence[company]
    )
    return EvidenceGateResult(
        passed=not missing,
        coverage=coverage,
        missing_requirements=missing,
    )


def _catalog_by_name(catalog: Sequence[PlannerToolSpec]) -> dict[str, PlannerToolSpec]:
    return {tool.name: tool for tool in catalog}


def _validate_step(step: PlanStep, catalog: Mapping[str, PlannerToolSpec]) -> None:
    tool = catalog.get(step.capability)
    if tool is None:
        raise ValueError(f"capability_not_permitted: {step.capability}")
    _validate_arguments(step.arguments, tool.input_schema)
    if not step.purpose.strip():
        raise ValueError(f"purpose_not_descriptive: step {step.step_id}")
    if not step.expected_evidence or any(not label.strip() for label in step.expected_evidence):
        raise ValueError(f"expected_evidence_not_descriptive: step {step.step_id}")


def _validate_dependencies(
    step: PlanStep,
    known_ids: set[int],
    *,
    prior_ids: set[int],
) -> None:
    if any(dependency not in known_ids or dependency not in prior_ids for dependency in step.depends_on):
        raise ValueError(f"dependencies_not_prior: step {step.step_id}")


def _validate_arguments(arguments: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    _validate_schema_value(dict(arguments), schema, "arguments")


def _validate_schema_value(value: Any, schema: Mapping[str, Any], path: str) -> None:
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(value, Mapping):
            raise ValueError(f"arguments_not_accepted: {path} must be an object")
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            raise ValueError(f"arguments_not_accepted: invalid properties schema at {path}")
        required = schema.get("required", ())
        if not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
            raise ValueError(f"arguments_not_accepted: invalid required schema at {path}")
        for name in required:
            if name not in value:
                raise ValueError(f"arguments_not_accepted: missing required argument {name}")
        additional_properties = schema.get("additionalProperties")
        if additional_properties is not None and not isinstance(
            additional_properties, (bool, Mapping)
        ):
            raise ValueError(
                f"arguments_not_accepted: invalid additionalProperties schema at {path}"
            )
        if additional_properties is False:
            unknown = set(value) - set(properties)
            if unknown:
                raise ValueError(
                    f"arguments_not_accepted: additional properties {sorted(unknown)!r}"
                )
        for name, item in value.items():
            if name in properties:
                child_schema = properties[name]
                if not isinstance(child_schema, Mapping):
                    raise ValueError(f"arguments_not_accepted: invalid schema for {path}.{name}")
                _validate_schema_value(item, child_schema, f"{path}.{name}")
            elif isinstance(additional_properties, Mapping):
                _validate_schema_value(item, additional_properties, f"{path}.{name}")
    elif expected_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"arguments_not_accepted: {path} must be an array")
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, f"{path}[{index}]")
    elif expected_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"arguments_not_accepted: {path} must be a string")
    elif expected_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"arguments_not_accepted: {path} must be an integer")
    elif expected_type == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"arguments_not_accepted: {path} must be a number")
    elif expected_type not in (None, "object", "array", "string", "integer", "number"):
        raise ValueError(f"arguments_not_accepted: unsupported schema type {expected_type!r}")

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        raise ValueError(f"arguments_not_accepted: {path} is not an allowed value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < minimum:
            raise ValueError(f"arguments_not_accepted: {path} is below minimum")
        if maximum is not None and value > maximum:
            raise ValueError(f"arguments_not_accepted: {path} is above maximum")
