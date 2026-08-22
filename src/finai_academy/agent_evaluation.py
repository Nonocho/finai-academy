"""Pure contracts for evaluating the public Lesson 11 agent boundary."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from hashlib import sha256
from hmac import compare_digest
from pathlib import Path
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
METRIC_NAMES: tuple[MetricName, ...] = (
    "tool_call_correctness",
    "tool_call_efficiency",
    "answer_relevance",
    "answer_completeness",
    "citation_integrity",
)

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
    """Reject credential-shaped strings recursively in public evaluation fields."""

    if isinstance(value, str):
        if _SECRET_PATTERN.search(value):
            raise ValueError("evaluation fields must not contain secret-shaped text")
        return
    if isinstance(value, BaseModel):
        _reject_secret_shaped_strings(value.model_dump(mode="python"))
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

    @model_validator(mode="before")
    @classmethod
    def reject_secret_shaped_text(cls, value: Any) -> Any:
        _reject_secret_shaped_strings(value)
        return value


class AgentEvaluationDataset(BaseModel):
    """One verified version of agent expectations."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1]
    dataset_version: Literal["agent-cases-v1"]
    scorer_contract_version: Literal["agent-scorers-v1"]
    dataset_sha256: str
    cases: tuple[AgentEvaluationCase, ...]


class RecordedAgentConfiguration(BaseModel):
    """One complete recorded configuration over the evaluation dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    configuration_id: Literal["bounded-agent-v1", "regressed-agent-v0"]
    agent_version: str
    provider: Literal["recorded"]
    agent_model: str
    prompt_version: str
    max_steps: int = Field(ge=1)
    max_replans: int = Field(ge=0)
    predictions: tuple[AgentEvaluationPrediction, ...]


class RecordedAgentRuns(BaseModel):
    """Two aligned recorded configurations for the offline lesson."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1]
    fixture_version: Literal["agent-runs-v1"]
    dataset_version: Literal["agent-cases-v1"]
    dataset_sha256: str
    configurations: tuple[RecordedAgentConfiguration, ...]


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


_CASE_KEYS = {
    "case_id",
    "mission",
    "expected_final_status",
    "expected_tool_calls",
    "expected_error_codes",
    "expected_replan_count",
    "max_tool_calls",
    "required_companies",
    "required_evidence_ids",
    "required_fact_kinds",
    "required_limitations",
    "allow_briefing",
}
_EXPECTED_CALL_KEYS = {
    "call_id",
    "capability",
    "arguments",
    "prerequisite_call_ids",
}
_CONFIGURATION_KEYS = {
    "configuration_id",
    "agent_version",
    "provider",
    "agent_model",
    "prompt_version",
    "max_steps",
    "max_replans",
    "predictions",
}
_PREDICTION_KEYS = {
    "case_id",
    "dataset_version",
    "dataset_sha256",
    "configuration_id",
    "agent_version",
    "provider",
    "agent_model",
    "prompt_version",
    "max_steps",
    "max_replans",
    "status",
    "initial_plan",
    "final_steps",
    "observations",
    "trajectory",
    "replan_count",
    "evidence_gate",
    "briefing",
}
_PLAN_KEYS = {"goal", "steps"}
_STEP_KEYS = {
    "step_id",
    "capability",
    "arguments",
    "purpose",
    "expected_evidence",
    "depends_on",
}
_OBSERVATION_KEYS = {
    "attempt_id",
    "step_id",
    "plan_revision",
    "capability",
    "arguments",
    "status",
    "result",
    "error_code",
    "evidence_ids",
    "source_references",
    "duration_ms",
}
_TRAJECTORY_KEYS = {
    "index",
    "phase",
    "status",
    "summary",
    "step_id",
    "attempt_id",
    "duration_ms",
}
_EVIDENCE_GATE_KEYS = {"passed", "coverage", "missing_requirements"}
_BRIEFING_KEYS = {
    "reported_facts",
    "cross_company_observations",
    "interpretation",
    "limitations",
    "source_references",
}
_FACT_KEYS = {
    "claim",
    "provenance_kind",
    "source_references",
    "evidence_ids",
}


def _object(value: object, *, path: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be an object")
    unknown = set(value) - keys
    if unknown:
        names = ", ".join(sorted(str(item) for item in unknown))
        raise ValueError(f"unknown field at {path}: {names}")
    return value


def _array(value: object, *, path: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be an array")
    return value


def _screen_step(value: object, *, path: str) -> None:
    step = _object(value, path=path, keys=_STEP_KEYS)
    if not isinstance(step.get("arguments"), dict):
        raise TypeError(f"{path}.arguments must be an object")
    _array(step.get("expected_evidence"), path=f"{path}.expected_evidence")
    _array(step.get("depends_on"), path=f"{path}.depends_on")


def _screen_prediction(value: object, *, path: str) -> None:
    prediction = _object(value, path=path, keys=_PREDICTION_KEYS)
    plan = _object(prediction.get("initial_plan"), path=f"{path}.initial_plan", keys=_PLAN_KEYS)
    for index, step in enumerate(_array(plan.get("steps"), path=f"{path}.initial_plan.steps")):
        _screen_step(step, path=f"{path}.initial_plan.steps[{index}]")
    for index, step in enumerate(_array(prediction.get("final_steps"), path=f"{path}.final_steps")):
        _screen_step(step, path=f"{path}.final_steps[{index}]")
    for index, raw_observation in enumerate(
        _array(prediction.get("observations"), path=f"{path}.observations")
    ):
        observation_path = f"{path}.observations[{index}]"
        observation = _object(raw_observation, path=observation_path, keys=_OBSERVATION_KEYS)
        arguments = observation.get("arguments")
        if not isinstance(arguments, dict):
            raise TypeError(f"{observation_path}.arguments must be an object")
        result = observation.get("result")
        if result is not None and not isinstance(result, dict):
            raise ValueError(f"{observation_path}.result must be an object or null")
        _array(observation.get("evidence_ids"), path=f"{observation_path}.evidence_ids")
        _array(
            observation.get("source_references"),
            path=f"{observation_path}.source_references",
        )
    for index, event in enumerate(_array(prediction.get("trajectory"), path=f"{path}.trajectory")):
        _object(event, path=f"{path}.trajectory[{index}]", keys=_TRAJECTORY_KEYS)
    gate = _object(
        prediction.get("evidence_gate"),
        path=f"{path}.evidence_gate",
        keys=_EVIDENCE_GATE_KEYS,
    )
    if not isinstance(gate.get("coverage"), dict):
        raise TypeError(f"{path}.evidence_gate.coverage must be an object")
    _array(
        gate.get("missing_requirements"),
        path=f"{path}.evidence_gate.missing_requirements",
    )
    briefing_value = prediction.get("briefing")
    if briefing_value is None:
        return
    briefing = _object(briefing_value, path=f"{path}.briefing", keys=_BRIEFING_KEYS)
    for index, fact in enumerate(
        _array(briefing.get("reported_facts"), path=f"{path}.briefing.reported_facts")
    ):
        _object(
            fact,
            path=f"{path}.briefing.reported_facts[{index}]",
            keys=_FACT_KEYS,
        )
    for field_name in (
        "cross_company_observations",
        "interpretation",
        "limitations",
        "source_references",
    ):
        _array(briefing.get(field_name), path=f"{path}.briefing.{field_name}")


def _screen_cases_payload(value: object) -> dict[str, object]:
    payload = _object(
        value,
        path="dataset",
        keys={"schema_version", "dataset_version", "scorer_contract_version", "cases"},
    )
    for case_index, raw_case in enumerate(_array(payload.get("cases"), path="dataset.cases")):
        case_path = f"dataset.cases[{case_index}]"
        case = _object(raw_case, path=case_path, keys=_CASE_KEYS)
        for field_name in (
            "expected_error_codes",
            "required_companies",
            "required_evidence_ids",
            "required_fact_kinds",
            "required_limitations",
        ):
            _array(case.get(field_name), path=f"{case_path}.{field_name}")
        for call_index, raw_call in enumerate(
            _array(case.get("expected_tool_calls"), path=f"{case_path}.expected_tool_calls")
        ):
            call_path = f"{case_path}.expected_tool_calls[{call_index}]"
            call = _object(raw_call, path=call_path, keys=_EXPECTED_CALL_KEYS)
            if not isinstance(call.get("arguments"), dict):
                raise TypeError(f"{call_path}.arguments must be an object")
            _array(
                call.get("prerequisite_call_ids"),
                path=f"{call_path}.prerequisite_call_ids",
            )
    return payload


def _screen_runs_payload(value: object) -> dict[str, object]:
    payload = _object(
        value,
        path="recorded runs",
        keys={
            "schema_version",
            "fixture_version",
            "dataset_version",
            "dataset_sha256",
            "configurations",
        },
    )
    for config_index, raw_configuration in enumerate(
        _array(payload.get("configurations"), path="recorded runs.configurations")
    ):
        config_path = f"recorded runs.configurations[{config_index}]"
        configuration = _object(raw_configuration, path=config_path, keys=_CONFIGURATION_KEYS)
        for prediction_index, prediction in enumerate(
            _array(configuration.get("predictions"), path=f"{config_path}.predictions")
        ):
            _screen_prediction(prediction, path=f"{config_path}.predictions[{prediction_index}]")
    return payload


def _verified_json(path: Path, expected_sha256: str) -> object:
    payload = path.read_bytes()
    actual = sha256(payload).hexdigest()
    if not compare_digest(actual, expected_sha256):
        raise ValueError(f"dataset SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return json.loads(payload.decode("utf-8"))


def _validate_expected_call_graph(case: AgentEvaluationCase) -> None:
    call_ids = tuple(call.call_id for call in case.expected_tool_calls)
    if len(set(call_ids)) != len(call_ids):
        raise ValueError(f"duplicate normalized expected call ID in {case.case_id}")
    known = set(call_ids)
    dependencies = {call.call_id: call.prerequisite_call_ids for call in case.expected_tool_calls}
    for call_id, prerequisite_ids in dependencies.items():
        if len(set(prerequisite_ids)) != len(prerequisite_ids):
            raise ValueError(f"duplicate prerequisite call ID for {call_id}")
        unknown = set(prerequisite_ids) - known
        if unknown:
            raise ValueError(f"unknown prerequisite call ID for {call_id}: {sorted(unknown)}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(call_id: str) -> None:
        if call_id in visiting:
            raise ValueError(f"cycle in expected call prerequisites for {case.case_id}")
        if call_id in visited:
            return
        visiting.add(call_id)
        for prerequisite_id in dependencies[call_id]:
            visit(prerequisite_id)
        visiting.remove(call_id)
        visited.add(call_id)

    for call_id in call_ids:
        visit(call_id)


def load_agent_evaluation_dataset(path: Path, *, expected_sha256: str) -> AgentEvaluationDataset:
    """Verify exact bytes, screen raw JSON, then parse versioned expectations."""

    raw = _screen_cases_payload(_verified_json(path, expected_sha256))
    _reject_secret_shaped_strings(raw)
    dataset = AgentEvaluationDataset.model_validate({**raw, "dataset_sha256": expected_sha256})
    case_ids = tuple(case.case_id for case in dataset.cases)
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("duplicate normalized case ID or empty dataset")
    for case in dataset.cases:
        _validate_expected_call_graph(case)
    return dataset


def load_recorded_agent_runs(
    path: Path,
    *,
    cases: AgentEvaluationDataset,
    expected_sha256: str,
) -> RecordedAgentRuns:
    """Verify and align the two complete recorded prediction configurations."""

    raw = _screen_runs_payload(_verified_json(path, expected_sha256))
    _reject_secret_shaped_strings(raw)
    runs = RecordedAgentRuns.model_validate(raw)
    if runs.dataset_version != cases.dataset_version or runs.dataset_sha256 != cases.dataset_sha256:
        raise ValueError("recorded-run alignment mismatch for dataset version or hash")
    configuration_ids = tuple(item.configuration_id for item in runs.configurations)
    if configuration_ids != ("bounded-agent-v1", "regressed-agent-v0"):
        raise ValueError("recorded-run alignment requires both configurations in order")
    for configuration in runs.configurations:
        for prediction in configuration.predictions:
            if (
                prediction.configuration_id != configuration.configuration_id
                or prediction.agent_version != configuration.agent_version
                or prediction.provider != configuration.provider
                or prediction.agent_model != configuration.agent_model
                or prediction.prompt_version != configuration.prompt_version
                or prediction.max_steps != configuration.max_steps
                or prediction.max_replans != configuration.max_replans
            ):
                raise ValueError("recorded-run alignment mismatch in configuration metadata")
        align_cases_and_predictions(
            cases.cases,
            configuration.predictions,
            dataset_version=cases.dataset_version,
            dataset_sha256=cases.dataset_sha256,
        )
    return runs


def align_cases_and_predictions(
    cases: Sequence[AgentEvaluationCase],
    predictions: Sequence[AgentEvaluationPrediction],
    *,
    dataset_version: str,
    dataset_sha256: str,
) -> tuple[tuple[AgentEvaluationCase, AgentEvaluationPrediction], ...]:
    """Return exact case-order alignment or reject the entire prediction table."""

    case_ids = tuple(case.case_id for case in cases)
    prediction_ids = tuple(prediction.case_id for prediction in predictions)
    if (
        not case_ids
        or len(set(case_ids)) != len(case_ids)
        or len(set(prediction_ids)) != len(prediction_ids)
        or set(prediction_ids) != set(case_ids)
        or len(prediction_ids) != len(case_ids)
    ):
        raise ValueError("alignment requires one unique prediction for every case")
    for prediction in predictions:
        if (
            prediction.dataset_version != dataset_version
            or prediction.dataset_sha256 != dataset_sha256
        ):
            raise ValueError("alignment requires one dataset version and SHA-256")
    predictions_by_id = {prediction.case_id: prediction for prediction in predictions}
    return tuple((case, predictions_by_id[case.case_id]) for case in cases)


def canonical_call_signature(capability: str, arguments: Mapping[str, Any]) -> str:
    """Return a stable public signature for one capability invocation."""

    cleaned_capability = _clean_strings(capability)
    return f"{cleaned_capability}:{json.dumps(arguments, sort_keys=True, separators=(',', ':'))}"


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


def _metric_score(checks: Sequence[tuple[str, bool | float]]) -> MetricScore:
    satisfied: list[str] = []
    missing: list[str] = []
    earned = 0.0
    for name, result in checks:
        value = float(result)
        earned += value
        if value >= 1.0:
            satisfied.append(name)
        elif value <= 0.0:
            missing.append(name)
        else:
            missing.append(f"{name} ({value:.2f} covered)")
    value = earned / len(checks) if checks else 1.0
    satisfied_text = ", ".join(satisfied) if satisfied else "none"
    missing_text = ", ".join(missing) if missing else "none"
    return MetricScore(
        value=max(0.0, min(1.0, value)),
        rationale=f"Satisfied: {satisfied_text}. Missing: {missing_text}.",
    )


def _coverage(required: Sequence[str], observed: set[str]) -> float:
    if not required:
        return 1.0
    normalized_observed = {item.casefold() for item in observed}
    return sum(item.casefold() in normalized_observed for item in required) / len(required)


def _briefing_text(briefing: CandidateBriefing | None) -> str:
    if briefing is None:
        return ""
    values = (
        *(fact.claim for fact in briefing.reported_facts),
        *briefing.cross_company_observations,
        *briefing.interpretation,
        *briefing.limitations,
    )
    return " ".join(values).casefold()


def _tool_call_correctness(
    case: AgentEvaluationCase, prediction: AgentEvaluationPrediction
) -> MetricScore:
    observed_positions: dict[str, list[int]] = {}
    for position, observation in enumerate(prediction.observations):
        signature = canonical_call_signature(observation.capability, observation.arguments)
        observed_positions.setdefault(signature, []).append(position)
    expected_by_id = {call.call_id: call for call in case.expected_tool_calls}
    checks: list[tuple[str, bool | float]] = []
    for call in case.expected_tool_calls:
        signature = canonical_call_signature(call.capability, call.arguments)
        checks.append((f"expected call {call.call_id}", signature in observed_positions))
    observed_errors = {
        observation.error_code
        for observation in prediction.observations
        if observation.status == "error" and observation.error_code is not None
    }
    for error_code in case.expected_error_codes:
        checks.append((f"typed error {error_code}", error_code in observed_errors))
    checks.append(
        (
            f"replan count {case.expected_replan_count}",
            prediction.replan_count == case.expected_replan_count,
        )
    )
    for call in case.expected_tool_calls:
        call_signature = canonical_call_signature(call.capability, call.arguments)
        for prerequisite_id in call.prerequisite_call_ids:
            prerequisite = expected_by_id[prerequisite_id]
            prerequisite_signature = canonical_call_signature(
                prerequisite.capability, prerequisite.arguments
            )
            ordered = (
                prerequisite_signature in observed_positions
                and call_signature in observed_positions
                and min(observed_positions[prerequisite_signature])
                < min(observed_positions[call_signature])
            )
            checks.append((f"dependency {prerequisite_id} before {call.call_id}", ordered))
    return _metric_score(checks)


def _terminal_attempt_ids(prediction: AgentEvaluationPrediction) -> set[int]:
    terminal_seen = False
    post_terminal: set[int] = set()
    for event in prediction.trajectory:
        if event.phase in {"evidence_gate", "guardrail"} and event.status in {
            "blocked",
            "error",
        }:
            terminal_seen = True
        elif terminal_seen and event.phase == "execution" and event.attempt_id is not None:
            post_terminal.add(event.attempt_id)
    return post_terminal


def _tool_call_efficiency(
    case: AgentEvaluationCase, prediction: AgentEvaluationPrediction
) -> tuple[MetricScore, int]:
    successful_counts: dict[str, int] = {}
    for observation in prediction.observations:
        if observation.status != "ok":
            continue
        signature = canonical_call_signature(observation.capability, observation.arguments)
        successful_counts[signature] = successful_counts.get(signature, 0) + 1
    redundant = sum(max(0, count - 1) for count in successful_counts.values())
    above_budget = max(0, len(prediction.observations) - case.max_tool_calls)
    after_terminal = len(_terminal_attempt_ids(prediction))
    excess_replans = max(0, prediction.replan_count - case.expected_replan_count)
    penalties = redundant + above_budget + after_terminal + excess_replans
    denominator = max(1, case.max_tool_calls)
    value = max(0.0, 1.0 - penalties / denominator)
    checks = (
        f"redundant successful calls={redundant}",
        f"calls above budget={above_budget}",
        f"calls after terminal event={after_terminal}",
        f"excess replans={excess_replans}",
    )
    missing = [item for item in checks if not item.endswith("=0")]
    satisfied = [item for item in checks if item.endswith("=0")]
    return (
        MetricScore(
            value=value,
            rationale=(
                f"Satisfied: {', '.join(satisfied) if satisfied else 'none'}. "
                f"Missing: {', '.join(missing) if missing else 'none'}."
            ),
        ),
        redundant,
    )


def _answer_relevance(
    case: AgentEvaluationCase, prediction: AgentEvaluationPrediction
) -> MetricScore:
    if not case.allow_briefing:
        return _metric_score(
            (
                ("expected typed status", prediction.status == case.expected_final_status),
                ("no briefing emitted", prediction.briefing is None),
            )
        )
    text = _briefing_text(prediction.briefing)
    checks: list[tuple[str, bool | float]] = [
        (f"company {company}", company.casefold() in text) for company in case.required_companies
    ]
    checks.extend(
        (
            ("mission dimension valuation", "valuation" in text or "p/e" in text),
            (
                "mission dimension operating growth",
                "operating growth" in text
                or "operating-growth" in text
                or "revenue growth" in text,
            ),
        )
    )
    return _metric_score(checks)


def _limitation_coverage(required: Sequence[str], limitations: Sequence[str]) -> float:
    if not required:
        return 1.0
    text = " ".join(limitations).casefold()
    variants = {
        "currencies": ("currencies", "currency"),
        "reporting periods": ("reporting periods", "reporting period"),
        "business definitions": (
            "business definitions",
            "business definition",
            "business mixes",
            "business mix",
        ),
    }
    covered = 0
    for phrase in required:
        candidates = variants.get(phrase.casefold(), (phrase.casefold(),))
        covered += any(candidate in text for candidate in candidates)
    return covered / len(required)


def _answer_completeness(
    case: AgentEvaluationCase, prediction: AgentEvaluationPrediction
) -> MetricScore:
    briefing = prediction.briefing
    evidence_ids = {
        evidence_id
        for fact in (briefing.reported_facts if briefing is not None else ())
        for evidence_id in fact.evidence_ids
    }
    fact_kinds = {
        fact.provenance_kind
        for fact in (briefing.reported_facts if briefing is not None else ())
        if fact.provenance_kind is not None
    }
    text = _briefing_text(briefing)
    company_coverage = (
        sum(company.casefold() in text for company in case.required_companies)
        / len(case.required_companies)
        if case.required_companies
        else 1.0
    )
    checks: tuple[tuple[str, bool | float], ...] = (
        ("expected final status", prediction.status == case.expected_final_status),
        ("required evidence IDs", _coverage(case.required_evidence_ids, evidence_ids)),
        (
            "required fact kinds",
            _coverage(case.required_fact_kinds, {str(item) for item in fact_kinds}),
        ),
        ("required companies", company_coverage),
        (
            "cross-company comparison",
            briefing is not None and bool(briefing.cross_company_observations),
        ),
        (
            "required limitations",
            _limitation_coverage(
                case.required_limitations,
                briefing.limitations if briefing is not None else (),
            ),
        ),
    )
    return _metric_score(checks)


def _citation_integrity(
    case: AgentEvaluationCase, prediction: AgentEvaluationPrediction
) -> MetricScore:
    if prediction.briefing is None:
        return _metric_score(
            (
                (
                    "expected no-briefing stop",
                    not case.allow_briefing and prediction.status == case.expected_final_status,
                ),
            )
        )

    metric_sources = {
        source
        for observation in prediction.observations
        if observation.status == "ok" and observation.capability == "get_company_metric"
        for source in observation.source_references
    }
    document_pairs: set[tuple[str, str]] = set()
    for observation in prediction.observations:
        if observation.status != "ok" or observation.capability != "search_financial_documents":
            continue
        hits = observation.result.get("hits") if observation.result is not None else None
        if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
            continue
        for hit in hits:
            if not isinstance(hit, Mapping):
                continue
            source = hit.get("source")
            evidence_id = hit.get("evidence_id")
            if (
                isinstance(source, str)
                and isinstance(evidence_id, str)
                and source in observation.source_references
                and evidence_id in observation.evidence_ids
            ):
                document_pairs.add((source, evidence_id))

    fact_checks: list[bool] = []
    for fact in prediction.briefing.reported_facts:
        if fact.provenance_kind == "metric":
            fact_checks.append(
                len(fact.source_references) == 1
                and not fact.evidence_ids
                and fact.source_references[0] in metric_sources
            )
        elif fact.provenance_kind == "document":
            fact_checks.append(
                len(fact.source_references) == 1
                and len(fact.evidence_ids) == 1
                and (fact.source_references[0], fact.evidence_ids[0]) in document_pairs
            )
        else:
            fact_checks.append(False)
    ordered_union = tuple(
        dict.fromkeys(
            source
            for fact in prediction.briefing.reported_facts
            for source in fact.source_references
        )
    )
    facts_valid = bool(fact_checks) and all(fact_checks)
    union_valid = prediction.briefing.source_references == ordered_union
    if facts_valid and union_valid:
        return _metric_score(
            (
                ("every fact has observation-backed provenance", True),
                ("aggregate source union", True),
            )
        )
    missing = []
    if not facts_valid:
        missing.append("observation-backed fact provenance")
    if not union_valid:
        missing.append("aggregate source union")
    return MetricScore(
        value=0.0,
        rationale=f"Satisfied: none. Missing: {', '.join(missing)}.",
    )


def classify_failure(
    case: AgentEvaluationCase,
    prediction: AgentEvaluationPrediction,
    scores: AgentCaseScores,
) -> FailureStage:
    """Assign the earliest actionable owner visible in public state."""

    if case.case_id != prediction.case_id:
        return "dataset"
    if prediction.status == "plan_blocked" or not prediction.initial_plan.steps:
        return "planner"
    observed_errors = {
        observation.error_code
        for observation in prediction.observations
        if observation.error_code is not None
    }
    if any(code not in observed_errors for code in case.expected_error_codes):
        return "tool_boundary"
    if (
        scores.redundant_tool_calls > 0
        or prediction.replan_count != case.expected_replan_count
        or (
            case.expected_final_status == "execution_stopped"
            and prediction.status != case.expected_final_status
        )
    ):
        return "replanner"
    if (
        prediction.status == "insufficient_evidence"
        or not prediction.evidence_gate.passed
        or (not case.allow_briefing and prediction.briefing is not None)
    ):
        return "evidence_gate"
    if (
        scores.citation_integrity.value < 1.0
        or scores.answer_relevance.value < 1.0
        or scores.answer_completeness.value < 1.0
    ):
        return "report_writer"
    return "none"


def score_agent_case(
    case: AgentEvaluationCase, prediction: AgentEvaluationPrediction
) -> AgentCaseScores:
    """Compute five deterministic public scores and the release decision."""

    if case.case_id != prediction.case_id:
        raise ValueError("alignment mismatch between case and prediction")
    correctness = _tool_call_correctness(case, prediction)
    efficiency, redundant = _tool_call_efficiency(case, prediction)
    relevance = _answer_relevance(case, prediction)
    completeness = _answer_completeness(case, prediction)
    citation = _citation_integrity(case, prediction)
    release_passed = citation.value == 1.0 and (case.allow_briefing or prediction.briefing is None)
    latency = sum(event.duration_ms for event in prediction.trajectory)
    if not prediction.trajectory:
        latency = sum(observation.duration_ms for observation in prediction.observations)
    provisional = AgentCaseScores(
        case_id=case.case_id,
        configuration_id=prediction.configuration_id,
        tool_call_correctness=correctness,
        tool_call_efficiency=efficiency,
        answer_relevance=relevance,
        answer_completeness=completeness,
        citation_integrity=citation,
        failure_stage="none",
        release_passed=release_passed,
        total_tool_calls=len(prediction.observations),
        redundant_tool_calls=redundant,
        latency_ms=latency,
    )
    return provisional.model_copy(
        update={"failure_stage": classify_failure(case, prediction, provisional)}
    )


def summarize_agent_evaluation(
    scores: Sequence[AgentCaseScores],
    *,
    dataset_version: str,
    dataset_sha256: str,
) -> AgentEvaluationSummary:
    """Aggregate one complete aligned configuration without hiding case failures."""

    if not scores:
        raise ValueError("summary requires at least one aligned case score")
    configuration_ids = {score.configuration_id for score in scores}
    case_ids = {score.case_id for score in scores}
    if len(configuration_ids) != 1 or len(case_ids) != len(scores):
        raise ValueError("summary alignment requires one configuration and unique cases")
    metric_means = {
        name: sum(getattr(score, name).value for score in scores) / len(scores)
        for name in METRIC_NAMES
    }
    metric_pass_counts = {
        name: sum(getattr(score, name).value == 1.0 for score in scores) for name in METRIC_NAMES
    }
    latencies = tuple(score.latency_ms for score in scores)
    return AgentEvaluationSummary(
        configuration_id=next(iter(configuration_ids)),
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        case_count=len(scores),
        metric_means=metric_means,
        metric_pass_counts=metric_pass_counts,
        mean_tool_calls=sum(score.total_tool_calls for score in scores) / len(scores),
        mean_latency_ms=sum(latencies) / len(latencies),
        max_latency_ms=max(latencies),
        release_passed=all(score.release_passed for score in scores),
    )
