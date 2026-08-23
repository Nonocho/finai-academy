"""MLflow-only integration for Lesson 12 agent evaluation."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import mlflow
from mlflow.entities import SpanEvent, Trace
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

from finai_academy import agent_evaluation
from finai_academy.agent_evaluation import (
    METRIC_NAMES,
    AgentCaseScores,
    AgentEvaluationCase,
    AgentEvaluationPrediction,
    align_cases_and_predictions,
    score_agent_case,
    summarize_agent_evaluation,
)
from finai_academy.research_planning import ResearchObservation, TrajectoryEvent

ARTIFACT_PATHS = (
    "evaluation/case_scores.json",
    "evaluation/failure_rows.json",
    "evaluation/dataset_manifest.json",
)
PHASE_SPAN_TYPES = {
    "planning": "CHAIN",
    "plan_gate": "CHAIN",
    "execution": "TOOL",
    "replanning": "CHAIN",
    "evidence_gate": "CHAIN",
    "report": "CHAIN",
}
_REQUIRED_CHAIN_PHASES = (
    "planning",
    "plan_gate",
    "replanning",
    "evidence_gate",
    "report",
)
_PERSONAL_PATH_PATTERN = re.compile(
    r"(?i)(?:(?:/Users|/home)/[^\r\n]*|[A-Z]:\\Users\\[^\r\n]*)"
)
_HIDDEN_REASONING_PATTERN = re.compile(r"(?i)chain[- ]of[- ]thought|hidden reasoning")
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)\b[a-z0-9_-]*api[_-]?key\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_AUTHORIZATION_HEADER_PATTERN = re.compile(
    r"(?i)\bauthorization\s*:\s*(?:basic|bearer)\s+"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_SANITIZED_SECRET_PATTERN = re.compile(
    r"(?i)(bearer\s+[a-z0-9._-]+|sk-[a-z0-9_-]{8,})"
)
_SECRET_SHAPED_PATTERN = re.compile(r"(?i)\bsk-[a-z0-9_-]{8,}\b")
_JUDGE_MODEL_PATTERN = re.compile(r"^(openai|ollama_chat):/([^\s]+)$")
_JUDGE_SCORER_NAMES = (
    "ToolCallCorrectness",
    "ToolCallEfficiency",
    "RelevanceToQuery",
    "Completeness",
)
_JUDGE_METRIC_NAMES = {
    "ToolCallCorrectness": "judge_tool_call_correctness",
    "ToolCallEfficiency": "judge_tool_call_efficiency",
    "RelevanceToQuery": "judge_relevance_to_query",
    "Completeness": "judge_completeness",
}
_JUDGE_AVAILABLE_TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "get_company_metric",
            "description": "Return a controlled company metric with date and source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "metric": {"type": "string"},
                },
                "required": ["ticker", "metric"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_financial_documents",
            "description": "Search controlled company financial evidence passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 3,
                        "default": 2,
                    },
                },
                "required": ["company", "query"],
                "additionalProperties": False,
            },
        },
    },
)


class _ArtifactLocationError(ValueError):
    """Raised only for the caller-visible local artifact-location contract."""


class AgentEvaluationConfiguration(BaseModel):
    """Complete reproducibility metadata for one agent evaluation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    configuration_id: str
    dataset_version: str
    dataset_sha256: str
    agent_version: str
    provider: Literal["recorded", "openai", "ollama"]
    agent_model: str
    judge_provider: Literal["none", "openai", "ollama"] = "none"
    judge_model: str = "none"
    prompt_version: str
    max_steps: int = Field(ge=1)
    max_replans: int = Field(ge=0)
    scorer_contract_version: str = "agent-scorers-v1"


class LocalMLflowStore(BaseModel):
    """Resolved local SQLite backend and artifact paths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root_directory: Path
    database_path: Path
    artifact_directory: Path
    tracking_uri: str
    ui_command: str


class MLflowAgentEvaluationSummary(BaseModel):
    """Notebook-ready summary of one persisted evaluation configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    experiment_id: str
    tracking_uri: str
    trace_count: int
    trace_ids: dict[str, str]
    parameters: dict[str, str | int]
    metrics: dict[str, float]
    artifact_paths: tuple[str, ...]
    case_scores_by_id: dict[str, AgentCaseScores]
    failure_rows: tuple[dict[str, object], ...]


class AgentEvaluationComparison(BaseModel):
    """Aligned scorecard rows for comparing complete MLflow configurations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    configuration_ids: tuple[str, ...]
    dataset_version: str
    dataset_sha256: str
    metric_mean_rows: tuple[dict[str, object], ...]
    metric_pass_rows: tuple[dict[str, object], ...]
    case_metric_rows: tuple[dict[str, object], ...]
    tool_call_rows: tuple[dict[str, object], ...]
    latency_rows: tuple[dict[str, object], ...]
    failure_rows: tuple[dict[str, object], ...]


class JudgeConfiguration(BaseModel):
    """One explicitly selected MLflow judge provider and model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["openai", "ollama"]
    model_uri: str
    model: str


class JudgeResult(BaseModel):
    """Truthful aggregate outcome for one optional MLflow judge."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scorer_name: Literal[
        "ToolCallCorrectness",
        "ToolCallEfficiency",
        "RelevanceToQuery",
        "Completeness",
    ]
    provider: Literal["openai", "ollama"] | None
    model: str | None
    mlflow_version: str
    latency_ms: float = Field(ge=0)
    status: Literal["COMPLETED", "ERROR", "NOT RUN"]
    score: float | None = Field(default=None, ge=0, le=1)
    rationale: str


class JudgeScorerSet(BaseModel):
    """Optional judge configuration paired with constructed MLflow scorers."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    configuration: JudgeConfiguration | None
    scorers: tuple[object, ...]


def load_judge_configuration(
    environment: Mapping[str, str] = os.environ,
) -> JudgeConfiguration | None:
    """Load only an explicitly configured OpenAI or Ollama MLflow model URI."""

    model_uri = environment.get("FINAI_EVAL_JUDGE_MODEL")
    if model_uri is None or model_uri == "":
        return None
    match = _JUDGE_MODEL_PATTERN.fullmatch(model_uri)
    if match is None:
        raise ValueError(
            "FINAI_EVAL_JUDGE_MODEL must use openai:/<model> or "
            "ollama_chat:/<model>"
        )
    _validate_safe_payload(model_uri)
    route, model = match.groups()
    provider: Literal["openai", "ollama"] = (
        "openai" if route == "openai" else "ollama"
    )
    return JudgeConfiguration(
        provider=provider,
        model_uri=model_uri,
        model=model,
    )


def build_optional_genai_scorers(
    configuration: JudgeConfiguration | None,
) -> JudgeScorerSet:
    """Build the four current MLflow scorers only for an explicit judge route."""

    if configuration is None:
        return JudgeScorerSet(configuration=None, scorers=())

    from mlflow.genai.scorers import (
        Completeness,
        RelevanceToQuery,
        ToolCallCorrectness,
        ToolCallEfficiency,
    )

    runtime_model_uri = (
        f"ollama:/{configuration.model}"
        if configuration.provider == "ollama"
        else configuration.model_uri
    )
    return JudgeScorerSet(
        configuration=configuration,
        scorers=(
            ToolCallCorrectness(model=runtime_model_uri),
            ToolCallEfficiency(model=runtime_model_uri),
            RelevanceToQuery(model=runtime_model_uri),
            Completeness(model=runtime_model_uri),
        ),
    )


def _judge_score(value: object) -> float:
    candidate = getattr(value, "value", value)
    if isinstance(candidate, bool):
        return float(candidate)
    if isinstance(candidate, (int, float)):
        score = float(candidate)
        if 0.0 <= score <= 1.0:
            return score
    if isinstance(candidate, str):
        normalized = candidate.casefold()
        if normalized in {"yes", "true", "pass", "passed"}:
            return 1.0
        if normalized in {"no", "false", "fail", "failed"}:
            return 0.0
    raise ValueError("MLflow judge returned an unsupported score")


def _normalize_trace_for_judges(trace: object) -> object:
    """Clone a Task 3 trace into MLflow's current judge-facing tool schema."""

    if not isinstance(trace, Trace):
        return trace
    trace_payload = deepcopy(trace.to_dict())
    span_payloads = trace_payload["data"]["spans"]
    for span, span_payload in zip(trace.data.spans, span_payloads, strict=True):
        attributes = span_payload["attributes"]
        if span.parent_id is None:
            root_inputs = span.inputs if isinstance(span.inputs, dict) else {}
            attributes["mlflow.spanType"] = json.dumps("CHAT_MODEL")
            attributes["mlflow.spanInputs"] = json.dumps(
                {
                    "mission": root_inputs.get("mission"),
                    "tools": _JUDGE_AVAILABLE_TOOLS,
                }
            )
        elif span.span_type == "TOOL":
            inputs = span.inputs if isinstance(span.inputs, dict) else {}
            capability = inputs.get("capability")
            arguments = inputs.get("arguments")
            if not isinstance(capability, str) or not isinstance(arguments, dict):
                raise ValueError("Task 3 TOOL span lacks judge-safe capability metadata")
            span_payload["name"] = capability
            attributes["mlflow.spanInputs"] = json.dumps(arguments)
    return Trace.from_dict(trace_payload)


def _judge_error_reason(exc: Exception) -> str:
    reason = _sanitized_error_reason(exc)
    try:
        _validate_safe_payload(reason)
    except ValueError:
        reason = "details redacted"
    return f"{type(exc).__name__}: {reason}"


def _judge_is_unavailable(exc: Exception) -> bool:
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return True
    if isinstance(exc, MlflowException):
        reason = str(exc).casefold()
        return any(
            marker in reason
            for marker in (
                "no suitable adapter found",
                "environment variable must be set",
                "failed to connect to",
                " is required to use ",
            )
        ) or (
            "provider" in reason and "not found" in reason
        )
    return False


def _not_run_judge_results(
    *,
    configuration: JudgeConfiguration | None,
    rationale: str,
) -> tuple[JudgeResult, ...]:
    return tuple(
        JudgeResult(
            scorer_name=scorer_name,
            provider=configuration.provider if configuration is not None else None,
            model=configuration.model if configuration is not None else None,
            mlflow_version=mlflow.__version__,
            latency_ms=0.0,
            status="NOT RUN",
            score=None,
            rationale=rationale,
        )
        for scorer_name in _JUDGE_SCORER_NAMES
    )


def _run_one_judge(
    *,
    scorer_name: str,
    scorer: object,
    configuration: JudgeConfiguration,
    traces: Sequence[object],
) -> JudgeResult:
    started = perf_counter()
    try:
        scores: list[float] = []
        rationales: list[str] = []
        for trace in traces:
            judge_trace = _normalize_trace_for_judges(trace)
            feedback = scorer(trace=judge_trace)  # type: ignore[operator]
            feedback_error = getattr(feedback, "error", None)
            if feedback_error is not None:
                raise RuntimeError(str(feedback_error))
            scores.append(_judge_score(getattr(feedback, "value", None)))
            rationale = getattr(feedback, "rationale", None)
            if rationale:
                rationales.append(str(rationale))
        score = sum(scores) / len(scores)
        rationale = (
            f"Completed {len(scores)} trace(s). " + " | ".join(rationales)
            if rationales
            else f"Completed {len(scores)} trace(s)."
        )
        _validate_safe_payload(rationale)
        status: Literal["COMPLETED", "ERROR", "NOT RUN"] = "COMPLETED"
    except Exception as exc:  # noqa: BLE001 - one failed judge stays observational.
        score = None
        rationale = _judge_error_reason(exc)
        status = "NOT RUN" if _judge_is_unavailable(exc) else "ERROR"
    return JudgeResult(
        scorer_name=scorer_name,
        provider=configuration.provider,
        model=configuration.model,
        mlflow_version=mlflow.__version__,
        latency_ms=(perf_counter() - started) * 1000,
        status=status,
        score=score,
        rationale=rationale,
    )


def run_optional_judges(
    *,
    run_id: str,
    configuration: JudgeConfiguration | None,
    traces: Sequence[object],
) -> tuple[JudgeResult, ...]:
    """Run and log optional judges without changing deterministic evaluation state."""

    if configuration is None:
        results = _not_run_judge_results(
            configuration=None,
            rationale="FINAI_EVAL_JUDGE_MODEL is not configured.",
        )
    elif not traces:
        results = _not_run_judge_results(
            configuration=configuration,
            rationale="No MLflow traces were supplied.",
        )
    else:
        try:
            scorer_set = build_optional_genai_scorers(configuration)
        except Exception as exc:  # noqa: BLE001 - unavailable optional API is observable.
            results = _not_run_judge_results(
                configuration=configuration,
                rationale=_judge_error_reason(exc),
            )
        else:
            results = tuple(
                _run_one_judge(
                    scorer_name=scorer_name,
                    scorer=scorer,
                    configuration=configuration,
                    traces=traces,
                )
                for scorer_name, scorer in zip(
                    _JUDGE_SCORER_NAMES,
                    scorer_set.scorers,
                    strict=True,
                )
            )

    client = MlflowClient()
    for result in results:
        if result.status == "COMPLETED" and result.score is not None:
            client.log_metric(
                run_id,
                _JUDGE_METRIC_NAMES[result.scorer_name],
                result.score,
            )
    rows = [result.model_dump(mode="json") for result in results]
    _validate_safe_payload(rows)
    client.log_dict(run_id, {"rows": rows}, "evaluation/judge_results.json")
    return results


def initialize_local_mlflow(tracking_directory: Path | None = None) -> LocalMLflowStore:
    """Initialize and select a resolved local SQLite MLflow store."""

    selected = tracking_directory
    if selected is None:
        configured = os.environ.get("FINAI_MLFLOW_DIR")
        selected = (
            Path(configured)
            if configured
            else Path(tempfile.mkdtemp(prefix="finai-lesson12-mlflow-"))
        )
    root_directory = selected.resolve()
    database_path = (root_directory / "mlflow.db").resolve()
    artifact_directory = (root_directory / "artifacts").resolve()
    tracking_uri = f"sqlite:///{database_path}"
    try:
        root_directory.mkdir(parents=True, exist_ok=True)
        artifact_directory.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
    except Exception as exc:  # noqa: BLE001 - sanitize every local-backend failure.
        reason = _sanitized_error_reason(exc)
        raise RuntimeError(
            f"MLflow backend initialization failed for {database_path}: {reason}"
        ) from None
    return LocalMLflowStore(
        root_directory=root_directory,
        database_path=database_path,
        artifact_directory=artifact_directory,
        tracking_uri=tracking_uri,
        ui_command=f"mlflow ui --backend-store-uri {tracking_uri}",
    )


def _parameters(configuration: AgentEvaluationConfiguration) -> dict[str, str | int]:
    return configuration.model_dump(mode="python")


def _sanitized_error_reason(exc: Exception) -> str:
    reason = _CREDENTIAL_ASSIGNMENT_PATTERN.sub("[credential redacted]", str(exc))
    reason = _AUTHORIZATION_HEADER_PATTERN.sub("[credential redacted]", reason)
    reason = _SANITIZED_SECRET_PATTERN.sub("[credential redacted]", reason)
    reason = _PERSONAL_PATH_PATTERN.sub("[personal path redacted]", reason)
    return reason or type(exc).__name__


def _validate_safe_payload(value: Any) -> Any:
    """Reject secret, personal-path, or hidden-reasoning text recursively."""

    agent_evaluation._reject_secret_shaped_strings(value)
    if isinstance(value, BaseModel):
        _validate_safe_payload(value.model_dump(mode="python"))
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_safe_payload(key)
            _validate_safe_payload(item)
    elif isinstance(value, (tuple, list, set)):
        for item in value:
            _validate_safe_payload(item)
    elif isinstance(value, str):
        if _SECRET_SHAPED_PATTERN.search(value):
            raise ValueError("MLflow payload must not contain secret-shaped text")
        if _PERSONAL_PATH_PATTERN.search(value):
            raise ValueError("MLflow payload must not contain a personal path")
        if _HIDDEN_REASONING_PATTERN.search(value):
            raise ValueError("MLflow payload must not contain hidden reasoning")
    return value


def _validate_configuration_alignment(
    configuration: AgentEvaluationConfiguration,
    predictions: Sequence[AgentEvaluationPrediction],
) -> None:
    expected = {
        "configuration_id": configuration.configuration_id,
        "dataset_version": configuration.dataset_version,
        "dataset_sha256": configuration.dataset_sha256,
        "agent_version": configuration.agent_version,
        "provider": configuration.provider,
        "agent_model": configuration.agent_model,
        "prompt_version": configuration.prompt_version,
        "max_steps": configuration.max_steps,
        "max_replans": configuration.max_replans,
    }
    for prediction in predictions:
        if any(getattr(prediction, name) != value for name, value in expected.items()):
            raise ValueError("configuration metadata must align with every prediction")


def _require_local_artifact_uri(artifact_uri: str) -> None:
    parsed = urlparse(artifact_uri)
    local_path = Path(unquote(parsed.path))
    if (
        parsed.scheme != "file"
        or parsed.netloc not in {"", "localhost"}
        or not local_path.is_absolute()
    ):
        raise _ArtifactLocationError(
            "MLflow experiment requires an absolute local artifact URI"
        )


def _experiment_id(
    *, client: MlflowClient, experiment_name: str, store: LocalMLflowStore
) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=store.artifact_directory.as_uri(),
        )
        experiment = client.get_experiment(experiment_id)
    _require_local_artifact_uri(experiment.artifact_location)
    return experiment.experiment_id


def _phase_output(event: TrajectoryEvent) -> dict[str, object]:
    return {
        "status": event.status,
        "summary": event.summary,
        "duration_ms": event.duration_ms,
    }


def _guardrails_by_owner(
    trajectory: Sequence[TrajectoryEvent],
) -> dict[int, tuple[TrajectoryEvent, ...]]:
    guardrails: dict[int, list[TrajectoryEvent]] = {}
    owner_index: int | None = None
    for event in trajectory:
        if event.phase == "guardrail":
            if owner_index is not None:
                guardrails.setdefault(owner_index, []).append(event)
        else:
            owner_index = event.index
    return {index: tuple(events) for index, events in guardrails.items()}


def _add_guardrail_events(span: Any, events: Sequence[TrajectoryEvent]) -> None:
    for event in events:
        attributes = _validate_safe_payload(
            {
                "status": event.status,
                "summary": event.summary,
                "duration_ms": event.duration_ms,
                "trajectory_index": event.index,
            }
        )
        span.add_event(SpanEvent(name="guardrail", attributes=attributes))


def _revision_after_replanning(event: TrajectoryEvent, current_revision: int) -> int:
    """Return the post-decision revision; unchanged decisions retain their revision."""

    if "replace_remaining" in event.summary.casefold():
        return current_revision + 1
    return current_revision


def _trace_phase(
    *,
    phase: str,
    event: TrajectoryEvent,
    plan_revision: int,
    guardrails: Sequence[TrajectoryEvent],
) -> None:
    attributes: dict[str, object] = {"trajectory_index": event.index}
    if phase == "replanning":
        attributes["plan_revision"] = plan_revision
    with mlflow.start_span(
        name=phase,
        span_type=PHASE_SPAN_TYPES[phase],
        attributes=attributes,
    ) as phase_span:
        phase_span.set_inputs(
            _validate_safe_payload(
                {
                    "step_id": event.step_id,
                    "attempt_id": event.attempt_id,
                }
            )
        )
        phase_span.set_outputs(_validate_safe_payload(_phase_output(event)))
        _add_guardrail_events(phase_span, guardrails)


def _trace_reconstructed_phase(
    *,
    phase: str,
    case_id: str,
    observed_status: str,
    plan_revision: int,
) -> None:
    attributes: dict[str, object] = {"reconstructed": True}
    if phase == "replanning":
        attributes["plan_revision"] = plan_revision
    with mlflow.start_span(
        name=phase,
        span_type=PHASE_SPAN_TYPES[phase],
        attributes=attributes,
    ) as phase_span:
        phase_span.set_inputs(_validate_safe_payload({"case_id": case_id}))
        phase_span.set_outputs(
            _validate_safe_payload(
                {
                    "status": "not_emitted",
                    "observed_status": observed_status,
                }
            )
        )


def _plan_revisions(prediction: AgentEvaluationPrediction) -> list[dict[str, object]]:
    return [
        {
            "plan_revision": 0,
            "steps": [step.model_dump(mode="json") for step in prediction.initial_plan.steps],
        },
        {
            "plan_revision": prediction.replan_count,
            "steps": [step.model_dump(mode="json") for step in prediction.final_steps],
        },
    ]


def _score_payload(score: AgentCaseScores) -> dict[str, dict[str, object]]:
    return {
        name: getattr(score, name).model_dump(mode="json") for name in METRIC_NAMES
    }


def _trace_case(
    *,
    run_id: str,
    configuration: AgentEvaluationConfiguration,
    case: AgentEvaluationCase,
    prediction: AgentEvaluationPrediction,
    score: AgentCaseScores,
) -> None:
    root_inputs = _validate_safe_payload(
        {
            "case_id": case.case_id,
            "mission": case.mission,
            "configuration_id": configuration.configuration_id,
            "expected_tool_calls": [
                call.model_dump(mode="json") for call in case.expected_tool_calls
            ],
        }
    )
    root_outputs = _validate_safe_payload(
        {
            "observed_status": prediction.status,
            "expected_status": case.expected_final_status,
            "plan_revisions": _plan_revisions(prediction),
            "evidence_gate": prediction.evidence_gate.model_dump(mode="json"),
            "briefing": (
                prediction.briefing.model_dump(mode="json")
                if prediction.briefing is not None
                else None
            ),
            "scores": _score_payload(score),
            "failure_stage": score.failure_stage,
            "latency_ms": score.latency_ms,
        }
    )
    with mlflow.start_span(
        name=f"case:{case.case_id}",
        span_type="CHAIN",
        run_id=run_id,
        attributes={"case_id": case.case_id, "source_kind": configuration.provider},
    ) as root_span:
        root_span.set_inputs(root_inputs)
        emitted_chain_phases: set[str] = set()
        observations_by_attempt = {
            observation.attempt_id: observation for observation in prediction.observations
        }
        guardrails_by_owner = _guardrails_by_owner(prediction.trajectory)
        emitted_attempts: set[int] = set()
        current_revision = 0

        def ensure_phases(phases: Sequence[str]) -> None:
            for missing_phase in phases:
                if missing_phase in emitted_chain_phases:
                    continue
                _trace_reconstructed_phase(
                    phase=missing_phase,
                    case_id=case.case_id,
                    observed_status=prediction.status,
                    plan_revision=current_revision,
                )
                emitted_chain_phases.add(missing_phase)

        for event in prediction.trajectory:
            if event.phase == "guardrail":
                continue
            if event.phase == "execution" and event.attempt_id in observations_by_attempt:
                ensure_phases(("planning", "plan_gate"))
                observation = observations_by_attempt[event.attempt_id]
                _trace_execution(
                    observation,
                    guardrails=guardrails_by_owner.get(event.index, ()),
                )
                emitted_attempts.add(observation.attempt_id)
                current_revision = max(current_revision, observation.plan_revision)
                continue

            mapped_phase = "plan_gate" if event.phase == "policy" else event.phase
            if mapped_phase not in PHASE_SPAN_TYPES:
                continue
            if mapped_phase == "plan_gate":
                ensure_phases(("planning",))
            elif mapped_phase in {"replanning", "evidence_gate", "report"}:
                ensure_phases(("planning", "plan_gate"))
            if mapped_phase in {"evidence_gate", "report"}:
                ensure_phases(("replanning",))
            if mapped_phase == "report":
                ensure_phases(("evidence_gate",))
            if mapped_phase == "replanning":
                current_revision = _revision_after_replanning(event, current_revision)
            _trace_phase(
                phase=mapped_phase,
                event=event,
                plan_revision=current_revision,
                guardrails=guardrails_by_owner.get(event.index, ()),
            )
            emitted_chain_phases.add(mapped_phase)

        for observation in prediction.observations:
            if observation.attempt_id not in emitted_attempts:
                ensure_phases(("planning", "plan_gate"))
                _trace_execution(observation, guardrails=())
                current_revision = max(current_revision, observation.plan_revision)

        ensure_phases(_REQUIRED_CHAIN_PHASES)
        root_span.set_outputs(root_outputs)


def _trace_execution(
    observation: ResearchObservation,
    *,
    guardrails: Sequence[TrajectoryEvent],
) -> None:
    inputs = _validate_safe_payload(
        {
            "capability": observation.capability,
            "arguments": observation.arguments,
            "step_id": observation.step_id,
            "attempt_id": observation.attempt_id,
            "plan_revision": observation.plan_revision,
        }
    )
    outputs = _validate_safe_payload(
        {
            "status": observation.status,
            "error_code": observation.error_code,
            "evidence_ids": list(observation.evidence_ids),
            "source_references": list(observation.source_references),
            "duration_ms": observation.duration_ms,
        }
    )
    with mlflow.start_span(
        name=f"execution:{observation.attempt_id}",
        span_type=PHASE_SPAN_TYPES["execution"],
        attributes={
            "attempt_id": observation.attempt_id,
            "plan_revision": observation.plan_revision,
        },
    ) as execution_span:
        execution_span.set_inputs(inputs)
        execution_span.set_outputs(outputs)
        _add_guardrail_events(execution_span, guardrails)


def _failure_row(score: AgentCaseScores) -> dict[str, object]:
    row: dict[str, object] = {
        "case_id": score.case_id,
        "configuration_id": score.configuration_id,
        "failure_stage": score.failure_stage,
    }
    for name in METRIC_NAMES:
        metric = getattr(score, name)
        row[name] = float(metric.value)
        row[f"{name}_rationale"] = metric.rationale
    return row


def _trace_ids_by_case(traces: Sequence[Trace]) -> dict[str, str]:
    trace_ids: dict[str, str] = {}
    for trace in traces:
        root = next(span for span in trace.data.spans if span.parent_id is None)
        case_id = root.inputs["case_id"]
        trace_id = getattr(trace.info, "trace_id", root.trace_id)
        trace_ids[case_id] = trace_id
    return trace_ids


def run_mlflow_agent_evaluation(
    *,
    tracking_directory: Path | None,
    experiment_name: str,
    configuration: AgentEvaluationConfiguration,
    cases: Sequence[AgentEvaluationCase],
    predictions: Sequence[AgentEvaluationPrediction],
) -> MLflowAgentEvaluationSummary:
    """Score and persist one aligned agent evaluation configuration."""

    _validate_safe_payload(experiment_name)
    _validate_safe_payload(configuration)
    _validate_safe_payload(cases)
    _validate_safe_payload(predictions)
    _validate_configuration_alignment(configuration, predictions)
    aligned = align_cases_and_predictions(
        cases,
        predictions,
        dataset_version=configuration.dataset_version,
        dataset_sha256=configuration.dataset_sha256,
    )
    scores = tuple(score_agent_case(case, prediction) for case, prediction in aligned)
    scores_by_case = {score.case_id: score for score in scores}
    evaluation_summary = summarize_agent_evaluation(
        scores,
        dataset_version=configuration.dataset_version,
        dataset_sha256=configuration.dataset_sha256,
    )
    parameters = _parameters(configuration)
    metrics = {
        **{
            f"{name}_mean": float(evaluation_summary.metric_means[name])
            for name in METRIC_NAMES
        },
        "mean_tool_calls": float(evaluation_summary.mean_tool_calls),
        "mean_latency_ms": float(evaluation_summary.mean_latency_ms),
    }
    store = initialize_local_mlflow(tracking_directory)
    try:
        client = MlflowClient()
        experiment_id = _experiment_id(
            client=client,
            experiment_name=experiment_name,
            store=store,
        )
    except _ArtifactLocationError:
        raise
    except Exception as exc:  # noqa: BLE001 - sanitize every MLflow-client failure.
        reason = _sanitized_error_reason(exc)
        raise RuntimeError(
            f"MLflow backend initialization failed for {store.database_path}: {reason}"
        ) from None

    score_rows = [score.model_dump(mode="json") for score in scores]
    failure_rows = [
        _failure_row(score)
        for score in scores
        if score.failure_stage != "none"
    ]
    dataset_manifest = {
        "dataset_version": configuration.dataset_version,
        "dataset_sha256": configuration.dataset_sha256,
        "scorer_contract_version": configuration.scorer_contract_version,
        "case_ids": [case.case_id for case in cases],
        "fixture_version": "agent-runs-v1",
        "configuration_id": configuration.configuration_id,
    }
    _validate_safe_payload(parameters)
    _validate_safe_payload(metrics)
    _validate_safe_payload(score_rows)
    _validate_safe_payload(failure_rows)
    _validate_safe_payload(dataset_manifest)
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=configuration.configuration_id,
    ) as active_run:
        run_id = active_run.info.run_id
        mlflow.log_params(parameters)
        for case, prediction in aligned:
            _trace_case(
                run_id=run_id,
                configuration=configuration,
                case=case,
                prediction=prediction,
                score=scores_by_case[case.case_id],
            )
        mlflow.flush_trace_async_logging()
        traces = mlflow.search_traces(
            run_id=run_id,
            locations=[experiment_id],
            return_type="list",
            flush=True,
        )
        trace_ids = _trace_ids_by_case(traces)
        if set(trace_ids) != {case.case_id for case in cases}:
            raise RuntimeError("MLflow did not persist one trace for every aligned case")
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"rows": score_rows}, ARTIFACT_PATHS[0])
        mlflow.log_dict({"rows": failure_rows}, ARTIFACT_PATHS[1])
        mlflow.log_dict(dataset_manifest, ARTIFACT_PATHS[2])

    return MLflowAgentEvaluationSummary(
        run_id=run_id,
        experiment_id=experiment_id,
        tracking_uri=store.tracking_uri,
        trace_count=len(trace_ids),
        trace_ids=trace_ids,
        parameters=parameters,
        metrics=metrics,
        artifact_paths=ARTIFACT_PATHS,
        case_scores_by_id=scores_by_case,
        failure_rows=tuple(failure_rows),
    )


def compare_agent_configurations(
    summaries: Sequence[MLflowAgentEvaluationSummary],
) -> AgentEvaluationComparison:
    """Return aligned notebook-ready rows across persisted configurations."""

    if len(summaries) < 2:
        raise ValueError("comparison requires at least two configurations")
    dataset_keys = {
        (
            summary.parameters["dataset_version"],
            summary.parameters["dataset_sha256"],
        )
        for summary in summaries
    }
    if len(dataset_keys) != 1:
        raise ValueError("comparison requires one dataset version and SHA-256")
    dataset_version, dataset_sha256 = next(iter(dataset_keys))
    case_id_sets = {frozenset(summary.case_scores_by_id) for summary in summaries}
    if len(case_id_sets) != 1:
        raise ValueError("comparison requires identical aligned case IDs")
    configuration_ids = tuple(
        str(summary.parameters["configuration_id"]) for summary in summaries
    )
    if len(set(configuration_ids)) != len(configuration_ids):
        raise ValueError("comparison requires unique configuration IDs")

    metric_mean_rows: list[dict[str, object]] = []
    metric_pass_rows: list[dict[str, object]] = []
    case_metric_rows: list[dict[str, object]] = []
    tool_call_rows: list[dict[str, object]] = []
    latency_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    for configuration_id, summary in zip(configuration_ids, summaries, strict=True):
        for name in METRIC_NAMES:
            metric_mean_rows.append(
                {
                    "configuration_id": configuration_id,
                    "metric": name,
                    "mean": summary.metrics[f"{name}_mean"],
                }
            )
            metric_pass_rows.append(
                {
                    "configuration_id": configuration_id,
                    "metric": name,
                    "pass_count": sum(
                        getattr(score, name).value == 1.0
                        for score in summary.case_scores_by_id.values()
                    ),
                    "case_count": len(summary.case_scores_by_id),
                }
            )
        for case_id, score in summary.case_scores_by_id.items():
            case_metric_rows.append(
                {
                    "configuration_id": configuration_id,
                    "case_id": case_id,
                    **{name: getattr(score, name).value for name in METRIC_NAMES},
                    "failure_stage": score.failure_stage,
                    "trace_id": summary.trace_ids[case_id],
                    "run_id": summary.run_id,
                }
            )
            tool_call_rows.append(
                {
                    "configuration_id": configuration_id,
                    "case_id": case_id,
                    "total_tool_calls": score.total_tool_calls,
                    "redundant_tool_calls": score.redundant_tool_calls,
                }
            )
        latencies = [score.latency_ms for score in summary.case_scores_by_id.values()]
        latency_rows.append(
            {
                "configuration_id": configuration_id,
                "mean_latency_ms": summary.metrics["mean_latency_ms"],
                "max_latency_ms": max(latencies),
            }
        )
        failure_rows.extend(summary.failure_rows)

    comparison = AgentEvaluationComparison(
        configuration_ids=configuration_ids,
        dataset_version=str(dataset_version),
        dataset_sha256=str(dataset_sha256),
        metric_mean_rows=tuple(metric_mean_rows),
        metric_pass_rows=tuple(metric_pass_rows),
        case_metric_rows=tuple(case_metric_rows),
        tool_call_rows=tuple(tool_call_rows),
        latency_rows=tuple(latency_rows),
        failure_rows=tuple(failure_rows),
    )
    _validate_safe_payload(comparison)
    return comparison
