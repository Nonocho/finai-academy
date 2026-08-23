"""MLflow-only integration for Lesson 12 agent evaluation."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlparse

import mlflow
from mlflow.entities import Trace
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
_PERSONAL_PATH_PATTERN = re.compile(r"(?i)(?:/Users/|/home/|[A-Z]:\\Users\\)")
_HIDDEN_REASONING_PATTERN = re.compile(r"(?i)chain[- ]of[- ]thought|hidden reasoning")
_SANITIZED_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer\s+[a-z0-9._-]+|sk-[a-z0-9_-]{8,})"
)
_SECRET_SHAPED_PATTERN = re.compile(r"(?i)\bsk-[a-z0-9_-]{8,}\b")


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
    root_directory.mkdir(parents=True, exist_ok=True)
    artifact_directory.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{database_path}"
    try:
        mlflow.set_tracking_uri(tracking_uri)
    except Exception as exc:
        reason = _sanitized_error_reason(exc)
        raise RuntimeError(
            f"MLflow backend initialization failed for {database_path}: {reason}"
        ) from exc
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
    reason = _SANITIZED_SECRET_PATTERN.sub("[redacted]", str(exc))
    reason = _PERSONAL_PATH_PATTERN.sub("[personal-path]/", reason)
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
        raise ValueError("MLflow experiment requires an absolute local artifact URI")


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
        emitted_attempts: set[int] = set()
        owning_phase = "planning"

        for event in prediction.trajectory:
            if event.phase == "execution" and event.attempt_id in observations_by_attempt:
                observation = observations_by_attempt[event.attempt_id]
                _trace_execution(observation)
                emitted_attempts.add(observation.attempt_id)
                owning_phase = "execution"
                continue

            mapped_phase = "plan_gate" if event.phase == "policy" else event.phase
            if mapped_phase == "guardrail":
                mapped_phase = owning_phase
            if mapped_phase == "execution":
                continue
            if mapped_phase not in PHASE_SPAN_TYPES:
                continue
            with mlflow.start_span(
                name=mapped_phase,
                span_type=PHASE_SPAN_TYPES[mapped_phase],
                attributes={"trajectory_index": event.index},
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
            emitted_chain_phases.add(mapped_phase)
            owning_phase = mapped_phase

        for observation in prediction.observations:
            if observation.attempt_id not in emitted_attempts:
                _trace_execution(observation)

        for phase in _REQUIRED_CHAIN_PHASES:
            if phase in emitted_chain_phases:
                continue
            with mlflow.start_span(
                name=phase,
                span_type=PHASE_SPAN_TYPES[phase],
                attributes={"reconstructed": True},
            ) as phase_span:
                phase_span.set_inputs({"case_id": case.case_id})
                phase_span.set_outputs(
                    _validate_safe_payload(
                        {
                            "status": "not_emitted",
                            "observed_status": prediction.status,
                        }
                    )
                )
        root_span.set_outputs(root_outputs)


def _trace_execution(observation: ResearchObservation) -> None:
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
    except ValueError:
        raise
    except Exception as exc:
        reason = _sanitized_error_reason(exc)
        raise RuntimeError(
            f"MLflow backend initialization failed for {store.database_path}: {reason}"
        ) from exc

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
