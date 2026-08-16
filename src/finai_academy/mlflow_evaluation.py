"""MLflow-only integration boundary for Lesson 07 evaluation and tracing."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import mlflow
from mlflow.tracking import MlflowClient

from finai_academy.evaluation import (
    EvaluationCase,
    evaluate_case,
    summarize_evaluations,
)
from finai_academy.measurement import MeasurementValue, StageObserver, StageSpan
from finai_academy.retrieval_pipeline import RetrievalResult


def _normalized_text(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


@dataclass(frozen=True)
class EvaluationConfiguration:
    """Complete reproducibility metadata for one comparable evaluation run."""

    configuration_id: str
    dataset_version: str
    provider: str
    chat_model: str
    embedding_model: str
    index_version: str
    prompt_version: str
    candidate_k: int
    final_k: int
    rrf_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in (
            "configuration_id",
            "dataset_version",
            "provider",
            "chat_model",
            "embedding_model",
            "index_version",
            "prompt_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalized_text(getattr(self, field_name), field_name=field_name),
            )
        for field_name in ("candidate_k", "final_k"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.final_k > self.candidate_k:
            raise ValueError("final_k must not exceed candidate_k")
        weights = dict(self.rrf_weights)
        if set(weights) != {"keyword", "dense"}:
            raise ValueError("rrf_weights must contain keyword and dense")
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
            for value in weights.values()
        ):
            raise ValueError("rrf_weights must be positive numbers")
        object.__setattr__(self, "rrf_weights", MappingProxyType(weights))


@dataclass(frozen=True)
class EvaluationPrediction:
    """One answer plus the real retrieval result and assembled contexts."""

    retrieval: RetrievalResult
    answer: str
    contexts: tuple[str, ...]

    def __post_init__(self) -> None:
        normalized_answer = _normalized_text(self.answer, field_name="answer")
        normalized_contexts = tuple(context.strip() for context in self.contexts)
        if any(not context for context in normalized_contexts):
            raise ValueError("contexts must not contain empty text")
        object.__setattr__(self, "answer", normalized_answer)
        object.__setattr__(self, "contexts", normalized_contexts)


class EvaluationPredictor(Protocol):
    """Predict one golden-set case while using the supplied stage observer."""

    def __call__(
        self,
        case: EvaluationCase,
        observer: StageObserver,
    ) -> EvaluationPrediction: ...


@dataclass(frozen=True)
class MLflowEvaluationSummary:
    """Inline equivalent of the essential MLflow run-comparison view."""

    run_id: str
    parameters: Mapping[str, str | int | float]
    metrics: Mapping[str, float]
    trace_count: int
    failure_rows: tuple[Mapping[str, object], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(
            self,
            "failure_rows",
            tuple(MappingProxyType(dict(row)) for row in self.failure_rows),
        )


class MLflowStageObserver:
    """Create nested MLflow spans at the provider-neutral Lesson 06 boundaries."""

    def __init__(self, *, run_id: str) -> None:
        self.run_id = _normalized_text(run_id, field_name="run_id")

    @contextmanager
    def span(
        self,
        name: str,
        *,
        inputs: Mapping[str, MeasurementValue],
    ) -> Iterator[StageSpan]:
        normalized_name = _normalized_text(name, field_name="span name")
        safe_inputs = dict(inputs)
        with mlflow.start_span(
            name=normalized_name,
            span_type="RETRIEVER" if normalized_name in {"eligibility", "keyword", "dense", "fusion", "rerank"} else "CHAIN",
        ) as live_span:
            live_span.set_inputs(safe_inputs)
            yield StageSpan(name=normalized_name, inputs=safe_inputs)
            live_span.set_outputs({"status": "completed"})


def _configuration_parameters(
    configuration: EvaluationConfiguration,
) -> dict[str, str | int | float]:
    return {
        "configuration_id": configuration.configuration_id,
        "dataset_version": configuration.dataset_version,
        "provider": configuration.provider,
        "chat_model": configuration.chat_model,
        "embedding_model": configuration.embedding_model,
        "index_version": configuration.index_version,
        "prompt_version": configuration.prompt_version,
        "candidate_k": configuration.candidate_k,
        "final_k": configuration.final_k,
        "rrf_weights": json.dumps(dict(configuration.rrf_weights), sort_keys=True),
    }


def _filter_payload(case: EvaluationCase) -> dict[str, str | None]:
    return {
        "company": case.filters.company,
        "period": case.filters.period,
        "document_type": case.filters.document_type,
        "section": case.filters.section,
    }


def run_mlflow_evaluation(
    *,
    tracking_path: Path,
    experiment_name: str,
    configuration: EvaluationConfiguration,
    cases: Sequence[EvaluationCase],
    predict_fn: EvaluationPredictor,
) -> MLflowEvaluationSummary:
    """Trace every case, log reproducibility metadata, and return the run summary."""

    if not cases:
        raise ValueError("cases must not be empty")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be unique")
    normalized_experiment = _normalized_text(
        experiment_name,
        field_name="experiment_name",
    )
    tracking_path.mkdir(parents=True, exist_ok=True)
    artifact_path = tracking_path / "artifacts"
    artifact_path.mkdir(parents=True, exist_ok=True)
    database_path = (tracking_path / "mlflow.db").resolve()
    mlflow.set_tracking_uri(f"sqlite:///{database_path}")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(normalized_experiment)
    if experiment is None:
        experiment_id = client.create_experiment(
            normalized_experiment,
            artifact_location=artifact_path.resolve().as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id
    parameters = _configuration_parameters(configuration)

    evaluations = []
    failure_rows = []
    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name=configuration.configuration_id,
    ) as active_run:
        run_id = active_run.info.run_id
        mlflow.log_params(parameters)
        for case in cases:
            observer = MLflowStageObserver(run_id=run_id)
            with mlflow.start_span(
                name=f"case:{case.case_id}",
                span_type="CHAIN",
                run_id=run_id,
            ) as root_span:
                root_span.set_inputs(
                    {
                        "case_id": case.case_id,
                        "question": case.question,
                        "filters": _filter_payload(case),
                        "configuration_id": configuration.configuration_id,
                    }
                )
                prediction = predict_fn(case, observer)
                result = evaluate_case(case, prediction.retrieval, prediction.answer)
                root_span.set_outputs(
                    {
                        "answer": prediction.answer,
                        "context_count": len(prediction.contexts),
                        "retrieved_ids": list(result.retrieved_ids),
                        "citations": list(result.parsed_citations),
                        "rerank_scores": [
                            hit.score for hit in prediction.retrieval.reranked_hits
                        ],
                        "stage_duration_ms": {
                            name: measurement.duration_ms
                            for name, measurement in prediction.retrieval.stage_measurements.items()
                        },
                        "failure_stage": result.failure_stage,
                    }
                )
            evaluations.append(result)
            if result.failure_stage != "none":
                failure_rows.append(
                    {
                        "case_id": case.case_id,
                        "configuration_id": configuration.configuration_id,
                        "failure_stage": result.failure_stage,
                        "expected_ids": list(case.expected_evidence_ids),
                        "retrieved_ids": list(result.retrieved_ids),
                        "citations": list(result.parsed_citations),
                        "retrieval_recall_at_k": result.retrieval_recall_at_k,
                        "reciprocal_rank": result.reciprocal_rank,
                        "filter_correctness": result.filter_correctness,
                        "citation_correctness": result.citation_correctness,
                        "grounded_fact_coverage": result.grounded_fact_coverage,
                        "abstention_correctness": result.abstention_correctness,
                    }
                )

        evaluation_summary = summarize_evaluations(evaluations)
        metrics = {
            key: float(value)
            for key, value in asdict(evaluation_summary).items()
            if key != "case_count"
        }
        mlflow.log_metrics(metrics)
        mlflow.log_dict(
            {"rows": failure_rows},
            "evaluation/failure_rows.json",
        )

    mlflow.flush_trace_async_logging()
    return MLflowEvaluationSummary(
        run_id=run_id,
        parameters=parameters,
        metrics=metrics,
        trace_count=len(cases),
        failure_rows=tuple(failure_rows),
    )
