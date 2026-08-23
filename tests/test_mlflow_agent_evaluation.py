"""SQLite-backed MLflow integration for Lesson 12 agent evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from finai_academy.agent_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationPrediction,
    load_agent_evaluation_dataset,
    load_recorded_agent_runs,
)
from finai_academy.mlflow_agent_evaluation import (
    AgentEvaluationConfiguration,
    compare_agent_configurations,
    initialize_local_mlflow,
    run_mlflow_agent_evaluation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "assets/course-data/evaluation/agent_cases_v1.json"
RUNS_PATH = PROJECT_ROOT / "assets/course-data/evaluation/agent_runs_v1.json"
MANIFEST_PATH = PROJECT_ROOT / "assets/course-data/manifest.json"


def _manifest_entry(section: str, key: str, value: str) -> dict[str, object]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return next(entry for entry in manifest[section] if entry[key] == value)


def fixture_run_inputs(
    configuration_id: str,
) -> tuple[
    AgentEvaluationConfiguration,
    tuple[AgentEvaluationCase, ...],
    tuple[AgentEvaluationPrediction, ...],
]:
    case_entry = _manifest_entry(
        "evaluation_datasets", "dataset_version", "agent-cases-v1"
    )
    run_entry = _manifest_entry(
        "evaluation_run_fixtures", "fixture_version", "agent-runs-v1"
    )
    dataset = load_agent_evaluation_dataset(
        CASES_PATH, expected_sha256=str(case_entry["sha256"])
    )
    runs = load_recorded_agent_runs(
        RUNS_PATH,
        cases=dataset,
        expected_sha256=str(run_entry["sha256"]),
    )
    recorded = next(
        item for item in runs.configurations if item.configuration_id == configuration_id
    )
    configuration = AgentEvaluationConfiguration(
        configuration_id=recorded.configuration_id,
        dataset_version=dataset.dataset_version,
        dataset_sha256=dataset.dataset_sha256,
        agent_version=recorded.agent_version,
        provider=recorded.provider,
        agent_model=recorded.agent_model,
        prompt_version=recorded.prompt_version,
        max_steps=recorded.max_steps,
        max_replans=recorded.max_replans,
    )
    return configuration, dataset.cases, recorded.predictions


def run_fixture_configuration(tmp_path: Path, configuration_id: str):
    configuration, cases, predictions = fixture_run_inputs(configuration_id)
    return run_mlflow_agent_evaluation(
        tracking_directory=tmp_path / "lesson12-mlflow",
        experiment_name="lesson-12-agent-evaluation-test",
        configuration=configuration,
        cases=cases,
        predictions=predictions,
    )


def run_both_fixture_configurations(tmp_path: Path):
    return tuple(
        run_fixture_configuration(tmp_path, configuration_id)
        for configuration_id in ("bounded-agent-v1", "regressed-agent-v0")
    )


def serialized_traces(run_id: str) -> str:
    experiment_id = mlflow.get_run(run_id).info.experiment_id
    traces = mlflow.search_traces(
        run_id=run_id,
        locations=[experiment_id],
        return_type="list",
        flush=True,
    )
    return json.dumps([trace.to_dict() for trace in traces], sort_keys=True)


def test_local_store_uses_resolved_sqlite_database_and_local_artifacts(
    tmp_path: Path,
) -> None:
    store = initialize_local_mlflow(tmp_path / "lesson12-mlflow")

    assert store.database_path == (tmp_path / "lesson12-mlflow" / "mlflow.db").resolve()
    assert store.tracking_uri == f"sqlite:///{store.database_path}"
    assert store.artifact_directory == (
        tmp_path / "lesson12-mlflow" / "artifacts"
    ).resolve()
    assert store.ui_command == f"mlflow ui --backend-store-uri sqlite:///{store.database_path}"


def test_one_configuration_logs_required_parameters_metrics_and_artifacts(
    tmp_path: Path,
) -> None:
    summary = run_fixture_configuration(tmp_path, "bounded-agent-v1")
    run = mlflow.get_run(summary.run_id)

    assert set(run.data.params) >= {
        "configuration_id",
        "dataset_version",
        "dataset_sha256",
        "agent_version",
        "provider",
        "agent_model",
        "judge_provider",
        "judge_model",
        "prompt_version",
        "max_steps",
        "max_replans",
        "scorer_contract_version",
    }
    assert set(run.data.metrics) >= {
        "tool_call_correctness_mean",
        "tool_call_efficiency_mean",
        "answer_relevance_mean",
        "answer_completeness_mean",
        "citation_integrity_mean",
        "mean_tool_calls",
        "mean_latency_ms",
    }
    assert set(summary.artifact_paths) == {
        "evaluation/case_scores.json",
        "evaluation/failure_rows.json",
        "evaluation/dataset_manifest.json",
    }


def test_each_case_has_one_root_trace_and_required_public_child_spans(
    tmp_path: Path,
) -> None:
    summary = run_fixture_configuration(tmp_path, "bounded-agent-v1")
    traces = mlflow.search_traces(
        run_id=summary.run_id,
        locations=[summary.experiment_id],
        return_type="list",
        flush=True,
    )

    assert len(traces) == 6
    assert set(summary.trace_ids) == set(summary.case_scores_by_id)
    assert set(summary.trace_ids.values()) == {trace.info.trace_id for trace in traces}
    required_chain = {"planning", "plan_gate", "replanning", "evidence_gate", "report"}
    for trace in traces:
        names = {span.name for span in trace.data.spans}
        assert required_chain <= names
        assert sum(span.span_type == "TOOL" for span in trace.data.spans) >= 1
        execution_spans = [
            span for span in trace.data.spans if span.name.startswith("execution:")
        ]
        assert all(span.attributes["attempt_id"] >= 1 for span in execution_spans)
        assert all(span.attributes["plan_revision"] >= 0 for span in execution_spans)


def test_trace_contains_only_public_safe_agent_fields(tmp_path: Path) -> None:
    summary = run_fixture_configuration(tmp_path, "bounded-agent-v1")
    payload = serialized_traces(summary.run_id)

    assert "OPENAI_API_KEY" not in payload
    assert "Authorization" not in payload
    assert "/Users/" not in payload
    assert "chain-of-thought" not in payload.casefold()
    assert "unsupported_metric" in payload
    assert "NVDA-FY2026-DATA-CENTER-001" in payload
    assert "source_kind" in payload
    assert "recorded" in payload


def test_trace_logging_flushes_before_counting_or_returning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    real_search_traces = mlflow.search_traces

    monkeypatch.setattr(
        mlflow,
        "flush_trace_async_logging",
        lambda: events.append("flush"),
    )

    def ordered_search(*args: object, **kwargs: object):
        events.append("search")
        return real_search_traces(*args, **kwargs)

    monkeypatch.setattr(mlflow, "search_traces", ordered_search)
    summary = run_fixture_configuration(tmp_path, "bounded-agent-v1")

    assert events == ["flush", "search"]
    assert summary.trace_count == 6


def test_logging_rejects_secret_shaped_configuration_before_starting_a_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configuration, cases, predictions = fixture_run_inputs("bounded-agent-v1")
    unsafe = configuration.model_copy(update={"agent_model": "sk-secret-shaped-value"})
    start_run = Mock(wraps=mlflow.start_run)
    monkeypatch.setattr(mlflow, "start_run", start_run)

    with pytest.raises(ValueError, match="secret-shaped"):
        run_mlflow_agent_evaluation(
            tracking_directory=tmp_path,
            experiment_name="lesson-12-secret-test",
            configuration=unsafe,
            cases=cases,
            predictions=predictions,
        )

    start_run.assert_not_called()


def test_logging_rejects_nested_secret_shaped_prediction_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configuration, cases, predictions = fixture_run_inputs("bounded-agent-v1")
    first = predictions[0]
    observation = first.observations[0].model_copy(
        update={"arguments": {"metadata": {"Authorization": "Bearer public-looking-token"}}}
    )
    unsafe = first.model_copy(
        update={"observations": (observation, *first.observations[1:])}
    )
    start_run = Mock(wraps=mlflow.start_run)
    monkeypatch.setattr(mlflow, "start_run", start_run)

    with pytest.raises(ValueError, match="secret-shaped"):
        run_mlflow_agent_evaluation(
            tracking_directory=tmp_path,
            experiment_name="lesson-12-nested-secret-test",
            configuration=configuration,
            cases=cases,
            predictions=(unsafe, *predictions[1:]),
        )

    start_run.assert_not_called()


def test_backend_failure_names_resolved_store_and_fails_verification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configuration, cases, predictions = fixture_run_inputs("bounded-agent-v1")
    monkeypatch.setattr(
        mlflow,
        "set_tracking_uri",
        Mock(side_effect=OSError("store unavailable")),
    )

    with pytest.raises(
        RuntimeError,
        match=r"lesson12-store.*mlflow\.db.*store unavailable",
    ):
        run_mlflow_agent_evaluation(
            tracking_directory=tmp_path / "lesson12-store",
            experiment_name="lesson-12-backend-test",
            configuration=configuration,
            cases=cases,
            predictions=predictions,
        )


def test_backend_failure_sanitizes_secret_and_personal_path_from_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configuration, cases, predictions = fixture_run_inputs("bounded-agent-v1")
    monkeypatch.setattr(
        mlflow,
        "set_tracking_uri",
        Mock(
            side_effect=OSError(
                "sk-secret-shaped-value unavailable at /Users/private/mlflow"
            )
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_mlflow_agent_evaluation(
            tracking_directory=tmp_path / "lesson12-store",
            experiment_name="lesson-12-backend-sanitizer-test",
            configuration=configuration,
            cases=cases,
            predictions=predictions,
        )

    message = str(exc_info.value)
    assert "sk-secret-shaped-value" not in message
    assert "/Users/" not in message


def test_existing_experiment_must_use_an_absolute_local_artifact_uri(
    tmp_path: Path,
) -> None:
    configuration, cases, predictions = fixture_run_inputs("bounded-agent-v1")
    initialize_local_mlflow(tmp_path / "lesson12-remote-artifacts")
    MlflowClient().create_experiment(
        "lesson-12-remote-artifact-test",
        artifact_location="s3://private-bucket/lesson12",
    )

    with pytest.raises(ValueError, match="absolute local artifact URI"):
        run_mlflow_agent_evaluation(
            tracking_directory=tmp_path / "lesson12-remote-artifacts",
            experiment_name="lesson-12-remote-artifact-test",
            configuration=configuration,
            cases=cases,
            predictions=predictions,
        )


def test_json_artifacts_have_exact_public_contracts(tmp_path: Path) -> None:
    summary = run_fixture_configuration(tmp_path, "bounded-agent-v1")
    client = MlflowClient()
    artifact_payloads = {
        artifact_path: json.loads(
            Path(client.download_artifacts(summary.run_id, artifact_path)).read_text(
                encoding="utf-8"
            )
        )
        for artifact_path in summary.artifact_paths
    }

    score_rows = artifact_payloads["evaluation/case_scores.json"]["rows"]
    failure_rows = artifact_payloads["evaluation/failure_rows.json"]["rows"]
    dataset_manifest = artifact_payloads["evaluation/dataset_manifest.json"]
    assert len(score_rows) == 6
    assert {row["case_id"] for row in score_rows} == set(summary.case_scores_by_id)
    assert dataset_manifest == {
        "dataset_version": "agent-cases-v1",
        "dataset_sha256": summary.parameters["dataset_sha256"],
        "scorer_contract_version": "agent-scorers-v1",
        "case_ids": list(summary.case_scores_by_id),
        "fixture_version": "agent-runs-v1",
        "configuration_id": "bounded-agent-v1",
    }
    for row in failure_rows:
        assert set(row) == {
            "case_id",
            "configuration_id",
            "failure_stage",
            "tool_call_correctness",
            "tool_call_correctness_rationale",
            "tool_call_efficiency",
            "tool_call_efficiency_rationale",
            "answer_relevance",
            "answer_relevance_rationale",
            "answer_completeness",
            "answer_completeness_rationale",
            "citation_integrity",
            "citation_integrity_rationale",
        }
        assert all(
            isinstance(row[name], float)
            for name in (
                "tool_call_correctness",
                "tool_call_efficiency",
                "answer_relevance",
                "answer_completeness",
                "citation_integrity",
            )
        )
    serialized_artifacts = json.dumps(artifact_payloads, sort_keys=True)
    assert str((tmp_path / "lesson12-mlflow").resolve()) not in serialized_artifacts


def test_comparison_rejects_different_case_hashes_and_returns_heatmap_rows(
    tmp_path: Path,
) -> None:
    bounded, regressed = run_both_fixture_configurations(tmp_path)
    comparison = compare_agent_configurations((bounded, regressed))

    assert len(comparison.case_metric_rows) == 12
    assert comparison.configuration_ids == (
        "bounded-agent-v1",
        "regressed-agent-v0",
    )
    assert len(comparison.metric_mean_rows) == 10
    assert len(comparison.metric_pass_rows) == 10
    assert len(comparison.tool_call_rows) == 12
    assert len(comparison.latency_rows) == 2
    with pytest.raises(ValueError, match="dataset version and SHA-256"):
        compare_agent_configurations(
            (
                bounded,
                regressed.model_copy(
                    update={
                        "parameters": {
                            **regressed.parameters,
                            "dataset_sha256": "b" * 64,
                        }
                    }
                ),
            )
        )
