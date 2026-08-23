"""SQLite-backed MLflow integration for Lesson 12 agent evaluation."""

from __future__ import annotations

import inspect
import json
import traceback
from pathlib import Path
from unittest.mock import Mock

import mlflow
import pytest
from mlflow.entities import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai import scorers as genai_scorers
from mlflow.genai.judges.adapters.gateway_adapter import GatewayAdapter
from mlflow.genai.judges.adapters.utils import get_adapter
from mlflow.genai.scorers import builtin_scorers
from mlflow.genai.utils import trace_utils
from mlflow.tracking import MlflowClient

from finai_academy import mlflow_agent_evaluation
from finai_academy.agent_evaluation import (
    AgentEvaluationCase,
    AgentEvaluationPrediction,
    load_agent_evaluation_dataset,
    load_recorded_agent_runs,
)
from finai_academy.mlflow_agent_evaluation import (
    AgentEvaluationConfiguration,
    build_optional_genai_scorers,
    compare_agent_configurations,
    initialize_local_mlflow,
    load_judge_configuration,
    run_mlflow_agent_evaluation,
    run_optional_judges,
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


def traces_by_case(summary: object) -> dict[str, object]:
    traces = mlflow.search_traces(
        run_id=summary.run_id,
        locations=[summary.experiment_id],
        return_type="list",
        flush=True,
    )
    result = {}
    for trace in traces:
        root = next(span for span in trace.data.spans if span.parent_id is None)
        result[root.inputs["case_id"]] = trace
    return result


def test_no_explicit_judge_model_returns_four_not_run_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    logged: list[tuple[str, object]] = []
    initialize_local_mlflow(tmp_path / "judge-not-configured")
    monkeypatch.setattr(
        MlflowClient,
        "log_dict",
        lambda self, run_id, dictionary, artifact_file: logged.append(
            (artifact_file, dictionary)
        ),
    )

    assert load_judge_configuration({"OPENAI_API_KEY": "present-but-ambient"}) is None
    results = run_optional_judges(
        run_id="test-run",
        configuration=None,
        traces=(),
    )

    assert [result.scorer_name for result in results] == [
        "ToolCallCorrectness",
        "ToolCallEfficiency",
        "RelevanceToQuery",
        "Completeness",
    ]
    assert all(result.status == "NOT RUN" for result in results)
    assert all(result.score is None for result in results)
    assert logged[0][0] == "evaluation/judge_results.json"


@pytest.mark.parametrize(
    ("uri", "provider", "model"),
    [
        ("openai:/gpt-5-mini", "openai", "gpt-5-mini"),
        ("ollama_chat:/qwen3:8b", "ollama", "qwen3:8b"),
    ],
)
def test_judge_model_uri_selects_exactly_one_explicit_provider(
    uri: str,
    provider: str,
    model: str,
) -> None:
    config = load_judge_configuration({"FINAI_EVAL_JUDGE_MODEL": uri})

    assert config is not None
    assert (config.provider, config.model_uri, config.model) == (
        provider,
        uri,
        model,
    )


def test_invalid_or_credential_only_judge_configuration_never_falls_back() -> None:
    with pytest.raises(ValueError, match="FINAI_EVAL_JUDGE_MODEL"):
        load_judge_configuration({"FINAI_EVAL_JUDGE_MODEL": "gpt-5-mini"})


def test_explicit_judge_model_rejects_secret_shaped_text() -> None:
    with pytest.raises(ValueError, match="secret-shaped"):
        load_judge_configuration(
            {"FINAI_EVAL_JUDGE_MODEL": "openai:/sk-secret-shaped-value"}
        )


def test_installed_mlflow_judge_signatures_support_explicit_model_and_trace() -> None:
    for scorer_class in (
        genai_scorers.ToolCallCorrectness,
        genai_scorers.ToolCallEfficiency,
        genai_scorers.RelevanceToQuery,
        genai_scorers.Completeness,
    ):
        assert "model" in inspect.signature(scorer_class).parameters
        assert "trace" in inspect.signature(scorer_class.__call__).parameters


def test_optional_judge_scorers_use_the_explicit_model_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[tuple[str, str]] = []

    def fake_scorer_class(name: str) -> type:
        class FakeScorer:
            def __init__(self, *, model: str) -> None:
                constructed.append((name, model))

        FakeScorer.__name__ = name
        return FakeScorer

    for name in (
        "ToolCallCorrectness",
        "ToolCallEfficiency",
        "RelevanceToQuery",
        "Completeness",
    ):
        monkeypatch.setattr(genai_scorers, name, fake_scorer_class(name))

    configuration = load_judge_configuration(
        {"FINAI_EVAL_JUDGE_MODEL": "ollama_chat:/qwen3:8b"}
    )
    scorer_set = build_optional_genai_scorers(configuration)

    assert scorer_set.configuration == configuration
    assert [type(scorer).__name__ for scorer in scorer_set.scorers] == [
        "ToolCallCorrectness",
        "ToolCallEfficiency",
        "RelevanceToQuery",
        "Completeness",
    ]
    assert constructed == [
        ("ToolCallCorrectness", "ollama:/qwen3:8b"),
        ("ToolCallEfficiency", "ollama:/qwen3:8b"),
        ("RelevanceToQuery", "ollama:/qwen3:8b"),
        ("Completeness", "ollama:/qwen3:8b"),
    ]


def test_ollama_judge_course_uri_selects_native_same_provider_adapter_without_network() -> None:
    configuration = load_judge_configuration(
        {"FINAI_EVAL_JUDGE_MODEL": "ollama_chat:/qwen3:8b"}
    )
    scorer_set = build_optional_genai_scorers(configuration)

    assert configuration is not None
    assert (
        configuration.provider,
        configuration.model_uri,
        configuration.model,
    ) == ("ollama", "ollama_chat:/qwen3:8b", "qwen3:8b")
    assert {scorer.model for scorer in scorer_set.scorers} == {"ollama:/qwen3:8b"}
    assert isinstance(get_adapter("ollama:/qwen3:8b", prompt="adapter probe"), GatewayAdapter)


def test_judge_trace_normalization_matches_installed_tool_extractors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    summary = run_fixture_configuration(tmp_path, "bounded-agent-v1")
    persisted = traces_by_case(summary)["reference_completed"]
    normalized = mlflow_agent_evaluation._normalize_trace_for_judges(persisted)

    monkeypatch.setattr(
        trace_utils,
        "get_default_model",
        Mock(side_effect=AssertionError("implicit default judge model reached")),
    )
    called = trace_utils.extract_tools_called_from_trace(normalized)
    available = trace_utils.extract_available_tools_from_trace(normalized)

    assert [call.name for call in called] == [
        "get_company_metric",
        "get_company_metric",
        "get_company_metric",
        "search_financial_documents",
        "search_financial_documents",
    ]
    assert called[0].arguments == {"ticker": "NVDA", "metric": "P/E"}
    assert [tool.function.name for tool in available] == [
        "get_company_metric",
        "search_financial_documents",
    ]
    assert available[0].function.parameters.required == ["ticker", "metric"]
    observed_models: list[str] = []

    def accept_tool_calls(**kwargs: object) -> Feedback:
        observed_models.append(str(kwargs["model"]))
        assert [call.name for call in kwargs["tools_called"]] == [
            call.name for call in called
        ]
        assert [tool.function.name for tool in kwargs["available_tools"]] == [
            tool.function.name for tool in available
        ]
        return Feedback(value=True, rationale="normalized trace accepted")

    monkeypatch.setattr(
        builtin_scorers.judges,
        "is_tool_call_correct",
        accept_tool_calls,
    )
    monkeypatch.setattr(
        builtin_scorers.judges,
        "is_tool_call_efficient",
        accept_tool_calls,
    )
    scorer_set = build_optional_genai_scorers(
        load_judge_configuration(
            {"FINAI_EVAL_JUDGE_MODEL": "ollama_chat:/qwen3:8b"}
        )
    )
    for scorer in scorer_set.scorers[:2]:
        assert scorer(trace=normalized).value is True
    assert observed_models == ["ollama:/qwen3:8b", "ollama:/qwen3:8b"]
    persisted_root = next(
        span for span in persisted.data.spans if span.parent_id is None
    )
    assert persisted_root.span_type == "CHAIN"
    assert [span.name for span in persisted.search_spans(span_type="TOOL")] == [
        "execution:1",
        "execution:2",
        "execution:3",
        "execution:4",
        "execution:5",
    ]
    assert persisted.search_spans(span_type="TOOL")[0].inputs["capability"] == (
        "get_company_metric"
    )


def test_completed_judges_log_four_truthful_rows_and_separate_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_logs: list[tuple[str, object]] = []
    metric_logs: list[tuple[str, float]] = []
    initialize_local_mlflow(tmp_path / "judge-completed")

    class PassingScorer:
        def __init__(self, *, model: str) -> None:
            self.model = model

        def __call__(self, *, trace: object) -> object:
            return Feedback(value=True, rationale=f"accepted {trace}")

    for name in (
        "ToolCallCorrectness",
        "ToolCallEfficiency",
        "RelevanceToQuery",
        "Completeness",
    ):
        monkeypatch.setattr(genai_scorers, name, PassingScorer)
    monkeypatch.setattr(mlflow, "__version__", "3.15.1-test")
    monkeypatch.setattr(
        MlflowClient,
        "log_dict",
        lambda self, run_id, dictionary, artifact_file: artifact_logs.append(
            (artifact_file, dictionary)
        ),
    )
    monkeypatch.setattr(
        MlflowClient,
        "log_metric",
        lambda self, run_id, key, value: metric_logs.append((key, value)),
    )
    configuration = load_judge_configuration(
        {"FINAI_EVAL_JUDGE_MODEL": "openai:/gpt-5-mini"}
    )

    results = run_optional_judges(
        run_id="test-run",
        configuration=configuration,
        traces=("trace-a", "trace-b"),
    )

    assert len(results) == 4
    assert all(result.status == "COMPLETED" for result in results)
    assert all(result.score == 1.0 for result in results)
    assert all(result.provider == "openai" for result in results)
    assert all(result.model == "gpt-5-mini" for result in results)
    assert all(result.mlflow_version == "3.15.1-test" for result in results)
    assert all(result.latency_ms >= 0 for result in results)
    assert artifact_logs == [
        (
            "evaluation/judge_results.json",
            {"rows": [result.model_dump(mode="json") for result in results]},
        )
    ]
    assert metric_logs == [
        ("judge_tool_call_correctness", 1.0),
        ("judge_tool_call_efficiency", 1.0),
        ("judge_relevance_to_query", 1.0),
        ("judge_completeness", 1.0),
    ]


@pytest.mark.parametrize(
    ("error_type", "reason"),
    [
        (ImportError, "provider client unavailable"),
        (MlflowException, "No suitable adapter found for explicit provider"),
        (
            MlflowException,
            "OPENAI_API_KEY environment variable must be set to use the openai provider",
        ),
        (MlflowException, "Failed to connect to http://localhost:11434/v1"),
    ],
)
def test_call_time_unavailable_judge_is_not_run_and_logs_no_metric(
    error_type: type[Exception],
    reason: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_logs: list[tuple[str, object]] = []
    metric_logs: list[tuple[str, float]] = []
    constructed: list[str] = []
    initialize_local_mlflow(tmp_path / "judge-unavailable")

    class UnavailableScorer:
        def __init__(self, *, model: str) -> None:
            constructed.append(model)

        def __call__(self, *, trace: object) -> object:
            raise error_type(reason)

    for name in (
        "ToolCallCorrectness",
        "ToolCallEfficiency",
        "RelevanceToQuery",
        "Completeness",
    ):
        monkeypatch.setattr(genai_scorers, name, UnavailableScorer)
    monkeypatch.setattr(
        MlflowClient,
        "log_dict",
        lambda self, run_id, dictionary, artifact_file: artifact_logs.append(
            (artifact_file, dictionary)
        ),
    )
    monkeypatch.setattr(
        MlflowClient,
        "log_metric",
        lambda self, run_id, key, value: metric_logs.append((key, value)),
    )
    configuration = load_judge_configuration(
        {"FINAI_EVAL_JUDGE_MODEL": "openai:/gpt-5-mini"}
    )

    results = run_optional_judges(
        run_id="test-run",
        configuration=configuration,
        traces=("trace-a",),
    )

    assert len(results) == 4
    assert len(constructed) == 4
    assert all(result.status == "NOT RUN" for result in results)
    assert all(result.score is None for result in results)
    assert all(error_type.__name__ in result.rationale for result in results)
    assert metric_logs == []
    assert artifact_logs[0][0] == "evaluation/judge_results.json"


@pytest.mark.parametrize("error_type", [TimeoutError, RuntimeError])
def test_judge_error_does_not_mutate_deterministic_metrics_or_release_status(
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FailingScorer:
        def __init__(self, *, model: str) -> None:
            self.model = model

        def __call__(self, *, trace: object) -> object:
            raise error_type(
                "hidden reasoning; Authorization: Bearer private-token at "
                "/Users/alice/private"
            )

    for name in (
        "ToolCallCorrectness",
        "ToolCallEfficiency",
        "RelevanceToQuery",
        "Completeness",
    ):
        monkeypatch.setattr(genai_scorers, name, FailingScorer)
    store = initialize_local_mlflow(tmp_path / "judge-separation")
    client = MlflowClient(tracking_uri=store.tracking_uri)
    experiment_id = client.create_experiment(
        "judge-separation-test",
        artifact_location=store.artifact_directory.as_uri(),
    )
    run = client.create_run(experiment_id)
    deterministic_metrics = {
        "tool_call_correctness_mean": 1.0,
        "tool_call_efficiency_mean": 1.0,
        "answer_relevance_mean": 1.0,
        "answer_completeness_mean": 0.75,
        "citation_integrity_mean": 1.0,
        "release_passed": 1.0,
    }
    for key, value in deterministic_metrics.items():
        client.log_metric(run.info.run_id, key, value)
    configuration = load_judge_configuration(
        {"FINAI_EVAL_JUDGE_MODEL": "ollama_chat:/qwen3:8b"}
    )

    results = run_optional_judges(
        run_id=run.info.run_id,
        configuration=configuration,
        traces=("trace-a",),
    )

    persisted_metrics = client.get_run(run.info.run_id).data.metrics
    assert all(result.status == "ERROR" for result in results)
    assert all(result.score is None for result in results)
    assert all(result.provider == "ollama" for result in results)
    assert all(result.model == "qwen3:8b" for result in results)
    assert all(error_type.__name__ in result.rationale for result in results)
    assert all("Authorization" not in result.rationale for result in results)
    assert all("/Users/" not in result.rationale for result in results)
    assert all("hidden reasoning" not in result.rationale for result in results)
    assert persisted_metrics == deterministic_metrics


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

    expected_parameters = {
        "configuration_id": "bounded-agent-v1",
        "dataset_version": "agent-cases-v1",
        "dataset_sha256": "c8f81fc59b182df8b2044c70d759fcb1fdac1fa90faead4bb70812b409ba0131",
        "agent_version": "lesson11-certified-v1",
        "provider": "recorded",
        "agent_model": "recorded-public-fixture-v1",
        "judge_provider": "none",
        "judge_model": "none",
        "prompt_version": "lesson11-recorded-policies-v1",
        "max_steps": 6,
        "max_replans": 1,
        "scorer_contract_version": "agent-scorers-v1",
    }
    expected_metrics = {
        "tool_call_correctness_mean": 1.0,
        "tool_call_efficiency_mean": 1.0,
        "answer_relevance_mean": 1.0,
        "answer_completeness_mean": 0.7777777777777778,
        "citation_integrity_mean": 1.0,
        "mean_tool_calls": 4.5,
        "mean_latency_ms": 53.0,
    }
    assert summary.parameters == expected_parameters
    assert run.data.params == {key: str(value) for key, value in expected_parameters.items()}
    assert summary.metrics == expected_metrics
    assert run.data.metrics == pytest.approx(expected_metrics)
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

    _, _, predictions = fixture_run_inputs("bounded-agent-v1")
    predictions_by_case = {prediction.case_id: prediction for prediction in predictions}
    assert len(traces) == 6
    assert set(summary.trace_ids) == set(summary.case_scores_by_id)
    assert set(summary.trace_ids.values()) == {trace.info.trace_id for trace in traces}
    required_chain = {"planning", "plan_gate", "replanning", "evidence_gate", "report"}
    for trace in traces:
        roots = [span for span in trace.data.spans if span.parent_id is None]
        assert len(roots) == 1
        root = roots[0]
        case_id = root.inputs["case_id"]
        prediction = predictions_by_case[case_id]
        assert root.name == f"case:{case_id}"
        assert root.inputs.keys() == {
            "case_id",
            "mission",
            "configuration_id",
            "expected_tool_calls",
        }
        assert root.outputs.keys() == {
            "observed_status",
            "expected_status",
            "plan_revisions",
            "evidence_gate",
            "briefing",
            "scores",
            "failure_stage",
            "latency_ms",
        }
        children = [span for span in trace.data.spans if span.parent_id is not None]
        assert all(span.parent_id == root.span_id for span in children)
        assert trace.info.trace_metadata["mlflow.sourceRun"] == summary.run_id
        names = {span.name for span in trace.data.spans}
        assert required_chain <= names
        execution_spans = [
            span for span in trace.data.spans if span.name.startswith("execution:")
        ]
        assert len(execution_spans) == len(prediction.observations)
        assert {span.name for span in execution_spans} == {
            f"execution:{observation.attempt_id}"
            for observation in prediction.observations
        }
        assert all(span.span_type == "TOOL" for span in execution_spans)
        assert all(span.attributes["attempt_id"] >= 1 for span in execution_spans)
        assert all(span.attributes["plan_revision"] >= 0 for span in execution_spans)
        assert all(
            span.inputs.keys()
            == {"capability", "arguments", "step_id", "attempt_id", "plan_revision"}
            for span in execution_spans
        )
        assert all(
            span.outputs.keys()
            == {
                "status",
                "error_code",
                "evidence_ids",
                "source_references",
                "duration_ms",
            }
            for span in execution_spans
        )
        replanning_spans = [span for span in children if span.name == "replanning"]
        assert all("plan_revision" in span.attributes for span in replanning_spans)
        ordered_names = [
            span.name for span in sorted(children, key=lambda span: span.start_time_ns)
        ]
        assert ordered_names.index("planning") < ordered_names.index("plan_gate")
        assert ordered_names.index("evidence_gate") < ordered_names.index("report")

    reference_trace = next(
        trace
        for trace in traces
        if next(span for span in trace.data.spans if span.parent_id is None).inputs["case_id"]
        == "reference_completed"
    )
    reference_replanning = sorted(
        (span for span in reference_trace.data.spans if span.name == "replanning"),
        key=lambda span: span.start_time_ns,
    )
    assert [span.attributes["plan_revision"] for span in reference_replanning] == [
        0,
        0,
        1,
        1,
        1,
    ]


def test_guardrail_event_is_preserved_on_owning_execution_span(tmp_path: Path) -> None:
    summary = run_fixture_configuration(tmp_path, "bounded-agent-v1")
    trace = traces_by_case(summary)["unsupported_metric_not_recovered"]
    execution = next(span for span in trace.data.spans if span.name == "execution:3")

    assert [(event.name, event.attributes) for event in execution.events] == [
        (
            "guardrail",
            {
                "status": "blocked",
                "summary": "Execution stopped after the unsupported metric was not recovered.",
                "duration_ms": 1.0,
                "trajectory_index": 8,
            },
        )
    ]


def test_reconstructed_evidence_gate_precedes_existing_report(tmp_path: Path) -> None:
    summary = run_fixture_configuration(tmp_path, "regressed-agent-v0")
    trace = traces_by_case(summary)["unsupported_metric_not_recovered"]
    root = next(span for span in trace.data.spans if span.parent_id is None)
    ordered = sorted(
        (span for span in trace.data.spans if span.parent_id == root.span_id),
        key=lambda span: span.start_time_ns,
    )
    names = [span.name for span in ordered]
    evidence_gate = ordered[names.index("evidence_gate")]
    report = ordered[names.index("report")]

    assert evidence_gate.start_time_ns < report.start_time_ns
    assert evidence_gate.attributes["reconstructed"] is True
    assert report.outputs == {
        "status": "error",
        "summary": "Briefing emitted without a recovery strategy.",
        "duration_ms": 4.0,
    }


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
    formatted = "".join(traceback.format_exception(exc_info.value))
    assert "sk-secret-shaped-value" not in message
    assert "/Users/" not in message
    assert "sk-secret-shaped-value" not in formatted
    assert "/Users/private/mlflow" not in formatted
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize(
    ("reason", "generic_reason", "forbidden_fragments"),
    [
        (
            (
                "credential rejected: OPENAI_API_KEY=topsecretvalue at "
                "/Users/alice/private-cache/secret-store.sqlite"
            ),
            "credential rejected",
            (
                "OPENAI_API_KEY",
                "topsecretvalue",
                "alice",
                "private-cache",
                "secret-store.sqlite",
            ),
        ),
        (
            (
                "authentication failed: Authorization: Basic dXNlcjpwYXNz at "
                "/home/bob/.config/mlflow/private.sqlite"
            ),
            "authentication failed",
            (
                "Authorization",
                "Basic dXNlcjpwYXNz",
                "bob",
                ".config/mlflow",
                "private.sqlite",
            ),
        ),
        (
            (
                "authentication failed: Authorization: Bearer bearer-token-value at "
                "/Users/carol/Library/Application Support/mlflow/private.db"
            ),
            "authentication failed",
            (
                "Authorization",
                "Bearer bearer-token-value",
                "carol",
                "Library/Application Support",
                "mlflow/private.db",
            ),
        ),
    ],
    ids=("api-key-macos", "basic-linux", "bearer-macos-spaces"),
)
def test_backend_failure_redacts_complete_credentials_and_home_paths(
    reason: str,
    generic_reason: str,
    forbidden_fragments: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tracking_directory = tmp_path / "lesson12-sanitized-store"
    database_path = (tracking_directory / "mlflow.db").resolve()
    monkeypatch.setattr(
        mlflow,
        "set_tracking_uri",
        Mock(side_effect=OSError(reason)),
    )

    with pytest.raises(RuntimeError) as exc_info:
        initialize_local_mlflow(tracking_directory)

    formatted = "".join(traceback.format_exception(exc_info.value))
    assert str(database_path) in formatted
    assert generic_reason in formatted
    assert "[credential redacted]" in formatted
    assert "[personal path redacted]" in formatted
    assert all(fragment not in formatted for fragment in forbidden_fragments)
    assert exc_info.value.__cause__ is None


def test_unusable_tracking_file_is_wrapped_by_safe_initialization(tmp_path: Path) -> None:
    tracking_file = tmp_path / "lesson12-store-file"
    tracking_file.write_text("not a directory", encoding="utf-8")
    database_path = (tracking_file / "mlflow.db").resolve()

    with pytest.raises(RuntimeError) as exc_info:
        initialize_local_mlflow(tracking_file)

    assert str(database_path) in str(exc_info.value)
    assert "File exists" in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.parametrize("error_type", [OSError, ValueError])
def test_client_initialization_traceback_does_not_retain_raw_backend_cause(
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configuration, cases, predictions = fixture_run_inputs("bounded-agent-v1")
    monkeypatch.setattr(
        mlflow_agent_evaluation,
        "MlflowClient",
        Mock(
            side_effect=error_type(
                "sk-client-secret-value unavailable at /Users/private/client"
            )
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_mlflow_agent_evaluation(
            tracking_directory=tmp_path / "lesson12-client-store",
            experiment_name="lesson-12-client-failure-test",
            configuration=configuration,
            cases=cases,
            predictions=predictions,
        )

    formatted = "".join(traceback.format_exception(exc_info.value))
    assert "sk-client-secret-value" not in formatted
    assert "/Users/private/client" not in formatted
    assert exc_info.value.__cause__ is None


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
    assert [row["case_id"] for row in failure_rows] == [
        "unsupported_metric_not_recovered",
        "missing_schneider_document",
    ]
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
    regressed_mean = next(
        row
        for row in comparison.metric_mean_rows
        if row["configuration_id"] == "regressed-agent-v0"
        and row["metric"] == "tool_call_efficiency"
    )
    assert regressed_mean == {
        "configuration_id": "regressed-agent-v0",
        "metric": "tool_call_efficiency",
        "mean": 0.9333333333333332,
    }
    regressed_pass = next(
        row
        for row in comparison.metric_pass_rows
        if row["configuration_id"] == "regressed-agent-v0"
        and row["metric"] == "citation_integrity"
    )
    assert regressed_pass == {
        "configuration_id": "regressed-agent-v0",
        "metric": "citation_integrity",
        "pass_count": 4,
        "case_count": 6,
    }
    redundant_case = next(
        row
        for row in comparison.case_metric_rows
        if row["configuration_id"] == "regressed-agent-v0"
        and row["case_id"] == "redundant_metric_call"
    )
    assert redundant_case == {
        "configuration_id": "regressed-agent-v0",
        "case_id": "redundant_metric_call",
        "tool_call_correctness": 1.0,
        "tool_call_efficiency": 0.6,
        "answer_relevance": 1.0,
        "answer_completeness": 1.0,
        "citation_integrity": 1.0,
        "failure_stage": "replanner",
        "trace_id": regressed.trace_ids["redundant_metric_call"],
        "run_id": regressed.run_id,
    }
    assert next(
        row
        for row in comparison.tool_call_rows
        if row["configuration_id"] == "regressed-agent-v0"
        and row["case_id"] == "redundant_metric_call"
    ) == {
        "configuration_id": "regressed-agent-v0",
        "case_id": "redundant_metric_call",
        "total_tool_calls": 6,
        "redundant_tool_calls": 1,
    }
    assert next(
        row
        for row in comparison.latency_rows
        if row["configuration_id"] == "regressed-agent-v0"
    ) == {
        "configuration_id": "regressed-agent-v0",
        "mean_latency_ms": 55.833333333333336,
        "max_latency_ms": 70.0,
    }
    assert comparison.failure_rows
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

    with pytest.raises(ValueError, match="dataset version and SHA-256"):
        compare_agent_configurations(
            (
                bounded,
                regressed.model_copy(
                    update={
                        "parameters": {
                            **regressed.parameters,
                            "dataset_version": "agent-cases-v2",
                        }
                    }
                ),
            )
        )

    missing_case_id = "wrong_source_evidence_pair"
    with pytest.raises(ValueError, match="identical aligned case IDs"):
        compare_agent_configurations(
            (
                bounded,
                regressed.model_copy(
                    update={
                        "case_scores_by_id": {
                            case_id: score
                            for case_id, score in regressed.case_scores_by_id.items()
                            if case_id != missing_case_id
                        },
                        "trace_ids": {
                            case_id: trace_id
                            for case_id, trace_id in regressed.trace_ids.items()
                            if case_id != missing_case_id
                        },
                    }
                ),
            )
        )

    with pytest.raises(ValueError, match="unique configuration IDs"):
        compare_agent_configurations(
            (
                bounded,
                regressed.model_copy(
                    update={
                        "parameters": {
                            **regressed.parameters,
                            "configuration_id": "bounded-agent-v1",
                        }
                    }
                ),
            )
        )
