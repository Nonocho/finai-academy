from __future__ import annotations

import inspect
import json
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import ValidationError

from finai_academy import agent_evaluation
from finai_academy.agent_evaluation import (
    AgentCaseScores,
    AgentEvaluationCase,
    AgentEvaluationPrediction,
    AgentEvaluationSummary,
    CandidateFact,
    ExpectedToolCall,
    MetricScore,
    canonical_call_signature,
    prediction_from_plan_execute_result,
)
from finai_academy.plan_execute_graph import PlanExecuteResult
from finai_academy.research_planning import (
    AnalystBriefing,
    CitedFact,
    EvidenceGateResult,
    PlanStep,
    ResearchObservation,
    ResearchPlan,
    TrajectoryEvent,
)

MISSION = (
    "Produce a concise NVIDIA and Schneider Electric briefing. Compare their available "
    "valuation metrics and latest operating-growth evidence."
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = PROJECT_ROOT / "assets/course-data/evaluation/agent_cases_v1.json"
RUNS_PATH = PROJECT_ROOT / "assets/course-data/evaluation/agent_runs_v1.json"
MANIFEST_PATH = PROJECT_ROOT / "assets/course-data/manifest.json"
METRIC_NAMES = (
    "tool_call_correctness",
    "tool_call_efficiency",
    "answer_relevance",
    "answer_completeness",
    "citation_integrity",
)


def load_agent_evaluation_dataset(*args: object, **kwargs: object):
    return agent_evaluation.load_agent_evaluation_dataset(*args, **kwargs)


def load_recorded_agent_runs(*args: object, **kwargs: object):
    return agent_evaluation.load_recorded_agent_runs(*args, **kwargs)


def align_cases_and_predictions(*args: object, **kwargs: object):
    return agent_evaluation.align_cases_and_predictions(*args, **kwargs)


def score_agent_case(*args: object, **kwargs: object):
    return agent_evaluation.score_agent_case(*args, **kwargs)


def summarize_agent_evaluation(*args: object, **kwargs: object):
    return agent_evaluation.summarize_agent_evaluation(*args, **kwargs)


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def manifest_case_hash() -> str:
    entries = _manifest()["evaluation_datasets"]
    assert isinstance(entries, list)
    return next(
        str(item["sha256"]) for item in entries if item["dataset_version"] == "agent-cases-v1"
    )


def manifest_run_hash() -> str:
    entries = _manifest()["evaluation_run_fixtures"]
    assert isinstance(entries, list)
    return next(
        str(item["sha256"]) for item in entries if item["fixture_version"] == "agent-runs-v1"
    )


def initial_plan() -> ResearchPlan:
    return ResearchPlan(
        goal=MISSION,
        steps=(
            PlanStep(
                step_id=1,
                capability="get_company_metric",
                arguments={"ticker": "NVDA", "metric": "P/E"},
                purpose="Collect NVIDIA valuation evidence.",
                expected_evidence=("NVIDIA P/E",),
            ),
            PlanStep(
                step_id=2,
                capability="search_financial_documents",
                arguments={
                    "company": "Schneider Electric",
                    "query": "revenue growth",
                    "top_k": 2,
                },
                purpose="Collect Schneider Electric operating evidence.",
                expected_evidence=("Schneider Electric revenue evidence",),
                depends_on=(1,),
            ),
        ),
    )


def completed_plan_execute_result() -> PlanExecuteResult:
    plan = initial_plan()
    observations = (
        ResearchObservation(
            attempt_id=1,
            step_id=1,
            plan_revision=0,
            capability="get_company_metric",
            arguments={"ticker": "NVDA", "metric": "P/E"},
            status="ok",
            result={"company": "NVIDIA", "metric": "P/E", "value": 47.2},
            source_references=("NVIDIA metrics snapshot",),
            duration_ms=1.5,
        ),
        ResearchObservation(
            attempt_id=2,
            step_id=2,
            plan_revision=1,
            capability="search_financial_documents",
            arguments={
                "company": "Schneider Electric",
                "query": "revenue growth",
                "top_k": 2,
            },
            status="ok",
            result={
                "company": "Schneider Electric",
                "hits": [
                    {
                        "evidence_id": "se-fy2025",
                        "source": "Schneider Electric FY2025 excerpt",
                    }
                ],
            },
            evidence_ids=("se-fy2025",),
            source_references=("Schneider Electric FY2025 excerpt",),
            duration_ms=2.5,
        ),
    )
    briefing = AnalystBriefing(
        reported_facts=(
            CitedFact(
                claim="NVIDIA's maintained P/E is 47.2x.",
                provenance_kind="metric",
                source_references=("NVIDIA metrics snapshot",),
            ),
            CitedFact(
                claim="Schneider Electric reported maintained revenue-growth evidence.",
                provenance_kind="document",
                source_references=("Schneider Electric FY2025 excerpt",),
                evidence_ids=("se-fy2025",),
            ),
        ),
        cross_company_observations=("The maintained evidence uses different periods.",),
        interpretation=("The evidence is descriptive, not investment advice.",),
        limitations=("Currencies, periods, and business definitions differ.",),
        source_references=(
            "NVIDIA metrics snapshot",
            "Schneider Electric FY2025 excerpt",
        ),
    )
    return PlanExecuteResult(
        status="completed",
        initial_plan=plan,
        final_steps=plan.steps,
        observations=observations,
        trajectory=(
            TrajectoryEvent(
                index=1,
                phase="planning",
                status="ok",
                summary="Planner proposed two research steps.",
                duration_ms=0.5,
            ),
            TrajectoryEvent(
                index=2,
                phase="evidence_gate",
                status="ok",
                summary="Evidence requirements passed.",
                duration_ms=0.25,
            ),
        ),
        replan_count=1,
        evidence_gate=EvidenceGateResult(
            passed=True,
            coverage={
                "NVIDIA": ("metric",),
                "Schneider Electric": ("document",),
            },
        ),
        briefing=briefing,
    )


def prediction_metadata() -> dict[str, object]:
    return {
        "case_id": "reference_completed",
        "dataset_version": "agent-cases-v1",
        "dataset_sha256": "a" * 64,
        "configuration_id": "bounded-agent-v1",
        "agent_version": "lesson11-certified-v1",
        "provider": "recorded",
        "agent_model": "recorded-public-fixture-v1",
        "prompt_version": "lesson11-recorded-policies-v1",
        "max_steps": 6,
        "max_replans": 1,
    }


def test_prediction_conversion_preserves_every_public_plan_execute_field() -> None:
    result = completed_plan_execute_result()
    prediction = prediction_from_plan_execute_result(
        result,
        case_id="reference_completed",
        dataset_version="agent-cases-v1",
        dataset_sha256="a" * 64,
        configuration_id="bounded-agent-v1",
        agent_version="lesson11-certified-v1",
        provider="recorded",
        agent_model="recorded-public-fixture-v1",
        prompt_version="lesson11-recorded-policies-v1",
        max_steps=6,
        max_replans=1,
    )
    assert prediction.status == result.status
    assert prediction.initial_plan == result.initial_plan
    assert prediction.final_steps == result.final_steps
    assert prediction.observations == result.observations
    assert prediction.trajectory == result.trajectory
    assert prediction.replan_count == result.replan_count
    assert prediction.evidence_gate == result.evidence_gate
    assert prediction.briefing is not None and result.briefing is not None
    assert prediction.briefing.model_dump(mode="python") == result.briefing.model_dump(
        mode="python"
    )


def test_prediction_conversion_has_a_lossless_json_round_trip() -> None:
    prediction = prediction_from_plan_execute_result(
        completed_plan_execute_result(), **prediction_metadata()
    )

    restored = AgentEvaluationPrediction.model_validate_json(prediction.model_dump_json())

    assert restored == prediction


def test_candidate_fact_can_represent_missing_document_provenance_for_scoring() -> None:
    fact = CandidateFact(
        claim="Schneider Electric revenue grew in the maintained evidence.",
        provenance_kind="document",
        source_references=("assets/course-data/fixtures/schneider_fy2025_excerpt.pdf",),
        evidence_ids=(),
    )
    assert fact.evidence_ids == ()


def test_candidate_fact_rejects_blank_text_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CandidateFact.model_validate({"claim": " ", "invented": True})


@pytest.mark.parametrize(
    "payload",
    [
        {"claim": "Bearer abcdefghijklmnop"},
        {"claim": "Valid claim", "source_references": ("authorization",)},
        {"claim": "Valid claim", "evidence_ids": ("sk-abcdefghijkl",)},
    ],
)
def test_candidate_fact_rejects_secret_shaped_text(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        CandidateFact.model_validate(payload)


def test_prediction_rejects_secret_shaped_metadata_before_logging() -> None:
    metadata = prediction_metadata()
    metadata["agent_model"] = "sk-abcdefghijkl"

    with pytest.raises(ValidationError):
        prediction_from_plan_execute_result(completed_plan_execute_result(), **metadata)


@pytest.mark.parametrize(
    ("path", "secret_value"),
    [
        (("initial_plan", "goal"), "Bearer abcdefghijklmnop"),
        (("final_steps", 0, "arguments", "credential"), "Bearer abcdefghijklmnop"),
        (("observations", 0, "result", "credential"), "Bearer abcdefghijklmnop"),
        (("trajectory", 0, "summary"), "Bearer abcdefghijklmnop"),
        (("evidence_gate", "coverage", "authorization"), ("Bearer abcdefghijklmnop",)),
    ],
    ids=("plan", "tool", "observation", "trajectory", "evidence_gate"),
)
def test_prediction_rejects_secret_shaped_nested_public_state_before_logging(
    path: tuple[str | int, ...],
    secret_value: object,
) -> None:
    payload = completed_plan_execute_result().model_dump(mode="python")
    target: object = payload
    for part in path[:-1]:
        if isinstance(part, int):
            assert isinstance(target, tuple)
            target = target[part]
        else:
            assert isinstance(target, dict)
            target = target[part]
    assert isinstance(target, dict)
    final_key = path[-1]
    assert isinstance(final_key, str)
    target[final_key] = secret_value
    result = PlanExecuteResult.model_validate(payload)

    with pytest.raises(ValidationError):
        prediction_from_plan_execute_result(result, **prediction_metadata())


def test_candidate_fact_strips_text_and_preserves_tuple_order() -> None:
    fact = CandidateFact(
        claim="  Maintained claim.  ",
        provenance_kind="document",
        source_references=(" source-b ", " source-a "),
        evidence_ids=(" evidence-b ", " evidence-a "),
    )

    assert fact.claim == "Maintained claim."
    assert fact.source_references == ("source-b", "source-a")
    assert fact.evidence_ids == ("evidence-b", "evidence-a")


def test_canonical_call_signature_sorts_nested_arguments() -> None:
    left = canonical_call_signature("get_company_metric", {"metric": "P/E", "ticker": "NVDA"})
    right = canonical_call_signature("get_company_metric", {"ticker": "NVDA", "metric": "P/E"})
    assert left == right == 'get_company_metric:{"metric":"P/E","ticker":"NVDA"}'


def test_regression_prediction_accepts_missing_provenance_only_at_candidate_boundary() -> None:
    valid = prediction_from_plan_execute_result(
        completed_plan_execute_result(), **prediction_metadata()
    ).model_dump(mode="python")
    valid["briefing"]["reported_facts"][1]["evidence_ids"] = ()

    regression = AgentEvaluationPrediction.model_validate(valid)

    assert regression.briefing is not None
    assert regression.briefing.reported_facts[1].evidence_ids == ()
    with pytest.raises(ValidationError):
        AnalystBriefing.model_validate(valid["briefing"])


def test_contract_models_are_strict_frozen_and_validate_numeric_bounds() -> None:
    expected_call = ExpectedToolCall(
        call_id=" metric-nvda ",
        capability=" get_company_metric ",
        arguments={"ticker": "NVDA", "metric": "P/E"},
    )
    case = AgentEvaluationCase(
        case_id=" reference_completed ",
        mission=MISSION,
        expected_final_status="completed",
        expected_tool_calls=(expected_call,),
        expected_error_codes=(),
        expected_replan_count=1,
        max_tool_calls=2,
        required_companies=(" NVIDIA ", " Schneider Electric "),
        required_evidence_ids=(" se-fy2025 ",),
        required_fact_kinds=("metric", "document"),
        required_limitations=(" periods ",),
        allow_briefing=True,
    )

    assert case.case_id == "reference_completed"
    assert case.expected_tool_calls[0].capability == "get_company_metric"
    assert case.required_companies == ("NVIDIA", "Schneider Electric")
    with pytest.raises(ValidationError):
        AgentEvaluationCase.model_validate({**case.model_dump(), "unknown": True})
    with pytest.raises(ValidationError):
        AgentEvaluationCase.model_validate({**case.model_dump(), "expected_replan_count": -1})
    with pytest.raises(ValidationError):
        ExpectedToolCall.model_validate(
            {**expected_call.model_dump(), "prerequisite_call_ids": (" ",)}
        )
    with pytest.raises(ValidationError):
        case.case_id = "changed"


def test_score_contracts_enforce_bounds_and_retain_metric_keys() -> None:
    passing = MetricScore(value=1, rationale=" All required evidence is present. ")
    scores = AgentCaseScores(
        case_id="reference_completed",
        configuration_id="bounded-agent-v1",
        tool_call_correctness=passing,
        tool_call_efficiency=passing,
        answer_relevance=passing,
        answer_completeness=passing,
        citation_integrity=passing,
        failure_stage="none",
        release_passed=True,
        total_tool_calls=2,
        redundant_tool_calls=0,
        latency_ms=4.0,
    )
    summary = AgentEvaluationSummary(
        configuration_id="bounded-agent-v1",
        dataset_version="agent-cases-v1",
        dataset_sha256="a" * 64,
        case_count=1,
        metric_means={"citation_integrity": 1.0},
        metric_pass_counts={"citation_integrity": 1},
        mean_tool_calls=2.0,
        mean_latency_ms=4.0,
        max_latency_ms=4.0,
        release_passed=True,
    )

    assert scores.tool_call_correctness.rationale == "All required evidence is present."
    assert summary.metric_means == {"citation_integrity": 1.0}
    with pytest.raises(ValidationError):
        MetricScore(value=1.01, rationale="Out of bounds.")
    with pytest.raises(ValidationError):
        AgentEvaluationSummary.model_validate(
            {**summary.model_dump(), "metric_means": {"unknown": 1.0}}
        )


def test_agent_evaluation_module_does_not_import_mlflow() -> None:
    source = inspect.getsource(agent_evaluation)

    assert "import mlflow" not in source
    assert "from mlflow" not in source


def aligned_fixture() -> tuple[
    tuple[AgentEvaluationCase, ...], tuple[AgentEvaluationPrediction, ...]
]:
    dataset = load_agent_evaluation_dataset(CASES_PATH, expected_sha256=manifest_case_hash())
    runs = load_recorded_agent_runs(RUNS_PATH, cases=dataset, expected_sha256=manifest_run_hash())
    return dataset.cases, runs.configurations[0].predictions


def mutate_alignment(
    predictions: tuple[AgentEvaluationPrediction, ...], mutation: str
) -> tuple[AgentEvaluationPrediction, ...]:
    if mutation == "missing":
        return predictions[:-1]
    if mutation == "extra":
        return predictions + (predictions[0].model_copy(update={"case_id": "unexpected_case"}),)
    if mutation == "duplicate":
        return predictions[:-1] + (predictions[0],)
    if mutation == "wrong_hash":
        return (
            predictions[0].model_copy(update={"dataset_sha256": "b" * 64}),
            *predictions[1:],
        )
    if mutation == "wrong_version":
        return (
            predictions[0].model_copy(update={"dataset_version": "agent-cases-v2"}),
            *predictions[1:],
        )
    raise AssertionError(f"unknown mutation: {mutation}")


def fixture_pair(
    case_id: str, configuration_id: str
) -> tuple[AgentEvaluationCase, AgentEvaluationPrediction]:
    dataset = load_agent_evaluation_dataset(CASES_PATH, expected_sha256=manifest_case_hash())
    runs = load_recorded_agent_runs(RUNS_PATH, cases=dataset, expected_sha256=manifest_run_hash())
    case = next(item for item in dataset.cases if item.case_id == case_id)
    configuration = next(
        item for item in runs.configurations if item.configuration_id == configuration_id
    )
    prediction = next(item for item in configuration.predictions if item.case_id == case_id)
    return case, prediction


def test_agent_cases_v1_has_exactly_six_required_cases_and_expected_calls() -> None:
    dataset = load_agent_evaluation_dataset(CASES_PATH, expected_sha256=manifest_case_hash())
    assert dataset.dataset_version == "agent-cases-v1"
    assert tuple(case.case_id for case in dataset.cases) == (
        "reference_completed",
        "unsupported_metric_not_recovered",
        "redundant_metric_call",
        "missing_schneider_document",
        "document_fact_without_evidence_id",
        "wrong_source_evidence_pair",
    )
    assert all(call.call_id for case in dataset.cases for call in case.expected_tool_calls)


def test_dataset_loader_rejects_byte_hash_mismatch_before_parsing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "agent_cases_v1.json"
    path.write_bytes(CASES_PATH.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        load_agent_evaluation_dataset(path, expected_sha256=manifest_case_hash())


def test_recorded_runs_have_two_configurations_and_six_aligned_predictions_each() -> None:
    dataset = load_agent_evaluation_dataset(CASES_PATH, expected_sha256=manifest_case_hash())
    runs = load_recorded_agent_runs(RUNS_PATH, cases=dataset, expected_sha256=manifest_run_hash())
    assert tuple(config.configuration_id for config in runs.configurations) == (
        "bounded-agent-v1",
        "regressed-agent-v0",
    )
    assert all(len(config.predictions) == 6 for config in runs.configurations)
    assert (
        runs.configurations[0].agent_version,
        runs.configurations[1].agent_version,
    ) == ("lesson11-certified-v1", "lesson11-regression-fixtures-v0")
    assert all(config.agent_model == "recorded-public-fixture-v1" for config in runs.configurations)
    assert tuple(config.prompt_version for config in runs.configurations) == (
        "lesson11-recorded-policies-v1",
        "lesson11-regression-policies-v0",
    )
    assert all((config.max_steps, config.max_replans) == (6, 1) for config in runs.configurations)


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "duplicate", "wrong_hash", "wrong_version"]
)
def test_alignment_rejects_partial_or_mismatched_prediction_tables(
    mutation: str,
) -> None:
    cases, predictions = aligned_fixture()
    changed = mutate_alignment(predictions, mutation)
    with pytest.raises(ValueError, match="alignment"):
        align_cases_and_predictions(
            cases,
            changed,
            dataset_version="agent-cases-v1",
            dataset_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    ("fixture", "nested_path"),
    [
        ("cases", ("cases", 0, "expected_tool_calls", 0)),
        ("runs", ("configurations", 0, "predictions", 0, "initial_plan", "steps", 0)),
        ("runs", ("configurations", 0, "predictions", 0, "observations", 0)),
    ],
    ids=("expected-call", "reused-plan-step", "reused-observation"),
)
def test_raw_json_loader_rejects_unknown_nested_fields_before_model_parsing(
    fixture: str, nested_path: tuple[str | int, ...], tmp_path: Path
) -> None:
    source = CASES_PATH if fixture == "cases" else RUNS_PATH
    payload = json.loads(source.read_text(encoding="utf-8"))
    target: object = payload
    for part in nested_path:
        if isinstance(part, int):
            assert isinstance(target, list)
            target = target[part]
        else:
            assert isinstance(target, dict)
            target = target[part]
    assert isinstance(target, dict)
    target["unknown_nested_field"] = True
    changed = json.dumps(payload, indent=2).encode("utf-8") + b"\n"
    path = tmp_path / source.name
    path.write_bytes(changed)
    changed_hash = sha256(changed).hexdigest()

    with pytest.raises(ValueError, match="unknown field"):
        if fixture == "cases":
            load_agent_evaluation_dataset(path, expected_sha256=changed_hash)
        else:
            dataset = load_agent_evaluation_dataset(
                CASES_PATH, expected_sha256=manifest_case_hash()
            )
            load_recorded_agent_runs(path, cases=dataset, expected_sha256=changed_hash)


def test_tool_call_correctness_is_dependency_aware_not_list_position_based() -> None:
    case, prediction = fixture_pair("reference_completed", "bounded-agent-v1")
    independent = tuple(reversed(prediction.observations[:2])) + prediction.observations[2:]
    changed = prediction.model_copy(update={"observations": independent})
    assert score_agent_case(case, changed).tool_call_correctness.value == 1.0


def test_tool_call_correctness_requires_expected_typed_error_and_replan() -> None:
    case, prediction = fixture_pair("reference_completed", "bounded-agent-v1")
    changed = prediction.model_copy(update={"replan_count": 0})
    score = score_agent_case(case, changed).tool_call_correctness
    assert 0.0 <= score.value < 1.0
    assert "replan" in score.rationale.casefold()


def test_tool_call_efficiency_penalizes_duplicate_budget_and_post_terminal_calls() -> None:
    case, prediction = fixture_pair("redundant_metric_call", "regressed-agent-v0")
    score = score_agent_case(case, prediction)
    assert score.tool_call_efficiency.value < 1.0
    assert score.redundant_tool_calls >= 1


def test_answer_relevance_scores_expected_typed_stop_without_a_briefing() -> None:
    case, prediction = fixture_pair("missing_schneider_document", "bounded-agent-v1")
    assert prediction.briefing is None
    assert score_agent_case(case, prediction).answer_relevance.value == 1.0


def test_answer_completeness_requires_companies_evidence_fact_kinds_comparison_and_limits() -> None:
    case, prediction = fixture_pair("reference_completed", "bounded-agent-v1")
    assert score_agent_case(case, prediction).answer_completeness.value == 1.0
    assert prediction.briefing is not None
    changed = prediction.model_copy(
        update={"briefing": prediction.briefing.model_copy(update={"limitations": ()})}
    )
    assert score_agent_case(case, changed).answer_completeness.value < 1.0


def test_citation_integrity_accepts_metric_source_and_exact_document_pair() -> None:
    case, prediction = fixture_pair("reference_completed", "bounded-agent-v1")
    assert score_agent_case(case, prediction).citation_integrity.value == 1.0


def test_citation_integrity_returns_zero_for_document_fact_without_evidence_id() -> None:
    case, prediction = fixture_pair("document_fact_without_evidence_id", "regressed-agent-v0")
    assert score_agent_case(case, prediction).citation_integrity.value == 0.0


def test_citation_integrity_returns_zero_for_cross_paired_source_and_evidence() -> None:
    case, prediction = fixture_pair("wrong_source_evidence_pair", "regressed-agent-v0")
    assert score_agent_case(case, prediction).citation_integrity.value == 0.0


def test_release_fails_when_required_gate_stop_emits_a_briefing() -> None:
    case, prediction = fixture_pair("missing_schneider_document", "regressed-agent-v0")
    assert prediction.briefing is not None
    assert score_agent_case(case, prediction).release_passed is False


def test_summary_preserves_per_case_failures_and_computes_five_means() -> None:
    dataset = load_agent_evaluation_dataset(CASES_PATH, expected_sha256=manifest_case_hash())
    runs = load_recorded_agent_runs(RUNS_PATH, cases=dataset, expected_sha256=manifest_run_hash())
    configuration = runs.configurations[0]
    aligned = align_cases_and_predictions(
        dataset.cases,
        configuration.predictions,
        dataset_version=dataset.dataset_version,
        dataset_sha256=dataset.dataset_sha256,
    )
    scores = tuple(score_agent_case(case, prediction) for case, prediction in aligned)
    summary = summarize_agent_evaluation(
        scores,
        dataset_version=dataset.dataset_version,
        dataset_sha256=dataset.dataset_sha256,
    )
    assert summary.case_count == 6
    assert set(summary.metric_means) == set(METRIC_NAMES)


def test_failure_classification_assigns_expected_fixture_owners() -> None:
    expected = {
        "unsupported_metric_not_recovered": "replanner",
        "redundant_metric_call": "replanner",
        "missing_schneider_document": "evidence_gate",
        "document_fact_without_evidence_id": "report_writer",
        "wrong_source_evidence_pair": "report_writer",
    }
    for case_id, owner in expected.items():
        case, prediction = fixture_pair(case_id, "regressed-agent-v0")
        scores = score_agent_case(case, prediction)
        assert scores.failure_stage == owner
