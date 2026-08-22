from __future__ import annotations

import inspect

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
    assert prediction.briefing.reported_facts[0].claim == result.briefing.reported_facts[0].claim
    assert (
        prediction.briefing.reported_facts[0].provenance_kind
        == result.briefing.reported_facts[0].provenance_kind
    )
    assert prediction.briefing.source_references == result.briefing.source_references


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
    left = canonical_call_signature(
        "get_company_metric", {"metric": "P/E", "ticker": "NVDA"}
    )
    right = canonical_call_signature(
        "get_company_metric", {"ticker": "NVDA", "metric": "P/E"}
    )
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
        AgentEvaluationCase.model_validate(
            {**case.model_dump(), "expected_replan_count": -1}
        )
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
