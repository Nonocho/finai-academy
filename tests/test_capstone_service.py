from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from finai_academy.capstone import (
    FinancialAnalystCopilot,
    ResearchRequest,
    ResearchRunResult,
    build_reference_copilot,
)
from finai_academy.capstone.models import CapstoneEvidenceHit
from finai_academy.capstone.tools import (
    MANDATORY_ANALYST_TOOLS,
    AnalystToolRegistry,
)
from finai_academy.research_planning import PlanStep


class MissingSchneiderRetriever:
    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def search(self, company: str, query: str, top_k: int = 2) -> tuple[CapstoneEvidenceHit, ...]:
        if company == "Schneider Electric":
            return ()
        return self._wrapped.search(company, query, top_k)


class MalformedRegistry:
    def discover(self) -> tuple[str, ...]:
        return tuple(sorted(MANDATORY_ANALYST_TOOLS))

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> object:
        del name, arguments
        return object()


def registry() -> AnalystToolRegistry:
    return AnalystToolRegistry(discovered=tuple(MANDATORY_ANALYST_TOOLS))


def test_recorded_reference_run_is_complete_cited_and_bounded() -> None:
    result = build_reference_copilot(run_id_factory=lambda: "reference-run-001").run(
        ResearchRequest.reference()
    )

    assert result.status == "completed"
    assert result.run_id == "reference-run-001"
    assert result.replan_count == 1
    assert len(result.initial_plan) == 5
    assert len(result.final_plan) <= 6
    assert result.evidence_gate.passed
    assert result.briefing is not None
    assert set(result.briefing.company_evidence) == {"NVIDIA", "Schneider Electric"}
    assert {fact.company for fact in result.briefing.cited_facts} == {
        "NVIDIA",
        "Schneider Electric",
    }
    assert all(fact.source_reference for fact in result.briefing.cited_facts)
    assert all(
        fact.evidence_id
        for fact in result.briefing.cited_facts
        if fact.provenance_kind == "document"
    )
    assert [metric.name for metric in result.deterministic_evaluation.metrics] == [
        "tool_call_correctness",
        "tool_call_efficiency",
        "answer_relevance",
        "answer_completeness",
        "citation_integrity",
    ]
    assert [metric.value for metric in result.deterministic_evaluation.metrics] == [1.0] * 5
    assert result.deterministic_evaluation.release_passed
    assert result.judge_evaluation is not None
    assert result.judge_evaluation.status == "not_run"


def test_unsupported_revenue_metric_replans_to_document_search() -> None:
    result = build_reference_copilot().run(ResearchRequest.reference())

    errors = [event for event in result.trajectory if event.status == "error"]
    assert [event.error_code for event in errors] == ["unsupported_metric"]
    assert [
        observation.error_code for observation in result.observations if observation.error_code
    ] == ["unsupported_metric"]
    assert [observation.capability for observation in result.observations] == [
        "get_company_metric",
        "get_company_metric",
        "get_company_metric",
        "search_financial_documents",
        "search_financial_documents",
    ]
    assert result.observations[3].arguments["company"] == "NVIDIA"
    assert result.observations[4].arguments["company"] == "Schneider Electric"
    assert result.initial_plan != result.final_plan
    assert result.replan_count == 1
    assert [observation.attempt_id for observation in result.observations] == [1, 2, 3, 4, 5]
    assert [observation.plan_revision for observation in result.observations] == [0, 0, 0, 1, 1]


def test_successful_tool_signatures_are_unique() -> None:
    result = build_reference_copilot().run(ResearchRequest.reference())

    signatures = [
        (observation.capability, tuple(sorted(observation.arguments.items())))
        for observation in result.observations
        if observation.status == "ok"
    ]
    assert len(signatures) == len(set(signatures))


def test_failed_evidence_gate_returns_no_briefing() -> None:
    complete = build_reference_copilot()
    service = build_reference_copilot(retriever=MissingSchneiderRetriever(complete.retriever))

    result = service.run(ResearchRequest.reference())

    assert result.status == "insufficient_evidence"
    assert not result.evidence_gate.passed
    assert result.evidence_gate.missing_requirements == ("Schneider Electric document evidence",)
    assert result.briefing is None
    assert not result.deterministic_evaluation.release_passed


def test_unknown_capability_stops_at_the_plan_gate() -> None:
    unknown_plan = (
        PlanStep(
            step_id=1,
            capability="place_order",
            arguments={"ticker": "NVDA"},
            purpose="Attempt an unknown capability.",
            expected_evidence=("None",),
        ),
    )
    service = FinancialAnalystCopilot(
        retriever=build_reference_copilot().retriever,
        registry=registry(),
        initial_plan=unknown_plan,
    )

    result = service.run(ResearchRequest.reference())

    assert result.status == "plan_blocked"
    assert result.observations == ()
    assert result.trajectory[-1].failure_owner == "planner"


def test_duplicate_successful_call_stops_before_repeating_it() -> None:
    duplicate_plan = (
        PlanStep(
            step_id=1,
            capability="get_company_metric",
            arguments={"ticker": "NVDA", "metric": "P/E"},
            purpose="Collect NVIDIA P/E.",
            expected_evidence=("NVIDIA P/E",),
        ),
        PlanStep(
            step_id=2,
            capability="get_company_metric",
            arguments={"metric": "P/E", "ticker": "NVDA"},
            purpose="Repeat NVIDIA P/E.",
            expected_evidence=("NVIDIA P/E",),
        ),
    )
    service = FinancialAnalystCopilot(
        retriever=build_reference_copilot().retriever,
        registry=registry(),
        initial_plan=duplicate_plan,
    )

    result = service.run(ResearchRequest.reference())

    assert result.status == "execution_stopped"
    assert len(result.observations) == 1
    assert result.trajectory[-1].summary == "duplicate_successful_call"
    assert result.trajectory[-1].failure_owner == "replanner"


def test_exhausted_replan_budget_stops_truthfully() -> None:
    result = build_reference_copilot().run(ResearchRequest.reference(max_replans=0))

    assert result.status == "replan_budget_exhausted"
    assert result.replan_count == 0
    assert len(result.observations) == 3
    assert result.briefing is None


def test_malformed_tool_outcome_stops_truthfully() -> None:
    service = FinancialAnalystCopilot(
        retriever=build_reference_copilot().retriever,
        registry=MalformedRegistry(),
    )

    result = service.run(ResearchRequest.reference())

    assert result.status == "execution_stopped"
    assert result.observations[-1].error_code == "malformed_tool_outcome"
    assert result.trajectory[-1].failure_owner == "tool_boundary"
    assert result.briefing is None


def test_public_serialization_contains_no_secrets_or_personal_paths() -> None:
    payload = build_reference_copilot().run(ResearchRequest.reference()).model_dump_json()

    assert "Authorization: Bearer" not in payload
    assert "api_key" not in payload
    assert "/Users/" not in payload
    assert "/home/" not in payload


def test_unsafe_injected_plan_is_blocked_without_echoing_its_query() -> None:
    unsafe_plan = (
        PlanStep(
            step_id=1,
            capability="search_financial_documents",
            arguments={
                "company": "NVIDIA",
                "query": "api_key=provider-secret-value",
                "top_k": 2,
            },
            purpose="Collect document evidence.",
            expected_evidence=("NVIDIA evidence",),
        ),
    )
    service = FinancialAnalystCopilot(
        retriever=build_reference_copilot().retriever,
        registry=registry(),
        initial_plan=unsafe_plan,
    )

    result = service.run(ResearchRequest.reference())
    payload = result.model_dump_json()

    assert result.status == "plan_blocked"
    assert result.initial_plan == ()
    assert "provider-secret-value" not in payload
    assert "api_key" not in payload


def test_custom_question_stays_in_the_two_company_universe_and_keeps_result_shape() -> None:
    request = ResearchRequest(
        mode="custom",
        question="Compare the available operating-growth evidence for both companies.",
        companies=("NVIDIA", "Schneider Electric"),
        provider="recorded",
        model="recorded-capstone-v1",
        data_mode="certified",
    )

    result = build_reference_copilot().run(request)

    assert isinstance(result, ResearchRunResult)
    assert result.request == request
    assert result.status == "completed"
    assert {
        observation.result["company"]
        for observation in result.observations
        if observation.status == "ok" and observation.result is not None
    } == {"NVIDIA", "Schneider Electric"}


def test_custom_question_with_an_unapproved_company_is_blocked() -> None:
    request = ResearchRequest(
        mode="custom",
        question="Compare NVIDIA with another issuer.",
        companies=("NVIDIA", "Other Issuer"),
        provider="recorded",
        model="recorded-capstone-v1",
        data_mode="certified",
    )

    result = build_reference_copilot().run(request)

    assert result.status == "plan_blocked"
    assert result.observations == ()
    assert result.briefing is None
