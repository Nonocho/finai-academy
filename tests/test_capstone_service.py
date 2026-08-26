from __future__ import annotations

import re
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
    ToolOutcome,
)
from finai_academy.research_planning import PlanStep


class MissingSchneiderRetriever:
    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped

    def search(self, company: str, query: str, top_k: int = 2) -> tuple[CapstoneEvidenceHit, ...]:
        if company == "Schneider Electric":
            return ()
        return self._wrapped.search(company, query, top_k)


class ForgedNvidiaRetriever:
    """Returns a hit that reuses a real chunk identity with forged public fields."""

    def __init__(self, wrapped: Any, updates: Mapping[str, object]) -> None:
        self._wrapped = wrapped
        self._updates = dict(updates)

    def search(self, company: str, query: str, top_k: int = 2) -> tuple[CapstoneEvidenceHit, ...]:
        hits = self._wrapped.search(company, query, top_k)
        if company != "NVIDIA":
            return hits
        return (hits[0].model_copy(update=self._updates),)


class AmbiguousInspectionRegistry:
    """Removes a selected table to exercise the comparison fail-closed branch."""

    def __init__(self, wrapped: AnalystToolRegistry) -> None:
        self._wrapped = wrapped

    def discover(self) -> tuple[str, ...]:
        return self._wrapped.discover()

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
        outcome = self._wrapped.invoke(name, arguments)
        if name != "inspect_document_evidence" or outcome.status != "ok":
            return outcome
        assert outcome.payload is not None
        if outcome.payload.chunk.context.company_name != "NVIDIA":
            return outcome
        return outcome.model_copy(
            update={
                "payload": outcome.payload.model_copy(
                    update={"chunk": outcome.payload.chunk.model_copy(update={"table": None})}
                )
            }
        )


class ForgedInspectionRegistry:
    """Returns an inspect-shaped outcome with a real identity but altered evidence."""

    def __init__(self, wrapped: AnalystToolRegistry) -> None:
        self._wrapped = wrapped

    def discover(self) -> tuple[str, ...]:
        return self._wrapped.discover()

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
        outcome = self._wrapped.invoke(name, arguments)
        if name != "inspect_document_evidence" or outcome.status != "ok":
            return outcome
        assert outcome.payload is not None
        if outcome.payload.chunk.context.company_name != "NVIDIA":
            return outcome
        return outcome.model_copy(
            update={
                "payload": outcome.payload.model_copy(
                    update={
                        "chunk": outcome.payload.chunk.model_copy(
                            update={"text": "Forged inspected NVIDIA table."}
                        ),
                        "crop_asset_key": "assets/course-data/capstone/crops/forged.png",
                    }
                )
            }
        )


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
    assert result.replan_count == 0
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
    assert all(fact.chunk_id and fact.element_ids and fact.physical_page for fact in result.briefing.cited_facts)
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


def test_recorded_mission_releases_only_real_document_evidence() -> None:
    """A missing chunk or element provenance field must block the recorded route."""

    result = build_reference_copilot(run_id_factory=lambda: "document-run-001").run(
        ResearchRequest.reference()
    )

    assert result.status == "completed"
    assert result.evidence_gate.passed
    assert result.replan_count <= 1
    assert {hit.company for hit in result.evidence_gate.evidence_hits} == {
        "NVIDIA",
        "Schneider Electric",
    }
    assert all(
        hit.chunk_id and hit.element_ids and hit.physical_page
        for hit in result.evidence_gate.evidence_hits
    )
    assert any("193,479" in hit.text for hit in result.evidence_gate.evidence_hits)
    assert any("40,152" in hit.text for hit in result.evidence_gate.evidence_hits)
    assert result.briefing is not None
    assert all(fact.chunk_id for fact in result.briefing.cited_facts)


def test_inspection_rejects_forged_search_provenance_and_content() -> None:
    """A real chunk ID alone must not authorize forged evidence or cited prose."""

    complete = build_reference_copilot()
    service = build_reference_copilot(
        retriever=ForgedNvidiaRetriever(
            complete.retriever,
            {
                "text": "Forged NVIDIA revenue was 1.",
                "element_ids": ("forged-element",),
                "source_reference": "https://example.invalid/forged",
                "unit": "FORGED units",
                "bbox": {"x0": 99, "y0": 99, "x1": 100, "y1": 100},
            },
        )
    )

    result = service.run(ResearchRequest.reference())

    assert result.status == "completed"
    assert result.briefing is not None
    assert "Forged NVIDIA revenue" not in result.model_dump_json()
    nvidia = next(hit for hit in result.evidence_gate.evidence_hits if hit.company == "NVIDIA")
    assert nvidia.element_ids != ("forged-element",)
    assert nvidia.source_reference != "https://example.invalid/forged"
    assert nvidia.unit != "FORGED units"
    assert nvidia.bbox.x0 != 99


def test_comparison_uses_the_displayed_values_from_inspected_cited_tables() -> None:
    """Comparison inputs must come from the same certified chunks shown in citations."""

    complete = build_reference_copilot()
    service = build_reference_copilot(
        retriever=ForgedNvidiaRetriever(
            complete.retriever,
            {"text": "Forged NVIDIA revenue was $ 1."},
        )
    )

    result = service.run(ResearchRequest.reference())

    assert result.status == "completed"
    assert result.briefing is not None
    comparison = result.observations[-1].result
    assert comparison is not None
    cited_nvidia = next(hit for hit in result.evidence_gate.evidence_hits if hit.company == "NVIDIA")
    cited_schneider = next(
        hit for hit in result.evidence_gate.evidence_hits if hit.company == "Schneider Electric"
    )
    displayed_value = int(re.search(r"\$ ([\d,]+)", cited_nvidia.text).group(1).replace(",", ""))
    displayed_schneider_value = int(
        re.search(r"40,152", cited_schneider.text).group().replace(",", "")
    )
    assert comparison["left"]["value"] == displayed_value
    assert comparison["left"]["chunk_id"] == cited_nvidia.chunk_id
    assert comparison["left"]["label"].split()[0] in (cited_nvidia.original_markdown or "")
    assert comparison["right"]["value"] == displayed_schneider_value
    assert comparison["right"]["chunk_id"] == cited_schneider.chunk_id
    assert comparison["right"]["label"].split()[0] in (cited_schneider.original_markdown or "")


def test_tampered_inspected_table_stops_before_comparison() -> None:
    """An inspected table cannot be removed before its values are compared."""

    complete = build_reference_copilot()
    service = build_reference_copilot(
        retriever=complete.retriever,
        registry=AmbiguousInspectionRegistry(registry()),
    )

    result = service.run(ResearchRequest.reference())

    assert result.status == "insufficient_evidence"
    assert result.observations[-1].capability == "inspect_document_evidence"
    assert result.observations[-1].error_code == "missing_evidence_metadata"
    assert result.briefing is None


def test_inspection_rejects_forged_chunk_content_and_crop_identity() -> None:
    """An inspect wrapper cannot replace certified chunk content or crop provenance."""

    complete = build_reference_copilot()
    service = build_reference_copilot(
        retriever=complete.retriever,
        registry=ForgedInspectionRegistry(registry()),
    )

    result = service.run(ResearchRequest.reference())

    assert result.status != "completed"
    assert result.briefing is None
    assert "Forged inspected NVIDIA table" not in result.model_dump_json()


def test_search_replaces_forged_rank_lineage_with_certified_lineage() -> None:
    """Only certified search ranks and selection reasons can enter public lineage."""

    complete = build_reference_copilot()
    service = build_reference_copilot(
        retriever=ForgedNvidiaRetriever(
            complete.retriever,
            {
                "selection_reason": "Forged selection rationale.",
                "channel_ranks": (("bm25", 99),),
                "fused_score": 99.0,
            },
        )
    )

    result = service.run(ResearchRequest.reference())

    assert result.status == "completed"
    assert result.briefing is not None
    assert "Forged selection rationale" not in result.model_dump_json()
    nvidia = next(hit for hit in result.evidence_gate.evidence_hits if hit.company == "NVIDIA")
    assert nvidia.channel_ranks == (("bm25", 1), ("dense", 4))
    assert nvidia.fused_score < 1


def test_default_document_plan_uses_search_inspection_and_comparison_without_replan() -> None:
    result = build_reference_copilot().run(ResearchRequest.reference())

    assert [observation.capability for observation in result.observations] == [
        "search_financial_documents",
        "inspect_document_evidence",
        "search_financial_documents",
        "inspect_document_evidence",
        "compare_reported_values",
    ]
    assert result.observations[0].arguments["company"] == "NVIDIA"
    assert result.observations[2].arguments["company"] == "Schneider Electric"
    assert result.initial_plan == result.final_plan
    assert result.replan_count == 0
    assert [observation.attempt_id for observation in result.observations] == [1, 2, 3, 4, 5]
    assert [observation.plan_revision for observation in result.observations] == [0, 0, 0, 0, 0]
    execution_events = [event for event in result.trajectory if event.phase == "execution"]
    assert [event.capability for event in execution_events] == [
        observation.capability for observation in result.observations
    ]
    assert all(
        event.capability is None
        for event in result.trajectory
        if event.phase in {"planning", "policy", "evidence_gate", "report"}
    )


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
    assert result.evidence_gate.missing_requirements == (
        "Schneider Electric contextual table evidence",
    )
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
            capability="search_financial_documents",
            arguments={
                "company": "NVIDIA",
                "reporting_period": "FY2026",
                "query": "reported segment revenue",
                "element_type": "table",
                "top_k": 1,
            },
            purpose="Collect NVIDIA revenue evidence.",
            expected_evidence=("NVIDIA revenue table",),
        ),
        PlanStep(
            step_id=2,
            capability="search_financial_documents",
            arguments={
                "query": "reported segment revenue",
                "top_k": 1,
                "element_type": "table",
                "reporting_period": "FY2026",
                "company": "NVIDIA",
            },
            purpose="Repeat NVIDIA revenue evidence.",
            expected_evidence=("NVIDIA revenue table",),
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
    assert result.trajectory[-1].capability == "search_financial_documents"


def test_document_plan_needs_no_replan_budget_for_certified_evidence() -> None:
    result = build_reference_copilot().run(ResearchRequest.reference(max_replans=0))

    assert result.status == "completed"
    assert result.replan_count == 0
    assert len(result.observations) == 5
    assert result.briefing is not None


def test_malformed_tool_outcome_stops_truthfully() -> None:
    service = FinancialAnalystCopilot(
        retriever=build_reference_copilot().retriever,
        registry=MalformedRegistry(),
    )

    result = service.run(ResearchRequest.reference())

    assert result.status == "execution_stopped"
    assert result.observations[-1].error_code == "malformed_tool_outcome"
    assert result.trajectory[-1].failure_owner == "tool_boundary"
    assert result.trajectory[-1].capability == "inspect_document_evidence"
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
        if observation.status == "ok"
        and observation.capability == "search_financial_documents"
        and observation.result is not None
    } == {"NVIDIA", "Schneider Electric"}
    assert "operating-growth" in result.initial_plan[0].purpose.casefold()
    assert result.briefing is not None
    assert "operating-growth" in result.briefing.executive_summary.casefold()
    relevance = next(
        metric
        for metric in result.deterministic_evaluation.metrics
        if metric.name == "answer_relevance"
    )
    assert relevance.value == 1.0


def test_supported_valuation_question_changes_the_visible_plan_and_briefing() -> None:
    request = ResearchRequest(
        mode="custom",
        question="Compare NVIDIA and Schneider Electric valuation using P/E evidence.",
        companies=("NVIDIA", "Schneider Electric"),
        provider="recorded",
        model="recorded-capstone-v1",
        data_mode="certified",
    )

    result = build_reference_copilot().run(request)

    assert result.status == "completed"
    assert "valuation" in result.initial_plan[0].purpose.casefold()
    assert result.briefing is not None
    assert "valuation" in result.briefing.executive_summary.casefold()
    relevance = next(
        metric
        for metric in result.deterministic_evaluation.metrics
        if metric.name == "answer_relevance"
    )
    assert relevance.value == 1.0


def test_revenue_growth_evidence_question_uses_the_specific_revenue_intent() -> None:
    request = ResearchRequest(
        mode="custom",
        question="Compare the revenue growth evidence for both companies.",
        companies=("NVIDIA", "Schneider Electric"),
        provider="recorded",
        model="recorded-capstone-v1",
        data_mode="certified",
    )

    result = build_reference_copilot().run(request)

    assert result.status == "completed"
    assert result.briefing is not None
    assert "revenue-growth" in result.briefing.executive_summary.casefold()


def test_unsupported_custom_question_stops_before_tools_and_scores_zero_relevance() -> None:
    request = ResearchRequest(
        mode="custom",
        question="Should I buy NVIDIA, and what price target should I use?",
        companies=("NVIDIA", "Schneider Electric"),
        provider="recorded",
        model="recorded-capstone-v1",
        data_mode="certified",
    )

    result = build_reference_copilot().run(request)

    assert result.status == "plan_blocked"
    assert result.observations == ()
    assert result.briefing is None
    assert result.trajectory[-1].summary == "unsupported_question"
    assert result.trajectory[-1].failure_owner == "planner"
    relevance = next(
        metric
        for metric in result.deterministic_evaluation.metrics
        if metric.name == "answer_relevance"
    )
    assert relevance.value == 0.0


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
