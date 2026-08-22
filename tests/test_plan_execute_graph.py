from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import pytest

from finai_academy.plan_execute_graph import build_plan_execute_graph, run_plan_execute
from finai_academy.research_planning import (
    AnalystBriefing,
    CitedFact,
    PlannerToolSpec,
    PlanStep,
    ReplanDecision,
    ResearchObservation,
    ResearchPlan,
)

MISSION = (
    "Produce a concise NVIDIA and Schneider Electric briefing. Compare their available "
    "valuation metrics and latest operating-growth evidence."
)


class SdkPolicyError(Exception):
    """Representative provider SDK exception outside built-in validation error families."""


def tool_catalog() -> tuple[PlannerToolSpec, ...]:
    return (
        PlannerToolSpec(
            name="get_company_metric",
            description="Return one controlled company metric.",
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "metric": {"type": "string"},
                },
                "required": ["ticker", "metric"],
                "additionalProperties": False,
            },
        ),
        PlannerToolSpec(
            name="search_financial_documents",
            description="Search controlled financial evidence.",
            input_schema={
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 3},
                },
                "required": ["company", "query"],
                "additionalProperties": False,
            },
        ),
    )


def initial_plan() -> ResearchPlan:
    return ResearchPlan(
        goal="Compare available NVIDIA and Schneider Electric evidence.",
        steps=(
            metric_step(1, "NVDA"),
            metric_step(2, "SU.PA"),
            metric_step(3, "NVDA", metric="Revenue"),
            document_step(4, "Schneider Electric", depends_on=(2,)),
        ),
    )


def metric_step(step_id: int, ticker: str, *, metric: str = "P/E") -> PlanStep:
    return PlanStep(
        step_id=step_id,
        capability="get_company_metric",
        arguments={"ticker": ticker, "metric": metric},
        purpose=f"Collect {ticker} valuation evidence.",
        expected_evidence=(f"{ticker} {metric}",),
    )


def document_step(
    step_id: int,
    company: str,
    *,
    depends_on: tuple[int, ...] = (),
) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        capability="search_financial_documents",
        arguments={"company": company, "query": "revenue growth", "top_k": 2},
        purpose=f"Collect {company} operating evidence.",
        expected_evidence=(f"{company} revenue evidence",),
        depends_on=depends_on,
    )


def replacement_tail() -> tuple[PlanStep, ...]:
    return (
        document_step(5, "NVIDIA", depends_on=(1,)),
        document_step(6, "Schneider Electric", depends_on=(2, 5)),
    )


class FakeExecutor:
    def __init__(self) -> None:
        self.catalog = tool_catalog()
        self.calls: list[tuple[int, int, int]] = []

    async def execute(
        self,
        step: PlanStep,
        *,
        attempt_id: int,
        plan_revision: int,
    ) -> ResearchObservation:
        self.calls.append((step.step_id, attempt_id, plan_revision))
        if step.arguments.get("metric") == "Revenue":
            return ResearchObservation(
                attempt_id=attempt_id,
                step_id=step.step_id,
                plan_revision=plan_revision,
                capability=step.capability,
                arguments=step.arguments,
                status="error",
                error_code="unsupported_metric",
                result={"valid_values": ["EPS", "P/E"]},
                duration_ms=2,
            )

        company = (
            "NVIDIA"
            if step.arguments.get("ticker") == "NVDA"
            else step.arguments.get("company", "Schneider Electric")
        )
        is_document = step.capability == "search_financial_documents"
        evidence_id = f"evidence-{step.step_id}"
        source = f"source-{step.step_id}"
        result: dict[str, Any] = {"company": company, "evidence_id": evidence_id}
        if is_document:
            result["hits"] = [{"evidence_id": evidence_id, "source": source}]
        return ResearchObservation(
            attempt_id=attempt_id,
            step_id=step.step_id,
            plan_revision=plan_revision,
            capability=step.capability,
            arguments=step.arguments,
            status="ok",
            result=result,
            evidence_ids=(evidence_id,),
            source_references=(source,),
            duration_ms=2,
        )


async def recorded_planner(
    question: str,
    catalog: tuple[PlannerToolSpec, ...],
) -> ResearchPlan:
    assert question == MISSION
    assert tuple(tool.name for tool in catalog) == (
        "get_company_metric",
        "search_financial_documents",
    )
    return initial_plan()


async def recorded_replanner(state: Mapping[str, Any]) -> ReplanDecision:
    observations = tuple(state["observations"])
    if observations[-1].error_code == "unsupported_metric":
        return ReplanDecision(
            action="replace_remaining",
            reasoning="Replace unsupported structured revenue with document evidence.",
            replacement_steps=replacement_tail(),
        )
    if int(state["current_index"]) == len(state["active_steps"]):
        return ReplanDecision(action="finish", reasoning="Required evidence is collected.")
    return ReplanDecision(action="continue", reasoning="Execute the next validated step.")


async def recorded_report_writer(
    question: str,
    observations: tuple[ResearchObservation, ...],
) -> AnalystBriefing:
    assert question == MISSION
    facts = tuple(
        CitedFact(
            claim=f"Evidence from successful step {item.step_id}.",
            source_references=item.source_references,
            evidence_ids=(
                item.evidence_ids
                if item.capability == "search_financial_documents"
                else ()
            ),
        )
        for item in observations
        if item.status == "ok" and item.source_references
    )
    return AnalystBriefing(
        reported_facts=facts,
        cross_company_observations=("The reporting periods differ.",),
        interpretation=("The controlled evidence is descriptive, not investment advice.",),
        limitations=("Currencies, periods, and business definitions differ.",),
        source_references=tuple(
            dict.fromkeys(
                source
                for fact in facts
                for source in fact.source_references
            )
        ),
    )


async def finish_after_last_step(state: Mapping[str, Any]) -> ReplanDecision:
    if int(state["current_index"]) == len(state["active_steps"]):
        return ReplanDecision(action="finish", reasoning="All planned steps were attempted.")
    return ReplanDecision(action="continue", reasoning="Continue with unfinished work.")


class ReportRecorder:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(
        self,
        question: str,
        observations: tuple[ResearchObservation, ...],
    ) -> AnalystBriefing:
        self.calls += 1
        return await recorded_report_writer(question, observations)


def test_successful_revision_preserves_completed_prefix_and_finishes() -> None:
    """Breaks if replanning repeats completed work or loses the failed attempt."""
    executor = FakeExecutor()

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=executor,
            planner=recorded_planner,
            replanner=recorded_replanner,
            report_writer=recorded_report_writer,
        )
    )

    assert result.status == "completed"
    assert result.replan_count == 1
    assert [item.attempt_id for item in result.observations] == [1, 2, 3, 4, 5]
    assert [item.step_id for item in result.observations] == [1, 2, 3, 5, 6]
    assert result.observations[2].error_code == "unsupported_metric"
    assert result.evidence_gate.passed is True
    assert result.briefing is not None
    assert [call[0] for call in executor.calls].count(1) == 1
    assert [call[0] for call in executor.calls].count(2) == 1


def test_replacement_rejects_a_canonical_duplicate_of_successful_work() -> None:
    """Breaks if a revised tail can repeat a successful capability call."""
    executor = FakeExecutor()

    async def planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        return ResearchPlan(
            goal="Collect evidence without repeating successful calls.",
            steps=(metric_step(1, "NVDA"), metric_step(2, "NVDA", metric="Revenue")),
        )

    async def replanner(state: Mapping[str, Any]) -> ReplanDecision:
        observations = tuple(state["observations"])
        if observations[-1].error_code == "unsupported_metric":
            duplicate = metric_step(3, "NVDA").model_copy(
                update={"arguments": {"metric": "P/E", "ticker": "NVDA"}}
            )
            return ReplanDecision(
                action="replace_remaining",
                reasoning="Incorrectly repeat completed metric work.",
                replacement_steps=(duplicate,),
            )
        return ReplanDecision(action="continue", reasoning="Reach the failed step.")

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=executor,
            planner=planner,
            replanner=replanner,
            report_writer=recorded_report_writer,
        )
    )

    assert result.status == "plan_blocked"
    assert [item.step_id for item in result.observations] == [1, 2]
    assert [call[0] for call in executor.calls] == [1, 2]


def test_initial_plan_stops_at_the_total_step_budget() -> None:
    """Breaks if an oversized initial plan reaches the executor."""
    executor = FakeExecutor()

    async def oversized_planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        return ResearchPlan(
            goal="Exceed the host-owned step budget.",
            steps=tuple(metric_step(step_id, "NVDA") for step_id in range(1, 8)),
        )

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=executor,
            planner=oversized_planner,
            replanner=finish_after_last_step,
            report_writer=recorded_report_writer,
        )
    )

    assert result.status == "plan_blocked"
    assert result.observations == ()
    assert executor.calls == []


def test_second_tail_replacement_stops_at_the_replan_budget() -> None:
    """Breaks if more than one replacement revision is accepted."""
    executor = FakeExecutor()

    async def failing_planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        return ResearchPlan(
            goal="Exercise the bounded revision path.",
            steps=(metric_step(1, "NVDA", metric="Revenue"),),
        )

    async def repeated_replanner(state: Mapping[str, Any]) -> ReplanDecision:
        if len(state["observations"]) > 2:
            return ReplanDecision(action="stop", reasoning="Test safety stop.")
        next_id = max(state["all_step_ids"]) + 1
        return ReplanDecision(
            action="replace_remaining",
            reasoning="Try another replacement after an error.",
            replacement_steps=(metric_step(next_id, "NVDA", metric="Revenue"),),
        )

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=executor,
            planner=failing_planner,
            replanner=repeated_replanner,
            report_writer=recorded_report_writer,
        )
    )

    assert result.status == "replan_budget_exhausted"
    assert result.replan_count == 1
    assert [item.step_id for item in result.observations] == [1, 2]


def test_missing_document_evidence_blocks_report_generation() -> None:
    """Breaks if the report writer runs without both evidence kinds per company."""
    executor = FakeExecutor()
    report_writer = ReportRecorder()

    async def incomplete_planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        return ResearchPlan(
            goal="Collect an intentionally incomplete evidence set.",
            steps=(
                metric_step(1, "NVDA"),
                metric_step(2, "SU.PA"),
                document_step(3, "NVIDIA"),
            ),
        )

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=executor,
            planner=incomplete_planner,
            replanner=finish_after_last_step,
            report_writer=report_writer,
        )
    )

    assert result.status == "insufficient_evidence"
    assert result.briefing is None
    assert "Schneider Electric document evidence" in result.evidence_gate.missing_requirements
    assert report_writer.calls == 0


def test_unreportable_document_provenance_blocks_report_generation() -> None:
    """Breaks if the graph can complete from document metadata absent from its hits."""

    class MismatchedDocumentExecutor(FakeExecutor):
        async def execute(
            self,
            step: PlanStep,
            *,
            attempt_id: int,
            plan_revision: int,
        ) -> ResearchObservation:
            observation = await super().execute(
                step,
                attempt_id=attempt_id,
                plan_revision=plan_revision,
            )
            if step.capability != "search_financial_documents":
                return observation
            return observation.model_copy(
                update={
                    "result": {
                        "company": observation.result["company"],
                        "hits": [
                            {"source": "returned-source", "evidence_id": "returned-id"}
                        ],
                    },
                    "source_references": ("declared-source",),
                    "evidence_ids": ("declared-id",),
                }
            )

    report_writer = ReportRecorder()
    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=MismatchedDocumentExecutor(),
            planner=recorded_planner,
            replanner=recorded_replanner,
            report_writer=report_writer,
        )
    )

    assert result.status == "insufficient_evidence"
    assert result.briefing is None
    assert result.evidence_gate.passed is False
    assert report_writer.calls == 0


def test_explicit_replanner_stop_returns_a_typed_bounded_result() -> None:
    """Breaks if a model-requested stop is routed back into execution."""
    executor = FakeExecutor()

    async def one_step_planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        return ResearchPlan(goal="Stop after one observation.", steps=(metric_step(1, "NVDA"),))

    async def stop_replanner(state: Mapping[str, Any]) -> ReplanDecision:
        del state
        return ReplanDecision(action="stop", reasoning="The requested comparison is unavailable.")

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=executor,
            planner=one_step_planner,
            replanner=stop_replanner,
            report_writer=recorded_report_writer,
        )
    )

    assert result.status == "execution_stopped"
    assert [item.step_id for item in result.observations] == [1]


def test_compiled_graph_exposes_the_six_instructional_nodes() -> None:
    """Breaks if the classroom graph collapses a visible host-control stage."""
    graph = build_plan_execute_graph(
        executor=FakeExecutor(),
        planner=recorded_planner,
        replanner=recorded_replanner,
        report_writer=recorded_report_writer,
    )

    assert set(graph.get_graph().nodes) == {
        "__start__",
        "planner",
        "plan_gate",
        "executor",
        "replanner",
        "evidence_gate",
        "report",
        "__end__",
    }


def test_replanner_receives_the_graph_owned_step_limit() -> None:
    """Breaks if a policy cannot align its replacement budget with the graph validator."""
    executor = FakeExecutor()
    received_limits: list[int] = []

    async def limit_aware_replanner(state: Mapping[str, Any]) -> ReplanDecision:
        received_limits.append(state["max_steps"])
        return ReplanDecision(action="stop", reasoning="The configured budget permits no revision.")

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=executor,
            planner=recorded_planner,
            replanner=limit_aware_replanner,
            report_writer=recorded_report_writer,
            max_steps=4,
        )
    )

    assert result.status == "execution_stopped"
    assert received_limits == [4]


def test_result_trajectory_is_sequential_and_contains_no_runtime_configuration() -> None:
    """Breaks if displayable output captures executor internals, environment, or secrets."""
    executor = FakeExecutor()
    executor.OPENAI_API_KEY = "course-secret"
    executor.server_parameters = {
        "type": "StdioServerParameters",
        "command": "sys.executable",
        "env": {"COURSE_TOKEN": "environment mapping"},
    }

    async def unsafe_reasoning_replanner(state: Mapping[str, Any]) -> ReplanDecision:
        decision = await recorded_replanner(state)
        return decision.model_copy(
            update={"reasoning": "OPENAI_API_KEY must never enter a display trajectory."}
        )

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=executor,
            planner=recorded_planner,
            replanner=unsafe_reasoning_replanner,
            report_writer=recorded_report_writer,
        )
    )

    assert [event.index for event in result.trajectory] == list(
        range(1, len(result.trajectory) + 1)
    )
    assert {event.phase for event in result.trajectory} <= {
        "planning",
        "policy",
        "execution",
        "replanning",
        "evidence_gate",
        "report",
        "guardrail",
    }
    serialized = json.dumps(result.model_dump(mode="json"))
    for unsafe_text in (
        "OPENAI_API_KEY",
        "course-secret",
        "StdioServerParameters",
        "sys.executable",
        "environment mapping",
    ):
        assert unsafe_text not in serialized


def test_planner_custom_exception_returns_provider_error_without_raw_text() -> None:
    """Breaks if malformed planner output escapes the graph or leaks provider content."""
    executor = FakeExecutor()

    async def failing_planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        raise SdkPolicyError("OPENAI_API_KEY=planner-secret")

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=executor,
            planner=failing_planner,
            replanner=recorded_replanner,
            report_writer=recorded_report_writer,
        )
    )

    assert result.status == "provider_error"
    assert result.observations == ()
    assert executor.calls == []
    assert "planner-secret" not in json.dumps(result.model_dump(mode="json"))


def test_replanner_custom_exception_returns_a_typed_provider_error() -> None:
    """Breaks if a provider failure after execution crashes instead of stopping safely."""
    executor = FakeExecutor()

    async def one_step_planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        return ResearchPlan(goal="Collect one observation.", steps=(metric_step(1, "NVDA"),))

    async def failing_replanner(state: Mapping[str, Any]) -> ReplanDecision:
        del state
        raise SdkPolicyError("provider response was malformed")

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=executor,
            planner=one_step_planner,
            replanner=failing_replanner,
            report_writer=recorded_report_writer,
        )
    )

    assert result.status == "provider_error"
    assert [item.step_id for item in result.observations] == [1]


def test_report_custom_exception_returns_provider_error_without_a_briefing() -> None:
    """Breaks if synthesis failure discards the completed evidence trajectory."""
    executor = FakeExecutor()

    async def complete_planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        return ResearchPlan(
            goal="Collect the complete evidence gate inputs.",
            steps=(
                metric_step(1, "NVDA"),
                metric_step(2, "SU.PA"),
                document_step(3, "NVIDIA"),
                document_step(4, "Schneider Electric"),
            ),
        )

    async def failing_report_writer(
        question: str,
        observations: tuple[ResearchObservation, ...],
    ) -> AnalystBriefing:
        del question, observations
        raise SdkPolicyError("provider response was malformed")

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=executor,
            planner=complete_planner,
            replanner=finish_after_last_step,
            report_writer=failing_report_writer,
        )
    )

    assert result.status == "provider_error"
    assert result.evidence_gate.passed is True
    assert result.briefing is None
    assert [item.step_id for item in result.observations] == [1, 2, 3, 4]


def test_planner_malformed_return_becomes_a_typed_provider_error() -> None:
    """Breaks if a wrong-shaped planner return raises AttributeError outside the boundary."""

    async def malformed_planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> Any:
        del question, catalog
        return object()

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=FakeExecutor(),
            planner=malformed_planner,
            replanner=recorded_replanner,
            report_writer=recorded_report_writer,
        )
    )

    assert result.status == "provider_error"
    assert result.initial_plan.steps == ()
    assert result.final_steps == ()


def test_replanner_malformed_return_becomes_a_typed_provider_error() -> None:
    """Breaks if a wrong-shaped replanner return raises AttributeError outside the boundary."""

    async def one_step_planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        return ResearchPlan(goal="Collect one metric.", steps=(metric_step(1, "NVDA"),))

    async def malformed_replanner(state: Mapping[str, Any]) -> Any:
        del state
        return object()

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=FakeExecutor(),
            planner=one_step_planner,
            replanner=malformed_replanner,
            report_writer=recorded_report_writer,
        )
    )

    assert result.status == "provider_error"
    assert [item.step_id for item in result.observations] == [1]


def test_report_malformed_return_becomes_provider_error_without_a_briefing() -> None:
    """Breaks if a wrong-shaped report reaches terminal result validation."""

    async def complete_planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        return ResearchPlan(
            goal="Collect complete evidence.",
            steps=(
                metric_step(1, "NVDA"),
                metric_step(2, "SU.PA"),
                document_step(3, "NVIDIA"),
                document_step(4, "Schneider Electric"),
            ),
        )

    async def malformed_report_writer(
        question: str,
        observations: tuple[ResearchObservation, ...],
    ) -> Any:
        del question, observations
        return object()

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=FakeExecutor(),
            planner=complete_planner,
            replanner=finish_after_last_step,
            report_writer=malformed_report_writer,
        )
    )

    assert result.status == "provider_error"
    assert result.evidence_gate.passed is True
    assert result.briefing is None


def test_report_with_unsupported_provenance_is_redacted_and_rejected() -> None:
    """Breaks if an otherwise typed provider report can invent citations at completion."""

    async def complete_planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        return ResearchPlan(
            goal="Collect complete evidence.",
            steps=(
                metric_step(1, "NVDA"),
                metric_step(2, "SU.PA"),
                document_step(3, "NVIDIA"),
                document_step(4, "Schneider Electric"),
            ),
        )

    async def unsupported_report_writer(
        question: str,
        observations: tuple[ResearchObservation, ...],
    ) -> AnalystBriefing:
        del question, observations
        return AnalystBriefing(
            reported_facts=(
                CitedFact(
                    claim="Unsupported provider claim.",
                    source_references=("OPENAI_API_KEY=invented-source",),
                    evidence_ids=("invented-evidence",),
                ),
            ),
            cross_company_observations=("Different periods.",),
            interpretation=("No advice.",),
            limitations=("Periods differ.",),
            source_references=("OPENAI_API_KEY=invented-source",),
        )

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=FakeExecutor(),
            planner=complete_planner,
            replanner=finish_after_last_step,
            report_writer=unsupported_report_writer,
        )
    )

    serialized = json.dumps(result.model_dump(mode="json"))
    assert result.status == "provider_error"
    assert result.evidence_gate.passed is True
    assert result.briefing is None
    assert "invented-source" not in serialized
    assert "invented-evidence" not in serialized
    assert "OPENAI_API_KEY" not in serialized


@pytest.mark.parametrize("max_steps", [1, 6])
def test_configured_step_ceiling_allows_limits_up_to_six(max_steps: int) -> None:
    """Breaks if a caller cannot choose a stricter limit or the hard ceiling itself."""
    graph = build_plan_execute_graph(
        executor=FakeExecutor(),
        planner=recorded_planner,
        replanner=recorded_replanner,
        report_writer=recorded_report_writer,
        max_steps=max_steps,
    )

    assert graph is not None


def test_configured_step_ceiling_rejects_values_above_six() -> None:
    """Breaks if a caller can expand the host-owned six-step safety ceiling."""
    with pytest.raises(ValueError, match="between 1 and 6"):
        build_plan_execute_graph(
            executor=FakeExecutor(),
            planner=recorded_planner,
            replanner=recorded_replanner,
            report_writer=recorded_report_writer,
            max_steps=7,
        )


@pytest.mark.parametrize("max_replans", [0, 1])
def test_configured_replan_ceiling_allows_limits_up_to_one(max_replans: int) -> None:
    """Breaks if a caller cannot disable revisions or allow the maintained one revision."""
    graph = build_plan_execute_graph(
        executor=FakeExecutor(),
        planner=recorded_planner,
        replanner=recorded_replanner,
        report_writer=recorded_report_writer,
        max_replans=max_replans,
    )

    assert graph is not None


def test_configured_replan_ceiling_rejects_values_above_one() -> None:
    """Breaks if a caller can expand the host-owned one-revision safety ceiling."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        build_plan_execute_graph(
            executor=FakeExecutor(),
            planner=recorded_planner,
            replanner=recorded_replanner,
            report_writer=recorded_report_writer,
            max_replans=2,
        )


def test_executor_exception_returns_safe_provider_error_without_replanning() -> None:
    """Breaks if an executor runtime failure escapes, leaks, or reaches the replanner."""

    class RaisingExecutor(FakeExecutor):
        async def execute(
            self,
            step: PlanStep,
            *,
            attempt_id: int,
            plan_revision: int,
        ) -> ResearchObservation:
            del step, attempt_id, plan_revision
            raise OSError("OPENAI_API_KEY=executor-secret stderr=raw-server-output")

    executor = RaisingExecutor()
    replanner_calls: list[bool] = []

    async def one_step_planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        return ResearchPlan(goal="Collect one metric.", steps=(metric_step(1, "NVDA"),))

    async def replanner(state: Mapping[str, Any]) -> ReplanDecision:
        del state
        replanner_calls.append(True)
        return ReplanDecision(action="stop", reasoning="Should not be called.")

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=executor,
            planner=one_step_planner,
            replanner=replanner,
            report_writer=recorded_report_writer,
        )
    )

    assert result.status == "provider_error"
    assert result.observations == ()
    assert replanner_calls == []
    assert result.trajectory[-1].phase == "execution"
    assert result.trajectory[-1].status == "error"
    serialized = json.dumps(result.model_dump(mode="json"))
    assert "OPENAI_API_KEY" not in serialized
    assert "executor-secret" not in serialized
    assert "stderr" not in serialized
    assert "raw-server-output" not in serialized


def test_non_json_replacement_arguments_are_validated_before_canonicalization() -> None:
    """Breaks if duplicate-signature serialization runs on an unvalidated replacement."""

    async def planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        return ResearchPlan(
            goal="Reach a replacement after one successful call.",
            steps=(metric_step(1, "NVDA"), metric_step(2, "NVDA", metric="Revenue")),
        )

    async def replanner(state: Mapping[str, Any]) -> ReplanDecision:
        observations = tuple(state["observations"])
        if observations[-1].status == "ok":
            return ReplanDecision(action="continue", reasoning="Reach the failing step.")
        malformed = document_step(3, "NVIDIA").model_copy(
            update={
                "arguments": {
                    "company": "NVIDIA",
                    "query": {"OPENAI_API_KEY=non-json-secret"},
                    "top_k": 2,
                }
            }
        )
        return ReplanDecision(
            action="replace_remaining",
            reasoning="Propose malformed non-JSON arguments.",
            replacement_steps=(malformed,),
        )

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=FakeExecutor(),
            planner=planner,
            replanner=replanner,
            report_writer=recorded_report_writer,
        )
    )

    assert result.status == "plan_blocked"
    assert [item.step_id for item in result.observations] == [1, 2]
    assert result.replan_count == 0
    serialized = json.dumps(result.model_dump(mode="json"))
    assert "OPENAI_API_KEY" not in serialized
    assert "non-json-secret" not in serialized


def test_invalid_initial_plan_is_replaced_before_result_serialization() -> None:
    """Breaks if rejected planner content or validation details remain displayable."""

    async def unsafe_planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        return ResearchPlan(
            goal="stderr=raw-initial-provider-output",
            steps=(
                PlanStep(
                    step_id=1,
                    capability="OPENAI_API_KEY=initial-secret",
                    arguments={"stderr": "raw-initial-arguments"},
                    purpose="Retain no rejected provider content.",
                    expected_evidence=("none",),
                ),
            ),
        )

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=FakeExecutor(),
            planner=unsafe_planner,
            replanner=recorded_replanner,
            report_writer=recorded_report_writer,
        )
    )

    assert result.status == "plan_blocked"
    assert result.initial_plan.steps == ()
    assert result.final_steps == ()
    assert result.trajectory[-1].summary == "initial_plan_rejected"
    serialized = json.dumps(result.model_dump(mode="json"))
    for unsafe_text in (
        "OPENAI_API_KEY",
        "initial-secret",
        "stderr",
        "raw-initial-provider-output",
        "raw-initial-arguments",
    ):
        assert unsafe_text not in serialized


def test_invalid_replacement_plan_is_excluded_from_result_serialization() -> None:
    """Breaks if a rejected replacement or raw validation message becomes displayable."""

    async def planner(
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        del question, catalog
        return ResearchPlan(
            goal="Reach a rejected replacement.",
            steps=(metric_step(1, "NVDA"), metric_step(2, "NVDA", metric="Revenue")),
        )

    async def unsafe_replanner(state: Mapping[str, Any]) -> ReplanDecision:
        observations = tuple(state["observations"])
        if observations[-1].status == "ok":
            return ReplanDecision(action="continue", reasoning="Reach the failing step.")
        return ReplanDecision(
            action="replace_remaining",
            reasoning="stderr=raw-replanner-output",
            replacement_steps=(
                PlanStep(
                    step_id=3,
                    capability="OPENAI_API_KEY=replacement-secret",
                    arguments={"stderr": "raw-replacement-arguments"},
                    purpose="Retain no rejected replacement content.",
                    expected_evidence=("none",),
                ),
            ),
        )

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=FakeExecutor(),
            planner=planner,
            replanner=unsafe_replanner,
            report_writer=recorded_report_writer,
        )
    )

    assert result.status == "plan_blocked"
    assert [step.step_id for step in result.final_steps] == [1, 2]
    assert result.trajectory[-1].summary == "replacement_plan_rejected"
    serialized = json.dumps(result.model_dump(mode="json"))
    for unsafe_text in (
        "OPENAI_API_KEY",
        "replacement-secret",
        "stderr",
        "raw-replanner-output",
        "raw-replacement-arguments",
    ):
        assert unsafe_text not in serialized
