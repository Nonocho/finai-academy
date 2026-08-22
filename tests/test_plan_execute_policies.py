from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

from finai_academy.plan_execute_graph import run_plan_execute
from finai_academy.plan_execute_policies import (
    INITIAL_RECORDED_STEPS,
    MISSION,
    RECORDED_REPLACEMENT_STEPS,
    build_live_plan_execute_policies,
    recorded_planner,
    recorded_replanner,
    recorded_report_writer,
)
from finai_academy.research_planning import (
    AnalystBriefing,
    CitedFact,
    PlannerToolSpec,
    PlanStep,
    ReplanDecision,
    ResearchObservation,
    ResearchPlan,
)
from finai_academy.settings import Settings


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


def successful_observations() -> tuple[ResearchObservation, ...]:
    return (
        ResearchObservation(
            attempt_id=1,
            step_id=1,
            plan_revision=0,
            capability="get_company_metric",
            arguments={"ticker": "NVDA", "metric": "P/E"},
            status="ok",
            result={
                "company": "NVIDIA",
                "metric": "P/E",
                "value": 47.2,
                "unit": "x",
                "as_of": "2026-08-15",
                "OPENAI_API_KEY": "secret",
            },
            source_references=("NVIDIA metrics snapshot",),
            duration_ms=1,
        ),
        ResearchObservation(
            attempt_id=2,
            step_id=2,
            plan_revision=0,
            capability="get_company_metric",
            arguments={"ticker": "SU.PA", "metric": "P/E"},
            status="ok",
            result={
                "company": "Schneider Electric",
                "metric": "P/E",
                "value": 24.7,
                "unit": "x",
                "as_of": "2026-07-31",
            },
            source_references=("Schneider metrics snapshot",),
            duration_ms=1,
        ),
        ResearchObservation(
            attempt_id=4,
            step_id=4,
            plan_revision=0,
            capability="search_financial_documents",
            arguments={"company": "Schneider Electric", "query": "revenue growth", "top_k": 2},
            status="ok",
            result={
                "company": "Schneider Electric",
                "hits": [
                    {
                        "evidence_id": "se-h1",
                        "text": "Energy Management revenue grew 8% in the period.",
                        "period": "H1 2026",
                        "source": "Schneider public report",
                    }
                ],
            },
            evidence_ids=("se-h1",),
            source_references=("Schneider public report",),
            duration_ms=1,
        ),
        ResearchObservation(
            attempt_id=5,
            step_id=5,
            plan_revision=1,
            capability="search_financial_documents",
            arguments={"company": "NVIDIA", "query": "revenue growth", "top_k": 2},
            status="ok",
            result={
                "company": "NVIDIA",
                "hits": [
                    {
                        "evidence_id": "nvda-q2",
                        "text": "Data Center revenue grew 56% year over year.",
                        "period": "Q2 2026",
                        "source": "NVIDIA public filing",
                    }
                ],
            },
            evidence_ids=("nvda-q2",),
            source_references=("NVIDIA public filing",),
            duration_ms=1,
        ),
    )


def test_recorded_planner_returns_the_maintained_plan() -> None:
    """Breaks if the classroom route loses its controlled failed Revenue attempt."""
    plan = asyncio.run(recorded_planner(MISSION, tool_catalog()))

    assert [step.step_id for step in plan.steps] == [1, 2, 3, 4]
    assert plan.steps[2].arguments == {"ticker": "NVDA", "metric": "Revenue"}


def test_recorded_replanner_replaces_unsupported_revenue_with_document_evidence() -> None:
    """Breaks if the known metric failure does not switch to the safe evidence route."""
    failed_observation = ResearchObservation(
        attempt_id=3,
        step_id=3,
        plan_revision=0,
        capability="get_company_metric",
        arguments={"ticker": "NVDA", "metric": "Revenue"},
        status="error",
        result={"valid_values": ["EPS", "P/E"]},
        error_code="unsupported_metric",
        duration_ms=1.0,
    )

    decision = asyncio.run(
        recorded_replanner(
            {
                "observations": (failed_observation,),
                "active_steps": INITIAL_RECORDED_STEPS,
                "current_index": 3,
            }
        )
    )

    assert decision.action == "replace_remaining"
    assert [step.step_id for step in decision.replacement_steps] == [5, 6]
    assert decision.replacement_steps[0].capability == "search_financial_documents"
    assert decision.replacement_steps[0].arguments == {
        "company": "NVIDIA",
        "query": "revenue growth",
        "top_k": 2,
    }


def test_recorded_replanner_does_not_replace_for_an_unrelated_metric_failure() -> None:
    """Breaks if a different failed step is mistaken for the maintained Revenue lesson route."""
    unrelated_failure = ResearchObservation(
        attempt_id=2,
        step_id=2,
        plan_revision=0,
        capability="get_company_metric",
        arguments={"ticker": "SU.PA", "metric": "Revenue"},
        status="error",
        result={"valid_values": ["EPS", "P/E"]},
        error_code="unsupported_metric",
        duration_ms=1,
    )

    decision = asyncio.run(
        recorded_replanner(
            {
                "observations": (unrelated_failure,),
                "active_steps": INITIAL_RECORDED_STEPS,
                "current_index": 2,
            }
        )
    )

    assert decision.action == "continue"


def test_recorded_report_uses_verified_facts_and_states_comparison_limits() -> None:
    """Breaks if the offline briefing omits public provenance or material comparison limits."""
    report = asyncio.run(recorded_report_writer(MISSION, successful_observations()))

    assert any("NVIDIA" in fact.claim for fact in report.reported_facts)
    assert any("Schneider Electric" in fact.claim for fact in report.reported_facts)
    assert CitedFact(
        claim="NVIDIA (Q2 2026): Data Center revenue grew 56% year over year.",
        source_references=("NVIDIA public filing",),
        evidence_ids=("nvda-q2",),
    ) in report.reported_facts
    assert report.source_references == tuple(
        dict.fromkeys(
            source
            for fact in report.reported_facts
            for source in fact.source_references
        )
    )
    assert len(report.source_references) >= 2
    assert any("currency" in limitation.casefold() for limitation in report.limitations)
    assert any("period" in limitation.casefold() for limitation in report.limitations)
    assert any("business mix" in limitation.casefold() for limitation in report.limitations)


def test_recorded_metric_fact_keeps_source_without_document_evidence_id() -> None:
    """Breaks if a metric claim is forced into the document citation-pair contract."""
    metric = successful_observations()[0].model_copy(
        update={"evidence_ids": ("metric-row-id",)}
    )

    report = asyncio.run(recorded_report_writer(MISSION, (metric,)))

    assert report.reported_facts[0].source_references == ("NVIDIA metrics snapshot",)
    assert report.reported_facts[0].evidence_ids == ()


class FakeStructuredModel:
    def __init__(self, response: object, schemas: list[type[Any]], prompts: list[object]) -> None:
        self._response = response
        self._schemas = schemas
        self._prompts = prompts

    def with_structured_output(self, schema: type[Any]) -> FakeStructuredModel:
        self._schemas.append(schema)
        return self

    async def ainvoke(self, payload: object) -> object:
        self._prompts.append(payload)
        return self._response


class FakeModelFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.schemas: list[type[Any]] = []
        self.prompts: list[object] = []

    def __call__(self, _settings: Settings) -> FakeStructuredModel:
        self.calls += 1
        responses: tuple[object, ...] = (
            ResearchPlan(goal=MISSION, steps=INITIAL_RECORDED_STEPS),
            ReplanDecision(action="finish", reasoning="The supplied observations are complete."),
            AnalystBriefing(
                reported_facts=(
                    CitedFact(
                        claim="NVIDIA P/E is 47.2x.",
                        source_references=("NVIDIA metrics snapshot",),
                    ),
                ),
                cross_company_observations=("The observations have different periods.",),
                interpretation=("This is descriptive evidence, not investment advice.",),
                limitations=("Currencies differ.",),
                source_references=("NVIDIA metrics snapshot",),
            ),
        )
        return FakeStructuredModel(responses[self.calls - 1], self.schemas, self.prompts)


def test_live_policies_use_lazy_structured_output_and_safe_prompt_context() -> None:
    """Breaks if live prompts absorb runtime configuration instead of typed research context."""
    factory = FakeModelFactory()
    observations = successful_observations() + (
        ResearchObservation(
            attempt_id=3,
            step_id=3,
            plan_revision=0,
            capability="get_company_metric",
            arguments={"ticker": "NVDA", "metric": "Revenue"},
            status="error",
            result={"valid_values": ["EPS", "P/E"]},
            error_code="unsupported_metric",
            duration_ms=1,
        ),
    )
    replanning_state: Mapping[str, Any] = {
        "observations": observations,
        "active_steps": INITIAL_RECORDED_STEPS,
        "current_index": 4,
        "server_parameters": {"command": "sys.executable", "env": {"TOKEN": "secret"}},
        "OPENAI_API_KEY": "secret",
    }

    async def scenario() -> tuple[ResearchPlan, ReplanDecision, AnalystBriefing]:
        planner, replanner, writer = build_live_plan_execute_policies(
            Settings(provider="ollama"), model_factory=factory
        )
        return (
            await planner(MISSION, tool_catalog()),
            await replanner(replanning_state),
            await writer(MISSION, observations),
        )

    plan, decision, report = asyncio.run(scenario())

    assert isinstance(plan, ResearchPlan)
    assert isinstance(decision, ReplanDecision)
    assert isinstance(report, AnalystBriefing)
    assert factory.calls == 3
    assert factory.schemas == [ResearchPlan, ReplanDecision, AnalystBriefing]
    rendered_prompts = json.dumps(factory.prompts)
    for unsafe_text in (
        "OPENAI_API_KEY",
        "sys.executable",
        '"TOKEN"',
        "secret",
        "active_steps",
        "current_index",
    ):
        assert unsafe_text not in rendered_prompts
    for safe_text in (
        MISSION,
        "get_company_metric",
        "unsupported_metric",
        "NVIDIA public filing",
        "Data Center revenue grew 56% year over year.",
        "nvda-q2",
        "source_references",
        "evidence_ids",
    ):
        assert safe_text in rendered_prompts


class ReplacementModel:
    def __init__(self, prompts: list[object]) -> None:
        self._prompts = prompts
        self._responses = iter(
            (
                ReplanDecision(action="continue", reasoning="Continue to the next research step."),
                ReplanDecision(action="continue", reasoning="Continue to the next research step."),
                ReplanDecision(
                    action="replace_remaining",
                    reasoning="Replace the unsupported metric with two document searches.",
                    replacement_steps=RECORDED_REPLACEMENT_STEPS,
                ),
                ReplanDecision(action="continue", reasoning="Continue to the final research step."),
                ReplanDecision(action="finish", reasoning="The revised plan is complete."),
            )
        )

    def with_structured_output(self, _schema: type[Any]) -> ReplacementModel:
        return self

    async def ainvoke(self, payload: object) -> ReplanDecision:
        self._prompts.append(payload)
        return next(self._responses)


class ReplacementFactory:
    def __init__(self) -> None:
        self.prompts: list[object] = []

    def __call__(self, _settings: Settings) -> ReplacementModel:
        return ReplacementModel(self.prompts)


class CapacityAwareModel:
    def __init__(self, prompts: list[object]) -> None:
        self._prompts = prompts

    def with_structured_output(self, _schema: type[Any]) -> CapacityAwareModel:
        return self

    async def ainvoke(self, payload: object) -> ReplanDecision:
        self._prompts.append(payload)
        context = json.loads(payload[1][1])
        errors = context["typed_errors"]
        if errors and context["remaining_step_capacity"] == 0:
            return ReplanDecision(
                action="stop",
                reasoning="The graph's remaining step budget is exhausted.",
            )
        return ReplanDecision(action="continue", reasoning="Continue within the graph budget.")


class CapacityAwareFactory:
    def __init__(self) -> None:
        self.prompts: list[object] = []

    def __call__(self, _settings: Settings) -> CapacityAwareModel:
        return CapacityAwareModel(self.prompts)


class ReplacementExecutor:
    catalog = tool_catalog()

    async def execute(
        self,
        step: PlanStep,
        *,
        attempt_id: int,
        plan_revision: int,
    ) -> ResearchObservation:
        if step.step_id == 3:
            return ResearchObservation(
                attempt_id=attempt_id,
                step_id=step.step_id,
                plan_revision=plan_revision,
                capability=step.capability,
                arguments=step.arguments,
                status="error",
                result={"valid_values": ["EPS", "P/E"], "OPENAI_API_KEY": "secret"},
                error_code="unsupported_metric",
                duration_ms=1,
            )
        company = (
            "NVIDIA"
            if step.arguments.get("ticker") == "NVDA"
            else "Schneider Electric"
            if step.arguments.get("ticker") == "SU.PA"
            else step.arguments.get("company")
        )
        if step.capability == "get_company_metric":
            result: dict[str, Any] = {
                "company": company,
                "metric": "P/E",
                "value": 42.0,
                "unit": "x",
                "as_of": "2026-08-15",
            }
        else:
            result = {
                "company": company,
                "hits": [
                    {
                        "evidence_id": f"evidence-{step.step_id}",
                        "text": "Public growth evidence.",
                        "period": "Q2 2026",
                        "source": f"public-source-{step.step_id}",
                    }
                ],
            }
        return ResearchObservation(
            attempt_id=attempt_id,
            step_id=step.step_id,
            plan_revision=plan_revision,
            capability=step.capability,
            arguments=step.arguments,
            status="ok",
            result=result,
            evidence_ids=(f"evidence-{step.step_id}",),
            source_references=(f"public-source-{step.step_id}",),
            duration_ms=1,
        )


def test_live_replanner_receives_safe_reserved_ids_and_validates_its_replacement_tail() -> None:
    """Breaks if the model cannot see Task 3's reserved IDs and remaining step budget."""
    factory = ReplacementFactory()
    _, replanner, _ = build_live_plan_execute_policies(
        Settings(provider="ollama"), model_factory=factory
    )

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=ReplacementExecutor(),
            planner=recorded_planner,
            replanner=replanner,
            report_writer=recorded_report_writer,
        )
    )

    assert result.status == "completed"
    assert [step.step_id for step in result.final_steps] == [1, 2, 3, 5, 6]
    assert result.replan_count == 1
    replanning_context = next(
        json.loads(messages[1][1])
        for messages in factory.prompts
        if "unsupported_metric" in messages[1][1]
    )
    assert replanning_context["reserved_step_ids"] == [1, 2, 3, 4]
    assert replanning_context["next_replacement_step_id"] == 5
    assert replanning_context["max_step_budget"] == 6
    assert replanning_context["remaining_step_capacity"] == 2
    rendered_prompt = json.dumps(factory.prompts)
    for unsafe_text in ("OPENAI_API_KEY", "secret", "server_parameters", "sys.executable"):
        assert unsafe_text not in rendered_prompt


def test_live_replanner_uses_a_stricter_graph_step_limit_for_remaining_capacity() -> None:
    """Breaks if the prompt advertises replacement slots that Task 3 will reject."""
    factory = CapacityAwareFactory()
    _, replanner, _ = build_live_plan_execute_policies(
        Settings(provider="ollama"), model_factory=factory
    )

    result = asyncio.run(
        run_plan_execute(
            question=MISSION,
            executor=ReplacementExecutor(),
            planner=recorded_planner,
            replanner=replanner,
            report_writer=recorded_report_writer,
            max_steps=4,
        )
    )

    assert result.status == "execution_stopped"
    failure_context = next(
        json.loads(messages[1][1])
        for messages in factory.prompts
        if "unsupported_metric" in messages[1][1]
    )
    assert failure_context["reserved_step_ids"] == [1, 2, 3, 4]
    assert failure_context["max_step_budget"] == 4
    assert failure_context["remaining_step_capacity"] == 0


def test_live_replanner_accepts_only_exact_safe_graph_progress() -> None:
    """Breaks if malformed direct IDs are filtered into a misleading replacement budget."""
    cases: tuple[tuple[str, dict[str, Any], bool, int], ...] = (
        ("valid default", {"all_step_ids": (1, 2, 3, 4), "max_steps": 6}, True, 2),
        ("valid stricter", {"all_step_ids": (1, 2, 3, 4), "max_steps": 4}, True, 0),
        ("missing", {"max_steps": 4}, False, 0),
        ("none", {"all_step_ids": None, "max_steps": 4}, False, 0),
        ("string", {"all_step_ids": "1,2", "max_steps": 4}, False, 0),
        ("bytes", {"all_step_ids": b"1,2", "max_steps": 4}, False, 0),
        ("bool", {"all_step_ids": (1, True), "max_steps": 4}, False, 0),
        ("non integer", {"all_step_ids": (1, "bad"), "max_steps": 4}, False, 0),
        ("below one", {"all_step_ids": (0,), "max_steps": 4}, False, 0),
        ("duplicate", {"all_step_ids": (1, 1), "max_steps": 4}, False, 0),
        ("non monotonic", {"all_step_ids": (2, 1), "max_steps": 4}, False, 0),
        ("empty direct state", {"all_step_ids": (), "max_steps": 4}, False, 0),
        ("more reserved IDs than the graph allows", {"all_step_ids": (1, 2, 3, 4, 5), "max_steps": 4}, False, 0),
        ("missing graph limit", {"all_step_ids": (1, 2, 3, 4)}, False, 0),
    )

    for label, state, valid, expected_capacity in cases:
        factory = CapacityAwareFactory()
        _, replanner, _ = build_live_plan_execute_policies(
            Settings(provider="ollama"), model_factory=factory
        )

        asyncio.run(replanner({**state, "observations": ()}))

        context = json.loads(factory.prompts[0][1][1])
        assert context["replanning_state_valid"] is valid, label
        assert context["remaining_step_capacity"] == expected_capacity, label
        if valid:
            assert context["reserved_step_ids"] == [1, 2, 3, 4], label
            assert context["next_replacement_step_id"] == 5, label
            assert context["max_step_budget"] == state["max_steps"], label
        else:
            assert context["reserved_step_ids"] == [], label
            assert "next_replacement_step_id" not in context, label
            assert "max_step_budget" not in context, label
