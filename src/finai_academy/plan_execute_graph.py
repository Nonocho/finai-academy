"""Bounded plan-execute-replan orchestration for Lesson 11 financial research."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from finai_academy.research_planning import (
    AnalystBriefing,
    EvidenceGateResult,
    PlannerToolSpec,
    PlanStep,
    ReplanDecision,
    ResearchObservation,
    ResearchPlan,
    TrajectoryEvent,
    evaluate_evidence_gate,
    validate_plan,
    validate_replacement,
)


class PlannerPolicy(Protocol):
    """Propose an initial research plan from planner-safe capability metadata."""

    async def __call__(
        self,
        question: str,
        catalog: tuple[PlannerToolSpec, ...],
    ) -> ResearchPlan:
        raise NotImplementedError


class ReplannerPolicy(Protocol):
    """Review serializable graph state and decide how unfinished work proceeds."""

    async def __call__(self, state: Mapping[str, Any]) -> ReplanDecision:
        raise NotImplementedError


class ReportPolicy(Protocol):
    """Create a briefing from the verified research scratchpad."""

    async def __call__(
        self,
        question: str,
        observations: tuple[ResearchObservation, ...],
    ) -> AnalystBriefing:
        raise NotImplementedError


class ResearchExecutor(Protocol):
    """Execute one validated plan step through a persistent runtime boundary."""

    catalog: tuple[PlannerToolSpec, ...]

    async def execute(
        self,
        step: PlanStep,
        *,
        attempt_id: int,
        plan_revision: int,
    ) -> ResearchObservation:
        raise NotImplementedError


class PlanExecuteState(TypedDict, total=False):
    """Serializable state shared by the plan-execute graph nodes."""

    question: str
    catalog: tuple[PlannerToolSpec, ...]
    max_steps: int
    initial_plan: ResearchPlan
    active_steps: tuple[PlanStep, ...]
    all_step_ids: tuple[int, ...]
    current_index: int
    observations: tuple[ResearchObservation, ...]
    trajectory: tuple[TrajectoryEvent, ...]
    plan_revision: int
    replan_count: int
    status: str
    briefing: AnalystBriefing | None
    evidence_gate: EvidenceGateResult


class PlanExecuteResult(BaseModel):
    """Typed terminal result returned by a bounded research run."""

    status: Literal[
        "completed",
        "plan_blocked",
        "execution_stopped",
        "replan_budget_exhausted",
        "insufficient_evidence",
        "provider_error",
    ]
    initial_plan: ResearchPlan
    final_steps: tuple[PlanStep, ...]
    observations: tuple[ResearchObservation, ...]
    trajectory: tuple[TrajectoryEvent, ...]
    replan_count: int = Field(ge=0)
    evidence_gate: EvidenceGateResult
    briefing: AnalystBriefing | None = None


def _safe_rejected_plan() -> ResearchPlan:
    return ResearchPlan(goal="Rejected plan content was removed.", steps=())


def _call_signature(capability: str, arguments: Mapping[str, Any]) -> tuple[str, str]:
    return capability, json.dumps(arguments, sort_keys=True, separators=(",", ":"))


def _append_event(
    state: Mapping[str, Any],
    *,
    phase: Literal[
        "planning",
        "policy",
        "execution",
        "replanning",
        "evidence_gate",
        "report",
        "guardrail",
    ],
    status: Literal["ok", "error", "blocked"],
    summary: str,
    step_id: int | None = None,
    attempt_id: int | None = None,
    duration_ms: float = 0,
) -> tuple[TrajectoryEvent, ...]:
    trajectory = tuple(state.get("trajectory", ()))
    return trajectory + (
        TrajectoryEvent(
            index=len(trajectory) + 1,
            phase=phase,
            status=status,
            summary=summary,
            step_id=step_id,
            attempt_id=attempt_id,
            duration_ms=duration_ms,
        ),
    )


def build_plan_execute_graph(
    *,
    executor: ResearchExecutor,
    planner: PlannerPolicy,
    replanner: ReplannerPolicy,
    report_writer: ReportPolicy,
    max_steps: int = 6,
    max_replans: int = 1,
) -> Any:
    """Compile the bounded graph with runtime dependencies captured in closures."""

    if not 1 <= max_steps <= 6:
        raise ValueError("max_steps must be between 1 and 6")
    if not 0 <= max_replans <= 1:
        raise ValueError("max_replans must be between 0 and 1")

    async def planner_node(state: PlanExecuteState) -> dict[str, Any]:
        try:
            plan = await planner(state["question"], state["catalog"])
            if not isinstance(plan, ResearchPlan):
                raise TypeError("planner returned an invalid plan")
        except Exception:  # noqa: BLE001 - provider boundary must fail closed
            empty_plan = _safe_rejected_plan()
            return {
                "initial_plan": empty_plan,
                "active_steps": (),
                "all_step_ids": (),
                "current_index": 0,
                "status": "provider_error",
                "trajectory": _append_event(
                    state,
                    phase="planning",
                    status="error",
                    summary="Planner policy failed to return a valid research plan.",
                ),
            }
        return {
            "initial_plan": plan,
            "active_steps": plan.steps,
            "all_step_ids": tuple(step.step_id for step in plan.steps),
            "current_index": 0,
            "status": "planning",
            "trajectory": _append_event(
                state,
                phase="planning",
                status="ok",
                summary=f"Planner proposed {len(plan.steps)} research steps.",
            ),
        }

    def plan_gate_node(state: PlanExecuteState) -> dict[str, Any]:
        if state["status"] == "provider_error":
            return {"status": "provider_error"}
        try:
            checked = validate_plan(state["initial_plan"], state["catalog"], max_steps=max_steps)
        except ValueError:
            return {
                "initial_plan": _safe_rejected_plan(),
                "active_steps": (),
                "all_step_ids": (),
                "current_index": 0,
                "status": "plan_blocked",
                "trajectory": _append_event(
                    state,
                    phase="guardrail",
                    status="blocked",
                    summary="initial_plan_rejected",
                ),
            }
        return {
            "active_steps": checked.steps,
            "status": "executing",
            "trajectory": _append_event(
                state,
                phase="policy",
                status="ok",
                summary="Initial research plan passed host validation.",
            ),
        }

    async def executor_node(state: PlanExecuteState) -> dict[str, Any]:
        step = state["active_steps"][state["current_index"]]
        attempt_id = len(state.get("observations", ())) + 1
        try:
            observation = await executor.execute(
                step,
                attempt_id=attempt_id,
                plan_revision=state.get("plan_revision", 0),
            )
            if not isinstance(observation, ResearchObservation):
                raise TypeError("executor returned an invalid observation")
        except Exception:  # noqa: BLE001 - runtime boundary must fail closed
            return {
                "status": "provider_error",
                "trajectory": _append_event(
                    state,
                    phase="execution",
                    status="error",
                    summary="Executor failed to return a valid observation.",
                    step_id=step.step_id,
                    attempt_id=attempt_id,
                ),
            }
        return {
            "observations": tuple(state.get("observations", ())) + (observation,),
            "current_index": state["current_index"] + 1,
            "trajectory": _append_event(
                state,
                phase="execution",
                status="ok" if observation.status == "ok" else observation.status,
                summary=(
                    f"Step {step.step_id} completed."
                    if observation.status == "ok"
                    else f"Step {step.step_id} returned {observation.error_code or observation.status}."
                ),
                step_id=step.step_id,
                attempt_id=attempt_id,
                duration_ms=observation.duration_ms,
            ),
        }

    async def replanner_node(state: PlanExecuteState) -> dict[str, Any]:
        if state.get("status") == "provider_error":
            return {"status": "provider_error"}
        try:
            decision = await replanner(state)
            if not isinstance(decision, ReplanDecision):
                raise TypeError("replanner returned an invalid decision")
        except Exception:  # noqa: BLE001 - provider boundary must fail closed
            return {
                "status": "provider_error",
                "trajectory": _append_event(
                    state,
                    phase="replanning",
                    status="error",
                    summary="Replanner policy failed to return a valid decision.",
                ),
            }
        trajectory = _append_event(
            state,
            phase="replanning",
            status="ok",
            summary=f"Replanner selected {decision.action}.",
        )
        if decision.action == "continue":
            return {"status": "executing", "trajectory": trajectory}
        if decision.action == "finish":
            return {"status": "evidence_ready", "trajectory": trajectory}
        if decision.action == "stop":
            return {"status": "execution_stopped", "trajectory": trajectory}

        if state.get("replan_count", 0) >= max_replans:
            return {
                "status": "replan_budget_exhausted",
                "trajectory": _append_event(
                    {**state, "trajectory": trajectory},
                    phase="guardrail",
                    status="blocked",
                    summary=f"Stopped after {max_replans} allowed plan revisions.",
                ),
            }

        successful_observations = tuple(
            item for item in state.get("observations", ()) if item.status == "ok"
        )
        try:
            checked = validate_replacement(
                decision.replacement_steps,
                catalog=state["catalog"],
                prior_step_ids=state["all_step_ids"],
                successful_step_ids=tuple(item.step_id for item in successful_observations),
                max_total_steps=max_steps,
            )
        except ValueError:
            return {
                "status": "plan_blocked",
                "trajectory": _append_event(
                    {**state, "trajectory": trajectory},
                    phase="guardrail",
                    status="blocked",
                    summary="replacement_plan_rejected",
                ),
            }
        try:
            successful_signatures = {
                _call_signature(item.capability, item.arguments)
                for item in successful_observations
            }
            repeated_signatures = {
                _call_signature(step.capability, step.arguments) for step in checked
            } & successful_signatures
        except Exception:  # noqa: BLE001 - validated signature boundary must fail closed
            return {
                "status": "plan_blocked",
                "trajectory": _append_event(
                    {**state, "trajectory": trajectory},
                    phase="guardrail",
                    status="blocked",
                    summary="replacement_signature_rejected",
                ),
            }
        if repeated_signatures:
            return {
                "status": "plan_blocked",
                "trajectory": _append_event(
                    {**state, "trajectory": trajectory},
                    phase="guardrail",
                    status="blocked",
                    summary="replacement_repeats_successful_call",
                ),
            }
        executed_prefix = state["active_steps"][: state["current_index"]]
        return {
            "active_steps": executed_prefix + checked,
            "all_step_ids": state["all_step_ids"] + tuple(step.step_id for step in checked),
            "current_index": len(executed_prefix),
            "plan_revision": state.get("plan_revision", 0) + 1,
            "replan_count": state.get("replan_count", 0) + 1,
            "status": "executing",
            "trajectory": trajectory,
        }

    def evidence_gate_node(state: PlanExecuteState) -> dict[str, Any]:
        evidence_gate = evaluate_evidence_gate(state.get("observations", ()))
        return {
            "evidence_gate": evidence_gate,
            "status": "report_ready" if evidence_gate.passed else "insufficient_evidence",
            "trajectory": _append_event(
                state,
                phase="evidence_gate",
                status="ok" if evidence_gate.passed else "blocked",
                summary=(
                    "Required metric and document evidence is present."
                    if evidence_gate.passed
                    else "Reporting blocked because required evidence is missing."
                ),
            ),
        }

    async def report_node(state: PlanExecuteState) -> dict[str, Any]:
        observations = tuple(state.get("observations", ()))
        try:
            briefing = await report_writer(state["question"], observations)
            if not isinstance(briefing, AnalystBriefing):
                raise TypeError("report writer returned an invalid briefing")
        except Exception:  # noqa: BLE001 - provider boundary must fail closed
            return {
                "briefing": None,
                "status": "provider_error",
                "trajectory": _append_event(
                    state,
                    phase="report",
                    status="error",
                    summary="Report policy failed to return a valid briefing.",
                ),
            }
        return {
            "briefing": briefing,
            "status": "completed",
            "trajectory": _append_event(
                state,
                phase="report",
                status="ok",
                summary="Analyst briefing created from verified observations.",
            ),
        }

    def route_after_plan_gate(state: PlanExecuteState) -> str:
        return "stop" if state["status"] in {"plan_blocked", "provider_error"} else "execute"

    def route_after_replanning(state: PlanExecuteState) -> str:
        if state["status"] in {
            "execution_stopped",
            "plan_blocked",
            "replan_budget_exhausted",
            "provider_error",
        }:
            return "stop"
        if state["status"] == "evidence_ready":
            return "evidence_gate"
        if state["current_index"] < len(state["active_steps"]):
            return "execute"
        return "evidence_gate"

    def route_after_evidence_gate(state: PlanExecuteState) -> str:
        return "report" if state["status"] == "report_ready" else "stop"

    workflow = StateGraph(PlanExecuteState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("plan_gate", plan_gate_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("replanner", replanner_node)
    workflow.add_node("evidence_gate", evidence_gate_node)
    workflow.add_node("report", report_node)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "plan_gate")
    workflow.add_conditional_edges(
        "plan_gate",
        route_after_plan_gate,
        {"execute": "executor", "stop": END},
    )
    workflow.add_edge("executor", "replanner")
    workflow.add_conditional_edges(
        "replanner",
        route_after_replanning,
        {"execute": "executor", "evidence_gate": "evidence_gate", "stop": END},
    )
    workflow.add_conditional_edges(
        "evidence_gate",
        route_after_evidence_gate,
        {"report": "report", "stop": END},
    )
    workflow.add_edge("report", END)
    return workflow.compile()


async def run_plan_execute(
    *,
    question: str,
    executor: ResearchExecutor,
    planner: PlannerPolicy,
    replanner: ReplannerPolicy,
    report_writer: ReportPolicy,
    max_steps: int = 6,
    max_replans: int = 1,
) -> PlanExecuteResult:
    """Run the compiled graph and return its typed, display-safe terminal state."""

    graph = build_plan_execute_graph(
        executor=executor,
        planner=planner,
        replanner=replanner,
        report_writer=report_writer,
        max_steps=max_steps,
        max_replans=max_replans,
    )
    final_state = await graph.ainvoke(
        {
            "question": question,
            "catalog": executor.catalog,
            "max_steps": max_steps,
            "observations": (),
            "trajectory": (),
            "plan_revision": 0,
            "replan_count": 0,
        }
    )
    return PlanExecuteResult(
        status=final_state["status"],
        initial_plan=final_state["initial_plan"],
        final_steps=final_state["active_steps"],
        observations=tuple(final_state.get("observations", ())),
        trajectory=tuple(final_state.get("trajectory", ())),
        replan_count=final_state.get("replan_count", 0),
        evidence_gate=final_state.get(
            "evidence_gate",
            EvidenceGateResult(
                passed=False,
                coverage={},
                missing_requirements=("not run",),
            ),
        ),
        briefing=final_state.get("briefing"),
    )
