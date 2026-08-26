from __future__ import annotations

import finai_academy.self_correcting_agent as self_correcting_agent
from finai_academy.self_correcting_agent import (
    AgentAction,
    MetricRequest,
    build_metric_registry,
    build_self_correcting_graph,
    recorded_correction_policy,
    run_self_correcting_agent,
)

SNAPSHOT = {
    "dataset_id": "lesson09-metrics-snapshot-v1",
    "notice": "Controlled course fixture; not live data or investment advice.",
    "as_of": "2026-08-20",
    "source": "First Finance controlled classroom fixture",
    "metrics": {
        "NVDA": {"company": "NVIDIA", "P/E": 52.4, "EPS": 4.08},
        "SU.PA": {"company": "Schneider Electric", "P/E": 31.8, "EPS": 9.12},
    },
}


def test_live_action_schema_is_strict_and_openai_compatible() -> None:
    """Optional schema keys would make strict Structured Outputs reject the request."""

    assert hasattr(self_correcting_agent, "ModelAgentAction")
    ModelAgentAction = self_correcting_agent.ModelAgentAction
    schema = ModelAgentAction.model_json_schema()

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "action",
        "ticker",
        "metric",
        "answer",
        "reason",
    }


def test_live_action_converts_to_the_internal_agent_contract() -> None:
    """A flat wire schema must still produce the existing validated internal action."""

    assert hasattr(self_correcting_agent, "ModelAgentAction")
    ModelAgentAction = self_correcting_agent.ModelAgentAction
    wire_action = ModelAgentAction(
        action="tool",
        ticker="NVDA",
        metric="P/E",
        answer=None,
        reason="Use the valid metric returned by the tool.",
    )

    action = wire_action.to_agent_action()

    assert action == AgentAction(
        action="tool",
        request=MetricRequest(ticker="NVDA", metric="P/E"),
        reason="Use the valid metric returned by the tool.",
    )


def test_invalid_metric_becomes_actionable_retryable_observation() -> None:
    """A regression to raising ValueError would crash the graph instead of teaching recovery."""

    registry = build_metric_registry(SNAPSHOT)

    result = registry.invoke(MetricRequest(ticker="NVDA", metric="PE"))

    assert result.status == "error"
    assert result.error_code == "unsupported_metric"
    assert result.retryable is True
    assert result.allowed_metrics == ("EPS", "P/E")
    assert "PE" in result.message
    assert "P/E" in result.message


def test_agent_corrects_pe_to_p_slash_e_after_structured_feedback() -> None:
    """Removing the error-feedback edge would leave the run failed after the first call."""

    result = run_self_correcting_agent(
        "Compare NVIDIA's P/E with Schneider Electric.",
        registry=build_metric_registry(SNAPSHOT),
        policy=recorded_correction_policy,
        max_retries=1,
        max_tool_calls=4,
    )

    assert result.status == "completed"
    assert result.error_count == 1
    assert result.tool_calls == 3
    assert result.answer is not None
    assert "NVIDIA" in result.answer
    assert "52.4" in result.answer
    assert "Schneider Electric" in result.answer
    assert "31.8" in result.answer
    assert [
        event.request.metric
        for event in result.trace
        if event.phase in {"tool_error", "tool_ok"} and event.request
    ] == [
        "PE",
        "P/E",
        "P/E",
    ]
    assert [event.phase for event in result.trace] == [
        "agent",
        "tool_error",
        "agent",
        "tool_ok",
        "agent",
        "tool_ok",
        "agent",
        "finish",
    ]


def test_agent_stops_after_the_single_allowed_retry() -> None:
    """An off-by-one retry guard would permit a third identical failing tool call."""

    def always_wrong(_state):
        return AgentAction(
            action="tool",
            request=MetricRequest(ticker="NVDA", metric="PE"),
            reason="Repeat the unsupported alias.",
        )

    result = run_self_correcting_agent(
        "Return NVIDIA's P/E.",
        registry=build_metric_registry(SNAPSHOT),
        policy=always_wrong,
        max_retries=1,
        max_tool_calls=4,
    )

    assert result.status == "retry_budget_exhausted"
    assert result.answer is None
    assert result.error_count == 2
    assert result.tool_calls == 2
    assert result.trace[-1].phase == "guardrail"


def test_agent_stops_before_exceeding_the_tool_call_budget() -> None:
    """Removing the call-budget route would allow an unbounded sequence of valid calls."""

    def never_finish(_state):
        return AgentAction(
            action="tool",
            request=MetricRequest(ticker="NVDA", metric="P/E"),
            reason="Keep requesting the same valid metric.",
        )

    result = run_self_correcting_agent(
        "Keep checking NVIDIA.",
        registry=build_metric_registry(SNAPSHOT),
        policy=never_finish,
        max_retries=1,
        max_tool_calls=2,
    )

    assert result.status == "tool_budget_exhausted"
    assert result.tool_calls == 2
    assert result.trace[-1].phase == "guardrail"


def test_agent_rejects_a_final_answer_without_successful_financial_evidence() -> None:
    """Dropping the evidence guard would allow an unsupported numeric answer."""

    def invent_answer(_state):
        return AgentAction(
            action="finish",
            answer="NVIDIA P/E is 99.9.",
            reason="Finish without calling the metric tool.",
        )

    result = run_self_correcting_agent(
        "Return NVIDIA's P/E.",
        registry=build_metric_registry(SNAPSHOT),
        policy=invent_answer,
        max_retries=1,
        max_tool_calls=2,
    )

    assert result.status == "insufficient_evidence"
    assert result.answer is None
    assert result.trace[-1].phase == "guardrail"


def test_builder_returns_an_invokable_langgraph_graph() -> None:
    """Replacing the LangGraph compiler with a plain loop would break the lesson contract."""

    graph = build_self_correcting_graph(
        registry=build_metric_registry(SNAPSHOT),
        policy=recorded_correction_policy,
        max_retries=1,
        max_tool_calls=3,
    )

    result = graph.invoke({"question": "Return NVIDIA's P/E."})

    assert result["status"] == "completed"
    assert result["tool_calls"] == 2
