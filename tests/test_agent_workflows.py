from __future__ import annotations

import math

from finai_academy.agent_workflows import (
    AgentDecision,
    OrchestrationResult,
    ToolRequest,
    WorkflowPlan,
    build_course_tool_registry,
    run_bounded_agent,
    run_one_pass_workflow,
)

SNAPSHOT = {
    "dataset_id": "lesson08-market-snapshot-v1",
    "notice": "Checked-in course snapshot; not a live quote or investment recommendation.",
    "prices": {
        "NVDA": {
            "company": "NVIDIA",
            "price": 180.0,
            "currency": "USD",
            "as_of": "2026-08-20",
            "source": "https://finance.yahoo.com/quote/NVDA/history/",
        },
        "SU.PA": {
            "company": "Schneider Electric",
            "price": 240.0,
            "currency": "EUR",
            "as_of": "2026-08-20",
            "source": "https://finance.yahoo.com/quote/SU.PA/history/",
        },
    },
    "fx": {
        "USD_EUR": {
            "rate": 0.86,
            "as_of": "2026-08-20",
            "source": "https://finance.yahoo.com/quote/EURUSD%3DX/history/",
        }
    },
}


def test_market_price_retains_provenance() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    result = registry.invoke(
        ToolRequest(name="get_market_price", arguments={"ticker": "NVDA"})
    )

    assert result.status == "ok"
    assert result.payload["ticker"] == "NVDA"
    assert result.payload["currency"] == "USD"
    assert result.payload["as_of"] == "2026-08-20"
    assert result.payload["source"].startswith("https://")


def test_registry_returns_actionable_unknown_tool_error() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    result = registry.invoke(ToolRequest(name="get_price", arguments={"ticker": "NVDA"}))

    assert result.status == "error"
    assert result.error is not None
    assert "get_market_price" in result.error
    assert "convert_currency" in result.error


def test_currency_conversion_rejects_non_positive_amount() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    result = registry.invoke(
        ToolRequest(
            name="convert_currency",
            arguments={"amount": 0, "from_currency": "USD", "to_currency": "EUR"},
        )
    )

    assert result.status == "error"
    assert result.error is not None
    assert "positive" in result.error.casefold()


def test_currency_conversion_uses_the_versioned_rate() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    result = registry.invoke(
        ToolRequest(
            name="convert_currency",
            arguments={"amount": 180.0, "from_currency": "USD", "to_currency": "EUR"},
        )
    )

    assert result.status == "ok"
    assert math.isclose(result.payload["output_amount"], 154.8)
    assert result.payload["rate"] == 0.86
    assert result.payload["rate_as_of"] == "2026-08-20"


def test_registry_rejects_unsupported_ticker_with_valid_choices() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    result = registry.invoke(
        ToolRequest(name="get_market_price", arguments={"ticker": "AAPL"})
    )

    assert result.status == "error"
    assert result.error is not None
    assert "NVDA" in result.error
    assert "SU.PA" in result.error


def test_one_pass_workflow_handles_a_direct_price_request() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    result = run_one_pass_workflow(
        "What is NVIDIA's share price?",
        planner=lambda _question: WorkflowPlan(
            route="tool",
            request=ToolRequest(name="get_market_price", arguments={"ticker": "NVDA"}),
            reason="One direct lookup is sufficient.",
        ),
        answer_writer=lambda _question, observations: (
            f"NVIDIA: {observations[0].payload['price']} "
            f"{observations[0].payload['currency']}."
        ),
        registry=registry,
    )

    assert isinstance(result, OrchestrationResult)
    assert result.status == "completed"
    assert result.answer == "NVIDIA: 180.0 USD."
    assert [step.phase for step in result.trajectory] == ["plan", "tool", "finish"]


def test_one_pass_workflow_exposes_an_unsupported_dependency_without_fabrication() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    result = run_one_pass_workflow(
        "What is NVIDIA's share price converted to euros?",
        planner=lambda _question: WorkflowPlan(
            route="unsupported_dependency",
            reason="The conversion amount depends on the unseen price observation.",
        ),
        answer_writer=lambda _question, _observations: "This must not be called.",
        registry=registry,
    )

    assert result.status == "unsupported_dependency"
    assert result.answer is None
    assert all(step.phase != "tool" for step in result.trajectory)
    assert "unseen price" in result.trajectory[0].summary


def test_bounded_agent_calls_price_before_currency_conversion() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    def policy(_question, trajectory):
        tool_steps = [step for step in trajectory if step.phase == "tool"]
        if not tool_steps:
            return AgentDecision(
                action="tool",
                request=ToolRequest(
                    name="get_market_price",
                    arguments={"ticker": "NVDA"},
                ),
            )
        if len(tool_steps) == 1:
            observation = tool_steps[0].observation
            assert observation is not None
            price = observation.payload["price"]
            return AgentDecision(
                action="tool",
                request=ToolRequest(
                    name="convert_currency",
                    arguments={
                        "amount": price,
                        "from_currency": "USD",
                        "to_currency": "EUR",
                    },
                ),
            )
        observation = tool_steps[-1].observation
        assert observation is not None
        converted = observation.payload["output_amount"]
        return AgentDecision(action="finish", answer=f"NVIDIA: EUR {converted:.2f}.")

    result = run_bounded_agent(
        "What is NVIDIA's share price converted to euros?",
        policy=policy,
        registry=registry,
        max_steps=4,
    )

    tool_steps = [step.tool_name for step in result.trajectory if step.phase == "tool"]
    assert tool_steps == ["get_market_price", "convert_currency"]
    assert result.status == "completed"
    assert result.answer == "NVIDIA: EUR 154.80."


def test_bounded_agent_stops_at_max_steps() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    result = run_bounded_agent(
        "Keep looking up NVIDIA forever.",
        policy=lambda _question, _trajectory: AgentDecision(
            action="tool",
            request=ToolRequest(name="get_market_price", arguments={"ticker": "NVDA"}),
        ),
        registry=registry,
        max_steps=2,
    )

    assert result.status == "step_budget_exhausted"
    assert result.answer is None
    assert result.trajectory[-1].phase == "guardrail"
    assert "MAX_STEPS=2" in result.trajectory[-1].summary


def test_bounded_agent_rejects_an_ungrounded_converted_answer() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    result = run_bounded_agent(
        "What is NVIDIA's share price converted to euros?",
        policy=lambda _question, _trajectory: AgentDecision(
            action="finish",
            answer="NVIDIA is worth EUR 150.00.",
        ),
        registry=registry,
        max_steps=4,
    )

    assert result.status == "error"
    assert result.answer is None
    assert result.trajectory[-1].phase == "guardrail"
    assert "price and conversion observations" in result.trajectory[-1].summary


def test_tool_error_is_retained_as_an_agent_observation() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    def policy(_question, trajectory):
        if not any(step.phase == "tool" for step in trajectory):
            return AgentDecision(
                action="tool",
                request=ToolRequest(name="get_market_price", arguments={"ticker": "AAPL"}),
            )
        return AgentDecision(action="finish", answer="The requested ticker is unsupported.")

    result = run_bounded_agent(
        "Look up AAPL.",
        policy=policy,
        registry=registry,
        max_steps=3,
    )

    tool_step = next(step for step in result.trajectory if step.phase == "tool")
    assert tool_step.observation is not None
    assert tool_step.observation.status == "error"
    assert tool_step.observation.error is not None
    assert "NVDA" in tool_step.observation.error
