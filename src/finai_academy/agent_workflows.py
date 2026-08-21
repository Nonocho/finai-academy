"""Typed tools and transparent orchestration records for the agent lessons."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator


class ToolRequest(BaseModel):
    """One validated request made to the course tool registry."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolObservation(BaseModel):
    """A serializable tool result that can be inspected by students and models."""

    tool_name: str
    status: Literal["ok", "error"]
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class MarketPrice(BaseModel):
    """Versioned market-price observation with its finance metadata."""

    ticker: str
    company: str
    price: float = Field(gt=0)
    currency: str
    as_of: str
    source: str


class CurrencyConversion(BaseModel):
    """Deterministic conversion calculated from a versioned FX observation."""

    input_amount: float = Field(gt=0)
    output_amount: float = Field(gt=0)
    rate: float = Field(gt=0)
    from_currency: str
    to_currency: str
    rate_as_of: str
    source: str


class WorkflowPlan(BaseModel):
    """One predetermined route selected before any tool observation exists."""

    route: Literal["tool", "unsupported_dependency", "finish"]
    request: ToolRequest | None = None
    answer: str | None = None
    reason: str

    @model_validator(mode="after")
    def validate_route_payload(self) -> WorkflowPlan:
        if self.route == "tool" and self.request is None:
            raise ValueError("tool route requires request")
        if self.route == "finish" and not self.answer:
            raise ValueError("finish route requires answer")
        return self


class AgentDecision(BaseModel):
    """One externally visible action selected by the bounded agent policy."""

    action: Literal["tool", "finish"]
    request: ToolRequest | None = None
    answer: str | None = None

    @model_validator(mode="after")
    def validate_action_payload(self) -> AgentDecision:
        if self.action == "tool" and self.request is None:
            raise ValueError("tool action requires request")
        if self.action == "finish" and not self.answer:
            raise ValueError("finish action requires answer")
        return self


class TraceStep(BaseModel):
    """One inspectable planning, execution, completion, or guardrail event."""

    index: int = Field(ge=1)
    phase: Literal["plan", "tool", "finish", "guardrail"]
    summary: str
    tool_name: str | None = None
    request: ToolRequest | None = None
    observation: ToolObservation | None = None


class OrchestrationResult(BaseModel):
    """Normalized result used to compare workflow and agent execution."""

    architecture: Literal["workflow", "agent"]
    status: Literal[
        "completed", "unsupported_dependency", "step_budget_exhausted", "error"
    ]
    answer: str | None
    trajectory: tuple[TraceStep, ...]
    latency_ms: float = Field(ge=0)


ToolHandler = Callable[..., BaseModel]


class ToolRegistry:
    """Execute narrow typed tools and turn every failure into an observation."""

    def __init__(self, handlers: Mapping[str, ToolHandler]) -> None:
        if not handlers:
            raise ValueError("ToolRegistry requires at least one tool")
        self._handlers = dict(handlers)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    def invoke(self, request: ToolRequest) -> ToolObservation:
        handler = self._handlers.get(request.name)
        if handler is None:
            choices = ", ".join(self.names)
            return ToolObservation(
                tool_name=request.name,
                status="error",
                error=f"Unknown tool {request.name!r}. Valid tools: {choices}.",
            )

        try:
            result = handler(**request.arguments)
        except (TypeError, ValueError, ValidationError) as error:
            return ToolObservation(
                tool_name=request.name,
                status="error",
                error=str(error),
            )
        return ToolObservation(
            tool_name=request.name,
            status="ok",
            payload=result.model_dump(mode="json"),
        )


def load_course_market_snapshot(path: Path) -> dict[str, Any]:
    """Load one checked-in course snapshot without performing network access."""

    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise TypeError("market snapshot must be a JSON object")
    if not snapshot.get("dataset_id"):
        raise ValueError("market snapshot requires dataset_id")
    return snapshot


def build_course_tool_registry(snapshot: Mapping[str, Any]) -> ToolRegistry:
    """Build the two Lesson 08 tools from a maintained snapshot."""

    prices = snapshot.get("prices")
    fx_rates = snapshot.get("fx")
    if not isinstance(prices, Mapping) or not prices:
        raise ValueError("market snapshot requires non-empty prices")
    if not isinstance(fx_rates, Mapping) or not fx_rates:
        raise ValueError("market snapshot requires non-empty fx rates")

    def get_market_price(ticker: str) -> MarketPrice:
        normalized_ticker = ticker.upper().strip()
        record = prices.get(normalized_ticker)
        if record is None:
            choices = ", ".join(sorted(str(value) for value in prices))
            raise ValueError(
                f"Unsupported ticker {normalized_ticker!r}. Valid tickers: {choices}."
            )
        if not isinstance(record, Mapping):
            raise TypeError(f"Price record for {normalized_ticker!r} must be an object")
        return MarketPrice(ticker=normalized_ticker, **dict(record))

    def convert_currency(
        amount: float,
        from_currency: str,
        to_currency: str,
    ) -> CurrencyConversion:
        numeric_amount = float(amount)
        if not math.isfinite(numeric_amount) or numeric_amount <= 0:
            raise ValueError("amount must be a positive finite number")
        source_currency = from_currency.upper().strip()
        target_currency = to_currency.upper().strip()
        pair = f"{source_currency}_{target_currency}"
        record = fx_rates.get(pair)
        if record is None:
            choices = ", ".join(sorted(str(value) for value in fx_rates))
            raise ValueError(f"Unsupported currency pair {pair!r}. Valid pairs: {choices}.")
        if not isinstance(record, Mapping):
            raise TypeError(f"FX record for {pair!r} must be an object")
        rate = float(record["rate"])
        return CurrencyConversion(
            input_amount=numeric_amount,
            output_amount=round(numeric_amount * rate, 4),
            rate=rate,
            from_currency=source_currency,
            to_currency=target_currency,
            rate_as_of=str(record["as_of"]),
            source=str(record["source"]),
        )

    return ToolRegistry(
        {
            "convert_currency": convert_currency,
            "get_market_price": get_market_price,
        }
    )


WorkflowPlanner = Callable[[str], WorkflowPlan]
AnswerWriter = Callable[[str, tuple[ToolObservation, ...]], str]
AgentPolicy = Callable[[str, tuple[TraceStep, ...]], AgentDecision]


def run_one_pass_workflow(
    question: str,
    *,
    planner: WorkflowPlanner,
    answer_writer: AnswerWriter,
    registry: ToolRegistry,
) -> OrchestrationResult:
    """Execute a route fixed before observations can influence another tool choice."""

    started = perf_counter()
    plan = planner(question)
    trajectory = [
        TraceStep(index=1, phase="plan", summary=plan.reason, request=plan.request)
    ]
    if plan.route == "unsupported_dependency":
        return OrchestrationResult(
            architecture="workflow",
            status="unsupported_dependency",
            answer=None,
            trajectory=tuple(trajectory),
            latency_ms=(perf_counter() - started) * 1_000,
        )
    if plan.route == "finish":
        trajectory.append(
            TraceStep(index=2, phase="finish", summary="Workflow returned a direct answer.")
        )
        return OrchestrationResult(
            architecture="workflow",
            status="completed",
            answer=plan.answer,
            trajectory=tuple(trajectory),
            latency_ms=(perf_counter() - started) * 1_000,
        )

    if plan.request is None:  # protected by validation; retained for static narrowing
        raise RuntimeError("tool route is missing its validated request")
    observation = registry.invoke(plan.request)
    trajectory.append(
        TraceStep(
            index=2,
            phase="tool",
            summary=f"{plan.request.name} returned {observation.status}.",
            tool_name=plan.request.name,
            request=plan.request,
            observation=observation,
        )
    )
    if observation.status == "error":
        return OrchestrationResult(
            architecture="workflow",
            status="error",
            answer=None,
            trajectory=tuple(trajectory),
            latency_ms=(perf_counter() - started) * 1_000,
        )

    answer = answer_writer(question, (observation,))
    trajectory.append(
        TraceStep(index=3, phase="finish", summary="Answer written from one observation.")
    )
    return OrchestrationResult(
        architecture="workflow",
        status="completed",
        answer=answer,
        trajectory=tuple(trajectory),
        latency_ms=(perf_counter() - started) * 1_000,
    )


def _question_requires_currency_conversion(question: str) -> bool:
    normalized = question.casefold()
    return "converted" in normalized or "convert" in normalized or "euro" in normalized


def _has_grounded_conversion(trajectory: list[TraceStep]) -> bool:
    successful_tools = [
        step.tool_name
        for step in trajectory
        if step.phase == "tool"
        and step.observation is not None
        and step.observation.status == "ok"
    ]
    return "get_market_price" in successful_tools and "convert_currency" in successful_tools


def run_bounded_agent(
    question: str,
    *,
    policy: AgentPolicy,
    registry: ToolRegistry,
    max_steps: int = 4,
) -> OrchestrationResult:
    """Run a transparent reason-act-observe-stop loop with a hard step budget."""

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    started = perf_counter()
    trajectory: list[TraceStep] = []

    for _agent_step in range(1, max_steps + 1):
        decision = policy(question, tuple(trajectory))
        trajectory.append(
            TraceStep(
                index=len(trajectory) + 1,
                phase="plan",
                summary=f"Agent selected {decision.action}.",
                request=decision.request,
            )
        )
        if decision.action == "finish":
            if _question_requires_currency_conversion(question) and not _has_grounded_conversion(
                trajectory
            ):
                trajectory.append(
                    TraceStep(
                        index=len(trajectory) + 1,
                        phase="guardrail",
                        summary=(
                            "Converted answer rejected: successful price and conversion "
                            "observations are required."
                        ),
                    )
                )
                return OrchestrationResult(
                    architecture="agent",
                    status="error",
                    answer=None,
                    trajectory=tuple(trajectory),
                    latency_ms=(perf_counter() - started) * 1_000,
                )
            trajectory.append(
                TraceStep(
                    index=len(trajectory) + 1,
                    phase="finish",
                    summary="Agent returned a final answer.",
                )
            )
            return OrchestrationResult(
                architecture="agent",
                status="completed",
                answer=decision.answer,
                trajectory=tuple(trajectory),
                latency_ms=(perf_counter() - started) * 1_000,
            )

        if decision.request is None:  # protected by validation; retained for static narrowing
            raise RuntimeError("tool action is missing its validated request")
        observation = registry.invoke(decision.request)
        trajectory.append(
            TraceStep(
                index=len(trajectory) + 1,
                phase="tool",
                summary=f"{decision.request.name} returned {observation.status}.",
                tool_name=decision.request.name,
                request=decision.request,
                observation=observation,
            )
        )

    trajectory.append(
        TraceStep(
            index=len(trajectory) + 1,
            phase="guardrail",
            summary=f"Stopped after MAX_STEPS={max_steps}.",
        )
    )
    return OrchestrationResult(
        architecture="agent",
        status="step_budget_exhausted",
        answer=None,
        trajectory=tuple(trajectory),
        latency_ms=(perf_counter() - started) * 1_000,
    )
