"""Typed state contracts for the Lesson 09 self-correcting financial agent."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator


class MetricRequest(BaseModel):
    ticker: str
    metric: str


class MetricObservation(BaseModel):
    status: Literal["ok", "error"]
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    retryable: bool = False
    allowed_metrics: tuple[str, ...] = ()


class AgentAction(BaseModel):
    action: Literal["tool", "finish"]
    reason: str
    request: MetricRequest | None = None
    answer: str | None = None

    @model_validator(mode="after")
    def validate_action(self) -> AgentAction:
        if self.action == "tool" and self.request is None:
            raise ValueError("tool action requires a request")
        if self.action == "finish" and not self.answer:
            raise ValueError("finish action requires an answer")
        return self


class RecoveryEvent(BaseModel):
    index: int = Field(ge=1)
    phase: Literal["agent", "tool_error", "tool_ok", "finish", "guardrail"]
    summary: str
    request: MetricRequest | None = None
    observation: MetricObservation | None = None


class RecoveryResult(BaseModel):
    status: Literal[
        "completed",
        "retry_budget_exhausted",
        "tool_budget_exhausted",
        "insufficient_evidence",
    ]
    answer: str | None
    error_count: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    trace: tuple[RecoveryEvent, ...]


class RecoveryState(TypedDict, total=False):
    """Explicit LangGraph state shared by the agent and tool nodes."""

    question: str
    decision: AgentAction
    observation: MetricObservation
    error_count: int
    tool_calls: int
    trace: tuple[RecoveryEvent, ...]
    status: str
    answer: str | None


AgentPolicy = Callable[[Mapping[str, Any]], AgentAction]


class MetricRegistry:
    """Return typed metric observations without leaking tool exceptions into the graph."""

    def __init__(self, snapshot: Mapping[str, Any]) -> None:
        metrics = snapshot.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            raise ValueError("metric snapshot requires non-empty metrics")
        if not snapshot.get("dataset_id"):
            raise ValueError("metric snapshot requires dataset_id")
        self._metrics = dict(metrics)
        self._as_of = str(snapshot.get("as_of", "unknown"))
        self._source = str(snapshot.get("source", "controlled course fixture"))

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(sorted(str(ticker) for ticker in self._metrics))

    @property
    def metric_names(self) -> tuple[str, ...]:
        names = {
            str(metric)
            for company_metrics in self._metrics.values()
            if isinstance(company_metrics, Mapping)
            for metric in company_metrics
            if metric != "company"
        }
        return tuple(sorted(names))

    def invoke(self, request: MetricRequest) -> MetricObservation:
        ticker = request.ticker.upper().strip()
        metric = request.metric.strip()
        record = self._metrics.get(ticker)
        if not isinstance(record, Mapping):
            choices = ", ".join(self.tickers)
            return MetricObservation(
                status="error",
                error_code="unsupported_ticker",
                message=f"Unknown ticker {ticker!r}. Valid tickers: {choices}.",
                retryable=True,
                allowed_metrics=self.metric_names,
            )
        if metric not in self.metric_names or metric not in record:
            choices = ", ".join(self.metric_names)
            return MetricObservation(
                status="error",
                error_code="unsupported_metric",
                message=f"Unknown metric {metric!r}. Valid metrics: {choices}.",
                retryable=True,
                allowed_metrics=self.metric_names,
            )
        company = str(record.get("company", ticker))
        payload = {
            "ticker": ticker,
            "company": company,
            "metric": metric,
            "value": record[metric],
            "as_of": self._as_of,
            "source": self._source,
        }
        return MetricObservation(
            status="ok",
            message=f"{company} {metric}: {record[metric]}",
            payload=payload,
        )


def build_metric_registry(snapshot: Mapping[str, Any]) -> MetricRegistry:
    """Build the Lesson 09 metric tool from one controlled finance snapshot."""

    return MetricRegistry(snapshot)


def _successful_observations(state: Mapping[str, Any]) -> list[MetricObservation]:
    return [
        event.observation
        for event in state.get("trace", ())
        if isinstance(event, RecoveryEvent)
        and event.phase == "tool_ok"
        and event.observation is not None
    ]


def recorded_correction_policy(state: Mapping[str, Any]) -> AgentAction:
    """Deterministic classroom policy that exposes one error and one correction."""

    trace = tuple(state.get("trace", ()))
    tool_events = [
        event
        for event in trace
        if isinstance(event, RecoveryEvent) and event.phase in {"tool_error", "tool_ok"}
    ]
    if not tool_events:
        return AgentAction(
            action="tool",
            request=MetricRequest(ticker="NVDA", metric="PE"),
            reason="Request the common PE alias first to expose validation feedback.",
        )

    last_observation = tool_events[-1].observation
    if last_observation is not None and last_observation.status == "error":
        corrected_metric = "P/E" if "P/E" in last_observation.allowed_metrics else "EPS"
        return AgentAction(
            action="tool",
            request=MetricRequest(ticker="NVDA", metric=corrected_metric),
            reason="Use the valid metric name returned by the tool error.",
        )

    observations = _successful_observations(state)
    observed_tickers = {item.payload.get("ticker") for item in observations}
    question = str(state.get("question", "")).casefold()
    compare_schneider = "schneider" in question or "compare" in question
    if compare_schneider and "SU.PA" not in observed_tickers:
        return AgentAction(
            action="tool",
            request=MetricRequest(ticker="SU.PA", metric="P/E"),
            reason="Use the same validated metric for Schneider Electric.",
        )

    values = {str(item.payload["ticker"]): item.payload for item in observations}
    nvda = values["NVDA"]
    if "SU.PA" in values:
        schneider = values["SU.PA"]
        answer = (
            f"NVIDIA P/E is {nvda['value']} and Schneider Electric P/E is "
            f"{schneider['value']} as of {nvda['as_of']}. "
            f"Source: {nvda['source']}."
        )
    else:
        answer = (
            f"NVIDIA P/E is {nvda['value']} as of {nvda['as_of']}. "
            f"Source: {nvda['source']}."
        )
    return AgentAction(
        action="finish",
        answer=answer,
        reason="Finish from successful typed observations only.",
    )


def _append_event(
    state: Mapping[str, Any],
    *,
    phase: Literal["agent", "tool_error", "tool_ok", "finish", "guardrail"],
    summary: str,
    request: MetricRequest | None = None,
    observation: MetricObservation | None = None,
) -> tuple[RecoveryEvent, ...]:
    trace = tuple(state.get("trace", ()))
    return trace + (
        RecoveryEvent(
            index=len(trace) + 1,
            phase=phase,
            summary=summary,
            request=request,
            observation=observation,
        ),
    )


def build_self_correcting_graph(
    *,
    registry: MetricRegistry,
    policy: AgentPolicy,
    max_retries: int,
    max_tool_calls: int,
):
    """Compile the bounded recovery loop as an explicit LangGraph state machine."""

    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater")
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be at least 1")

    def agent_node(state: RecoveryState) -> dict[str, Any]:
        prepared_state = {
            **state,
            "error_count": int(state.get("error_count", 0)),
            "tool_calls": int(state.get("tool_calls", 0)),
            "trace": tuple(state.get("trace", ())),
        }
        decision = policy(prepared_state)
        return {
            "decision": decision,
            "error_count": prepared_state["error_count"],
            "tool_calls": prepared_state["tool_calls"],
            "trace": _append_event(
                prepared_state,
                phase="agent",
                summary=decision.reason,
                request=decision.request,
            ),
        }

    def route_after_agent(state: RecoveryState) -> str:
        decision = state["decision"]
        if decision.action == "finish":
            return "finish" if _successful_observations(state) else "insufficient_evidence"
        if int(state.get("error_count", 0)) > max_retries:
            return "retry_guard"
        if int(state.get("tool_calls", 0)) >= max_tool_calls:
            return "tool_guard"
        return "tools"

    def tool_node(state: RecoveryState) -> dict[str, Any]:
        request = state["decision"].request
        if request is None:
            raise RuntimeError("tool route is missing a validated request")
        observation = registry.invoke(request)
        phase: Literal["tool_error", "tool_ok"] = (
            "tool_error" if observation.status == "error" else "tool_ok"
        )
        return {
            "observation": observation,
            "error_count": int(state.get("error_count", 0))
            + (1 if observation.status == "error" else 0),
            "tool_calls": int(state.get("tool_calls", 0)) + 1,
            "trace": _append_event(
                state,
                phase=phase,
                summary=observation.message,
                request=request,
                observation=observation,
            ),
        }

    def finish_node(state: RecoveryState) -> dict[str, Any]:
        answer = state["decision"].answer
        return {
            "status": "completed",
            "answer": answer,
            "trace": _append_event(
                state,
                phase="finish",
                summary="Answer accepted from successful tool evidence.",
            ),
        }

    def retry_guard_node(state: RecoveryState) -> dict[str, Any]:
        return {
            "status": "retry_budget_exhausted",
            "answer": None,
            "trace": _append_event(
                state,
                phase="guardrail",
                summary=f"Stopped after {max_retries} allowed retries.",
            ),
        }

    def tool_guard_node(state: RecoveryState) -> dict[str, Any]:
        return {
            "status": "tool_budget_exhausted",
            "answer": None,
            "trace": _append_event(
                state,
                phase="guardrail",
                summary=f"Stopped at MAX_TOOL_CALLS={max_tool_calls}.",
            ),
        }

    def insufficient_evidence_node(state: RecoveryState) -> dict[str, Any]:
        return {
            "status": "insufficient_evidence",
            "answer": None,
            "trace": _append_event(
                state,
                phase="guardrail",
                summary="Final answer rejected because no successful metric observation exists.",
            ),
        }

    workflow = StateGraph(RecoveryState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("finish", finish_node)
    workflow.add_node("retry_guard", retry_guard_node)
    workflow.add_node("tool_guard", tool_guard_node)
    workflow.add_node("insufficient_evidence", insufficient_evidence_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        route_after_agent,
        {
            "tools": "tools",
            "finish": "finish",
            "retry_guard": "retry_guard",
            "tool_guard": "tool_guard",
            "insufficient_evidence": "insufficient_evidence",
        },
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("finish", END)
    workflow.add_edge("retry_guard", END)
    workflow.add_edge("tool_guard", END)
    workflow.add_edge("insufficient_evidence", END)
    return workflow.compile()


def run_self_correcting_agent(
    question: str,
    *,
    registry,
    policy: AgentPolicy,
    max_retries: int = 1,
    max_tool_calls: int = 3,
) -> RecoveryResult:
    graph = build_self_correcting_graph(
        registry=registry,
        policy=policy,
        max_retries=max_retries,
        max_tool_calls=max_tool_calls,
    )
    state = graph.invoke({"question": question})
    return RecoveryResult(
        status=state["status"],
        answer=state.get("answer"),
        error_count=int(state.get("error_count", 0)),
        tool_calls=int(state.get("tool_calls", 0)),
        trace=tuple(state.get("trace", ())),
    )
