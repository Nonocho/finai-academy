from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Self

import pytest

from finai_academy import planning_mcp_executor
from finai_academy.financial_mcp_client import financial_stdio_transport
from finai_academy.planning_mcp_executor import FinancialMcpPlanningExecutor
from finai_academy.research_planning import PlanStep, ResearchObservation


def revenue_metric_step() -> PlanStep:
    return PlanStep(
        step_id=1,
        capability="get_company_metric",
        arguments={"ticker": "NVDA", "metric": "Revenue"},
        purpose="Attempt to collect NVIDIA revenue as a structured metric.",
        expected_evidence=("NVIDIA revenue",),
    )


def metric_step(step_id: int) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        capability="get_company_metric",
        arguments={"ticker": "NVDA", "metric": "P/E"},
        purpose="Collect NVIDIA valuation evidence.",
        expected_evidence=("NVDA P/E",),
    )


def test_executor_discovers_only_permitted_read_tools() -> None:
    """Breaks if discovery exposes a prompt or a non-allowlisted tool to the planner."""

    async def scenario() -> None:
        async with FinancialMcpPlanningExecutor() as executor:
            assert tuple(item.name for item in executor.catalog) == (
                "get_company_metric",
                "search_financial_documents",
            )
            assert executor.server_name == "First Finance Research"

    asyncio.run(scenario())


def test_executor_preserves_metric_and_document_provenance() -> None:
    """Breaks if successful protocol results lose source or evidence identifiers."""

    async def scenario() -> tuple[ResearchObservation, ResearchObservation]:
        async with FinancialMcpPlanningExecutor() as executor:
            metric = await executor.execute(metric_step(1), attempt_id=1, plan_revision=0)
            document = await executor.execute(
                PlanStep(
                    step_id=2,
                    capability="search_financial_documents",
                    arguments={"company": "NVIDIA", "query": "revenue growth", "top_k": 2},
                    purpose="Collect operating evidence.",
                    expected_evidence=("NVIDIA revenue",),
                ),
                attempt_id=2,
                plan_revision=0,
            )
            return metric, document

    metric, document = asyncio.run(scenario())

    assert metric.status == "ok"
    assert metric.source_references
    assert document.status == "ok"
    assert document.evidence_ids
    assert document.source_references


def test_executor_converts_unsupported_metric_to_observation() -> None:
    """Breaks if typed server validation errors crash the plan-execute graph."""

    async def scenario() -> ResearchObservation:
        async with FinancialMcpPlanningExecutor() as executor:
            return await executor.execute(revenue_metric_step(), attempt_id=1, plan_revision=0)

    observation = asyncio.run(scenario())

    assert observation.status == "error"
    assert observation.error_code == "unsupported_metric"
    assert observation.result is not None
    assert "P/E" in observation.result["valid_values"]


def test_executor_owns_one_transport_lifecycle_for_sequential_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Breaks if each execution launches and tears down its own stdio server."""
    lifecycle: list[str] = []

    @asynccontextmanager
    async def counted_transport():
        lifecycle.append("open")
        async with financial_stdio_transport() as streams:
            try:
                yield streams
            finally:
                lifecycle.append("close")

    monkeypatch.setattr(planning_mcp_executor, "financial_stdio_transport", counted_transport)

    async def scenario() -> None:
        async with FinancialMcpPlanningExecutor() as executor:
            for attempt_id in range(1, 5):
                observation = await executor.execute(
                    metric_step(attempt_id), attempt_id=attempt_id, plan_revision=0
                )
                assert observation.status == "ok"

    asyncio.run(scenario())

    assert lifecycle == ["open", "close"]


def test_executor_requires_open_context_before_execution() -> None:
    """Breaks if a tool can run before the executor has completed discovery."""

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="opened before execution"):
            await FinancialMcpPlanningExecutor().execute(metric_step(1), attempt_id=1, plan_revision=0)

    asyncio.run(scenario())


def test_executor_fails_closed_when_discovery_has_no_permitted_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Breaks if an empty or unrelated discovery response opens an executable session."""
    closed: list[bool] = []

    class EmptyDiscoveryClient:
        server_info = SimpleNamespace(name="Unrelated server")

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            closed.append(True)

        async def list_tools(self) -> SimpleNamespace:
            return SimpleNamespace(tools=())

    monkeypatch.setattr(planning_mcp_executor, "financial_stdio_transport", object)
    monkeypatch.setattr(planning_mcp_executor, "Client", lambda transport: EmptyDiscoveryClient())

    async def scenario() -> None:
        with pytest.raises(RuntimeError, match="no permitted tools"):
            async with FinancialMcpPlanningExecutor():
                pytest.fail("empty discovery must not enter the executor context")

    asyncio.run(scenario())

    assert closed == [True]


def test_executor_blocks_capabilities_outside_static_allowlist() -> None:
    """Breaks if a planned mutating capability reaches the MCP client."""

    async def scenario() -> ResearchObservation:
        async with FinancialMcpPlanningExecutor() as executor:
            return await executor.execute(
                PlanStep(
                    step_id=1,
                    capability="delete_portfolio",
                    arguments={},
                    purpose="Attempt a prohibited mutation.",
                    expected_evidence=("none",),
                ),
                attempt_id=1,
                plan_revision=0,
            )

    observation = asyncio.run(scenario())

    assert observation.status == "blocked"
    assert observation.error_code == "capability_not_permitted"
    assert observation.result is None
