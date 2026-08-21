from __future__ import annotations

import asyncio

import pytest

from finai_academy.financial_mcp_client import (
    call_allowlisted_tool,
    discover_and_run_financial_mcp,
)


def test_stdio_client_discovers_and_runs_course_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    """Breaks if the client stops using the discovered read-only MCP contract."""
    monkeypatch.setenv("LESSON10_TEST_SECRET", "must-not-appear-in-trace")

    run = asyncio.run(discover_and_run_financial_mcp())

    assert run.server_name == "First Finance Research"
    assert run.resource_names == ("finance://coverage",)
    assert run.tool_names == ("get_company_metric", "search_financial_documents")
    assert run.prompt_names == ("compare_companies",)
    assert run.coverage.dataset_id == "lesson10-financial-mcp-v1"
    assert run.metric.status == "ok"
    assert run.search.hits
    assert run.rendered_prompt
    assert run.failure.error_code == "unsupported_metric"
    assert [event.sequence for event in run.trace] == list(range(1, len(run.trace) + 1))
    assert all(event.duration_ms >= 0 for event in run.trace)
    assert "must-not-appear-in-trace" not in repr(run.trace)


def test_allowlisted_tool_rejects_names_outside_the_static_allowlist() -> None:
    """Breaks if a caller can invoke a mutating tool before MCP discovery."""
    with pytest.raises(ValueError, match="not allowlisted"):
        asyncio.run(call_allowlisted_tool("delete_portfolio", {}))
