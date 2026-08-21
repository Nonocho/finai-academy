from __future__ import annotations

import asyncio
import sys

import pytest
from mcp.types import TextContent

from finai_academy import financial_mcp_client
from finai_academy.financial_mcp_client import (
    McpOperationEvent,
    _record_async_operation,
    _resource_text,
    _validate_discovered_allowlisted_tool,
    call_allowlisted_tool,
    discover_and_run_financial_mcp,
    financial_stdio_transport,
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
    assert run.trace
    assert [event.operation for event in run.trace] == [
        "open_transport",
        "open_client_context",
        "discover_protocol_support",
        "discover_server_identity",
        "list_tools",
        "list_resources",
        "list_prompts",
        "read_resource",
        "call_tool",
        "call_tool",
        "get_prompt",
        "call_tool",
        "close_client_context",
        "close_transport",
    ]
    assert [event.capability for event in run.trace] == [
        "stdio",
        "client",
        "mcp:2026-07-28",
        "First Finance Research",
        "tools",
        "resources",
        "prompts",
        "finance://coverage",
        "get_company_metric",
        "search_financial_documents",
        "compare_companies",
        "get_company_metric",
        "client",
        "stdio",
    ]
    assert [event.status for event in run.trace] == [
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
        "error",
        "ok",
        "ok",
    ]
    assert [event.sequence for event in run.trace] == list(range(1, len(run.trace) + 1))
    assert all(event.duration_ms >= 0 for event in run.trace)
    assert run.trace[8].evidence_count == 1
    assert run.trace[9].evidence_count == len(run.search.hits)
    assert run.trace[11].error_code == "unsupported_metric"
    assert run.trace[-1].operation == "close_transport"
    assert "must-not-appear-in-trace" not in repr(run.trace)


def test_stdio_transport_uses_real_stderr_when_notebook_stderr_has_no_fileno(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Breaks if a Jupyter output wrapper is passed to the server subprocess."""

    class NotebookOutput:
        def fileno(self) -> int:
            raise OSError("Notebook output does not expose a process file descriptor.")

    captured: dict[str, object] = {}

    def capture_transport(*args, **kwargs):
        captured["parameters"] = args[0]
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(financial_mcp_client.sys, "stderr", NotebookOutput())
    monkeypatch.setattr(financial_mcp_client, "stdio_client", capture_transport)

    financial_stdio_transport()

    assert captured["errlog"] is sys.__stderr__
    assert callable(getattr(captured["errlog"], "fileno", None))


def test_allowlisted_tool_rejects_names_outside_the_static_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Breaks if a caller can invoke a mutating tool before MCP discovery."""
    monkeypatch.setattr(
        financial_mcp_client,
        "financial_stdio_transport",
        lambda: pytest.fail("static rejection must happen before transport launch"),
    )

    with pytest.raises(ValueError, match="not allowlisted"):
        asyncio.run(call_allowlisted_tool("delete_portfolio", {}))


def test_allowlisted_tool_rejects_static_name_missing_from_discovery() -> None:
    """Breaks if static allowlisting alone permits a tool call without discovery."""
    with pytest.raises(ValueError, match="not dynamically discovered"):
        _validate_discovered_allowlisted_tool(
            "get_company_metric", ("search_financial_documents",)
        )


def test_resource_parser_rejects_non_resource_text_blocks() -> None:
    """Breaks if a tool text block can be mistaken for the coverage resource payload."""
    with pytest.raises(TypeError, match="text resource content"):
        _resource_text([TextContent(type="text", text="{}")])


def test_measured_operation_records_a_safe_error_event_before_reraising() -> None:
    """Breaks if a failed measured operation leaves no visible safe trace event."""
    trace: list[McpOperationEvent] = []

    async def failing_operation() -> None:
        raise RuntimeError("synthetic secret must not enter the trace")

    with pytest.raises(RuntimeError, match="synthetic secret"):
        asyncio.run(
            _record_async_operation(
                trace,
                primitive="resource",
                operation="read_resource",
                capability="finance://coverage",
                attempt=failing_operation,
            )
        )

    assert len(trace) == 1
    assert trace[0].status == "error"
    assert trace[0].error_code == "runtime_error"
    assert "synthetic secret" not in repr(trace[0])
