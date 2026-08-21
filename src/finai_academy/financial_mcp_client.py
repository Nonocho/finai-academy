"""Discover and use the Lesson 10 financial MCP server over local stdio."""

from __future__ import annotations

import json
import sys
from collections.abc import Awaitable, Mapping, Sequence
from time import perf_counter
from typing import Any, Literal, TypeVar

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from finai_academy.financial_mcp_capabilities import (
    CapabilityError,
    CoverageSnapshot,
    DocumentSearchResult,
    MetricResult,
)

ALLOWED_TOOLS = frozenset({"get_company_metric", "search_financial_documents"})

T = TypeVar("T")


class DiscoveredCapability(BaseModel):
    """A named primitive returned by MCP capability discovery."""

    primitive: Literal["resource", "tool", "prompt"]
    name: str


class McpOperationEvent(BaseModel):
    """One safe-to-display operation from the client lifecycle."""

    sequence: int
    primitive: Literal["resource", "tool", "prompt", "discovery"]
    operation: str
    capability: str
    status: Literal["ok", "error"]
    duration_ms: float
    evidence_count: int = 0
    error_code: str | None = None


class FinancialMcpRun(BaseModel):
    """The deterministic Lesson 10 discovery and read-only protocol run."""

    server_name: str
    capabilities: tuple[DiscoveredCapability, ...]
    resource_names: tuple[str, ...]
    tool_names: tuple[str, ...]
    prompt_names: tuple[str, ...]
    coverage: CoverageSnapshot
    metric: MetricResult
    search: DocumentSearchResult
    rendered_prompt: str
    failure: CapabilityError
    trace: tuple[McpOperationEvent, ...]


def financial_stdio_transport():
    """Return the local server transport without recording process configuration."""
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "finai_academy.financial_mcp_server"],
    )
    return stdio_client(parameters)


async def call_allowlisted_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Call a tool only after static and runtime discovery allowlist checks."""
    if name not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {name!r} is not allowlisted.")

    async with Client(financial_stdio_transport()) as client:
        discovered_names = {tool.name for tool in (await client.list_tools()).tools}
        if name not in discovered_names:
            raise ValueError(f"Tool {name!r} was not dynamically discovered.")
        return await client.call_tool(name, arguments)


async def discover_and_run_financial_mcp() -> FinancialMcpRun:
    """Discover every course primitive, then run its read-only learning flow."""
    trace: list[McpOperationEvent] = []

    async with Client(financial_stdio_transport()) as client:
        tools, duration_ms = await _measure(client.list_tools())
        tool_names = tuple(tool.name for tool in tools.tools)
        _append_event(
            trace,
            primitive="discovery",
            operation="list_tools",
            capability="tools",
            duration_ms=duration_ms,
        )

        resources, duration_ms = await _measure(client.list_resources())
        resource_names = tuple(str(resource.uri) for resource in resources.resources)
        _append_event(
            trace,
            primitive="discovery",
            operation="list_resources",
            capability="resources",
            duration_ms=duration_ms,
        )

        prompts, duration_ms = await _measure(client.list_prompts())
        prompt_names = tuple(prompt.name for prompt in prompts.prompts)
        _append_event(
            trace,
            primitive="discovery",
            operation="list_prompts",
            capability="prompts",
            duration_ms=duration_ms,
        )

        coverage_result, duration_ms = await _measure(client.read_resource("finance://coverage"))
        coverage = CoverageSnapshot.model_validate(json.loads(_resource_text(coverage_result.contents)))
        _append_event(
            trace,
            primitive="resource",
            operation="read_resource",
            capability="finance://coverage",
            duration_ms=duration_ms,
        )

        metric_result, duration_ms = await _measure(
            _call_discovered_allowlisted_tool(
                client, tool_names, "get_company_metric", {"ticker": "NVDA", "metric": "P/E"}
            )
        )
        metric = MetricResult.model_validate(_structured_content(metric_result))
        _append_event(
            trace,
            primitive="tool",
            operation="call_tool",
            capability="get_company_metric",
            duration_ms=duration_ms,
        )

        search_result, duration_ms = await _measure(
            _call_discovered_allowlisted_tool(
                client,
                tool_names,
                "search_financial_documents",
                {
                    "company": "Schneider Electric",
                    "query": "energy management",
                    "top_k": 2,
                },
            )
        )
        search = DocumentSearchResult.model_validate(_structured_content(search_result))
        _append_event(
            trace,
            primitive="tool",
            operation="call_tool",
            capability="search_financial_documents",
            duration_ms=duration_ms,
            evidence_count=len(search.hits),
        )

        prompt_result, duration_ms = await _measure(
            client.get_prompt(
                "compare_companies",
                {"metric": "P/E", "question": "Compare valuation and operating evidence."},
            )
        )
        rendered_prompt = _prompt_text(prompt_result.messages)
        _append_event(
            trace,
            primitive="prompt",
            operation="get_prompt",
            capability="compare_companies",
            duration_ms=duration_ms,
        )

        failure_result, duration_ms = await _measure(
            _call_discovered_allowlisted_tool(
                client, tool_names, "get_company_metric", {"ticker": "NVDA", "metric": "PE"}
            )
        )
        failure = _capability_error(failure_result)
        _append_event(
            trace,
            primitive="tool",
            operation="call_tool",
            capability="get_company_metric",
            status="error",
            duration_ms=duration_ms,
            error_code=failure.error_code,
        )

        capabilities = (
            *(DiscoveredCapability(primitive="resource", name=name) for name in resource_names),
            *(DiscoveredCapability(primitive="tool", name=name) for name in tool_names),
            *(DiscoveredCapability(primitive="prompt", name=name) for name in prompt_names),
        )
        return FinancialMcpRun(
            server_name=client.server_info.name,
            capabilities=capabilities,
            resource_names=resource_names,
            tool_names=tool_names,
            prompt_names=prompt_names,
            coverage=coverage,
            metric=metric,
            search=search,
            rendered_prompt=rendered_prompt,
            failure=failure,
            trace=tuple(trace),
        )


async def _call_discovered_allowlisted_tool(
    client: Client,
    discovered_names: Sequence[str],
    name: str,
    arguments: dict[str, Any],
) -> CallToolResult:
    if name not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {name!r} is not allowlisted.")
    if name not in discovered_names:
        raise ValueError(f"Tool {name!r} was not dynamically discovered.")
    return await client.call_tool(name, arguments)


async def _measure(operation: Awaitable[T]) -> tuple[T, float]:
    started_at = perf_counter()
    result = await operation
    return result, round((perf_counter() - started_at) * 1000, 3)


def _append_event(
    trace: list[McpOperationEvent],
    *,
    primitive: Literal["resource", "tool", "prompt", "discovery"],
    operation: str,
    capability: str,
    duration_ms: float,
    status: Literal["ok", "error"] = "ok",
    evidence_count: int = 0,
    error_code: str | None = None,
) -> None:
    """Keep presentation traces sequential and free of request/environment data."""
    trace.append(
        McpOperationEvent(
            sequence=len(trace) + 1,
            primitive=primitive,
            operation=operation,
            capability=capability,
            status=status,
            duration_ms=duration_ms,
            evidence_count=evidence_count,
            error_code=error_code,
        )
    )


def _resource_text(contents: Sequence[Any]) -> str:
    for content in contents:
        text = getattr(content, "text", None)
        if isinstance(text, str):
            return text
    raise ValueError("Coverage resource did not return text content.")


def _structured_content(result: CallToolResult) -> Mapping[str, Any]:
    if result.is_error:
        raise ValueError(_text_content(result.content))
    if not isinstance(result.structured_content, Mapping):
        raise TypeError("Successful tool result did not include structured content.")
    return result.structured_content


def _capability_error(result: CallToolResult) -> CapabilityError:
    if not result.is_error:
        raise ValueError("Expected a validation error from the MCP tool.")
    text = _text_content(result.content)
    json_start = text.find("{")
    if json_start < 0:
        raise ValueError("Tool error did not contain a structured capability error.")
    return CapabilityError.model_validate(json.loads(text[json_start:]))


def _text_content(blocks: Sequence[Any]) -> str:
    text = "\n".join(block.text for block in blocks if isinstance(block, TextContent))
    if not text:
        raise ValueError("MCP response did not contain text content.")
    return text


def _prompt_text(messages: Sequence[Any]) -> str:
    text = "\n".join(
        message.content.text for message in messages if isinstance(message.content, TextContent)
    )
    if not text:
        raise ValueError("Prompt did not contain text content.")
    return text
