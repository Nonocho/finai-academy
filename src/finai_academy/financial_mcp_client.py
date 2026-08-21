"""Discover and use the Lesson 10 financial MCP server over local stdio."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any, Literal, TypeVar

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent, TextResourceContents
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
    primitive: Literal["resource", "tool", "prompt", "discovery", "lifecycle"]
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
    # Jupyter's active stderr wrapper has no file descriptor for subprocess pipes.
    return stdio_client(parameters, errlog=sys.__stderr__)


async def call_allowlisted_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Call a tool only after static and runtime discovery allowlist checks."""
    _validate_static_tool(name)
    async with Client(financial_stdio_transport()) as client:
        discovered_names = tuple(tool.name for tool in (await client.list_tools()).tools)
        _validate_discovered_allowlisted_tool(name, discovered_names)
        return await client.call_tool(name, arguments)


async def discover_and_run_financial_mcp() -> FinancialMcpRun:
    """Discover every course primitive, then run its read-only learning flow."""
    trace: list[McpOperationEvent] = []

    async with _traced_client_context(trace) as client:
        _record_sync_operation(
            trace,
            primitive="discovery",
            operation="discover_protocol_support",
            capability="protocol",
            capability_for=lambda version: f"mcp:{version}",
            attempt=lambda: client.protocol_version,
        )
        server_name = _record_sync_operation(
            trace,
            primitive="discovery",
            operation="discover_server_identity",
            capability="server",
            capability_for=lambda name: name,
            attempt=lambda: client.server_info.name,
        )
        tool_names = await _record_async_operation(
            trace,
            primitive="discovery",
            operation="list_tools",
            capability="tools",
            attempt=lambda: _discover_tool_names(client),
        )
        resource_names = await _record_async_operation(
            trace,
            primitive="discovery",
            operation="list_resources",
            capability="resources",
            attempt=lambda: _discover_resource_names(client),
        )
        prompt_names = await _record_async_operation(
            trace,
            primitive="discovery",
            operation="list_prompts",
            capability="prompts",
            attempt=lambda: _discover_prompt_names(client),
        )

        coverage = await _record_async_operation(
            trace,
            primitive="resource",
            operation="read_resource",
            capability="finance://coverage",
            attempt=lambda: _read_coverage(client),
        )
        metric = await _record_async_operation(
            trace,
            primitive="tool",
            operation="call_tool",
            capability="get_company_metric",
            attempt=lambda: _get_metric(client, tool_names),
            evidence_count_for=lambda _: 1,
        )
        search = await _record_async_operation(
            trace,
            primitive="tool",
            operation="call_tool",
            capability="search_financial_documents",
            attempt=lambda: _search_documents(client, tool_names),
            evidence_count_for=lambda result: len(result.hits),
        )
        rendered_prompt = await _record_async_operation(
            trace,
            primitive="prompt",
            operation="get_prompt",
            capability="compare_companies",
            attempt=lambda: _render_prompt(client),
        )
        failure = await _record_async_operation(
            trace,
            primitive="tool",
            operation="call_tool",
            capability="get_company_metric",
            attempt=lambda: _get_invalid_metric_error(client, tool_names),
            status="error",
            error_code_for=lambda result: result.error_code,
        )

        capabilities = (
            *(DiscoveredCapability(primitive="resource", name=name) for name in resource_names),
            *(DiscoveredCapability(primitive="tool", name=name) for name in tool_names),
            *(DiscoveredCapability(primitive="prompt", name=name) for name in prompt_names),
        )

    return FinancialMcpRun(
        server_name=server_name,
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


async def _read_coverage(client: Client) -> CoverageSnapshot:
    result = await client.read_resource("finance://coverage")
    return CoverageSnapshot.model_validate(json.loads(_resource_text(result.contents)))


async def _discover_tool_names(client: Client) -> tuple[str, ...]:
    return tuple(tool.name for tool in (await client.list_tools()).tools)


async def _discover_resource_names(client: Client) -> tuple[str, ...]:
    return tuple(str(resource.uri) for resource in (await client.list_resources()).resources)


async def _discover_prompt_names(client: Client) -> tuple[str, ...]:
    return tuple(prompt.name for prompt in (await client.list_prompts()).prompts)


async def _get_metric(client: Client, discovered_names: Sequence[str]) -> MetricResult:
    result = await _call_discovered_allowlisted_tool(
        client,
        discovered_names,
        "get_company_metric",
        {"ticker": "NVDA", "metric": "P/E"},
    )
    return MetricResult.model_validate(_structured_content(result))


async def _search_documents(
    client: Client, discovered_names: Sequence[str]
) -> DocumentSearchResult:
    result = await _call_discovered_allowlisted_tool(
        client,
        discovered_names,
        "search_financial_documents",
        {"company": "Schneider Electric", "query": "energy management", "top_k": 2},
    )
    return DocumentSearchResult.model_validate(_structured_content(result))


async def _render_prompt(client: Client) -> str:
    result = await client.get_prompt(
        "compare_companies",
        {"metric": "P/E", "question": "Compare valuation and operating evidence."},
    )
    return _prompt_text(result.messages)


async def _get_invalid_metric_error(
    client: Client, discovered_names: Sequence[str]
) -> CapabilityError:
    result = await _call_discovered_allowlisted_tool(
        client,
        discovered_names,
        "get_company_metric",
        {"ticker": "NVDA", "metric": "PE"},
    )
    return _capability_error(result)


async def _call_discovered_allowlisted_tool(
    client: Client,
    discovered_names: Sequence[str],
    name: str,
    arguments: dict[str, Any],
) -> CallToolResult:
    _validate_discovered_allowlisted_tool(name, discovered_names)
    return await client.call_tool(name, arguments)


def _validate_static_tool(name: str) -> None:
    if name not in ALLOWED_TOOLS:
        raise ValueError(f"Tool {name!r} is not allowlisted.")


def _validate_discovered_allowlisted_tool(name: str, discovered_names: Sequence[str]) -> None:
    _validate_static_tool(name)
    if name not in discovered_names:
        raise ValueError(f"Tool {name!r} was not dynamically discovered.")


@asynccontextmanager
async def _traced_client_context(trace: list[McpOperationEvent]) -> AsyncIterator[Client]:
    """Trace the v2 client context, which owns the stdio process lifecycle."""
    context = Client(financial_stdio_transport())
    started_at = perf_counter()
    try:
        value = await context.__aenter__()
    except BaseException as error:
        _append_event(
            trace,
            primitive="lifecycle",
            operation="open_transport",
            capability="stdio",
            status="error",
            duration_ms=_elapsed_ms(started_at),
            error_code=_safe_error_code(error),
        )
        _append_event(
            trace,
            primitive="lifecycle",
            operation="open_client_context",
            capability="client",
            status="error",
            duration_ms=_elapsed_ms(started_at),
            error_code=_safe_error_code(error),
        )
        raise

    _append_event(
        trace,
        primitive="lifecycle",
        operation="open_transport",
        capability="stdio",
        duration_ms=_elapsed_ms(started_at),
    )
    _append_event(
        trace,
        primitive="lifecycle",
        operation="open_client_context",
        capability="client",
        duration_ms=_elapsed_ms(started_at),
    )
    try:
        yield value
    except BaseException as body_error:
        suppressed = await _close_client_context(trace, context, body_error)
        if not suppressed:
            raise
    else:
        await _close_client_context(trace, context)


async def _close_client_context(
    trace: list[McpOperationEvent],
    context: Client,
    body_error: BaseException | None = None,
) -> bool:
    """Close the v2 client and its owned stdio transport with safe trace events."""
    started_at = perf_counter()
    try:
        if body_error is None:
            suppressed = await context.__aexit__(None, None, None)
        else:
            suppressed = await context.__aexit__(
                type(body_error), body_error, body_error.__traceback__
            )
    except BaseException as error:
        _append_event(
            trace,
            primitive="lifecycle",
            operation="close_client_context",
            capability="client",
            status="error",
            duration_ms=_elapsed_ms(started_at),
            error_code=_safe_error_code(error),
        )
        _append_event(
            trace,
            primitive="lifecycle",
            operation="close_transport",
            capability="stdio",
            status="error",
            duration_ms=_elapsed_ms(started_at),
            error_code=_safe_error_code(error),
        )
        raise

    _append_event(
        trace,
        primitive="lifecycle",
        operation="close_client_context",
        capability="client",
        duration_ms=_elapsed_ms(started_at),
    )
    _append_event(
        trace,
        primitive="lifecycle",
        operation="close_transport",
        capability="stdio",
        duration_ms=_elapsed_ms(started_at),
    )
    return suppressed


async def _record_async_operation(
    trace: list[McpOperationEvent],
    *,
    primitive: Literal["resource", "tool", "prompt", "discovery"],
    operation: str,
    capability: str,
    attempt: Callable[[], Awaitable[T]],
    status: Literal["ok", "error"] = "ok",
    evidence_count_for: Callable[[T], int] | None = None,
    error_code_for: Callable[[T], str | None] | None = None,
) -> T:
    """Record safe success or error metadata for an awaited protocol operation."""
    started_at = perf_counter()
    try:
        result = await attempt()
    except BaseException as error:
        _append_event(
            trace,
            primitive=primitive,
            operation=operation,
            capability=capability,
            status="error",
            duration_ms=_elapsed_ms(started_at),
            error_code=_safe_error_code(error),
        )
        raise

    _append_event(
        trace,
        primitive=primitive,
        operation=operation,
        capability=capability,
        status=status,
        duration_ms=_elapsed_ms(started_at),
        evidence_count=0 if evidence_count_for is None else evidence_count_for(result),
        error_code=None if error_code_for is None else error_code_for(result),
    )
    return result


def _record_sync_operation(
    trace: list[McpOperationEvent],
    *,
    primitive: Literal["resource", "tool", "prompt", "discovery"],
    operation: str,
    capability: str,
    attempt: Callable[[], T],
    capability_for: Callable[[T], str] | None = None,
) -> T:
    """Record safe success or error metadata for local result parsing/discovery."""
    started_at = perf_counter()
    try:
        result = attempt()
    except BaseException as error:
        _append_event(
            trace,
            primitive=primitive,
            operation=operation,
            capability=capability,
            status="error",
            duration_ms=_elapsed_ms(started_at),
            error_code=_safe_error_code(error),
        )
        raise

    _append_event(
        trace,
        primitive=primitive,
        operation=operation,
        capability=capability if capability_for is None else capability_for(result),
        duration_ms=_elapsed_ms(started_at),
    )
    return result


def _append_event(
    trace: list[McpOperationEvent],
    *,
    primitive: Literal["resource", "tool", "prompt", "discovery", "lifecycle"],
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


def _elapsed_ms(started_at: float) -> float:
    return round((perf_counter() - started_at) * 1000, 3)


def _safe_error_code(error: BaseException) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", type(error).__name__).lower()


def _resource_text(contents: Sequence[Any]) -> str:
    if len(contents) != 1 or not isinstance(contents[0], TextResourceContents):
        raise TypeError("Coverage resource must contain exactly one text resource content block.")
    return contents[0].text


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
