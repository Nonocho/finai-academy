"""Persistent, allowlisted MCP execution for Lesson 11 research plans."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any, Self

from mcp import Client
from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError

from finai_academy.financial_mcp_capabilities import CapabilityError
from finai_academy.financial_mcp_client import ALLOWED_TOOLS, financial_stdio_transport
from finai_academy.research_planning import PlannerToolSpec, PlanStep, ResearchObservation


class FinancialMcpPlanningExecutor:
    """Own one local MCP client lifecycle for a complete research run."""

    def __init__(self) -> None:
        self._client_context: Client | None = None
        self._client: Client | None = None
        self.catalog: tuple[PlannerToolSpec, ...] = ()
        self.server_name = ""

    async def __aenter__(self) -> Self:
        self._client_context = Client(financial_stdio_transport())
        try:
            self._client = await self._client_context.__aenter__()
            self.server_name = self._client.server_info.name
            discovered = await self._client.list_tools()
            self.catalog = tuple(
                PlannerToolSpec(
                    name=tool.name,
                    description=tool.description or "",
                    input_schema=dict(tool.input_schema),
                )
                for tool in discovered.tools
                if tool.name in ALLOWED_TOOLS
            )
            if not self.catalog:
                raise RuntimeError("MCP discovery returned no permitted tools")
        except BaseException as error:
            await self._client_context.__aexit__(type(error), error, error.__traceback__)
            self._client_context = None
            self._client = None
            raise
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._client_context is not None:
            await self._client_context.__aexit__(exc_type, exc, traceback)
        self._client_context = None
        self._client = None

    async def execute(
        self,
        step: PlanStep,
        *,
        attempt_id: int,
        plan_revision: int,
    ) -> ResearchObservation:
        """Dispatch one discovered, statically allowlisted research capability."""
        if self._client is None:
            raise RuntimeError("executor must be opened before execution")

        permitted = {item.name for item in self.catalog}
        if step.capability not in ALLOWED_TOOLS or step.capability not in permitted:
            return blocked_observation(step, attempt_id, plan_revision, "capability_not_permitted")

        started = perf_counter()
        result = await self._client.call_tool(step.capability, step.arguments)
        duration_ms = (perf_counter() - started) * 1000
        return observation_from_call_result(
            step=step,
            result=result,
            attempt_id=attempt_id,
            plan_revision=plan_revision,
            duration_ms=duration_ms,
        )


def extract_structured_tool_result(result: CallToolResult) -> dict[str, Any]:
    """Return a copied structured success payload, rejecting error protocol results."""
    if result.is_error:
        raise ValueError("Expected a successful MCP tool result.")
    if not isinstance(result.structured_content, Mapping):
        raise TypeError("Successful tool result did not include structured content.")
    return dict(result.structured_content)


def extract_capability_error(result: CallToolResult) -> CapabilityError:
    """Parse the server's typed JSON capability error from text-only MCP content."""
    if not result.is_error:
        raise ValueError("Expected a failed MCP tool result.")
    text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
    json_start = text.find("{")
    if json_start < 0:
        raise ValueError("Tool error did not contain a structured capability error.")
    return CapabilityError.model_validate(json.loads(text[json_start:]))


def observation_from_call_result(
    *,
    step: PlanStep,
    result: CallToolResult,
    attempt_id: int,
    plan_revision: int,
    duration_ms: float,
) -> ResearchObservation:
    """Convert an MCP result into the display-safe observation consumed by the graph."""
    if result.is_error:
        try:
            error = extract_capability_error(result)
        except (TypeError, ValueError, ValidationError, json.JSONDecodeError):
            return error_observation(
                step,
                attempt_id,
                plan_revision,
                duration_ms,
                error_code="invalid_tool_error",
            )
        return ResearchObservation(
            attempt_id=attempt_id,
            step_id=step.step_id,
            plan_revision=plan_revision,
            capability=step.capability,
            arguments=dict(step.arguments),
            status="error",
            result=error.model_dump(mode="json"),
            error_code=error.error_code,
            duration_ms=duration_ms,
        )

    try:
        payload = extract_structured_tool_result(result)
    except (TypeError, ValueError):
        return error_observation(
            step,
            attempt_id,
            plan_revision,
            duration_ms,
            error_code="invalid_tool_result",
        )
    return ResearchObservation(
        attempt_id=attempt_id,
        step_id=step.step_id,
        plan_revision=plan_revision,
        capability=step.capability,
        arguments=dict(step.arguments),
        status="ok",
        result=payload,
        evidence_ids=_evidence_ids(payload),
        source_references=_source_references(payload),
        duration_ms=duration_ms,
    )


def blocked_observation(
    step: PlanStep,
    attempt_id: int,
    plan_revision: int,
    error_code: str,
) -> ResearchObservation:
    """Record a denied capability without contacting the MCP server."""
    return ResearchObservation(
        attempt_id=attempt_id,
        step_id=step.step_id,
        plan_revision=plan_revision,
        capability=step.capability,
        arguments=dict(step.arguments),
        status="blocked",
        error_code=error_code,
        duration_ms=0,
    )


def error_observation(
    step: PlanStep,
    attempt_id: int,
    plan_revision: int,
    duration_ms: float,
    *,
    error_code: str,
) -> ResearchObservation:
    """Return a generic safe protocol error without retaining raw response content."""
    return ResearchObservation(
        attempt_id=attempt_id,
        step_id=step.step_id,
        plan_revision=plan_revision,
        capability=step.capability,
        arguments=dict(step.arguments),
        status="error",
        result={"status": "error", "error_code": error_code},
        error_code=error_code,
        duration_ms=duration_ms,
    )


def _evidence_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    hits = payload.get("hits")
    if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
        return ()
    return tuple(
        item["evidence_id"]
        for item in hits
        if isinstance(item, Mapping) and isinstance(item.get("evidence_id"), str)
    )


def _source_references(payload: Mapping[str, Any]) -> tuple[str, ...]:
    sources: list[str] = []
    source = payload.get("source")
    if isinstance(source, str):
        sources.append(source)
    hits = payload.get("hits")
    if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes)):
        sources.extend(
            item["source"]
            for item in hits
            if isinstance(item, Mapping) and isinstance(item.get("source"), str)
        )
    return tuple(dict.fromkeys(sources))
