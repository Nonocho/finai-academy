"""MCP adapter for the bounded certified-document research capabilities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.utilities.func_metadata import FuncMetadata
from mcp.types import CallToolResult, TextContent
from pydantic import ConfigDict, Field, ValidationError

from finai_academy.capstone.document_tools import (
    DocumentCapabilityRegistry,
    build_document_capability_registry,
)

_INVALID_ARGUMENTS_MESSAGE = "Tool arguments must match the approved schema."


class _SanitizedFuncMetadata(FuncMetadata):
    """Turn MCP pre-invocation validation failures into safe tool outcomes."""

    async def call_fn_with_arg_validation(
        self,
        fn: Callable[..., Any | Awaitable[Any]],
        fn_is_async: bool,
        arguments_to_validate: dict[str, Any],
        arguments_to_pass_directly: dict[str, Any] | None,
        pre_validated: dict[str, Any] | None = None,
    ) -> Any:
        try:
            return await super().call_fn_with_arg_validation(
                fn,
                fn_is_async,
                arguments_to_validate,
                arguments_to_pass_directly,
                pre_validated,
            )
        except (TypeError, ValidationError, ValueError):
            return _invalid_arguments_result()


def build_capstone_mcp_server(
    registry: DocumentCapabilityRegistry | None = None,
) -> MCPServer:
    """Expose only search and evidence inspection through the local MCP boundary."""

    active_registry = registry or build_document_capability_registry()
    server = MCPServer(
        "Financial Document Research",
        instructions=(
            "Read-only certified document research for NVIDIA and Schneider Electric. "
            "Search first, then inspect the selected evidence before making a claim."
        ),
    )

    @server.tool()
    def search_financial_documents(
        company: str,
        reporting_period: str,
        query: str,
        element_type: str | None = None,
        top_k: Annotated[int, Field(ge=1, le=5)] = 3,
    ) -> dict[str, object]:
        """Search certified financial-document evidence with explicit metadata filters."""

        return active_registry.search_financial_documents(
            company, reporting_period, query, element_type, top_k
        ).model_dump(mode="json")

    @server.tool()
    def inspect_document_evidence(chunk_id: str) -> dict[str, object]:
        """Inspect one exact certified evidence chunk selected from search."""

        return active_registry.inspect_document_evidence(chunk_id).model_dump(mode="json")

    _configure_strict_sanitized_tool(server, "search_financial_documents")
    _configure_strict_sanitized_tool(server, "inspect_document_evidence")
    return server


def _configure_strict_sanitized_tool(server: MCPServer, name: str) -> None:
    """Forbid protocol extras and coercion before a registered tool is invoked."""

    tool = server._tool_manager.get_tool(name)
    if tool is None:  # pragma: no cover - registration is immediately above
        raise RuntimeError("registered MCP tool is missing")
    original_metadata = tool.fn_metadata
    strict_argument_model = type(
        f"{name}Arguments",
        (original_metadata.arg_model,),
        {
            "model_config": ConfigDict(
                arbitrary_types_allowed=True,
                extra="forbid",
                strict=True,
            )
        },
    )
    tool.fn_metadata = _SanitizedFuncMetadata(
        arg_model=strict_argument_model,
        output_schema=original_metadata.output_schema,
        output_model=original_metadata.output_model,
        wrap_output=original_metadata.wrap_output,
    )
    tool.parameters = strict_argument_model.model_json_schema(by_alias=True)


def _invalid_arguments_result() -> CallToolResult:
    """Return a protocol-visible typed error without retaining rejected input."""

    payload = {
        "status": "error",
        "error_code": "invalid_arguments",
        "message": _INVALID_ARGUMENTS_MESSAGE,
        "retryable": True,
    }
    return CallToolResult(
        content=[TextContent(type="text", text=_INVALID_ARGUMENTS_MESSAGE)],
        structured_content=payload,
        is_error=True,
    )


mcp = build_capstone_mcp_server()


def main() -> None:
    """Run the capstone document MCP server without stdout logging."""

    mcp.run()


if __name__ == "__main__":
    main()
