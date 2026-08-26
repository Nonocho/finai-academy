"""MCP adapter for the bounded certified-document research capabilities."""

from __future__ import annotations

from typing import Annotated

from mcp.server import MCPServer
from pydantic import Field

from finai_academy.capstone.document_tools import (
    DocumentCapabilityRegistry,
    build_document_capability_registry,
)


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

    return server


mcp = build_capstone_mcp_server()


def main() -> None:
    """Run the capstone document MCP server without stdout logging."""

    mcp.run()


if __name__ == "__main__":
    main()
