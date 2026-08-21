"""MCP adapter for the Lesson 10 read-only financial capability registry."""

from __future__ import annotations

from typing import Annotated

from mcp.server import MCPServer
from pydantic import Field

from finai_academy.financial_mcp_capabilities import (
    FinancialCapabilityRegistry,
    build_financial_capability_registry,
)


def build_financial_mcp_server(
    registry: FinancialCapabilityRegistry | None = None,
) -> MCPServer:
    """Build the read-only server without duplicating registry business logic."""

    active_registry = registry or build_financial_capability_registry()
    server = MCPServer(
        "First Finance Research",
        instructions=(
            "Read-only financial research capabilities for NVIDIA and "
            "Schneider Electric. Preserve dates, sources, and evidence IDs."
        ),
    )

    @server.resource("finance://coverage", mime_type="application/json")
    def coverage() -> dict[str, object]:
        return active_registry.coverage().model_dump(mode="json")

    @server.tool()
    def get_company_metric(ticker: str, metric: str) -> dict[str, object]:
        """Return a controlled company metric with date and source."""
        return active_registry.get_company_metric(ticker, metric).model_dump(mode="json")

    @server.tool()
    def search_financial_documents(
        company: str,
        query: str,
        top_k: Annotated[int, Field(ge=1, le=3)] = 2,
    ) -> dict[str, object]:
        """Search controlled company financial evidence passages."""
        return active_registry.search_financial_documents(company, query, top_k).model_dump(
            mode="json"
        )

    @server.prompt()
    def compare_companies(metric: str, question: str) -> str:
        return (
            f"Compare NVIDIA and Schneider Electric using the metric {metric}.\n"
            f"Question: {question}\n"
            "Use only finance://coverage, get_company_metric, and "
            "search_financial_documents results. Cite every evidence ID, date, and "
            "source. State missing evidence. Do not make an investment recommendation."
        )

    return server


mcp = build_financial_mcp_server()


def main() -> None:
    """Run the MCP server over its configured transport without stdout logging."""

    mcp.run()


if __name__ == "__main__":
    main()
