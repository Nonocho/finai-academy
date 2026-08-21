from __future__ import annotations

import asyncio
import json

from mcp import Client
from mcp.types import TextContent

from finai_academy.financial_mcp_server import build_financial_mcp_server


def test_server_discovers_the_course_capabilities() -> None:
    """Breaks if an MCP primitive is added, removed, or renamed."""

    async def discover() -> None:
        server = build_financial_mcp_server()
        async with Client(server, raise_exceptions=True) as client:
            tools = await client.list_tools()
            resources = await client.list_resources()
            prompts = await client.list_prompts()

        assert [item.name for item in tools.tools] == [
            "get_company_metric",
            "search_financial_documents",
        ]
        assert [str(item.uri) for item in resources.resources] == ["finance://coverage"]
        assert [item.name for item in prompts.prompts] == ["compare_companies"]

        tool_schemas = {item.name: item.input_schema for item in tools.tools}
        tool_descriptions = {item.name: item.description for item in tools.tools}
        assert tool_descriptions == {
            "get_company_metric": "Return a controlled company metric with date and source.",
            "search_financial_documents": "Search controlled company financial evidence passages.",
        }
        assert tool_schemas["get_company_metric"] == {
            "properties": {
                "metric": {"title": "Metric", "type": "string"},
                "ticker": {"title": "Ticker", "type": "string"},
            },
            "required": ["ticker", "metric"],
            "title": "get_company_metricArguments",
            "type": "object",
        }
        assert tool_schemas["search_financial_documents"] == {
            "properties": {
                "company": {"title": "Company", "type": "string"},
                "query": {"title": "Query", "type": "string"},
                "top_k": {
                    "default": 2,
                    "maximum": 3,
                    "minimum": 1,
                    "title": "Top K",
                    "type": "integer",
                },
            },
            "required": ["company", "query"],
            "title": "search_financial_documentsArguments",
            "type": "object",
        }

    asyncio.run(discover())


def test_coverage_resource_returns_parseable_registry_coverage() -> None:
    """Breaks if the coverage resource stops returning the registry's JSON contract."""

    async def read_coverage() -> None:
        server = build_financial_mcp_server()
        async with Client(server, raise_exceptions=True) as client:
            result = await client.read_resource("finance://coverage")

        coverage = json.loads(result.contents[0].text)
        assert coverage["dataset_id"] == "lesson10-financial-mcp-v1"
        assert coverage["tickers"] == ["NVDA", "SU.PA"]
        assert coverage["source_notice"]

    asyncio.run(read_coverage())


def test_metric_tool_returns_structured_provenance() -> None:
    """Breaks if successful metric results lose structured provenance across MCP."""

    async def call_metric() -> None:
        server = build_financial_mcp_server()
        async with Client(server, raise_exceptions=True) as client:
            result = await client.call_tool(
                "get_company_metric", {"ticker": "NVDA", "metric": "P/E"}
            )

        assert result.is_error is False
        assert result.structured_content is not None
        assert result.structured_content["status"] == "ok"
        assert result.structured_content["ticker"] == "NVDA"
        assert result.structured_content["company"] == "NVIDIA"
        assert result.structured_content["metric"] == "P/E"
        assert result.structured_content["value"] == 52.4
        assert result.structured_content["as_of"] == "2026-08-20"
        assert result.structured_content["source"]

    asyncio.run(call_metric())


def test_document_search_tool_retains_evidence_ids() -> None:
    """Breaks if document search evidence IDs are lost in the protocol result."""

    async def search_documents() -> None:
        server = build_financial_mcp_server()
        async with Client(server, raise_exceptions=True) as client:
            result = await client.call_tool(
                "search_financial_documents",
                {
                    "company": "Schneider Electric",
                    "query": "energy management",
                    "top_k": 2,
                },
            )

        assert result.is_error is False
        assert result.structured_content is not None
        hits = result.structured_content["hits"]
        assert hits
        assert all(hit["evidence_id"].startswith("SU-") for hit in hits)

    asyncio.run(search_documents())


def test_compare_companies_prompt_returns_one_user_message() -> None:
    """Breaks if the reusable comparison prompt no longer renders one user message."""

    async def render_prompt() -> None:
        server = build_financial_mcp_server()
        async with Client(server, raise_exceptions=True) as client:
            result = await client.get_prompt(
                "compare_companies",
                {"metric": "P/E", "question": "Compare the available evidence."},
            )

        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert "NVIDIA" in result.messages[0].content.text
        assert "Schneider Electric" in result.messages[0].content.text
        assert "finance://coverage" in result.messages[0].content.text
        assert "get_company_metric" in result.messages[0].content.text
        assert "search_financial_documents" in result.messages[0].content.text
        assert "Do not make an investment recommendation." in result.messages[0].content.text

    asyncio.run(render_prompt())


def test_invalid_metric_returns_a_typed_protocol_error() -> None:
    """Breaks if registry validation fails to cross MCP as model-visible tool content."""

    async def call_invalid_metric() -> None:
        server = build_financial_mcp_server()
        async with Client(server, raise_exceptions=True) as client:
            result = await client.call_tool(
                "get_company_metric", {"ticker": "NVDA", "metric": "PE"}
            )

        assert result.is_error is True
        error_text = "\n".join(
            block.text for block in result.content if isinstance(block, TextContent)
        )
        assert "unsupported_metric" in error_text
        assert "P/E" in error_text

    asyncio.run(call_invalid_metric())
