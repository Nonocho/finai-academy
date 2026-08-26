"""Contracts for the bounded certified-document capability layer."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pytest
from mcp import Client
from mcp.types import TextContent

from finai_academy.capstone.document_tools import (
    ReportedValue,
    build_document_capability_registry,
    compare_reported_values,
)
from finai_academy.capstone.mcp_server import build_capstone_mcp_server

ROOT = Path(__file__).resolve().parents[1]


def test_search_then_inspect_returns_the_same_stable_chunk() -> None:
    """Changing inspection to a separate catalog lookup would lose stable evidence identity."""

    registry = build_document_capability_registry(ROOT)

    search = registry.search_financial_documents(
        company="NVIDIA",
        reporting_period="FY2026",
        query="segment revenue 193,479",
        element_type="table",
        top_k=1,
    )
    inspected = registry.inspect_document_evidence(search.hits[0].chunk_id)

    assert inspected.chunk_id == search.hits[0].chunk_id
    assert inspected.physical_page == 165
    assert inspected.crop_asset_key == (
        "assets/course-data/capstone/crops/nvidia_segment_table_page_165.png"
    )


def test_reported_value_comparison_uses_only_cited_inputs() -> None:
    """Comparing currencies without a supplied FX input would manufacture an unsupported value."""

    result = compare_reported_values(
        left=ReportedValue(
            label="NVIDIA total", value=215938, unit="USD millions", chunk_id="chunk-a"
        ),
        right=ReportedValue(
            label="Schneider total", value=40152, unit="EUR millions", chunk_id="chunk-b"
        ),
    )

    assert result.left.chunk_id == "chunk-a"
    assert result.right.chunk_id == "chunk-b"
    assert result.absolute_difference is None
    assert result.comparable is False
    assert result.formula == "215938 USD millions - 40152 EUR millions"
    assert result.reason == "Currencies differ; no FX rate was supplied."


def test_reported_value_comparison_calculates_only_matching_currency_and_scale() -> None:
    """Dropping unit matching would report arithmetic across incompatible displayed scales."""

    result = compare_reported_values(
        left=ReportedValue(label="Current", value=12.5, unit="USD millions", chunk_id="chunk-a"),
        right=ReportedValue(label="Prior", value=7.25, unit="USD millions", chunk_id="chunk-b"),
    )

    assert result.comparable is True
    assert result.absolute_difference == 5.25
    assert result.formula == "12.5 USD millions - 7.25 USD millions = 5.25 USD millions"
    assert result.reason == "Values use the same currency and scale."


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_reported_values_reject_non_finite_numbers(value: float) -> None:
    """Allowing NaN or infinity would make displayed deterministic arithmetic unreliable."""

    with pytest.raises(ValueError, match="finite"):
        ReportedValue(label="Reported", value=value, unit="USD millions", chunk_id="chunk-a")


def test_mcp_exposes_only_search_and_inspection_document_tools() -> None:
    """Publishing comparison or a non-document tool would widen the MCP protocol boundary."""

    async def discover() -> None:
        server = build_capstone_mcp_server(build_document_capability_registry(ROOT))
        async with Client(server, raise_exceptions=True) as client:
            tools = await client.list_tools()

        assert [tool.name for tool in tools.tools] == [
            "search_financial_documents",
            "inspect_document_evidence",
        ]
        assert tools.tools[0].input_schema["properties"]["top_k"]["maximum"] == 5
        assert "compare_reported_values" not in [tool.name for tool in tools.tools]

    asyncio.run(discover())


def test_mcp_direct_calls_reject_invalid_arguments_without_echoing_them() -> None:
    """Permissive MCP validation can coerce, ignore, or disclose untrusted direct inputs."""

    async def call_tools() -> None:
        server = build_capstone_mcp_server(build_document_capability_registry(ROOT))
        async with Client(server, raise_exceptions=True) as client:
            valid = await client.call_tool(
                "search_financial_documents",
                {
                    "company": "NVIDIA",
                    "reporting_period": "FY2026",
                    "query": "segment revenue",
                    "top_k": 1,
                },
            )
            valid_inspection = await client.call_tool(
                "inspect_document_evidence",
                {"chunk_id": valid.structured_content["hits"][0]["chunk_id"]},
            )
            invalid_results = []
            for arguments in (
                {
                    "company": "NVIDIA",
                    "reporting_period": "FY2026",
                    "query": "/Users/a",
                },
                {
                    "company": "NVIDIA",
                    "reporting_period": "FY2026",
                    "query": "api_key=x",
                },
                {
                    "company": "NVIDIA",
                    "reporting_period": "FY2026",
                    "query": "segment revenue",
                    "unexpected": "value",
                },
                {
                    "company": "NVIDIA",
                    "reporting_period": "FY2026",
                    "query": "segment revenue",
                    "top_k": "1",
                },
            ):
                invalid_results.append(
                    await client.call_tool("search_financial_documents", arguments)
                )
            invalid_results.append(
                await client.call_tool(
                    "inspect_document_evidence", {"chunk_id": "unknown-certified-chunk"}
                )
            )

        assert valid.is_error is False
        assert valid.structured_content is not None
        assert valid.structured_content["status"] == "ok"
        assert valid_inspection.is_error is False
        assert valid_inspection.structured_content is not None
        assert valid_inspection.structured_content["status"] == "ok"
        for result in invalid_results:
            assert result.is_error is True
            assert result.structured_content == {
                "status": "error",
                "error_code": "invalid_arguments",
                "message": "Tool arguments must match the approved schema.",
                "retryable": True,
            }
            error_text = "\n".join(
                block.text for block in result.content if isinstance(block, TextContent)
            )
            assert error_text == "Tool arguments must match the approved schema."
            assert "/Users/a" not in error_text
            assert "api_key=x" not in error_text

    asyncio.run(call_tools())
