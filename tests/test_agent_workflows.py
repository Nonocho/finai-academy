from __future__ import annotations

import math

from finai_academy.agent_workflows import ToolRequest, build_course_tool_registry

SNAPSHOT = {
    "dataset_id": "lesson08-market-snapshot-v1",
    "notice": "Checked-in course snapshot; not a live quote or investment recommendation.",
    "prices": {
        "NVDA": {
            "company": "NVIDIA",
            "price": 180.0,
            "currency": "USD",
            "as_of": "2026-08-20",
            "source": "https://finance.yahoo.com/quote/NVDA/history/",
        },
        "SU.PA": {
            "company": "Schneider Electric",
            "price": 240.0,
            "currency": "EUR",
            "as_of": "2026-08-20",
            "source": "https://finance.yahoo.com/quote/SU.PA/history/",
        },
    },
    "fx": {
        "USD_EUR": {
            "rate": 0.86,
            "as_of": "2026-08-20",
            "source": "https://finance.yahoo.com/quote/EURUSD%3DX/history/",
        }
    },
}


def test_market_price_retains_provenance() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    result = registry.invoke(
        ToolRequest(name="get_market_price", arguments={"ticker": "NVDA"})
    )

    assert result.status == "ok"
    assert result.payload["ticker"] == "NVDA"
    assert result.payload["currency"] == "USD"
    assert result.payload["as_of"] == "2026-08-20"
    assert result.payload["source"].startswith("https://")


def test_registry_returns_actionable_unknown_tool_error() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    result = registry.invoke(ToolRequest(name="get_price", arguments={"ticker": "NVDA"}))

    assert result.status == "error"
    assert result.error is not None
    assert "get_market_price" in result.error
    assert "convert_currency" in result.error


def test_currency_conversion_rejects_non_positive_amount() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    result = registry.invoke(
        ToolRequest(
            name="convert_currency",
            arguments={"amount": 0, "from_currency": "USD", "to_currency": "EUR"},
        )
    )

    assert result.status == "error"
    assert result.error is not None
    assert "positive" in result.error.casefold()


def test_currency_conversion_uses_the_versioned_rate() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    result = registry.invoke(
        ToolRequest(
            name="convert_currency",
            arguments={"amount": 180.0, "from_currency": "USD", "to_currency": "EUR"},
        )
    )

    assert result.status == "ok"
    assert math.isclose(result.payload["output_amount"], 154.8)
    assert result.payload["rate"] == 0.86
    assert result.payload["rate_as_of"] == "2026-08-20"


def test_registry_rejects_unsupported_ticker_with_valid_choices() -> None:
    registry = build_course_tool_registry(SNAPSHOT)

    result = registry.invoke(
        ToolRequest(name="get_market_price", arguments={"ticker": "AAPL"})
    )

    assert result.status == "error"
    assert result.error is not None
    assert "NVDA" in result.error
    assert "SU.PA" in result.error
