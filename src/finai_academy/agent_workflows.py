"""Typed tools and transparent orchestration records for the agent lessons."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


class ToolRequest(BaseModel):
    """One validated request made to the course tool registry."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolObservation(BaseModel):
    """A serializable tool result that can be inspected by students and models."""

    tool_name: str
    status: Literal["ok", "error"]
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class MarketPrice(BaseModel):
    """Versioned market-price observation with its finance metadata."""

    ticker: str
    company: str
    price: float = Field(gt=0)
    currency: str
    as_of: str
    source: str


class CurrencyConversion(BaseModel):
    """Deterministic conversion calculated from a versioned FX observation."""

    input_amount: float = Field(gt=0)
    output_amount: float = Field(gt=0)
    rate: float = Field(gt=0)
    from_currency: str
    to_currency: str
    rate_as_of: str
    source: str


ToolHandler = Callable[..., BaseModel]


class ToolRegistry:
    """Execute narrow typed tools and turn every failure into an observation."""

    def __init__(self, handlers: Mapping[str, ToolHandler]) -> None:
        if not handlers:
            raise ValueError("ToolRegistry requires at least one tool")
        self._handlers = dict(handlers)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._handlers))

    def invoke(self, request: ToolRequest) -> ToolObservation:
        handler = self._handlers.get(request.name)
        if handler is None:
            choices = ", ".join(self.names)
            return ToolObservation(
                tool_name=request.name,
                status="error",
                error=f"Unknown tool {request.name!r}. Valid tools: {choices}.",
            )

        try:
            result = handler(**request.arguments)
        except (TypeError, ValueError, ValidationError) as error:
            return ToolObservation(
                tool_name=request.name,
                status="error",
                error=str(error),
            )
        return ToolObservation(
            tool_name=request.name,
            status="ok",
            payload=result.model_dump(mode="json"),
        )


def load_course_market_snapshot(path: Path) -> dict[str, Any]:
    """Load one checked-in course snapshot without performing network access."""

    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise TypeError("market snapshot must be a JSON object")
    if not snapshot.get("dataset_id"):
        raise ValueError("market snapshot requires dataset_id")
    return snapshot


def build_course_tool_registry(snapshot: Mapping[str, Any]) -> ToolRegistry:
    """Build the two Lesson 08 tools from a maintained snapshot."""

    prices = snapshot.get("prices")
    fx_rates = snapshot.get("fx")
    if not isinstance(prices, Mapping) or not prices:
        raise ValueError("market snapshot requires non-empty prices")
    if not isinstance(fx_rates, Mapping) or not fx_rates:
        raise ValueError("market snapshot requires non-empty fx rates")

    def get_market_price(ticker: str) -> MarketPrice:
        normalized_ticker = ticker.upper().strip()
        record = prices.get(normalized_ticker)
        if record is None:
            choices = ", ".join(sorted(str(value) for value in prices))
            raise ValueError(
                f"Unsupported ticker {normalized_ticker!r}. Valid tickers: {choices}."
            )
        if not isinstance(record, Mapping):
            raise TypeError(f"Price record for {normalized_ticker!r} must be an object")
        return MarketPrice(ticker=normalized_ticker, **dict(record))

    def convert_currency(
        amount: float,
        from_currency: str,
        to_currency: str,
    ) -> CurrencyConversion:
        numeric_amount = float(amount)
        if not math.isfinite(numeric_amount) or numeric_amount <= 0:
            raise ValueError("amount must be a positive finite number")
        source_currency = from_currency.upper().strip()
        target_currency = to_currency.upper().strip()
        pair = f"{source_currency}_{target_currency}"
        record = fx_rates.get(pair)
        if record is None:
            choices = ", ".join(sorted(str(value) for value in fx_rates))
            raise ValueError(f"Unsupported currency pair {pair!r}. Valid pairs: {choices}.")
        if not isinstance(record, Mapping):
            raise TypeError(f"FX record for {pair!r} must be an object")
        rate = float(record["rate"])
        return CurrencyConversion(
            input_amount=numeric_amount,
            output_amount=round(numeric_amount * rate, 4),
            rate=rate,
            from_currency=source_currency,
            to_currency=target_currency,
            rate_as_of=str(record["as_of"]),
            source=str(record["source"]),
        )

    return ToolRegistry(
        {
            "convert_currency": convert_currency,
            "get_market_price": get_market_price,
        }
    )
