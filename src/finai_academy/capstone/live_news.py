"""Optional, sanitized Tavily news enrichment for the capstone."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict

from finai_academy.capstone.models import _clean_public_value

NewsSearchCallable = Callable[[str, str], Mapping[str, Any]]
_FAILURE_MESSAGE = "News enrichment failed; certified analysis remains available."
_UNAVAILABLE_MESSAGE = "Set TAVILY_API_KEY to enable optional live news enrichment."


class NewsItem(BaseModel):
    """The deliberately small public portion of one Tavily result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str
    url: str
    published_date: str | None = None
    provider: Literal["tavily"] = "tavily"
    retrieved_at: str


class NewsSearchOutcome(BaseModel):
    """A truthful optional-news result that never carries raw provider content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok", "unavailable", "error"]
    items: tuple[NewsItem, ...] = ()
    message: str


class TavilyNewsAdapter:
    """Read Tavily only when configured, keeping all failures outside certified analysis."""

    def __init__(
        self, api_key: str | None = None, search_callable: NewsSearchCallable | None = None
    ) -> None:
        self._api_key = api_key.strip() if api_key else None
        self._search_callable = search_callable or (
            _build_tavily_search_callable(self._api_key) if self._api_key else None
        )

    @classmethod
    def from_environment(
        cls, search_callable: NewsSearchCallable | None = None
    ) -> TavilyNewsAdapter:
        """Create the optional adapter without exposing the environment value in outputs."""

        return cls(os.environ.get("TAVILY_API_KEY"), search_callable=search_callable)

    def search(self, company: str, query: str) -> NewsSearchOutcome:
        """Return a sanitized best-effort result for a company-specific live-news query."""

        if not self._api_key or self._search_callable is None:
            return NewsSearchOutcome(status="unavailable", message=_UNAVAILABLE_MESSAGE)
        try:
            response = self._search_callable(company, query)
            raw_items = response.get("results", ())
            if not isinstance(raw_items, list):
                raise TypeError("Tavily results must be a list")
            retrieved_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            items = tuple(
                NewsItem(
                    title=_required_result_string(item, "title"),
                    url=_required_result_string(item, "url"),
                    published_date=_optional_result_string(item, "published_date"),
                    retrieved_at=retrieved_at,
                )
                for item in raw_items
                if isinstance(item, Mapping)
            )
            return NewsSearchOutcome(status="ok", items=items, message="Live news enrichment retrieved.")
        except Exception:  # noqa: BLE001 - provider failures must never block certified analysis
            return NewsSearchOutcome(status="error", message=_FAILURE_MESSAGE)


def _build_tavily_search_callable(api_key: str) -> NewsSearchCallable:
    """Build a small standard-library Tavily client so the optional dependency stays optional."""

    def search(company: str, query: str) -> Mapping[str, Any]:
        request = Request(
            "https://api.tavily.com/search",
            data=json.dumps({"api_key": api_key, "query": f"{company} {query}"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("Tavily response must be an object")
        return payload

    return search


def _required_result_string(item: Mapping[str, Any], field: str) -> str:
    value = item.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Tavily result requires {field}")
    return str(_clean_public_value(value))


def _optional_result_string(item: Mapping[str, Any], field: str) -> str | None:
    value = item.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Tavily result {field} must be a non-empty string")
    return str(_clean_public_value(value))
