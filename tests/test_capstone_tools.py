"""Contracts for the Financial Analyst Copilot's certified tool boundaries."""

from __future__ import annotations

import re

import pytest

from finai_academy.capstone.live_news import TavilyNewsAdapter
from finai_academy.capstone.tools import AnalystToolRegistry, build_certified_retriever


def test_certified_retriever_keeps_nvidia_evidence_in_its_exact_company_boundary() -> None:
    """Removing the exact company filter would leak Schneider evidence into this result."""

    hits = build_certified_retriever().search("NVIDIA", "growth revenue", top_k=3)

    assert hits
    assert {hit.company for hit in hits} == {"NVIDIA"}
    assert all(hit.evidence_id and hit.source_reference for hit in hits)


def test_certified_retriever_keeps_schneider_evidence_in_its_exact_company_boundary() -> None:
    """Removing the exact company filter would leak NVIDIA evidence into this result."""

    hits = build_certified_retriever().search("Schneider Electric", "energy management", top_k=3)

    assert hits
    assert {hit.company for hit in hits} == {"Schneider Electric"}
    assert all(hit.evidence_id.startswith("SU-") for hit in hits)


@pytest.mark.parametrize(
    ("company", "query", "top_k", "error_code"),
    [
        ("Unsupported Co", "revenue", 2, "unsupported_company"),
        ("NVIDIA", "   ", 2, "invalid_query"),
        ("NVIDIA", "revenue", 0, "invalid_top_k"),
    ],
)
def test_certified_retriever_preserves_typed_validation_failures(
    company: str, query: str, top_k: int, error_code: str
) -> None:
    """Replacing fixture validation with generic errors would hide safe recovery guidance."""

    from finai_academy.financial_mcp_capabilities import CapabilityValidationError

    with pytest.raises(CapabilityValidationError) as caught:
        build_certified_retriever().search(company, query, top_k)

    assert caught.value.error.error_code == error_code


def test_registry_discovers_only_the_mandatory_read_tools() -> None:
    """A policy regression must not expose runtime-discovered trading capabilities."""

    registry = AnalystToolRegistry(
        discovered=("search_financial_documents", "get_company_metric", "place_order")
    )

    assert registry.discover() == ("get_company_metric", "search_financial_documents")


def test_registry_fails_closed_for_a_trading_tool_without_echoing_arguments() -> None:
    """Allowing an untrusted capability or echoing its payload would violate the tool boundary."""

    registry = AnalystToolRegistry(discovered=("place_order", "get_company_metric"))

    with pytest.raises(ValueError, match="not allowlisted") as caught:
        registry.invoke("place_order", {"symbol": "NVDA", "quantity": 100})

    assert "quantity" not in str(caught.value)


def test_registry_fails_closed_without_echoing_a_malicious_tool_name() -> None:
    """Echoing an attacker-controlled tool name can expose credential-shaped public text."""

    malicious_name = "place_order Authorization: Bearer provider-secret-token"
    registry = AnalystToolRegistry(discovered=("get_company_metric",))

    with pytest.raises(ValueError, match="not allowlisted") as caught:
        registry.invoke(malicious_name, {})

    assert str(caught.value) == "Tool is not allowlisted."
    assert malicious_name not in str(caught.value)


def test_registry_returns_typed_metric_and_document_outcomes() -> None:
    """Returning raw capability models would make successful tool calls inconsistent to consumers."""

    registry = AnalystToolRegistry(
        discovered=("search_financial_documents", "get_company_metric")
    )

    metric = registry.invoke("get_company_metric", {"ticker": "NVDA", "metric": "P/E"})
    documents = registry.invoke(
        "search_financial_documents",
        {"company": "Schneider Electric", "query": "energy management", "top_k": 2},
    )

    assert metric.status == "ok"
    assert metric.payload is not None
    assert metric.payload.company == "NVIDIA"
    assert documents.status == "ok"
    assert documents.payload is not None
    assert documents.payload.company == "Schneider Electric"


def test_registry_document_search_uses_the_existing_default_top_k_when_omitted() -> None:
    """Requiring a redundant top_k field breaks the documented default-capability call."""

    outcome = AnalystToolRegistry(
        discovered=("get_company_metric", "search_financial_documents")
    ).invoke(
        "search_financial_documents",
        {"company": "Schneider Electric", "query": "energy management"},
    )

    assert outcome.status == "ok"
    assert outcome.payload is not None
    assert len(outcome.payload.hits) == 2


def test_registry_maps_unsupported_metric_to_a_retryable_typed_outcome() -> None:
    """Dropping capability validation details would prevent the later replan to document search."""

    outcome = AnalystToolRegistry(
        discovered=("get_company_metric", "search_financial_documents")
    ).invoke("get_company_metric", {"ticker": "NVDA", "metric": "Revenue"})

    assert outcome.status == "error"
    assert outcome.error_code == "unsupported_metric"
    assert outcome.retryable is True
    assert outcome.payload is None


@pytest.mark.parametrize(
    "arguments",
    [
        {"company": "NVIDIA", "query": "api_key=provider-secret", "top_k": 2},
        {"company": "NVIDIA", "query": "/Users/analyst/private-notes.txt", "top_k": 2},
    ],
)
def test_registry_rejects_unsafe_document_queries_before_they_reach_public_payloads(
    arguments: dict[str, object]
) -> None:
    """Returning the fixture result unchanged would echo unsafe query text in its public payload."""

    outcome = AnalystToolRegistry(
        discovered=("get_company_metric", "search_financial_documents")
    ).invoke("search_financial_documents", arguments)

    assert outcome.status == "error"
    assert outcome.error_code == "invalid_arguments"
    assert outcome.payload is None
    assert "provider-secret" not in outcome.model_dump_json()
    assert "/Users/" not in outcome.model_dump_json()


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("search_financial_documents", {"company": "NVIDIA", "query": 123, "top_k": 2}),
        ("search_financial_documents", {"company": None, "query": "revenue", "top_k": 2}),
        ("search_financial_documents", {"company": "NVIDIA", "query": "revenue", "top_k": "2"}),
        ("get_company_metric", {"ticker": None, "metric": "P/E"}),
        ("get_company_metric", {"ticker": "NVDA", "metric": {"name": "P/E"}}),
    ],
)
def test_registry_returns_a_stable_typed_error_for_malformed_arguments(
    name: str, arguments: dict[str, object]
) -> None:
    """Forwarding malformed values causes raw AttributeError or TypeError in capability code."""

    outcome = AnalystToolRegistry(
        discovered=("get_company_metric", "search_financial_documents")
    ).invoke(name, arguments)

    assert outcome.status == "error"
    assert outcome.error_code == "invalid_arguments"
    assert outcome.message == "Tool arguments must match the approved schema."
    assert outcome.payload is None


def test_tavily_without_a_key_is_explicitly_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treating an absent optional credential as successful news would mislabel certified runs."""

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    outcome = TavilyNewsAdapter.from_environment().search("NVIDIA", "AI demand")

    assert outcome.status == "unavailable"
    assert outcome.items == ()
    assert "TAVILY_API_KEY" in outcome.message


def test_tavily_injected_callable_cannot_bypass_missing_key() -> None:
    """A test double must not accidentally turn optional live news into a credential bypass."""

    outcome = TavilyNewsAdapter(
        search_callable=lambda company, query: {"results": []}
    ).search("NVIDIA", "AI demand")

    assert outcome.status == "unavailable"
    assert outcome.items == ()


def test_tavily_success_keeps_only_public_news_fields() -> None:
    """Passing the raw provider response through would expose unreviewed provider data."""

    adapter = TavilyNewsAdapter(
        api_key="test-key",
        search_callable=lambda company, query: {
            "results": [
                {
                    "title": "NVIDIA demand update",
                    "url": "https://example.test/nvidia",
                    "published_date": "2026-08-20",
                    "raw_content": "must not leave the adapter",
                }
            ],
            "query": f"{company} {query}",
        },
    )

    outcome = adapter.search("NVIDIA", "AI demand")

    assert outcome.status == "ok"
    assert len(outcome.items) == 1
    item = outcome.items[0]
    assert item.title == "NVIDIA demand update"
    assert item.url == "https://example.test/nvidia"
    assert item.published_date == "2026-08-20"
    assert item.provider == "tavily"
    assert re.fullmatch(r".+T.+(?:Z|[+-]\d\d:\d\d)", item.retrieved_at)
    assert "raw_content" not in outcome.model_dump_json()
    assert "test-key" not in outcome.model_dump_json()


def test_tavily_runtime_errors_do_not_affect_certified_analysis() -> None:
    """Provider exceptions must not surface raw failures or make live enrichment appear usable."""

    def fail_search(company: str, query: str) -> dict[str, object]:
        raise RuntimeError("credential test-key should not be exposed")

    outcome = TavilyNewsAdapter(api_key="test-key", search_callable=fail_search).search(
        "NVIDIA", "AI demand"
    )

    assert outcome.status == "error"
    assert outcome.items == ()
    assert outcome.message == "News enrichment failed; certified analysis remains available."
    assert "test-key" not in outcome.model_dump_json()


def test_tavily_drops_credential_shaped_provider_content() -> None:
    """Returning a provider title with a credential would violate the public output boundary."""

    outcome = TavilyNewsAdapter(
        api_key="test-key",
        search_callable=lambda company, query: {
            "results": [
                {
                    "title": "Authorization: Bearer provider-secret-token",
                    "url": "https://example.test/nvidia",
                }
            ]
        },
    ).search("NVIDIA", "AI demand")

    assert outcome.status == "error"
    assert outcome.items == ()
    assert "provider-secret-token" not in outcome.model_dump_json()
