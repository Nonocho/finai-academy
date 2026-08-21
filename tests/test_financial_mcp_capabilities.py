from __future__ import annotations

import pytest

from finai_academy.financial_mcp_capabilities import (
    CapabilityValidationError,
    build_financial_capability_registry,
)


@pytest.fixture
def registry():
    return build_financial_capability_registry()


def test_coverage_exposes_only_the_two_course_companies(registry) -> None:
    coverage = registry.coverage()

    assert coverage.dataset_id == "lesson10-financial-mcp-v1"
    assert {item.ticker for item in coverage.companies} == {"NVDA", "SU.PA"}
    assert coverage.supported_metrics == ("EPS", "P/E")


def test_metric_result_preserves_provenance(registry) -> None:
    result = registry.get_company_metric("NVDA", "P/E")

    assert result.status == "ok"
    assert result.value == 52.4
    assert result.as_of == "2026-08-20"
    assert result.source


def test_invalid_metric_is_a_typed_retryable_error(registry) -> None:
    with pytest.raises(CapabilityValidationError) as caught:
        registry.get_company_metric("NVDA", "PE")

    assert caught.value.error.error_code == "unsupported_metric"
    assert caught.value.error.rejected_value == "PE"
    assert "P/E" in caught.value.error.valid_values
    assert caught.value.error.retryable is True


def test_document_search_filters_company_and_keeps_evidence_ids(registry) -> None:
    result = registry.search_financial_documents("Schneider Electric", "energy management", 2)

    assert result.status == "ok"
    assert result.company == "Schneider Electric"
    assert 1 <= len(result.hits) <= 2
    assert all(hit.evidence_id.startswith("SU-") for hit in result.hits)
    assert all(hit.source for hit in result.hits)


@pytest.mark.parametrize("top_k", [0, 4])
def test_document_search_rejects_out_of_range_top_k(registry, top_k: int) -> None:
    with pytest.raises(CapabilityValidationError) as caught:
        registry.search_financial_documents("NVIDIA", "data center", top_k)

    assert caught.value.error.error_code == "invalid_top_k"


@pytest.mark.parametrize(
    ("alias", "expected_company"),
    [("nvidia", "NVIDIA"), ("schneider", "Schneider Electric")],
)
def test_document_search_normalizes_course_company_aliases(
    registry, alias: str, expected_company: str
) -> None:
    result = registry.search_financial_documents(alias, "revenue")

    assert result.company == expected_company


@pytest.mark.parametrize(
    ("company", "query", "error_code"),
    [
        ("Unknown Co", "revenue", "unsupported_company"),
        ("NVIDIA", "   ", "invalid_query"),
    ],
)
def test_document_search_rejects_unknown_companies_and_blank_queries(
    registry, company: str, query: str, error_code: str
) -> None:
    with pytest.raises(CapabilityValidationError) as caught:
        registry.search_financial_documents(company, query)

    assert caught.value.error.error_code == error_code
