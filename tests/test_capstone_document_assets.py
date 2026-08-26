from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import HttpUrl, ValidationError

from finai_academy.capstone.document_assets import (
    SourceAssetError,
    load_certified_document_sources,
    verify_source_asset,
)
from finai_academy.capstone.document_models import (
    BoundingBox,
    ContextualMetadata,
    DocumentElement,
    FinancialChunk,
    FinancialDocumentSource,
    FinancialMetadata,
    TableMatrix,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets/course-data/manifest.json"
SHA256 = "0" * 64


def sample_source(**overrides: object) -> FinancialDocumentSource:
    payload: dict[str, object] = {
        "document_id": "NVDA-FY2026-AR",
        "company_name": "NVIDIA",
        "ticker": "NVDA",
        "document_type": "Annual Report",
        "reporting_period": "FY2026",
        "publication_date": "2026-02-25",
        "official_source_url": "https://investor.nvidia.com/",
        "local_asset_key": "assets/course-data/downloads/report.pdf",
        "sha256": SHA256,
        "byte_size": 1,
        "page_count": 1,
    }
    payload.update(overrides)
    return FinancialDocumentSource.model_validate(payload)


def sample_context(**overrides: object) -> ContextualMetadata:
    payload: dict[str, object] = {
        "document_id": "NVDA-FY2026-AR",
        "company_name": "NVIDIA",
        "ticker": "NVDA",
        "document_type": "Annual Report",
        "reporting_period": "FY2026",
        "publication_date": date(2026, 2, 25),
        "official_source_url": "https://investor.nvidia.com/",
        "document_sha256": SHA256,
        "physical_page": 1,
        "element_type": "paragraph",
        "bbox": BoundingBox(x0=0, y0=0, x1=1, y1=1),
        "parser_name": "test-parser",
        "parser_version": "1.0",
        "extraction_method": "native_text",
    }
    payload.update(overrides)
    return ContextualMetadata.model_validate(payload)


def sample_financial(**overrides: object) -> FinancialMetadata:
    payload: dict[str, object] = {
        "periods": ("FY2026",),
        "source_element_ids": ("element-1",),
        "enrichment_method": "deterministic",
        "confidence": 1.0,
    }
    payload.update(overrides)
    return FinancialMetadata.model_validate(payload)


def test_capstone_manifest_certifies_both_complete_official_pdfs() -> None:
    sources = load_certified_document_sources(MANIFEST)

    assert [(source.company_name, source.page_count) for source in sources] == [
        ("NVIDIA", 175),
        ("Schneider Electric", 19),
    ]
    assert [source.sha256 for source in sources] == [
        "0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c",
        "5b17e6e8e5b8c47f697f5d352ed9639b98b261d9ecf886d3e8ce147493ebb00a",
    ]
    for source in sources:
        verify_source_asset(source, ROOT)


@pytest.mark.parametrize(
    "official_source_url",
    (
        "https://alice@example.com/report.pdf",
        "https://alice:correcthorsebatterystaple@example.com/report.pdf",
    ),
)
def test_public_source_contract_rejects_url_userinfo(official_source_url: str) -> None:
    with pytest.raises(ValueError, match="URL userinfo"):
        sample_source(official_source_url=official_source_url)


def test_public_source_contract_rejects_parsed_url_userinfo() -> None:
    with pytest.raises(ValueError, match="URL userinfo"):
        sample_source(
            official_source_url=HttpUrl(
                "https://alice:correcthorsebatterystaple@example.com/report.pdf"
            )
        )


@pytest.mark.parametrize(
    "company_name",
    (r"C:\Users\example", r"\Users\example\report.pdf"),
)
def test_public_contract_rejects_windows_personal_paths_in_all_public_strings(
    company_name: str,
) -> None:
    with pytest.raises(ValueError, match="personal filesystem paths"):
        sample_source(company_name=company_name)
    with pytest.raises(ValueError, match="personal filesystem paths"):
        sample_context(parser_name=company_name)


@pytest.mark.parametrize(
    "local_asset_key",
    (
        "/Users/example/report.pdf",
        r"C:\Users\example\report.pdf",
        r"C:\reports\report.pdf",
        r"C:reports\report.pdf",
        r"\reports\report.pdf",
    ),
)
def test_public_source_contract_rejects_rooted_or_drive_qualified_asset_keys(
    local_asset_key: str,
) -> None:
    with pytest.raises(ValueError, match="local_asset_key"):
        sample_source(local_asset_key=local_asset_key)


@pytest.mark.parametrize(
    "local_asset_key",
    ("assets/course-data/downloads/report.pdf", "assets/reports/report.pdf"),
)
def test_public_source_contract_accepts_safe_relative_asset_keys(local_asset_key: str) -> None:
    assert sample_source(local_asset_key=local_asset_key).local_asset_key == local_asset_key


def test_public_contract_rejects_blank_credential_and_non_json_values() -> None:
    with pytest.raises(ValueError, match="must not be blank"):
        sample_source(company_name=" ")
    with pytest.raises(ValueError, match="credential-shaped"):
        sample_source(company_name="api_key=super-secret")
    with pytest.raises(ValueError, match="JSON-compatible"):
        sample_source(document_id=Path("report.pdf"))


def test_public_contract_is_frozen_and_forbids_extra_fields() -> None:
    source = sample_source()

    with pytest.raises(ValidationError, match="frozen"):
        source.company_name = "Other"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FinancialDocumentSource.model_validate({**source.model_dump(), "extra": "no"})


def test_bounding_box_requires_positive_finite_area() -> None:
    with pytest.raises(ValueError, match="positive area"):
        BoundingBox(x0=10, y0=10, x1=10, y1=12)
    with pytest.raises(ValueError, match="finite"):
        BoundingBox(x0=0, y0=0, x1=float("inf"), y1=1)


def test_table_elements_require_a_consistent_table_matrix() -> None:
    table = TableMatrix(
        rows=(("Metric", "FY2026"), ("Revenue", "215.9")),
        row_count=2,
        column_count=2,
        markdown="| Metric | FY2026 |",
    )
    element = DocumentElement(
        element_id="element-1",
        document_id="NVDA-FY2026-AR",
        ordinal=0,
        physical_page=1,
        element_type="table",
        bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
        original_text="Revenue table",
        table=table,
    )

    assert element.table == table
    with pytest.raises(ValueError, match="table"):
        DocumentElement(
            element_id="element-1",
            document_id="NVDA-FY2026-AR",
            ordinal=0,
            physical_page=1,
            element_type="table",
            bbox=BoundingBox(x0=0, y0=0, x1=1, y1=1),
            original_text="Revenue table",
        )
    with pytest.raises(ValueError, match="row_count"):
        TableMatrix(
            rows=(("Metric", "FY2026"),),
            row_count=2,
            column_count=2,
            markdown="| Metric | FY2026 |",
        )


def test_financial_chunk_requires_matching_source_element_ids_and_finite_score() -> None:
    chunk = FinancialChunk(
        chunk_id="chunk-1",
        text="Revenue increased.",
        content_hash=SHA256,
        element_type="paragraph",
        source_element_ids=("element-1",),
        context=sample_context(),
        financial=sample_financial(),
    )

    assert chunk.financial.source_element_ids == chunk.source_element_ids
    with pytest.raises(ValueError, match="source_element_ids"):
        FinancialChunk(
            **{
                **chunk.model_dump(),
                "financial": sample_financial(source_element_ids=("element-2",)),
            }
        )
    with pytest.raises(ValueError, match="finite"):
        sample_financial(confidence=float("nan"))


def test_verify_source_asset_fails_closed_for_size_or_hash(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(b"certified")
    source = sample_source(
        local_asset_key="report.pdf",
        byte_size=len(b"certified"),
        sha256="0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c",
    )

    with pytest.raises(SourceAssetError, match="SHA-256"):
        verify_source_asset(source, tmp_path)
    with pytest.raises(SourceAssetError, match="byte size"):
        verify_source_asset(source.model_copy(update={"byte_size": 1}), tmp_path)
