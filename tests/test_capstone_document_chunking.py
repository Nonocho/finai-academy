from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from finai_academy.capstone.document_assets import load_certified_document_sources
from finai_academy.capstone.document_chunking import (
    MissingFinancialContextError,
    build_financial_chunks,
    build_table_chunk,
)
from finai_academy.capstone.document_ingestion import PyMuPDF4LLMParser
from finai_academy.capstone.document_models import FinancialChunk, ParsedDocument

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets/course-data/manifest.json"


def _parsed_page(company_name: str, page: int) -> ParsedDocument:
    source = next(
        item
        for item in load_certified_document_sources(MANIFEST)
        if item.company_name == company_name
    )
    return PyMuPDF4LLMParser().parse(source, project_root=ROOT, pages=(page,))


@pytest.fixture(scope="module")
def nvidia_page_165() -> ParsedDocument:
    return _parsed_page("NVIDIA", 165)


@pytest.fixture(scope="module")
def schneider_page_16() -> ParsedDocument:
    return _parsed_page("Schneider Electric", 16)


def test_nvidia_table_chunk_keeps_value_headers_period_unit_and_lineage(
    nvidia_page_165: ParsedDocument,
) -> None:
    chunks = build_financial_chunks(nvidia_page_165)

    chunk = next(item for item in chunks if "$ 193,479" in item.text)
    assert chunk.element_type == "table"
    assert chunk.financial.currency == "USD"
    assert chunk.financial.scale == "millions"
    assert "FY2026" in chunk.financial.periods
    assert chunk.context.company_name == "NVIDIA"
    assert chunk.context.physical_page == 165
    assert chunk.context.printed_page == 77
    assert chunk.table is not None and chunk.table.column_count == 4
    assert chunk.source_element_ids
    assert chunk.content_hash == sha256(chunk.text.encode("utf-8")).hexdigest()


def test_schneider_fy_table_is_atomic_and_keeps_financial_scope(
    schneider_page_16: ParsedDocument,
) -> None:
    chunks = build_financial_chunks(schneider_page_16)

    chunk = next(item for item in chunks if "33,130" in item.text and "40,152" in item.text)
    assert chunk.element_type == "table"
    assert chunk.financial.currency == "EUR"
    assert chunk.financial.scale == "millions"
    assert chunk.financial.periods == ("FY2025",)
    assert chunk.financial.segments == (
        "Energy Management",
        "Industrial Automation",
        "Group",
    )
    assert chunk.financial.metric_names == (
        "Revenues € million",
        "Organic growth",
        "FY 2025 Changes in scope of consolidation",
        "Currency effect",
        "Reported growth",
    )
    assert "Organic growth" in chunk.text
    assert chunk.table is not None and chunk.table.row_count == 4


@pytest.mark.parametrize(
    ("value", "periods"),
    [
        ("11,095", ("Q4 2025",)),
        ("20,816", ("H2 2025",)),
        ("40,152", ("FY2025",)),
    ],
)
def test_schneider_table_periods_are_limited_to_the_table_context(
    schneider_page_16: ParsedDocument, value: str, periods: tuple[str, ...]
) -> None:
    chunks = build_financial_chunks(schneider_page_16)

    chunk = next(item for item in chunks if value in item.text)

    assert chunk.financial.periods == periods


def test_identical_extraction_produces_identical_chunk_ids(
    nvidia_page_165: ParsedDocument,
) -> None:
    first = build_financial_chunks(nvidia_page_165)
    second = build_financial_chunks(nvidia_page_165)

    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]


def test_paragraph_chunks_do_not_span_physical_pages(
    nvidia_page_165: ParsedDocument,
) -> None:
    first, second = nvidia_page_165.elements[:2]
    next_page_paragraph = second.model_copy(update={"physical_page": 166})
    document = nvidia_page_165.model_copy(
        update={"elements": (first, next_page_paragraph)}
    )

    chunks = build_financial_chunks(document)

    assert [(chunk.context.physical_page, chunk.source_element_ids) for chunk in chunks] == [
        (165, (first.element_id,)),
        (166, (next_page_paragraph.element_id,)),
    ]


def test_table_value_without_the_explicit_unit_fails_closed(
    schneider_page_16: ParsedDocument,
) -> None:
    element = next(
        item
        for item in schneider_page_16.elements
        if item.table is not None and "40,152" in item.original_text
    )
    broken = element.model_copy(
        update={
            "original_text": element.original_text.replace("€ million", ""),
            "original_markdown": element.original_markdown.replace("€ million", ""),
        }
    )

    with pytest.raises(MissingFinancialContextError, match="table unit is missing"):
        build_table_chunk(
            schneider_page_16.model_copy(update={"elements": (broken,)}),
            broken,
        )


def test_unitless_bulk_table_chunk_is_not_financially_contextualized(
    schneider_page_16: ParsedDocument,
) -> None:
    element = next(item for item in schneider_page_16.elements if "40,152" in item.original_text)
    broken = element.model_copy(
        update={
            "original_text": element.original_text.replace("€ million", ""),
            "original_markdown": element.original_markdown.replace("€ million", ""),
        }
    )
    document = schneider_page_16.model_copy(update={"elements": (broken,)})

    chunk = build_financial_chunks(document)[0]

    assert chunk.financial.scale is None
    assert chunk.financially_contextualized is False
    with pytest.raises(ValueError, match="unitless tables"):
        FinancialChunk.model_validate({**chunk.model_dump(mode="python"), "financially_contextualized": True})


@pytest.fixture
def valid_table_chunk(nvidia_page_165: ParsedDocument) -> FinancialChunk:
    element = next(item for item in nvidia_page_165.elements if item.table is not None)
    return build_table_chunk(nvidia_page_165, element)


@pytest.mark.parametrize(
    "field",
    ["company_name", "reporting_period", "physical_page", "source_element_ids"],
)
def test_numeric_table_chunk_requires_source_context(
    valid_table_chunk: FinancialChunk, field: str
) -> None:
    payload = valid_table_chunk.model_dump(mode="python")
    target = payload["context"] if field in payload["context"] else payload
    target[field] = None if field != "source_element_ids" else ()

    with pytest.raises(ValueError):
        FinancialChunk.model_validate(payload)
