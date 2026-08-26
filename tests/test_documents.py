from __future__ import annotations

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from finai_academy.documents import (
    DocumentSource,
    build_nvidia_fy2026_context_pack,
    load_source_manifest,
    parse_html,
    parse_pdf,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "assets" / "course-data" / "fixtures"


@pytest.fixture
def nvidia_source() -> DocumentSource:
    return DocumentSource(
        source_id="NVDA-2026-10K-EXCERPT",
        company="NVIDIA",
        period="FY2026",
        document_type="10-K teaching extract",
        language="en",
        source_url=(
            "https://www.sec.gov/Archives/edgar/data/1045810/"
            "000104581026000021/nvda-20260125.htm"
        ),
    )


@pytest.fixture
def schneider_source() -> DocumentSource:
    return DocumentSource(
        source_id="SU-2025-FY-EXCERPT",
        company="Schneider Electric",
        period="FY2025",
        document_type="Full-year results teaching extract",
        language="en",
        source_url=(
            "https://www.se.com/ww/en/assets/564/document/528237/"
            "release-fy-results-2025.pdf"
        ),
    )


def test_html_parser_preserves_heading_table_order_and_source(
    nvidia_source: DocumentSource,
) -> None:
    blocks = parse_html(FIXTURES / "nvidia_fy2026_excerpt.html", nvidia_source)

    assert [block.block_type for block in blocks[:3]] == ["heading", "paragraph", "table"]
    assert blocks[2].table_rows[0] == ("Business", "Revenue", "Year-on-year growth")
    assert blocks[2].table_rows[1] == ("Data Center", "$193.7 billion", "68%")
    assert [block.ordinal for block in blocks] == list(range(len(blocks)))
    assert all(block.source_url == nvidia_source.source_url for block in blocks)
    assert all(block.section_path for block in blocks[1:])


def test_pdf_parser_preserves_page_table_and_provenance(
    schneider_source: DocumentSource,
) -> None:
    blocks = parse_pdf(FIXTURES / "schneider_fy2025_excerpt.pdf", schneider_source)

    assert {block.page_number for block in blocks} == {1, 2}
    table = next(block for block in blocks if block.block_type == "table")
    assert table.page_number == 2
    assert table.table_rows[0] == ("Metric", "FY2025", "Change")
    assert table.table_rows[1] == ("Revenue", "EUR 40.2bn", "+8.9% organic")
    assert all(block.block_id for block in blocks)
    assert all(block.source_id == schneider_source.source_id for block in blocks)
    assert all(block.source_url == schneider_source.source_url for block in blocks)


def test_manifest_matches_fixture_hashes() -> None:
    records = load_source_manifest(ROOT / "assets" / "course-data" / "manifest.json")

    assert {record.source_id for record in records} == {
        "NVDA-2026-10K-EXCERPT",
        "SU-2025-FY-EXCERPT",
    }
    assert all(record.fixture_sha256 for record in records)
    assert all(record.verify_fixture(ROOT) for record in records)


def test_nvidia_source_versions_the_complete_official_filing() -> None:
    records = load_source_manifest(ROOT / "assets" / "course-data" / "manifest.json")
    nvidia = next(record for record in records if record.company == "NVIDIA")

    assert nvidia.document_type == "Form 10-K"
    assert nvidia.official_path == (
        "assets/course-data/downloads/nvidia_fy2026_form_10k.html"
    )
    assert nvidia.official_sha256
    assert nvidia.verify_official(ROOT)
    assert nvidia.accession_number == "0001045810-26-000021"

    filing = ROOT / nvidia.official_path
    assert filing.stat().st_size > 1_500_000
    filing_text = BeautifulSoup(
        filing.read_text(encoding="utf-8"), "html.parser"
    ).get_text(" ", strip=True)
    assert "FORM 10-K" in filing_text


def test_nvidia_context_pack_is_derived_from_the_official_filing() -> None:
    filing = FIXTURES.parent / "downloads" / "nvidia_fy2026_form_10k.html"

    pack = build_nvidia_fy2026_context_pack(filing)

    assert pack.source_path == filing
    assert pack.anchor_count == 2
    assert "[F1] Fiscal Year 2026 Summary" in pack.text
    assert "$215.9 billion" in pack.text
    assert "[F2] Revenue by End Market" in pack.text
    assert "Data Center $193,737 million" in pack.text
    assert "Gaming $16,042 million" in pack.text
    assert "not a reproduction of the full filing" not in pack.text
    assert len(pack.text) > 5_000


def test_source_rejects_missing_provenance() -> None:
    with pytest.raises(ValueError, match="source_url must not be empty"):
        DocumentSource(
            source_id="SOURCE",
            company="NVIDIA",
            period="FY2026",
            document_type="10-K",
            language="en",
            source_url=" ",
        )
