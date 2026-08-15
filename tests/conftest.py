"""Shared evidence fixtures for the Lesson 06 hybrid retrieval tests."""

import pytest

from finai_academy.hybrid_retrieval import IndexedPassage


@pytest.fixture
def corpus() -> tuple[IndexedPassage, ...]:
    """Return the compact Lesson 05 teaching corpus with provenance."""

    return (
        IndexedPassage(
            passage_id="NVDA-TABLE",
            company="NVIDIA",
            period="FY2026",
            section="Revenue by business",
            text=(
                "Data Center revenue was $193.7 billion with 68% growth; "
                "Gaming revenue was $16.0 billion with 41% growth."
            ),
            source_url=(
                "https://www.sec.gov/Archives/edgar/data/1045810/"
                "000104581026000021/nvda-20260125.htm"
            ),
            document_type="10-K teaching extract",
        ),
        IndexedPassage(
            passage_id="NVDA-CONCENTRATION",
            company="NVIDIA",
            period="FY2026",
            section="Concentration question",
            text="Data Center represented most of total revenue and reported expansion.",
            source_url=(
                "https://www.sec.gov/Archives/edgar/data/1045810/"
                "000104581026000021/nvda-20260125.htm"
            ),
            document_type="10-K teaching extract",
        ),
        IndexedPassage(
            passage_id="SU-TABLE",
            company="Schneider Electric",
            period="FY2025",
            section="Key financial metrics",
            text=(
                "Revenue was EUR 40.2bn with 8.9% organic growth; Energy Management "
                "grew 10% organically; adjusted EBITA was EUR 7.5bn at an 18.7% margin."
            ),
            source_url=(
                "https://www.se.com/ww/en/assets/564/document/528237/"
                "release-fy-results-2025.pdf"
            ),
            document_type="Full-year results teaching extract",
        ),
        IndexedPassage(
            passage_id="SU-PARSING",
            company="Schneider Electric",
            period="FY2025",
            section="Key financial metrics",
            text="A naive character split can separate EUR 40.2bn from Revenue.",
            source_url=(
                "https://www.se.com/ww/en/assets/564/document/528237/"
                "release-fy-results-2025.pdf"
            ),
            document_type="Full-year results teaching extract",
        ),
    )
