from __future__ import annotations

from pathlib import Path

from finai_academy.documents import DocumentSource
from finai_academy.naive_rag import naive_fixed_windows, naive_parse_html
from finai_academy.retrieval import LexicalRetriever


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_FILING = (
    ROOT / "assets/course-data/downloads/nvidia_fy2026_form_10k.html"
)
SOURCE = DocumentSource(
    source_id="NVDA-2026-10K-EXCERPT",
    company="NVIDIA",
    period="FY2026",
    document_type="Form 10-K",
    language="en",
    source_url=(
        "https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581026000021/nvda-20260125.htm"
    ),
)


def test_naive_parser_flattens_the_real_filing_and_reports_lost_structure() -> None:
    parsed = naive_parse_html(OFFICIAL_FILING)

    assert parsed.source_path == OFFICIAL_FILING
    assert parsed.table_count == 64
    assert parsed.semantic_heading_count == 0
    assert len(parsed.text) > 300_000
    assert "Revenue by End Market: (In millions) Data Center" in parsed.text
    assert "Data Center $ 193,737" in parsed.text


def test_naive_windows_use_reproducible_overlap_and_keep_provenance() -> None:
    chunks = naive_fixed_windows(
        "ABCDEFGHIJKL",
        SOURCE,
        chunk_chars=5,
        overlap_chars=2,
    )

    assert [chunk.passage_id for chunk in chunks] == [
        "NVDA-C001",
        "NVDA-C002",
        "NVDA-C003",
        "NVDA-C004",
    ]
    assert [chunk.text for chunk in chunks] == ["ABCDE", "DEFGH", "GHIJK", "JKL"]
    assert all(chunk.source_url == SOURCE.source_url for chunk in chunks)
    assert all(chunk.section == "Naive character window" for chunk in chunks)


def test_naive_retrieval_exposes_the_real_table_miss() -> None:
    parsed = naive_parse_html(OFFICIAL_FILING)
    chunks = naive_fixed_windows(parsed.text, SOURCE, chunk_chars=1_600, overlap_chars=200)
    ranking = LexicalRetriever(chunks).rank(
        "How large was Data Center revenue compared with total revenue in fiscal 2026?"
    )

    precise_table_ranks = [
        rank
        for rank, hit in enumerate(ranking, start=1)
        if "193,737" in hit.passage.text
    ]
    assert precise_table_ranks
    assert min(precise_table_ranks) > 2
