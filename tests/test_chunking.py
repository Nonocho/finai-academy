from __future__ import annotations

from pathlib import Path

import pytest

from finai_academy.chunking import (
    compare_chunking_strategies,
    contextualize_chunks,
    fixed_chunks,
    hierarchical_chunks,
    proposition_chunks,
    recursive_chunks,
    semantic_chunks,
    structure_aware_chunks,
)
from finai_academy.documents import DocumentBlock, DocumentSource, parse_html
from finai_academy.lesson_support import RecordedChunkingModel

ROOT = Path(__file__).resolve().parents[1]
HTML_FIXTURE = ROOT / "assets" / "course-data" / "fixtures" / "nvidia_fy2026_excerpt.html"


@pytest.fixture
def source() -> DocumentSource:
    return DocumentSource(
        source_id="NVDA-2026-10K-EXCERPT",
        company="NVIDIA",
        period="FY2026",
        document_type="10-K teaching extract",
        language="en",
        source_url="https://www.sec.gov/example/nvidia.htm",
    )


@pytest.fixture
def blocks(source: DocumentSource) -> list[DocumentBlock]:
    return parse_html(HTML_FIXTURE, source)


def test_fixed_size_reproduces_table_split(blocks: list[DocumentBlock]) -> None:
    chunks = fixed_chunks(blocks, chunk_size=50, overlap=0)

    assert not any(
        "Business | Revenue | Year-on-year growth" in chunk.text
        and "Data Center | $193.7 billion | 68%" in chunk.text
        for chunk in chunks
    )
    assert all(chunk.source_block_ids for chunk in chunks)


def test_fixed_size_rejects_overlap_that_cannot_advance(blocks: list[DocumentBlock]) -> None:
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        fixed_chunks(blocks, chunk_size=80, overlap=80)


def test_recursive_chunks_respect_boundaries_and_size(blocks: list[DocumentBlock]) -> None:
    chunks = recursive_chunks(blocks, max_chars=180)

    assert all(len(chunk.text) <= 180 for chunk in chunks)
    assert all(chunk.source_block_ids for chunk in chunks)
    assert any(chunk.text.endswith("year.") for chunk in chunks)


def test_structure_aware_chunking_keeps_table_atomic(blocks: list[DocumentBlock]) -> None:
    chunks = structure_aware_chunks(blocks, max_chars=220)

    table_chunk = next(chunk for chunk in chunks if "Data Center" in chunk.text)
    assert "Business | Revenue | Year-on-year growth" in table_chunk.text
    assert "Gaming | $16.0 billion | 41%" in table_chunk.text
    assert len(table_chunk.source_block_ids) == 1
    assert table_chunk.section_path == ("Revenue by business",)


def test_semantic_boundary_uses_supplied_similarity_profile(
    source: DocumentSource,
) -> None:
    topic_blocks = [
        DocumentBlock(
            block_id=f"B{index}",
            source_id=source.source_id,
            company=source.company,
            period=source.period,
            document_type=source.document_type,
            language=source.language,
            source_url=source.source_url,
            ordinal=index,
            block_type="paragraph",
            text=text,
            section_path=("Performance",),
        )
        for index, text in enumerate(
            (
                "Revenue increased because Data Center demand remained strong.",
                "Data Center represented most of the reported revenue expansion.",
                "Supply concentration remains a separate operating risk.",
            )
        )
    ]

    chunks = semantic_chunks(topic_blocks, threshold=0.20, similarities=[0.72, 0.08])

    assert len(chunks) == 2
    assert "reported revenue expansion" in chunks[0].text
    assert chunks[1].text == "Supply concentration remains a separate operating risk."


def test_hierarchical_children_link_to_existing_parent(blocks: list[DocumentBlock]) -> None:
    chunks = hierarchical_chunks(blocks, child_max_chars=130)
    parent_ids = {chunk.chunk_id for chunk in chunks if chunk.role == "parent"}
    children = [chunk for chunk in chunks if chunk.role == "child"]

    assert parent_ids
    assert children
    assert {child.parent_id for child in children} <= parent_ids
    assert all(child.section_path for child in children)


def test_contextualization_adds_scope_without_losing_raw_text(
    blocks: list[DocumentBlock],
) -> None:
    base = structure_aware_chunks(blocks, max_chars=220)
    contextual = contextualize_chunks(base)

    assert contextual[0].text.startswith("NVIDIA | FY2026 | 10-K teaching extract")
    assert contextual[0].raw_text == base[0].text
    assert contextual[0].source_block_ids == base[0].source_block_ids


def test_recorded_propositions_preserve_source_identifiers(
    blocks: list[DocumentBlock],
) -> None:
    paragraph = next(block for block in blocks if block.block_type == "paragraph")
    chunks = proposition_chunks([paragraph], RecordedChunkingModel())

    assert len(chunks) == 2
    assert all(chunk.strategy == "llm_proposition" for chunk in chunks)
    assert all(chunk.source_block_ids == (paragraph.block_id,) for chunk in chunks)
    assert all(chunk.source_url == paragraph.source_url for chunk in chunks)


def test_comparison_scorecard_reports_integrity_and_provenance(
    blocks: list[DocumentBlock],
) -> None:
    strategies = {
        "fixed": fixed_chunks(blocks, chunk_size=50, overlap=0),
        "structure": structure_aware_chunks(blocks, max_chars=220),
    }

    scorecard = compare_chunking_strategies(strategies, blocks)

    fixed = scorecard.loc[scorecard["strategy"] == "fixed"].iloc[0]
    structured = scorecard.loc[scorecard["strategy"] == "structure"].iloc[0]
    assert fixed["table_integrity"] == 0.0
    assert structured["table_integrity"] == 1.0
    assert structured["provenance_completeness"] == 1.0
