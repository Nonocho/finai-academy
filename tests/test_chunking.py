from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from finai_academy.chunking import (
    compare_chunking_strategies,
    contextual_enrich_chunks,
    contextualize_chunks,
    embedding_similarity_profile,
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


class RecordedEmbeddings:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        assert texts
        return self.vectors


class RecordedContextModel:
    def __init__(self, contexts: dict[str, str], *, malformed: bool = False) -> None:
        self.contexts = contexts
        self.malformed = malformed

    def invoke(self, messages: list[tuple[str, str]]):
        import json
        from types import SimpleNamespace

        if self.malformed:
            return SimpleNamespace(content="not-json")
        payload = json.loads(messages[-1][1])
        return SimpleNamespace(
            content=json.dumps({"context": self.contexts[payload["chunk_id"]]})
        )


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


def test_embedding_similarity_profile_uses_adjacent_sentence_vectors(
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

    sentences, similarities = embedding_similarity_profile(
        topic_blocks,
        RecordedEmbeddings([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]),
    )

    assert len(sentences) == 3
    assert len(similarities) == 2
    assert similarities[0] == pytest.approx(0.9701425)
    assert similarities[1] == pytest.approx(0.2425356)


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([], "one vector per sentence"),
        ([[1.0, 0.0], [1.0], [0.0, 1.0]], "equal dimensions"),
        ([[1.0, 0.0], [float("nan"), 1.0], [0.0, 1.0]], "finite"),
    ],
)
def test_embedding_similarity_profile_rejects_invalid_provider_vectors(
    blocks: list[DocumentBlock],
    vectors: list[list[float]],
    message: str,
) -> None:
    three_sentences = [
        replace(blocks[0], block_id=f"VALIDATION-B{index}", text=f"Sentence {index}.")
        for index in range(3)
    ]
    with pytest.raises(ValueError, match=message):
        embedding_similarity_profile(three_sentences, RecordedEmbeddings(vectors))


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


def test_contextual_enrichment_preserves_source_and_separates_generated_context(
    blocks: list[DocumentBlock],
) -> None:
    source_chunk = structure_aware_chunks(blocks, max_chars=220)[0]
    context = "This passage situates NVIDIA fiscal 2026 Data Center performance."

    enriched = contextual_enrich_chunks(
        document_text="NVIDIA FY2026 filing excerpt",
        chunks=[source_chunk],
        model=RecordedContextModel({source_chunk.chunk_id: context}),
    )

    assert enriched[0].chunk_id == source_chunk.chunk_id
    assert enriched[0].raw_text == source_chunk.text
    assert enriched[0].generated_context == context
    assert enriched[0].text == f"{context}\n\n{source_chunk.text}"
    assert enriched[0].source_block_ids == source_chunk.source_block_ids
    assert enriched[0].page_numbers == source_chunk.page_numbers
    assert len(enriched[0].text) > len(source_chunk.text)


def test_contextual_enrichment_rejects_malformed_json(
    blocks: list[DocumentBlock],
) -> None:
    source_chunk = structure_aware_chunks(blocks, max_chars=220)[0]

    with pytest.raises(ValueError, match="valid JSON"):
        contextual_enrich_chunks(
            document_text="NVIDIA FY2026 filing excerpt",
            chunks=[source_chunk],
            model=RecordedContextModel({}, malformed=True),
        )


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
