from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from finai_academy.capstone.document_index import (
    CertifiedDocumentIndexError,
    load_certified_document_index,
)
from finai_academy.capstone.document_models import DocumentFilters

ROOT = Path(__file__).resolve().parents[1]


def test_company_period_and_table_filters_run_before_ranking() -> None:
    """Removing the eligibility boundary must leak non-Schneider table evidence."""

    index = load_certified_document_index(ROOT)

    hits = index.search(
        "reported revenue organic growth",
        filters=DocumentFilters(
            company_name="Schneider Electric",
            reporting_period="FY2025",
            element_type="table",
        ),
        top_k=3,
    )

    assert hits
    assert all(hit.chunk.context.company_name == "Schneider Electric" for hit in hits)
    assert all(hit.chunk.context.reporting_period == "FY2025" for hit in hits)
    assert all(hit.chunk.element_type == "table" for hit in hits)
    assert "40,152" in hits[0].chunk.text
    assert {name for name, _rank in hits[0].channel_ranks} == {"bm25", "dense"}
    assert hits[0].fused_score > 0
    assert (
        hits[0].selection_reason
        == "Matched exact financial terms and related document meaning."
    )


def test_exact_nvidia_figure_retrieves_the_atomic_segment_table() -> None:
    """Dropping numeric BM25 terms must stop the exact segment evidence from winning."""

    index = load_certified_document_index(ROOT)

    hit = index.search(
        "Which NVIDIA business generated 193,479 million?",
        filters=DocumentFilters(company_name="NVIDIA", element_type="table"),
        top_k=1,
    )[0]

    assert "$ 193,479" in hit.chunk.text
    assert hit.chunk.table is not None
    assert "Compute" in hit.chunk.table.rows[0][1]
    assert index.inspect(hit.chunk.chunk_id) == hit.chunk


def test_search_never_returns_a_unitless_table_as_contextual_financial_evidence() -> None:
    """Removing the contextualization boundary must expose unitless numeric table hits."""

    index = load_certified_document_index(ROOT)

    hits = index.search(
        "definitions business overview proxy statement 2026",
        filters=DocumentFilters(company_name="NVIDIA", element_type="table"),
        top_k=5,
    )

    assert hits
    assert all(hit.chunk.financially_contextualized for hit in hits)


@pytest.mark.parametrize(
    ("query", "top_k", "message"),
    [
        ("   ", 1, "query must not be blank"),
        ("revenue", 0, "top_k must be between 1 and 5"),
        ("revenue", True, "top_k must be between 1 and 5"),
        ("revenue", 6, "top_k must be between 1 and 5"),
    ],
)
def test_search_rejects_invalid_public_query_arguments(
    query: str, top_k: int, message: str
) -> None:
    """Relaxing public query validation must fail with a stable index error."""

    index = load_certified_document_index(ROOT)

    with pytest.raises(CertifiedDocumentIndexError, match=message):
        index.search(query, filters=DocumentFilters(), top_k=top_k)


def test_loader_rejects_a_chunk_artifact_whose_bytes_do_not_match_its_manifest(
    tmp_path: Path,
) -> None:
    """Skipping artifact verification must allow tampered certified evidence to load."""

    artifact_path = tmp_path / "assets/course-data/capstone/financial_chunks_v2.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(
        (ROOT / "assets/course-data/capstone/financial_chunks_v2.json").read_bytes()
    )
    manifest_path = tmp_path / "assets/course-data/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "capstone_derived_artifacts": [
                    {
                        "schema_version": 2,
                        "chunks": {
                            "path": "assets/course-data/capstone/financial_chunks_v2.json",
                            "sha256": "0" * 64,
                        },
                        "source_sha256s": [
                            "0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c",
                            "5b17e6e8e5b8c47f697f5d352ed9639b98b261d9ecf886d3e8ce147493ebb00a",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CertifiedDocumentIndexError, match="chunk artifact SHA-256 mismatch"):
        load_certified_document_index(tmp_path)


def test_loader_rejects_chunks_with_source_hashes_missing_from_the_manifest(
    tmp_path: Path,
) -> None:
    """Ignoring source identity must allow chunks from uncertified documents to load."""

    artifact_path = tmp_path / "assets/course-data/capstone/financial_chunks_v2.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_bytes = (
        ROOT / "assets/course-data/capstone/financial_chunks_v2.json"
    ).read_bytes()
    artifact_path.write_bytes(artifact_bytes)
    manifest_path = tmp_path / "assets/course-data/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "capstone_derived_artifacts": [
                    {
                        "schema_version": 2,
                        "chunks": {
                            "path": "assets/course-data/capstone/financial_chunks_v2.json",
                            "sha256": sha256(artifact_bytes).hexdigest(),
                        },
                        "source_sha256s": ["0" * 64],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CertifiedDocumentIndexError, match="source hashes do not match"):
        load_certified_document_index(tmp_path)


def test_loader_rejects_duplicate_chunk_ids_before_constructing_an_index(tmp_path: Path) -> None:
    """Omitting duplicate detection must permit ambiguous evidence inspection."""

    chunks = json.loads(
        (ROOT / "assets/course-data/capstone/financial_chunks_v2.json").read_text(
            encoding="utf-8"
        )
    )
    chunks.append(chunks[0])
    artifact_path = tmp_path / "assets/course-data/capstone/financial_chunks_v2.json"
    artifact_path.parent.mkdir(parents=True)
    artifact_bytes = json.dumps(chunks, sort_keys=True, separators=(",", ":")).encode("utf-8")
    artifact_path.write_bytes(artifact_bytes)
    manifest_path = tmp_path / "assets/course-data/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "capstone_derived_artifacts": [
                    {
                        "schema_version": 2,
                        "chunks": {
                            "path": "assets/course-data/capstone/financial_chunks_v2.json",
                            "sha256": sha256(artifact_bytes).hexdigest(),
                        },
                        "source_sha256s": [
                            "0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c",
                            "5b17e6e8e5b8c47f697f5d352ed9639b98b261d9ecf886d3e8ce147493ebb00a",
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CertifiedDocumentIndexError, match="duplicate chunk_id"):
        load_certified_document_index(tmp_path)
