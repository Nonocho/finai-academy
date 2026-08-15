"""Contract tests for provider-neutral hybrid retrieval indexes."""

import json
from dataclasses import replace

import numpy as np
import pytest

from finai_academy.hybrid_retrieval import (
    DenseIndex,
    DeterministicTeachingEmbeddings,
    IndexVersionError,
    KeywordIndex,
    RetrievalFilters,
)


class HighMagnitudeEmbeddings:
    """Return finite vectors whose direct squared norm overflows."""

    vector = (1e308, -1e308)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(self.vector) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return list(self.vector)


def build_offline_dense_index(corpus):
    """Build an offline index with the stable Lesson 06 identity metadata."""

    embeddings = DeterministicTeachingEmbeddings()
    return DenseIndex(
        corpus,
        embeddings,
        provider="offline",
        model=embeddings.model_name,
        chunking_strategy="contextual-structure",
    )


def write_empty_vectors(path):
    """Replace a vector artifact with an empty invalid file."""

    path.write_bytes(b"")


def write_npz_vectors(path):
    """Replace a vector artifact with an archive rather than one matrix."""

    with path.open("wb") as vector_file:
        np.savez(vector_file, vectors=np.zeros((1, 1)))


def test_dense_index_round_trip_preserves_vectors_and_version(corpus, tmp_path):
    """Persisted vectors must serve the same corpus and version identity."""

    original = build_offline_dense_index(corpus)
    original.save(tmp_path)

    restored = DenseIndex.from_artifact(
        tmp_path,
        corpus,
        DeterministicTeachingEmbeddings(),
        expected_version=original.version,
    )

    assert np.array_equal(restored.document_matrix, original.document_matrix)
    assert restored.version == original.version


def test_dense_index_version_hash_distinguishes_delimiter_containing_fields(corpus):
    """Corpus identity must not merge distinct fields containing old delimiters."""

    first_passage = corpus[0]
    first_corpus = (
        replace(first_passage, passage_id="first\x1fsecond", company="third"),
        *corpus[1:],
    )
    second_corpus = (
        replace(first_passage, passage_id="first", company="second\x1fthird"),
        *corpus[1:],
    )

    first_index = build_offline_dense_index(first_corpus)
    second_index = build_offline_dense_index(second_corpus)

    assert first_index.version.corpus_hash != second_index.version.corpus_hash


def test_document_matrix_is_defensive_and_cannot_diverge_from_persisted_vectors(corpus, tmp_path):
    """Public matrix mutation or reassignment must not alter index behavior or artifacts."""

    index = build_offline_dense_index(corpus)
    expected_matrix = index.document_matrix.copy()
    expected_hits = index.search("data center revenue", top_k=2)

    exposed_matrix = index.document_matrix
    exposed_matrix.fill(0.0)

    assert np.array_equal(index.document_matrix, expected_matrix)
    with pytest.raises(AttributeError):
        index.document_matrix = np.zeros_like(expected_matrix)

    index.save(tmp_path)
    restored = DenseIndex.from_artifact(
        tmp_path,
        corpus,
        DeterministicTeachingEmbeddings(),
        expected_version=index.version,
    )

    assert restored.search("data center revenue", top_k=2) == expected_hits
    assert np.array_equal(restored.document_matrix, expected_matrix)


@pytest.mark.parametrize("write_vectors", [write_empty_vectors, write_npz_vectors])
def test_loading_rejects_non_matrix_vector_artifacts(corpus, tmp_path, write_vectors):
    """Malformed and archive vector files must become domain-specific load errors."""

    index = build_offline_dense_index(corpus)
    index.save(tmp_path)
    write_vectors(tmp_path / "vectors.npy")

    with pytest.raises(IndexVersionError, match="vectors.npy"):
        DenseIndex.from_artifact(
            tmp_path,
            corpus,
            DeterministicTeachingEmbeddings(),
            expected_version=index.version,
        )


@pytest.mark.parametrize("invalid_schema_version", [True, 1.0])
def test_loading_rejects_non_integer_schema_version(corpus, tmp_path, invalid_schema_version):
    """Schema versions must be literal integers, never equivalent JSON values."""

    index = build_offline_dense_index(corpus)
    index.save(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = invalid_schema_version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IndexVersionError, match="schema_version"):
        DenseIndex.from_artifact(
            tmp_path,
            corpus,
            DeterministicTeachingEmbeddings(),
            expected_version=index.version,
        )


@pytest.mark.parametrize("invalid_schema_version", [True, 1.0])
def test_loading_rejects_non_integer_expected_schema_version(
    corpus, tmp_path, invalid_schema_version
):
    """Caller-provided index versions require the same literal schema type."""

    index = build_offline_dense_index(corpus)
    index.save(tmp_path)
    invalid_version = replace(index.version, schema_version=invalid_schema_version)

    with pytest.raises(IndexVersionError, match="schema_version"):
        DenseIndex.from_artifact(
            tmp_path,
            corpus,
            DeterministicTeachingEmbeddings(),
            expected_version=invalid_version,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("provider", "different-provider"),
        ("model", "different-model"),
        ("dimension", 99),
        ("corpus_hash", "0" * 64),
        ("chunking_strategy", "fixed"),
    ],
)
def test_loading_rejects_an_incompatible_index(corpus, tmp_path, field, replacement):
    """Expected identity fields must reject mismatched persisted indexes."""

    index = build_offline_dense_index(corpus)
    index.save(tmp_path)
    incompatible = replace(index.version, **{field: replacement})

    with pytest.raises(IndexVersionError, match=field):
        DenseIndex.from_artifact(
            tmp_path,
            corpus,
            DeterministicTeachingEmbeddings(),
            expected_version=incompatible,
        )


def test_company_and_period_filters_block_cross_company_candidates(corpus):
    index = DenseIndex(
        corpus,
        DeterministicTeachingEmbeddings(),
        provider="offline",
        model="financial-concepts-v1",
        chunking_strategy="contextual-structure",
    )

    hits = index.search(
        "data centre energy demand growth",
        top_k=3,
        filters=RetrievalFilters(company="NVIDIA", period="FY2026"),
    )

    assert hits
    assert {hit.passage.company for hit in hits} == {"NVIDIA"}
    assert {hit.passage.period for hit in hits} == {"FY2026"}


def test_keyword_retrieval_preserves_an_exact_financial_figure(corpus):
    hits = KeywordIndex(corpus).search("18.7% margin", top_k=1)

    assert hits[0].passage.company == "Schneider Electric"
    assert "18.7%" in hits[0].passage.text


@pytest.mark.parametrize(
    ("filters", "expected_ids"),
    [
        (RetrievalFilters(company="NVIDIA"), {"NVDA-TABLE", "NVDA-CONCENTRATION"}),
        (RetrievalFilters(period="FY2025"), {"SU-TABLE", "SU-PARSING"}),
        (
            RetrievalFilters(document_type="10-K teaching extract"),
            {"NVDA-TABLE", "NVDA-CONCENTRATION"},
        ),
        (RetrievalFilters(section="Concentration question"), {"NVDA-CONCENTRATION"}),
    ],
)
def test_each_supported_metadata_filter_is_enforced(corpus, filters, expected_ids):
    hits = KeywordIndex(corpus).search("revenue growth", top_k=10, filters=filters)

    assert {hit.passage.passage_id for hit in hits} == expected_ids


def test_equal_dense_scores_use_passage_id_as_stable_tie_break(corpus):
    index = DenseIndex(
        corpus,
        DeterministicTeachingEmbeddings(),
        provider="offline",
        model="financial-concepts-v1",
        chunking_strategy="contextual-structure",
    )

    hits = index.search("vocabulary outside the teaching concepts", top_k=len(corpus))

    tied_ids = [hit.passage.passage_id for hit in hits if hit.score == 0.5]
    assert tied_ids == sorted(tied_ids)


def test_identical_high_magnitude_vectors_have_cosine_similarity_one(corpus):
    index = DenseIndex(
        corpus[:1],
        HighMagnitudeEmbeddings(),
        provider="offline",
        model="teaching-test",
        chunking_strategy="test",
    )

    (_, cosine), = index.cosine_scores("finite high-magnitude vector")

    assert cosine == pytest.approx(1.0)
