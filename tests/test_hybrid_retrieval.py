"""Contract tests for provider-neutral hybrid retrieval indexes."""

import json
from dataclasses import replace

import numpy as np
import pytest

import finai_academy.retrieval as retrieval_module
from finai_academy.hybrid_retrieval import (
    BM25Index,
    DenseIndex,
    DeterministicTeachingEmbeddings,
    IndexVersionError,
    KeywordIndex,
    RetrievalFilters,
    reciprocal_rank_fusion,
)
from finai_academy.retrieval import RetrievalHit


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


def test_bm25_retrieval_preserves_an_exact_financial_figure(corpus):
    """Removing numeric lexical matching must make this financial query fail."""

    hits = BM25Index(corpus).search("18.7% margin", top_k=1)

    assert hits[0].passage.company == "Schneider Electric"
    assert "18.7%" in hits[0].passage.text
    assert hits[0].score > 0


def test_bm25_length_normalization_prefers_the_focused_passage(corpus):
    """Removing BM25 length normalization must let a padded passage tie or win."""

    focused = replace(
        corpus[0],
        passage_id="FOCUSED",
        text="revenue margin",
    )
    padded = replace(
        corpus[1],
        passage_id="PADDED",
        text="revenue margin " + "background " * 80,
    )

    hits = BM25Index((padded, focused)).search("revenue margin", top_k=2)

    assert [hit.passage.passage_id for hit in hits] == ["FOCUSED", "PADDED"]
    assert hits[0].score > hits[1].score


def test_bm25_applies_metadata_eligibility_before_ranking(corpus):
    """Removing the eligibility boundary must leak another company's evidence."""

    hits = BM25Index(corpus).search(
        "revenue growth margin",
        top_k=10,
        filters=RetrievalFilters(company="NVIDIA", period="FY2026"),
    )

    assert hits
    assert {hit.passage.company for hit in hits} == {"NVIDIA"}
    assert {hit.passage.period for hit in hits} == {"FY2026"}


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


def test_keyword_filter_limits_the_matrix_rows_passed_to_cosine_scoring(
    corpus, monkeypatch
):
    """Ineligible TF-IDF rows must never reach the cosine scorer."""

    scored_row_counts = []
    real_cosine_similarity = retrieval_module.cosine_similarity

    def record_scored_rows(query_vector, document_matrix):
        scored_row_counts.append(document_matrix.shape[0])
        return real_cosine_similarity(query_vector, document_matrix)

    monkeypatch.setattr(retrieval_module, "cosine_similarity", record_scored_rows)

    hits = KeywordIndex(corpus).search(
        "revenue growth",
        top_k=4,
        filters=RetrievalFilters(company="NVIDIA", period="FY2026"),
    )

    assert scored_row_counts == [2]
    assert {hit.passage.company for hit in hits} == {"NVIDIA"}


@pytest.mark.parametrize("index_kind", ["keyword", "dense"])
def test_public_indexes_reject_boolean_top_k(corpus, index_kind):
    """Boolean values must not pass the public integer result-budget contract."""

    index = KeywordIndex(corpus) if index_kind == "keyword" else build_offline_dense_index(corpus)

    with pytest.raises(ValueError, match="top_k"):
        index.search("revenue", top_k=True)


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


def test_rrf_deduplicates_and_preserves_channel_ranks(corpus):
    """Fusion must add one contribution per ranked channel for a shared passage."""

    lexical = [
        RetrievalHit(passage=corpus[0], score=0.9),
        RetrievalHit(passage=corpus[1], score=0.8),
    ]
    dense = [
        RetrievalHit(passage=corpus[1], score=0.95),
        RetrievalHit(passage=corpus[2], score=0.7),
    ]

    fused = reciprocal_rank_fusion(
        {"keyword": lexical, "dense": dense},
        k=60,
        weights={"keyword": 1.0, "dense": 1.0},
    )

    shared = next(hit for hit in fused if hit.passage.passage_id == corpus[1].passage_id)
    assert dict(shared.channel_ranks) == {"dense": 1, "keyword": 2}
    assert shared.rrf_score == pytest.approx(1 / 61 + 1 / 62)
    assert len({hit.passage.passage_id for hit in fused}) == len(fused)


def test_rrf_weight_changes_the_controlled_top_result(corpus):
    """Changing a channel weight must change the winner when ranks are opposed."""

    rankings = {
        "keyword": [
            RetrievalHit(passage=corpus[0], score=0.9),
            RetrievalHit(passage=corpus[1], score=0.8),
        ],
        "dense": [
            RetrievalHit(passage=corpus[1], score=0.95),
            RetrievalHit(passage=corpus[0], score=0.7),
        ],
    }

    keyword_heavy = reciprocal_rank_fusion(
        rankings, k=60, weights={"keyword": 2.0, "dense": 1.0}
    )
    dense_heavy = reciprocal_rank_fusion(
        rankings, k=60, weights={"keyword": 1.0, "dense": 2.0}
    )

    assert keyword_heavy[0].passage.passage_id != dense_heavy[0].passage.passage_id


def test_rrf_equal_scores_use_passage_id_as_stable_tie_break(corpus):
    """Equal RRF sums must not depend on mapping or retrieval order."""

    rankings = {
        "keyword": [RetrievalHit(passage=corpus[0], score=0.9)],
        "dense": [RetrievalHit(passage=corpus[1], score=0.9)],
    }

    fused = reciprocal_rank_fusion(rankings, k=60)

    assert [hit.passage.passage_id for hit in fused] == sorted(
        [corpus[0].passage_id, corpus[1].passage_id]
    )


@pytest.mark.parametrize(
    ("rankings", "k", "weights", "message"),
    [
        ({}, 60, None, "rankings"),
        ({"keyword": []}, 60, None, "keyword"),
        ({"keyword": []}, 0, None, "k"),
        ({"keyword": []}, 60, {"dense": 1.0}, "unknown"),
        ({"keyword": []}, 60, {"keyword": -1.0}, "negative"),
    ],
)
def test_rrf_rejects_invalid_ranking_inputs(rankings, k, weights, message):
    """Fusion must fail rather than silently turn malformed rank lists into evidence."""

    with pytest.raises(ValueError, match=message):
        reciprocal_rank_fusion(rankings, k=k, weights=weights)


def test_rrf_rejects_duplicate_ids_and_conflicting_passage_content(corpus):
    """A passage ID must have one occurrence per channel and one canonical record."""

    duplicate_channel = [
        RetrievalHit(passage=corpus[0], score=0.9),
        RetrievalHit(passage=corpus[0], score=0.8),
    ]
    conflicting_passage = replace(corpus[0], text="Different text with the same identifier.")

    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank_fusion({"keyword": duplicate_channel})
    with pytest.raises(ValueError, match="inconsistent"):
        reciprocal_rank_fusion(
            {
                "keyword": [RetrievalHit(passage=corpus[0], score=0.9)],
                "dense": [RetrievalHit(passage=conflicting_passage, score=0.8)],
            }
        )
