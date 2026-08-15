"""Contract tests for provider-neutral hybrid retrieval indexes."""

import pytest

from finai_academy.hybrid_retrieval import (
    DenseIndex,
    DeterministicTeachingEmbeddings,
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
