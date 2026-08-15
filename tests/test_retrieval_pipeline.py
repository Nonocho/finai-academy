"""Integration tests for the metadata-safe hybrid retrieval boundary."""

import pytest

from finai_academy.hybrid_retrieval import (
    DenseIndex,
    DeterministicTeachingEmbeddings,
    KeywordIndex,
    RetrievalFilters,
)
from finai_academy.retrieval_pipeline import retrieve_evidence


def build_indexes(corpus):
    """Build the offline indexes used by the complete pipeline contract."""

    embeddings = DeterministicTeachingEmbeddings()
    return (
        KeywordIndex(corpus),
        DenseIndex(
            corpus,
            embeddings,
            provider="offline",
            model=embeddings.model_name,
            chunking_strategy="contextual-structure",
        ),
    )


def test_pipeline_abstains_instead_of_broadening_empty_filters(corpus):
    """An impossible metadata constraint must return an explicit empty result."""

    keyword_index, dense_index = build_indexes(corpus)

    result = retrieve_evidence(
        "What was revenue?",
        keyword_index=keyword_index,
        dense_index=dense_index,
        filters=RetrievalFilters(company="Absent Company"),
        candidate_k=4,
        final_k=2,
    )

    assert result.reranked_hits == ()
    assert result.abstention_reason == "No passages matched the requested metadata filters."


def test_pipeline_abstention_checks_eligibility_before_searching_indexes(corpus):
    """The no-match safety barrier must prevent either retrieval stage from running."""

    class SearchForbiddenIndex:
        def __init__(self, passages):
            self.passages = passages

        def search(self, *args, **kwargs):
            raise AssertionError("search must not run when no passage is eligible")

    result = retrieve_evidence(
        "What was revenue?",
        keyword_index=SearchForbiddenIndex(corpus),
        dense_index=SearchForbiddenIndex(corpus),
        filters=RetrievalFilters(company="Absent Company"),
        candidate_k=4,
        final_k=2,
    )

    assert result.abstention_reason == "No passages matched the requested metadata filters."


def test_pipeline_keeps_all_rankings_visible_after_pre_filtering(corpus):
    """Results must show the filtered channel, fusion, and final evidence stages."""

    keyword_index, dense_index = build_indexes(corpus)

    result = retrieve_evidence(
        "What was revenue?",
        keyword_index=keyword_index,
        dense_index=dense_index,
        filters=RetrievalFilters(company="Schneider Electric"),
        candidate_k=4,
        final_k=1,
    )

    assert result.keyword_hits and result.dense_hits and result.fused_hits
    assert len(result.reranked_hits) == 1
    assert result.abstention_reason is None
    assert {hit.passage.company for hit in result.fused_hits} == {"Schneider Electric"}


def test_pipeline_does_not_abstain_when_only_the_dense_index_has_eligible_passages(corpus):
    """Eligibility must cover both channel corpora before the pipeline abstains."""

    embeddings = DeterministicTeachingEmbeddings()
    keyword_index = KeywordIndex(corpus[:1])
    dense_index = DenseIndex(
        corpus[2:],
        embeddings,
        provider="offline",
        model=embeddings.model_name,
        chunking_strategy="contextual-structure",
    )

    result = retrieve_evidence(
        "What margin reached 18.7%?",
        keyword_index=keyword_index,
        dense_index=dense_index,
        filters=RetrievalFilters(company="Schneider Electric"),
        candidate_k=2,
        final_k=1,
    )

    assert result.abstention_reason is None
    assert result.keyword_hits == ()
    assert result.dense_hits
    assert result.reranked_hits[0].passage.company == "Schneider Electric"


@pytest.mark.parametrize(("candidate_k", "final_k"), [(0, 1), (1, 0), (1, 2)])
def test_pipeline_rejects_an_invalid_candidate_and_final_budget(corpus, candidate_k, final_k):
    """The wider candidate stage cannot be smaller than the final evidence budget."""

    keyword_index, dense_index = build_indexes(corpus)

    with pytest.raises(ValueError, match="candidate_k"):
        retrieve_evidence(
            "What was revenue?",
            keyword_index=keyword_index,
            dense_index=dense_index,
            filters=RetrievalFilters(),
            candidate_k=candidate_k,
            final_k=final_k,
        )
