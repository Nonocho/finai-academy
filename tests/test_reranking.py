"""Contract tests for the transparent Lesson 06 reranker."""

from dataclasses import replace

import pytest

from finai_academy.hybrid_retrieval import FusedHit
from finai_academy.reranking import RERANK_FEATURE_WEIGHTS, rerank_candidates


def test_reranker_rewards_exact_numeric_evidence(corpus):
    """An exact requested number must outrank a slightly stronger fused non-match."""

    candidates = [
        FusedHit(
            passage=corpus[0],
            rrf_score=0.032,
            channel_ranks=(("dense", 1), ("keyword", 2)),
        ),
        FusedHit(
            passage=corpus[2],
            rrf_score=0.031,
            channel_ranks=(("dense", 2), ("keyword", 1)),
        ),
    ]

    reranked = rerank_candidates("What margin reached 18.7%?", candidates, top_k=1)

    assert reranked[0].passage.company == "Schneider Electric"
    assert reranked[0].features.numeric_coverage == 1.0
    assert "18.7%" in reranked[0].passage.text


def test_reranker_uses_fusion_score_then_passage_id_for_stable_ties(corpus):
    """Equal feature scores must retain fusion ordering before using the identifier."""

    candidates = [
        FusedHit(corpus[1], 0.02, (("dense", 1),)),
        FusedHit(corpus[0], 0.03, (("keyword", 1),)),
    ]

    reranked = rerank_candidates("unmatched query", candidates, top_k=2)

    assert [hit.passage.passage_id for hit in reranked] == ["NVDA-TABLE", "NVDA-CONCENTRATION"]
    assert all(0.0 <= value <= 1.0 for value in reranked[0].features.__dict__.values())


def test_reranker_requires_a_positive_result_budget(corpus):
    """A non-positive evidence budget cannot create a meaningful final ranking."""

    candidates = [FusedHit(corpus[0], 0.02, (("keyword", 1),))]

    with pytest.raises(ValueError, match="top_k"):
        rerank_candidates("revenue", candidates, top_k=0)


def test_reranker_preserves_sentence_final_and_thousands_separated_numbers(corpus):
    """Exact financial figures must survive surrounding punctuation and grouping commas."""

    punctuated = replace(
        corpus[0],
        text="Revenue was $1,234.56. The margin reached 18.7%.",
    )
    candidates = [FusedHit(punctuated, 0.02, (("keyword", 1),))]

    reranked = rerank_candidates(
        "Did revenue reach $1,234.56 and margin reach 18.7%?", candidates, top_k=1
    )

    assert reranked[0].features.numeric_coverage == 1.0


@pytest.mark.parametrize("rrf_score", [-0.01, float("inf"), float("nan")])
def test_fused_hit_rejects_scores_that_cannot_produce_normalized_features(corpus, rrf_score):
    """Reranking requires finite, non-negative fusion signals at its input boundary."""

    with pytest.raises(ValueError, match="rrf_score"):
        FusedHit(corpus[0], rrf_score, (("keyword", 1),))


def test_rerank_weights_are_immutable_and_keep_scores_bounded(corpus):
    """The declared fixed formula must remain immutable, normalized, and bounded."""

    candidates = [FusedHit(corpus[0], 0.02, (("keyword", 1),))]

    with pytest.raises(TypeError):
        RERANK_FEATURE_WEIGHTS["numeric_coverage"] = 2.0

    reranked = rerank_candidates("Data Center revenue", candidates, top_k=1)

    assert sum(RERANK_FEATURE_WEIGHTS.values()) == pytest.approx(1.0)
    assert 0.0 <= reranked[0].score <= 1.0
