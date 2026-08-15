"""Transparent deterministic reranking for Lesson 06 evidence candidates."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from finai_academy.hybrid_retrieval import FusedHit, IndexedPassage

# Matches standalone financial figures such as ``18.7%``, ``EUR 40.2bn``, and ``$193.7``.
NUMERIC_TOKEN_PATTERN = re.compile(
    r"(?<![\w.])(?:[$€£]\s*|(?:usd|eur|gbp)\s+)?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)(?:%|bn|m|b|x)?(?!\w)",
    re.IGNORECASE,
)
WORD_TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9]*", re.IGNORECASE)
QUERY_STOP_WORDS = frozenset(
    {"a", "an", "and", "at", "by", "did", "does", "for", "in", "is", "of", "on", "or", "the", "to", "was", "what", "with"}
)
RERANK_FEATURE_WEIGHTS = {
    "lexical_coverage": 0.25,
    "numeric_coverage": 0.45,
    "section_overlap": 0.10,
    "metadata_eligibility": 0.10,
    "fusion_signal": 0.10,
}


@dataclass(frozen=True)
class RerankFeatures:
    """Normalized, inspectable inputs to the deterministic rerank score."""

    lexical_coverage: float
    numeric_coverage: float
    section_overlap: float
    metadata_eligibility: float
    fusion_signal: float


@dataclass(frozen=True)
class RerankedHit:
    """A fused candidate paired with its transparent second-stage score."""

    passage: IndexedPassage
    score: float
    features: RerankFeatures
    fused_hit: FusedHit


def rerank_candidates(
    query: str,
    candidates: Sequence[FusedHit],
    *,
    top_k: int,
) -> list[RerankedHit]:
    """Score fused candidates with exact, normalized feature contributions."""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not candidates:
        return []

    query_words = _words(normalized_query)
    query_numbers = _numeric_tokens(normalized_query)
    maximum_fusion_score = max(candidate.rrf_score for candidate in candidates)
    reranked_hits = []
    for candidate in candidates:
        features = _features_for_candidate(
            candidate,
            query_words=query_words,
            query_numbers=query_numbers,
            maximum_fusion_score=maximum_fusion_score,
        )
        reranked_hits.append(
            RerankedHit(
                passage=candidate.passage,
                score=_weighted_score(features),
                features=features,
                fused_hit=candidate,
            )
        )
    return sorted(
        reranked_hits,
        key=lambda hit: (-hit.score, -hit.fused_hit.rrf_score, hit.passage.passage_id),
    )[:top_k]


def _features_for_candidate(
    candidate: FusedHit,
    *,
    query_words: frozenset[str],
    query_numbers: frozenset[str],
    maximum_fusion_score: float,
) -> RerankFeatures:
    passage_words = _words(candidate.passage.text)
    section_words = _words(candidate.passage.section)
    return RerankFeatures(
        lexical_coverage=_coverage(query_words, passage_words),
        numeric_coverage=_coverage(query_numbers, _numeric_tokens(candidate.passage.text)),
        section_overlap=_coverage(query_words, section_words),
        metadata_eligibility=1.0,
        fusion_signal=(candidate.rrf_score / maximum_fusion_score if maximum_fusion_score else 0.0),
    )


def _weighted_score(features: RerankFeatures) -> float:
    return sum(
        RERANK_FEATURE_WEIGHTS[name] * getattr(features, name)
        for name in RERANK_FEATURE_WEIGHTS
    )


def _coverage(query_tokens: frozenset[str], document_tokens: frozenset[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(query_tokens.intersection(document_tokens)) / len(query_tokens)


def _numeric_tokens(text: str) -> frozenset[str]:
    return frozenset(match.group().casefold().replace(" ", "") for match in NUMERIC_TOKEN_PATTERN.finditer(text))


def _words(text: str) -> frozenset[str]:
    return frozenset(
        token.casefold()
        for token in WORD_TOKEN_PATTERN.findall(text)
        if token.casefold() not in QUERY_STOP_WORDS
    )
