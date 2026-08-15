"""Metadata-safe orchestration for the complete Lesson 06 retrieval pipeline."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from finai_academy.hybrid_retrieval import (
    DenseIndex,
    FusedHit,
    KeywordIndex,
    RetrievalFilters,
    reciprocal_rank_fusion,
)
from finai_academy.reranking import RerankedHit, rerank_candidates
from finai_academy.retrieval import RetrievalHit

SUPPORTED_RRF_CHANNELS = frozenset({"keyword", "dense"})


@dataclass(frozen=True)
class RetrievalResult:
    """Every visible retrieval stage plus an explicit abstention, when applicable."""

    query: str
    filters: RetrievalFilters
    keyword_hits: tuple[RetrievalHit, ...]
    dense_hits: tuple[RetrievalHit, ...]
    fused_hits: tuple[FusedHit, ...]
    reranked_hits: tuple[RerankedHit, ...]
    abstention_reason: str | None = None


def retrieve_evidence(
    query: str,
    *,
    keyword_index: KeywordIndex,
    dense_index: DenseIndex,
    filters: RetrievalFilters,
    candidate_k: int,
    final_k: int,
    weights: Mapping[str, float] | None = None,
) -> RetrievalResult:
    """Pre-filter, retrieve widely, fuse, and transparently rerank final evidence."""

    _validate_budgets(candidate_k, final_k)
    _validate_weights(weights)
    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must not be empty")
    eligible_passages = (*keyword_index.passages, *dense_index.passages)
    if not any(filters.matches(passage) for passage in eligible_passages):
        return RetrievalResult(
            query=normalized_query,
            filters=filters,
            keyword_hits=(),
            dense_hits=(),
            fused_hits=(),
            reranked_hits=(),
            abstention_reason="No passages matched the requested metadata filters.",
        )

    keyword_hits = tuple(keyword_index.search(normalized_query, candidate_k, filters))
    dense_hits = tuple(dense_index.search(normalized_query, candidate_k, filters))
    rankings = {
        channel: hits
        for channel, hits in (("keyword", keyword_hits), ("dense", dense_hits))
        if hits
    }
    active_weights = (
        {channel: weights[channel] for channel in rankings if channel in weights}
        if weights is not None
        else None
    )
    fused_hits = tuple(reciprocal_rank_fusion(rankings, weights=active_weights)) if rankings else ()
    reranked_hits = tuple(rerank_candidates(normalized_query, fused_hits, top_k=final_k))
    return RetrievalResult(
        query=normalized_query,
        filters=filters,
        keyword_hits=keyword_hits,
        dense_hits=dense_hits,
        fused_hits=fused_hits,
        reranked_hits=reranked_hits,
    )


def _validate_budgets(candidate_k: int, final_k: int) -> None:
    valid_values = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 1
        for value in (candidate_k, final_k)
    )
    if not valid_values or candidate_k < final_k:
        raise ValueError("candidate_k must be greater than or equal to final_k, with both at least 1")


def _validate_weights(weights: Mapping[str, float] | None) -> None:
    if weights is None:
        return
    unsupported_channels = tuple(channel for channel in weights if channel not in SUPPORTED_RRF_CHANNELS)
    if unsupported_channels:
        raise ValueError(f"weights contain unsupported channels: {unsupported_channels!r}")
    for channel, weight in weights.items():
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight):
            raise ValueError(f"weight for {channel!r} must be a finite number")
        if weight < 0:
            raise ValueError(f"weight for {channel!r} must not be negative")
