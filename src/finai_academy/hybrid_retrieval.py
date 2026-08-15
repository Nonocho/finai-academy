"""Provider-neutral keyword and dense retrieval primitives for Lesson 06."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from finai_academy.retrieval import EvidencePassage, LexicalRetriever, RetrievalHit


class EmbeddingModel(Protocol):
    """The small embedding interface required by the dense teaching index."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of source documents."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a retrieval query."""


@dataclass(frozen=True)
class IndexedPassage(EvidencePassage):
    """An evidence passage with a disclosure-type eligibility field."""

    document_type: str = "financial disclosure"

    def __post_init__(self) -> None:
        """Preserve evidence validation and normalize the document type."""

        super().__post_init__()
        normalized_document_type = self.document_type.strip()
        if not normalized_document_type:
            raise ValueError("document_type must not be empty")
        object.__setattr__(self, "document_type", normalized_document_type)


@dataclass(frozen=True)
class RetrievalFilters:
    """Exact metadata constraints that determine passage eligibility."""

    company: str | None = None
    period: str | None = None
    document_type: str | None = None
    section: str | None = None

    def matches(self, passage: IndexedPassage) -> bool:
        """Return whether a passage satisfies every provided metadata value."""

        expected_values = {
            "company": self.company,
            "period": self.period,
            "document_type": self.document_type,
            "section": self.section,
        }
        return all(
            expected is None or _normalize_metadata(expected) == _normalize_metadata(getattr(passage, name))
            for name, expected in expected_values.items()
        )


class DeterministicTeachingEmbeddings:
    """A transparent 12-dimensional embedder for hybrid-retrieval exercises.

    Its dimensions represent NVIDIA, Schneider Electric, FY2026, FY2025, revenue,
    growth, margin, data center, gaming, energy management, adjusted EBITA, and
    organic growth. Numeric tokens are deliberately excluded so exact figures remain
    a keyword-retrieval teaching case.
    """

    model_name = "financial-concepts-v1"
    dimension = 12

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed each document with the fixed financial-concept vocabulary."""

        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the fixed financial-concept vocabulary."""

        return self._embed(text)

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        normalized_text = _normalize_text(text)
        matches = (
            ("nvidia",),
            ("schneider electric",),
            ("fy2026",),
            ("fy2025",),
            ("revenue",),
            ("growth", "grew", "expansion", "organic"),
            ("margin",),
            ("data center", "data centre"),
            ("gaming",),
            ("energy management", "energy"),
            ("adjusted ebita",),
            ("organic growth", "organically"),
        )
        vector = [
            float(any(concept in normalized_text for concept in dimension_concepts))
            for dimension_concepts in matches
        ]
        return _normalize_vector(vector)


class KeywordIndex:
    """TF-IDF lexical retrieval with optional exact metadata eligibility filters."""

    def __init__(self, passages: Sequence[IndexedPassage]) -> None:
        self.passages = tuple(passages)
        self._retriever = LexicalRetriever(self.passages)

    def search(
        self,
        query: str,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalHit]:
        """Return eligible lexical matches, descending by score then identifier."""

        _validate_top_k(top_k)
        ranked_hits = self._retriever.rank(query)
        eligible_hits = [
            hit for hit in ranked_hits if filters is None or filters.matches(hit.passage)
        ]
        return eligible_hits[:top_k]


class DenseIndex:
    """Dense retrieval that keeps provider metadata alongside document vectors."""

    def __init__(
        self,
        passages: Sequence[IndexedPassage],
        embeddings: EmbeddingModel,
        *,
        provider: str,
        model: str,
        chunking_strategy: str,
    ) -> None:
        self.passages = tuple(passages)
        _validate_passages(self.passages)
        self.embeddings = embeddings
        self.provider = provider
        self.model = model
        self.chunking_strategy = chunking_strategy

        document_vectors = embeddings.embed_documents([passage.text for passage in self.passages])
        if len(document_vectors) != len(self.passages):
            raise ValueError("embed_documents must return one vector per passage")
        self._document_vectors = tuple(
            _validate_and_normalize_vector(vector, expected_dimension=None)
            for vector in document_vectors
        )
        self.dimension = len(self._document_vectors[0])
        if self.dimension == 0:
            raise ValueError("embedding vectors must not be empty")
        if any(len(vector) != self.dimension for vector in self._document_vectors):
            raise ValueError("embedding vectors must have consistent dimensions")

    def search(
        self,
        query: str,
        top_k: int,
        filters: RetrievalFilters | None = None,
    ) -> list[RetrievalHit]:
        """Rank eligible passages with scores compatible with ``RetrievalHit``."""

        _validate_top_k(top_k)
        scores = self.cosine_scores(query, filters)
        hits = [
            RetrievalHit(passage=passage, score=(cosine + 1.0) / 2.0)
            for passage, cosine in scores
        ]
        return hits[:top_k]

    def cosine_scores(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
    ) -> list[tuple[IndexedPassage, float]]:
        """Return raw cosine similarities for teaching visualizations."""

        _validate_query(query)
        query_vector = _validate_and_normalize_vector(
            self.embeddings.embed_query(query), expected_dimension=self.dimension
        )
        scores = [
            (passage, _cosine_similarity(query_vector, vector))
            for passage, vector in zip(self.passages, self._document_vectors, strict=True)
            if filters is None or filters.matches(passage)
        ]
        return sorted(scores, key=lambda item: (-item[1], item[0].passage_id))


def _normalize_metadata(value: str) -> str:
    return value.casefold().strip()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold().replace("-", " ")).strip()


def _normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0.0:
        return vector
    return [component / magnitude for component in vector]


def _validate_and_normalize_vector(
    vector: Sequence[float], *, expected_dimension: int | None
) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in vector)
    except (TypeError, ValueError) as error:
        raise ValueError("embedding vectors must contain finite numeric values") from error
    if not values:
        raise ValueError("embedding vectors must not be empty")
    if expected_dimension is not None and len(values) != expected_dimension:
        raise ValueError("embedding vectors must have consistent dimensions")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("embedding vectors must contain finite numeric values")
    return tuple(_normalize_vector(list(values)))


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    cosine = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    return max(-1.0, min(1.0, cosine))


def _validate_passages(passages: Sequence[IndexedPassage]) -> None:
    if not passages:
        raise ValueError("passages must contain at least one item")
    passage_ids = [passage.passage_id for passage in passages]
    if len(passage_ids) != len(set(passage_ids)):
        raise ValueError("passage_id values must be unique")


def _validate_query(query: str) -> None:
    if not query.strip():
        raise ValueError("query must not be empty")


def _validate_top_k(top_k: int) -> None:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
