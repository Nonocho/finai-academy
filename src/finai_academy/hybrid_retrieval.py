"""Provider-neutral keyword and dense retrieval primitives for Lesson 06."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

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


@dataclass(frozen=True)
class FusedHit:
    """One deduplicated passage with its reciprocal-rank contributions."""

    passage: IndexedPassage
    rrf_score: float
    channel_ranks: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        """Keep the fusion signal suitable for normalized downstream features."""

        if not isinstance(self.rrf_score, (int, float)) or not math.isfinite(self.rrf_score):
            raise ValueError("rrf_score must be a finite number")
        if self.rrf_score < 0:
            raise ValueError("rrf_score must not be negative")


@dataclass(frozen=True)
class EmbeddingIndexVersion:
    """The identity metadata required to reuse locally stored document vectors."""

    schema_version: int
    provider: str
    model: str
    dimension: int
    corpus_hash: str
    chunking_strategy: str
    passage_ids: tuple[str, ...]


class IndexVersionError(ValueError):
    """Raised when persisted vectors do not describe the requested corpus."""


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
        eligible_indices = (
            range(len(self.passages))
            if filters is None
            else (
                index
                for index, passage in enumerate(self.passages)
                if filters.matches(passage)
            )
        )
        return self._retriever.rank_subset(query, eligible_indices)[:top_k]


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
        document_vectors = tuple(
            _validate_and_normalize_vector(vector, expected_dimension=None)
            for vector in document_vectors
        )
        self.dimension = len(document_vectors[0])
        if self.dimension == 0:
            raise ValueError("embedding vectors must not be empty")
        if any(len(vector) != self.dimension for vector in document_vectors):
            raise ValueError("embedding vectors must have consistent dimensions")
        self._document_matrix = np.asarray(document_vectors, dtype=float)
        self.version = _build_index_version(
            self.passages,
            provider=self.provider,
            model=self.model,
            dimension=self.dimension,
            chunking_strategy=self.chunking_strategy,
        )

    @property
    def document_matrix(self) -> np.ndarray:
        """Return a copy of the normalized document vectors for inspection."""

        return self._document_matrix.copy()

    def save(self, directory: str | Path) -> None:
        """Persist this index's identity metadata and normalized document vectors."""

        artifact_directory = Path(directory)
        artifact_directory.mkdir(parents=True, exist_ok=True)
        with (artifact_directory / "manifest.json").open("w", encoding="utf-8") as manifest_file:
            json.dump(asdict(self.version), manifest_file, sort_keys=True)
        np.save(artifact_directory / "vectors.npy", self._document_matrix)

    @classmethod
    def from_artifact(
        cls,
        directory: str | Path,
        passages: Sequence[IndexedPassage],
        embeddings: EmbeddingModel,
        *,
        expected_version: EmbeddingIndexVersion,
    ) -> DenseIndex:
        """Load vectors only when their manifest matches the requested index identity."""

        restored_passages = tuple(passages)
        _validate_passages(restored_passages)
        if not isinstance(expected_version, EmbeddingIndexVersion):
            raise IndexVersionError("expected_version must be an EmbeddingIndexVersion")
        _validate_embedding_index_version(expected_version)

        artifact_directory = Path(directory)
        stored_version = _load_embedding_index_version(artifact_directory / "manifest.json")
        _validate_matching_versions(stored_version, expected_version)
        _validate_manifest_corpus(stored_version, restored_passages)
        document_matrix = _load_document_matrix(
            artifact_directory / "vectors.npy",
            passage_count=len(restored_passages),
            dimension=stored_version.dimension,
        )
        _validate_query_vector_dimension(embeddings, stored_version.dimension)

        restored_index = cls.__new__(cls)
        restored_index.passages = restored_passages
        restored_index.embeddings = embeddings
        restored_index.provider = stored_version.provider
        restored_index.model = stored_version.model
        restored_index.chunking_strategy = stored_version.chunking_strategy
        restored_index.dimension = stored_version.dimension
        restored_index._document_matrix = document_matrix
        restored_index.version = stored_version
        return restored_index

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
            for passage, vector in zip(self.passages, self._document_matrix, strict=True)
            if filters is None or filters.matches(passage)
        ]
        return sorted(scores, key=lambda item: (-item[1], item[0].passage_id))


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[RetrievalHit]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[FusedHit]:
    """Combine ranked channels using deterministic, identifier-based RRF.

    Each ranking must contain every passage identifier at most once. A shared
    identifier contributes one weighted reciprocal-rank term per channel.
    """

    _validate_rrf_inputs(rankings, k=k, weights=weights)
    resolved_weights = {channel: 1.0 for channel in rankings}
    if weights is not None:
        resolved_weights.update(weights)

    passages_by_id: dict[str, IndexedPassage] = {}
    score_by_id: dict[str, float] = {}
    ranks_by_id: dict[str, dict[str, int]] = {}
    for channel, hits in rankings.items():
        seen_ids: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            passage = hit.passage
            passage_id = passage.passage_id
            if passage_id in seen_ids:
                raise ValueError(f"duplicate passage_id {passage_id!r} in {channel!r} ranking")
            seen_ids.add(passage_id)
            existing_passage = passages_by_id.get(passage_id)
            if existing_passage is not None and existing_passage != passage:
                raise ValueError(f"inconsistent passage content for passage_id {passage_id!r}")
            passages_by_id[passage_id] = passage
            score_by_id[passage_id] = score_by_id.get(passage_id, 0.0) + (
                resolved_weights[channel] / (k + rank)
            )
            ranks_by_id.setdefault(passage_id, {})[channel] = rank

    fused_hits = [
        FusedHit(
            passage=passages_by_id[passage_id],
            rrf_score=score_by_id[passage_id],
            channel_ranks=tuple(sorted(ranks_by_id[passage_id].items())),
        )
        for passage_id in passages_by_id
    ]
    return sorted(fused_hits, key=lambda hit: (-hit.rrf_score, hit.passage.passage_id))


def _normalize_metadata(value: str) -> str:
    return value.casefold().strip()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold().replace("-", " ")).strip()


def _normalize_vector(vector: list[float]) -> list[float]:
    scale = max(abs(component) for component in vector)
    if scale == 0.0:
        return vector
    scaled_vector = [component / scale for component in vector]
    magnitude = math.sqrt(sum(component * component for component in scaled_vector))
    return [component / magnitude for component in scaled_vector]


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


def _build_index_version(
    passages: Sequence[IndexedPassage],
    *,
    provider: str,
    model: str,
    dimension: int,
    chunking_strategy: str,
) -> EmbeddingIndexVersion:
    return EmbeddingIndexVersion(
        schema_version=1,
        provider=provider,
        model=model,
        dimension=dimension,
        corpus_hash=_hash_corpus(passages),
        chunking_strategy=chunking_strategy,
        passage_ids=tuple(passage.passage_id for passage in passages),
    )


def _hash_corpus(passages: Sequence[IndexedPassage]) -> str:
    records = [
        (
            passage.passage_id,
            passage.company,
            passage.period,
            passage.document_type,
            passage.section,
            passage.text,
            passage.source_url,
        )
        for passage in passages
    ]
    canonical_corpus = json.dumps(
        records,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_corpus.encode("utf-8")).hexdigest()


def _load_embedding_index_version(manifest_path: Path) -> EmbeddingIndexVersion:
    try:
        with manifest_path.open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        version = EmbeddingIndexVersion(
            schema_version=manifest["schema_version"],
            provider=manifest["provider"],
            model=manifest["model"],
            dimension=manifest["dimension"],
            corpus_hash=manifest["corpus_hash"],
            chunking_strategy=manifest["chunking_strategy"],
            passage_ids=tuple(manifest["passage_ids"]),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise IndexVersionError("manifest contains invalid index version metadata") from error

    _validate_embedding_index_version(version)
    return version


def _validate_embedding_index_version(version: EmbeddingIndexVersion) -> None:
    if type(version.schema_version) is not int:
        raise IndexVersionError("schema_version must be an integer")
    if version.schema_version != 1:
        raise IndexVersionError("schema_version is not supported")
    if not isinstance(version.dimension, int) or isinstance(version.dimension, bool):
        raise IndexVersionError("dimension must be an integer")
    if version.dimension <= 0:
        raise IndexVersionError("dimension must be positive")
    for field_name in ("provider", "model", "corpus_hash", "chunking_strategy"):
        if not isinstance(getattr(version, field_name), str):
            raise IndexVersionError(f"{field_name} must be a string")
    if not all(isinstance(passage_id, str) for passage_id in version.passage_ids):
        raise IndexVersionError("passage_ids must contain strings")


def _validate_matching_versions(
    stored_version: EmbeddingIndexVersion,
    expected_version: EmbeddingIndexVersion,
) -> None:
    for field_name in (
        "schema_version",
        "provider",
        "model",
        "dimension",
        "corpus_hash",
        "chunking_strategy",
        "passage_ids",
    ):
        if getattr(stored_version, field_name) != getattr(expected_version, field_name):
            raise IndexVersionError(f"index version mismatch for {field_name}")


def _validate_manifest_corpus(
    stored_version: EmbeddingIndexVersion,
    passages: Sequence[IndexedPassage],
) -> None:
    passage_ids = tuple(passage.passage_id for passage in passages)
    if stored_version.passage_ids != passage_ids:
        raise IndexVersionError("manifest passage_ids do not match the requested corpus order")
    if stored_version.corpus_hash != _hash_corpus(passages):
        raise IndexVersionError("manifest corpus_hash does not match the requested corpus")


def _load_document_matrix(
    vectors_path: Path,
    *,
    passage_count: int,
    dimension: int,
) -> np.ndarray:
    try:
        matrix = np.load(vectors_path, allow_pickle=False)
    except (EOFError, OSError, TypeError, ValueError, pickle.UnpicklingError, zipfile.BadZipFile) as error:
        raise IndexVersionError("vectors.npy could not be loaded") from error
    if not isinstance(matrix, np.ndarray):
        close = getattr(matrix, "close", None)
        if callable(close):
            close()
        raise IndexVersionError("vectors.npy must contain one ndarray matrix")
    if matrix.shape != (passage_count, dimension):
        raise IndexVersionError("vector matrix shape does not match the index version")
    if not np.issubdtype(matrix.dtype, np.number) or not np.all(np.isfinite(matrix)):
        raise IndexVersionError("vector matrix must contain finite numeric values")
    return np.asarray(matrix, dtype=float)


def _validate_query_vector_dimension(embeddings: EmbeddingModel, dimension: int) -> None:
    try:
        _validate_and_normalize_vector(
            embeddings.embed_query("embedding index dimension validation"),
            expected_dimension=dimension,
        )
    except ValueError as error:
        raise IndexVersionError("query vector dimension does not match the index version") from error


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
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


def _validate_rrf_inputs(
    rankings: Mapping[str, Sequence[RetrievalHit]],
    *,
    k: int,
    weights: Mapping[str, float] | None,
) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not rankings:
        raise ValueError("rankings must contain at least one channel")
    if weights is not None:
        unknown_channels = set(weights).difference(rankings)
        if unknown_channels:
            raise ValueError(f"weights contain unknown channels: {sorted(unknown_channels)!r}")
        for channel, weight in weights.items():
            if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(weight):
                raise ValueError(f"weight for {channel!r} must be a finite number")
            if weight < 0:
                raise ValueError(f"weight for {channel!r} must not be negative")
    for channel, hits in rankings.items():
        if not isinstance(channel, str) or not channel.strip():
            raise ValueError("ranking channel names must be non-empty strings")
        if not hits:
            raise ValueError(f"{channel!r} ranking must contain at least one hit")
