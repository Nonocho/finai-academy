"""Deterministic retrieval over certified, contextual financial document chunks."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from finai_academy.capstone.document_models import (
    DocumentFilters,
    DocumentRetrievalHit,
    FinancialChunk,
)
from finai_academy.hybrid_retrieval import (
    BM25Index,
    DenseIndex,
    DeterministicTeachingEmbeddings,
    FusedHit,
    IndexedPassage,
    reciprocal_rank_fusion,
)

_ARTIFACT_MANIFEST_PATH = Path("assets/course-data/manifest.json")
_SELECTION_REASON = "Matched exact financial terms and related document meaning."


class CertifiedDocumentIndexError(ValueError):
    """Raised when certified evidence or public document-index inputs are invalid."""


class CertifiedDocumentIndex:
    """Search immutable certified chunks after applying metadata eligibility."""

    def __init__(self, chunks: tuple[FinancialChunk, ...], *, index_version: str) -> None:
        if not chunks:
            raise CertifiedDocumentIndexError("certified document index contains no chunks")
        if not index_version.strip():
            raise CertifiedDocumentIndexError("index version must not be blank")

        self._chunks = chunks
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(self._chunks_by_id) != len(chunks):
            raise CertifiedDocumentIndexError("duplicate chunk_id in certified document index")
        self.index_version = index_version

    def search(
        self,
        query: str,
        *,
        filters: DocumentFilters,
        top_k: int = 3,
    ) -> tuple[DocumentRetrievalHit, ...]:
        """Return ranked contextual evidence constrained before either ranker is built."""

        _validate_search_arguments(query, filters, top_k)
        eligible = tuple(
            chunk
            for chunk in self._chunks
            if chunk.financially_contextualized and filters.matches(chunk)
        )
        if not eligible:
            return ()

        passages = tuple(_to_indexed_passage(chunk) for chunk in eligible)
        bm25_hits = BM25Index(passages).search(query, top_k=len(passages))
        embeddings = DeterministicTeachingEmbeddings()
        dense_hits = DenseIndex(
            passages,
            embeddings,
            provider="certified-fixture",
            model=embeddings.model_name,
            chunking_strategy="financial-context-v2",
        ).search(query, top_k=len(passages))
        fused_hits = reciprocal_rank_fusion(
            {"bm25": bm25_hits, "dense": dense_hits},
            weights={"bm25": 10.0, "dense": 1.0},
        )
        return tuple(_to_public_hit(hit, self._chunks_by_id, self.index_version) for hit in fused_hits[:top_k])

    def inspect(self, chunk_id: str) -> FinancialChunk:
        """Return the exact immutable chunk that a retrieval hit referenced."""

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise CertifiedDocumentIndexError("chunk_id must not be blank")
        try:
            return self._chunks_by_id[chunk_id]
        except KeyError as error:
            raise CertifiedDocumentIndexError("unknown certified chunk_id") from error


def load_certified_document_index(root: Path | None = None) -> CertifiedDocumentIndex:
    """Load chunks only after their recorded artifact and source identities verify."""

    project_root = _project_root(root)
    manifest = _load_manifest(project_root / _ARTIFACT_MANIFEST_PATH)
    artifact = _load_chunk_artifact_record(manifest)
    chunk_path, artifact_hash = _artifact_path_and_hash(project_root, artifact)
    raw_chunks = _load_verified_chunk_bytes(chunk_path, artifact_hash)
    chunks = _validate_chunks(raw_chunks)
    _verify_chunk_source_hashes(chunks, artifact)
    return CertifiedDocumentIndex(chunks, index_version=artifact_hash)


def _project_root(root: Path | None) -> Path:
    if root is None:
        return Path(__file__).resolve().parents[3]
    if not isinstance(root, Path):
        raise CertifiedDocumentIndexError("root must be a Path")
    return root


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CertifiedDocumentIndexError("certified artifact manifest could not be loaded") from error
    if not isinstance(payload, dict):
        raise CertifiedDocumentIndexError("certified artifact manifest must contain an object")
    return payload


def _load_chunk_artifact_record(manifest: dict[str, Any]) -> dict[str, Any]:
    records = manifest.get("capstone_derived_artifacts")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise CertifiedDocumentIndexError("certified artifact manifest has no unique chunk record")
    return records[0]


def _artifact_path_and_hash(project_root: Path, artifact: dict[str, Any]) -> tuple[Path, str]:
    chunk_record = artifact.get("chunks")
    if not isinstance(chunk_record, dict):
        raise CertifiedDocumentIndexError("certified artifact manifest has no chunk artifact")
    path_value = chunk_record.get("path")
    artifact_hash = chunk_record.get("sha256")
    if not isinstance(path_value, str) or not _is_relative_artifact_path(path_value):
        raise CertifiedDocumentIndexError("chunk artifact path must be repository-relative")
    if not isinstance(artifact_hash, str) or len(artifact_hash) != 64:
        raise CertifiedDocumentIndexError("chunk artifact SHA-256 is invalid")
    try:
        int(artifact_hash, 16)
    except ValueError as error:
        raise CertifiedDocumentIndexError("chunk artifact SHA-256 is invalid") from error
    return project_root / path_value, artifact_hash


def _is_relative_artifact_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value.strip()) and not path.is_absolute() and ".." not in path.parts


def _load_verified_chunk_bytes(path: Path, expected_hash: str) -> list[Any]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CertifiedDocumentIndexError("certified chunk artifact could not be loaded") from error
    if sha256(raw).hexdigest() != expected_hash:
        raise CertifiedDocumentIndexError("chunk artifact SHA-256 mismatch")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise CertifiedDocumentIndexError("certified chunk artifact is not valid JSON") from error
    if not isinstance(payload, list):
        raise CertifiedDocumentIndexError("certified chunk artifact must contain a list")
    return payload


def _validate_chunks(raw_chunks: list[Any]) -> tuple[FinancialChunk, ...]:
    try:
        chunks = tuple(FinancialChunk.model_validate(item) for item in raw_chunks)
    except ValidationError as error:
        raise CertifiedDocumentIndexError("certified chunk artifact contains an invalid chunk") from error
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    if len(chunk_ids) != len(set(chunk_ids)):
        raise CertifiedDocumentIndexError("duplicate chunk_id in certified chunk artifact")
    return chunks


def _verify_chunk_source_hashes(
    chunks: tuple[FinancialChunk, ...], artifact: dict[str, Any]
) -> None:
    expected_hashes = artifact.get("source_sha256s")
    if not isinstance(expected_hashes, list) or not expected_hashes:
        raise CertifiedDocumentIndexError("certified chunk artifact has no source hashes")
    if not all(isinstance(value, str) and len(value) == 64 for value in expected_hashes):
        raise CertifiedDocumentIndexError("certified chunk artifact has invalid source hashes")
    actual_hashes = {chunk.context.document_sha256 for chunk in chunks}
    if actual_hashes != set(expected_hashes):
        raise CertifiedDocumentIndexError("certified chunk source hashes do not match the manifest")


def _validate_search_arguments(query: str, filters: DocumentFilters, top_k: int) -> None:
    if not isinstance(query, str) or not query.strip():
        raise CertifiedDocumentIndexError("query must not be blank")
    if not isinstance(filters, DocumentFilters):
        raise CertifiedDocumentIndexError("filters must be DocumentFilters")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 5:
        raise CertifiedDocumentIndexError("top_k must be between 1 and 5")


def _to_indexed_passage(chunk: FinancialChunk) -> IndexedPassage:
    heading_path = " > ".join(chunk.context.heading_path) or chunk.element_type
    return IndexedPassage(
        passage_id=chunk.chunk_id,
        company=chunk.context.company_name,
        period=chunk.context.reporting_period,
        document_type=chunk.context.document_type,
        section=heading_path,
        text=chunk.text,
        source_url=chunk.context.official_source_url,
    )


def _to_public_hit(
    fused_hit: FusedHit,
    chunks_by_id: dict[str, FinancialChunk],
    index_version: str,
) -> DocumentRetrievalHit:
    return DocumentRetrievalHit(
        chunk=chunks_by_id[fused_hit.passage.passage_id],
        fused_score=fused_hit.rrf_score,
        channel_ranks=fused_hit.channel_ranks,
        index_version=index_version,
        selection_reason=_SELECTION_REASON,
    )
