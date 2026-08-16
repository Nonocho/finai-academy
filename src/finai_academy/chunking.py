"""Observable chunking strategies for financial document engineering."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from finai_academy.documents import DocumentBlock
from finai_academy.providers import EmbeddingModel

ChunkRole = Literal["standalone", "parent", "child", "proposition"]


class ChunkingModel(Protocol):
    """Minimal provider-neutral contract required by proposition chunking."""

    def invoke(self, messages: list[tuple[str, str]]) -> Any: ...


@dataclass(frozen=True)
class DocumentChunk:
    """One strategy-labelled chunk with complete source provenance."""

    chunk_id: str
    strategy: str
    text: str
    source_id: str
    company: str
    period: str
    document_type: str
    source_url: str
    source_block_ids: tuple[str, ...]
    section_path: tuple[str, ...] = ()
    page_numbers: tuple[int, ...] = ()
    role: ChunkRole = "standalone"
    parent_id: str | None = None
    raw_text: str | None = None
    generated_context: str | None = None

    def __post_init__(self) -> None:
        required = {
            "chunk_id": self.chunk_id,
            "strategy": self.strategy,
            "text": self.text,
            "source_id": self.source_id,
            "source_url": self.source_url,
        }
        for field_name, value in required.items():
            normalized = value.strip()
            if not normalized:
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, normalized)
        if not self.source_block_ids:
            raise ValueError("source_block_ids must not be empty")
        if self.role == "child" and not self.parent_id:
            raise ValueError("child chunks must reference parent_id")
        if self.generated_context is not None:
            context = self.generated_context.strip()
            if not context:
                raise ValueError("generated_context must not be empty when provided")
            object.__setattr__(self, "generated_context", context)


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _pages(blocks: Sequence[DocumentBlock]) -> tuple[int, ...]:
    return tuple(dict.fromkeys(block.page_number for block in blocks if block.page_number))


def _chunk_from_blocks(
    blocks: Sequence[DocumentBlock],
    *,
    strategy: str,
    index: int,
    text: str | None = None,
    role: ChunkRole = "standalone",
    parent_id: str | None = None,
) -> DocumentChunk:
    if not blocks:
        raise ValueError("blocks must not be empty")
    first = blocks[0]
    chunk_text = text if text is not None else "\n".join(block.text for block in blocks)
    return DocumentChunk(
        chunk_id=f"{first.source_id}-{strategy.upper()}-{index + 1:03d}",
        strategy=strategy,
        text=chunk_text,
        source_id=first.source_id,
        company=first.company,
        period=first.period,
        document_type=first.document_type,
        source_url=first.source_url,
        source_block_ids=_dedupe([block.block_id for block in blocks]),
        section_path=next((block.section_path for block in blocks if block.section_path), ()),
        page_numbers=_pages(blocks),
        role=role,
        parent_id=parent_id,
    )


def _hard_split(text: str, max_chars: int) -> list[str]:
    pieces = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        boundary = remaining.rfind(" ", 0, max_chars + 1)
        if boundary <= 0:
            boundary = max_chars
        pieces.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def fixed_chunks(
    blocks: Sequence[DocumentBlock],
    *,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    """Split every block at a fixed character boundary with optional overlap."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap
    for block in blocks:
        start = 0
        while start < len(block.text):
            segment = block.text[start : start + chunk_size].strip()
            if segment:
                chunks.append(
                    _chunk_from_blocks(
                        [block], strategy="fixed", index=len(chunks), text=segment
                    )
                )
            if start + chunk_size >= len(block.text):
                break
            start += step
    return chunks


def _sentence_units(block: DocumentBlock, max_chars: int) -> list[str]:
    if block.block_type == "table":
        return _hard_split(block.text, max_chars)
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+|\n+", block.text)
        if item.strip()
    ]
    return [piece for sentence in sentences for piece in _hard_split(sentence, max_chars)]


def recursive_chunks(
    blocks: Sequence[DocumentBlock],
    *,
    max_chars: int,
    strategy: str = "recursive",
) -> list[DocumentChunk]:
    """Prefer sentence boundaries before falling back to word boundaries."""

    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks = []
    for block in blocks:
        buffer = ""
        for unit in _sentence_units(block, max_chars):
            candidate = f"{buffer} {unit}".strip()
            if buffer and len(candidate) > max_chars:
                chunks.append(
                    _chunk_from_blocks(
                        [block], strategy=strategy, index=len(chunks), text=buffer
                    )
                )
                buffer = unit
            else:
                buffer = candidate
        if buffer:
            chunks.append(
                _chunk_from_blocks([block], strategy=strategy, index=len(chunks), text=buffer)
            )
    return chunks


def structure_aware_chunks(
    blocks: Sequence[DocumentBlock],
    *,
    max_chars: int,
) -> list[DocumentChunk]:
    """Use parser structure, keep tables atomic, and retain heading metadata."""

    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks: list[DocumentChunk] = []
    buffer: list[DocumentBlock] = []

    def flush() -> None:
        if not buffer:
            return
        chunks.append(
            _chunk_from_blocks(buffer, strategy="structure", index=len(chunks))
        )
        buffer.clear()

    for block in blocks:
        if block.block_type == "heading":
            flush()
            continue
        if block.block_type == "table":
            flush()
            chunks.append(
                _chunk_from_blocks([block], strategy="structure", index=len(chunks))
            )
            continue
        candidate = "\n".join(item.text for item in [*buffer, block])
        if buffer and len(candidate) > max_chars:
            flush()
        if len(block.text) <= max_chars:
            buffer.append(block)
        else:
            flush()
            chunks.extend(
                recursive_chunks([block], max_chars=max_chars, strategy="structure")
            )
    flush()
    return [replace(chunk, chunk_id=chunk.chunk_id.rsplit("-", 1)[0] + f"-{i + 1:03d}") for i, chunk in enumerate(chunks)]


def sentence_similarity_profile(
    blocks: Sequence[DocumentBlock],
) -> tuple[list[str], list[float]]:
    """Return observable adjacent-sentence similarities for semantic chunking."""

    sentences = [unit for block in blocks for unit in _sentence_units(block, 10_000)]
    if len(sentences) < 2:
        return sentences, []
    matrix = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    similarities = [
        float(cosine_similarity(matrix[index], matrix[index + 1])[0, 0])
        for index in range(len(sentences) - 1)
    ]
    return sentences, similarities


def embedding_similarity_profile(
    blocks: Sequence[DocumentBlock],
    embeddings: EmbeddingModel,
) -> tuple[list[str], list[float]]:
    """Return adjacent-sentence cosine similarities from configured embeddings."""

    sentences = [unit for block in blocks for unit in _sentence_units(block, 10_000)]
    if len(sentences) < 2:
        return sentences, []

    vectors = embeddings.embed_documents(sentences)
    if len(vectors) != len(sentences):
        raise ValueError("embedding provider must return one vector per sentence")
    if not vectors or any(not vector for vector in vectors):
        raise ValueError("embedding vectors must not be empty")
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("embedding vectors must have equal dimensions")

    matrix = np.asarray(vectors, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("embedding vectors must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix),
        where=norms > 0,
    )
    similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return sentences, [float(value) for value in similarities]


def semantic_chunks(
    blocks: Sequence[DocumentBlock],
    *,
    threshold: float,
    similarities: Sequence[float] | None = None,
) -> list[DocumentChunk]:
    """Start a new chunk when adjacent sentence similarity falls below threshold."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    units = [(unit, block) for block in blocks for unit in _sentence_units(block, 10_000)]
    if not units:
        return []
    sentences = [unit for unit, _block in units]
    if similarities is None:
        _sentences, calculated = sentence_similarity_profile(blocks)
        profile = calculated
    else:
        profile = [float(value) for value in similarities]
    if len(profile) != len(sentences) - 1:
        raise ValueError("similarities must contain one value per adjacent sentence pair")

    groups: list[list[tuple[str, DocumentBlock]]] = [[units[0]]]
    for index, item in enumerate(units[1:]):
        if profile[index] < threshold:
            groups.append([item])
        else:
            groups[-1].append(item)

    chunks = []
    for index, group in enumerate(groups):
        group_blocks = list(dict.fromkeys(block for _sentence, block in group))
        text = " ".join(sentence for sentence, _block in group)
        chunks.append(
            _chunk_from_blocks(group_blocks, strategy="semantic", index=index, text=text)
        )
    return chunks


def hierarchical_chunks(
    blocks: Sequence[DocumentBlock],
    *,
    child_max_chars: int,
) -> list[DocumentChunk]:
    """Emit large parent sections and smaller retrievable children."""

    if child_max_chars < 1:
        raise ValueError("child_max_chars must be positive")
    sections: dict[tuple[str, ...], list[DocumentBlock]] = {}
    for block in blocks:
        if block.block_type != "heading":
            sections.setdefault(block.section_path or ("Document",), []).append(block)

    output: list[DocumentChunk] = []
    for section_index, section_blocks in enumerate(sections.values()):
        parent = _chunk_from_blocks(
            section_blocks,
            strategy="hierarchical",
            index=len(output),
            role="parent",
        )
        output.append(parent)
        children = recursive_chunks(
            section_blocks,
            max_chars=child_max_chars,
            strategy="hierarchical",
        )
        for child_index, child in enumerate(children):
            output.append(
                replace(
                    child,
                    chunk_id=f"{parent.chunk_id}-C{child_index + 1:02d}",
                    role="child",
                    parent_id=parent.chunk_id,
                )
            )
    return output


def contextualize_chunks(chunks: Sequence[DocumentChunk]) -> list[DocumentChunk]:
    """Add compact document scope while keeping the original text observable."""

    contextual = []
    for chunk in chunks:
        section = " > ".join(chunk.section_path) if chunk.section_path else "Document"
        prefix = (
            f"{chunk.company} | {chunk.period} | {chunk.document_type} | "
            f"Section: {section}"
        )
        contextual.append(
            replace(
                chunk,
                chunk_id=chunk.chunk_id.replace(chunk.strategy.upper(), "CONTEXTUAL"),
                strategy="contextual",
                text=f"{prefix}\n{chunk.text}",
                raw_text=chunk.text,
            )
        )
    return contextual


def contextual_enrich_chunks(
    *,
    document_text: str,
    chunks: Sequence[DocumentChunk],
    model: ChunkingModel,
) -> list[DocumentChunk]:
    """Generate retrieval context while preserving raw evidence and provenance."""

    normalized_document = document_text.strip()
    if not normalized_document:
        raise ValueError("document_text must not be empty")

    enriched: list[DocumentChunk] = []
    for chunk in chunks:
        raw_text = chunk.raw_text or chunk.text
        prompt_payload = json.dumps(
            {
                "chunk_id": chunk.chunk_id,
                "document": normalized_document,
                "chunk": raw_text,
            }
        )
        response = model.invoke(
            [
                (
                    "system",
                    (
                        "Return JSON with one context string that situates the chunk "
                        "inside the supplied financial document. Do not rewrite the chunk."
                    ),
                ),
                ("human", prompt_payload),
            ]
        )
        try:
            payload = json.loads(response.content)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("model response must contain valid JSON") from error
        context = payload.get("context") if isinstance(payload, dict) else None
        if not isinstance(context, str) or not context.strip():
            raise ValueError("model response must contain a non-empty context string")
        normalized_context = context.strip()
        enriched.append(
            replace(
                chunk,
                strategy="llm_contextual",
                text=f"{normalized_context}\n\n{raw_text}",
                raw_text=raw_text,
                generated_context=normalized_context,
            )
        )
    return enriched


def proposition_chunks(
    blocks: Sequence[DocumentBlock],
    model: ChunkingModel,
) -> list[DocumentChunk]:
    """Convert source blocks into atomic statements with a validated JSON boundary."""

    output = []
    for block in blocks:
        response = model.invoke(
            [
                (
                    "system",
                    "Return JSON with a propositions array. Preserve every numeric qualifier.",
                ),
                ("human", block.text),
            ]
        )
        payload = json.loads(response.content)
        propositions = payload.get("propositions")
        if not isinstance(propositions, list) or not all(
            isinstance(item, str) and item.strip() for item in propositions
        ):
            raise ValueError("model response must contain a non-empty string proposition list")
        for proposition in propositions:
            output.append(
                _chunk_from_blocks(
                    [block],
                    strategy="llm_proposition",
                    index=len(output),
                    text=proposition.strip(),
                    role="proposition",
                )
            )
    return output


def compare_chunking_strategies(
    strategies: Mapping[str, Sequence[DocumentChunk]],
    blocks: Sequence[DocumentBlock],
) -> pd.DataFrame:
    """Return transparent construction metrics before retrieval evaluation."""

    table_blocks = [block for block in blocks if block.block_type == "table"]
    rows = []
    for strategy, chunks in strategies.items():
        sizes = [len(chunk.text) for chunk in chunks]
        intact_tables = sum(
            any(table.text in chunk.text for chunk in chunks) for table in table_blocks
        )
        rows.append(
            {
                "strategy": strategy,
                "chunk_count": len(chunks),
                "mean_chars": float(np.mean(sizes)) if sizes else 0.0,
                "max_chars": max(sizes, default=0),
                "heading_retention": (
                    sum(bool(chunk.section_path) for chunk in chunks) / len(chunks)
                    if chunks
                    else 0.0
                ),
                "table_integrity": (
                    intact_tables / len(table_blocks) if table_blocks else 1.0
                ),
                "provenance_completeness": (
                    sum(bool(chunk.source_block_ids and chunk.source_url) for chunk in chunks)
                    / len(chunks)
                    if chunks
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)
