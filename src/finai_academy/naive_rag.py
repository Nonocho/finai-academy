"""Intentionally naive real-document parsing and chunking for Lesson 04."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from finai_academy.documents import DocumentSource
from finai_academy.retrieval import EvidencePassage


@dataclass(frozen=True)
class NaiveHtmlParse:
    """A flattened HTML document plus the structure discarded by flattening."""

    source_path: Path
    text: str
    raw_html_characters: int
    table_count: int
    semantic_heading_count: int


def naive_parse_html(path: Path) -> NaiveHtmlParse:
    """Flatten an HTML filing to one whitespace-normalized text stream.

    This is deliberately a baseline, not the canonical parser used in Lesson 05.
    It records structure counts before discarding their boundaries so learners can
    see what the simplified representation lost.
    """

    raw_html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw_html, "html.parser")
    table_count = len(soup.find_all("table"))
    semantic_heading_count = sum(
        len(soup.find_all(f"h{level}")) for level in range(1, 7)
    )
    for element in soup(["script", "style"]):
        element.decompose()
    text = " ".join(soup.get_text(" ", strip=True).split())
    if not text:
        raise ValueError("HTML document did not contain visible text")
    return NaiveHtmlParse(
        source_path=path,
        text=text,
        raw_html_characters=len(raw_html),
        table_count=table_count,
        semantic_heading_count=semantic_heading_count,
    )


def naive_fixed_windows(
    text: str,
    source: DocumentSource,
    *,
    chunk_chars: int = 1_600,
    overlap_chars: int = 200,
) -> tuple[EvidencePassage, ...]:
    """Split flattened text into overlapping character windows with provenance."""

    normalized = text.strip()
    if not normalized:
        raise ValueError("text must not be empty")
    if chunk_chars < 1:
        raise ValueError("chunk_chars must be positive")
    if not 0 <= overlap_chars < chunk_chars:
        raise ValueError("overlap_chars must be between 0 and chunk_chars - 1")

    prefix = source.source_id.split("-", 1)[0].upper()
    chunks: list[EvidencePassage] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_chars, len(normalized))
        chunks.append(
            EvidencePassage(
                passage_id=f"{prefix}-C{len(chunks) + 1:03d}",
                company=source.company,
                period=source.period,
                section="Naive character window",
                text=normalized[start:end],
                source_url=source.source_url,
            )
        )
        if end == len(normalized):
            break
        start = end - overlap_chars
    return tuple(chunks)
