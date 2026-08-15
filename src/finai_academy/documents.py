"""Provenance-preserving parsers for the financial document laboratory."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bs4 import BeautifulSoup

BlockType = Literal["heading", "paragraph", "table", "list", "footnote", "page_marker"]


def _clean_text(value: str) -> str:
    return " ".join(value.split())


@dataclass(frozen=True)
class DocumentSource:
    """Source-level metadata retained by every parsed block and chunk."""

    source_id: str
    company: str
    period: str
    document_type: str
    language: str
    source_url: str
    retrieval_date: str | None = None
    fixture_path: str | None = None
    fixture_sha256: str | None = None
    provenance_mode: str = "fixture"

    def __post_init__(self) -> None:
        required = {
            "source_id": self.source_id,
            "company": self.company,
            "period": self.period,
            "document_type": self.document_type,
            "language": self.language,
            "source_url": self.source_url,
        }
        for field_name, value in required.items():
            normalized = value.strip()
            if not normalized:
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, normalized)

    def verify_fixture(self, root: Path) -> bool:
        """Verify the compact fixture against the versioned content hash."""

        if not self.fixture_path or not self.fixture_sha256:
            return False
        payload = (root / self.fixture_path).read_bytes()
        return hashlib.sha256(payload).hexdigest() == self.fixture_sha256


@dataclass(frozen=True)
class DocumentBlock:
    """One ordered structural unit emitted by a document parser."""

    block_id: str
    source_id: str
    company: str
    period: str
    document_type: str
    language: str
    source_url: str
    ordinal: int
    block_type: BlockType
    text: str
    page_number: int | None = None
    section_path: tuple[str, ...] = ()
    table_rows: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("block_id", "source_id", "company", "period", "source_url", "text"):
            value = getattr(self, field_name).strip()
            if not value:
                raise ValueError(f"{field_name} must not be empty")
            object.__setattr__(self, field_name, value)
        if self.ordinal < 0:
            raise ValueError("ordinal must not be negative")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be positive")
        if self.block_type == "table" and not self.table_rows:
            raise ValueError("table blocks must contain table_rows")


def _new_block(
    source: DocumentSource,
    *,
    ordinal: int,
    block_type: BlockType,
    text: str,
    page_number: int | None = None,
    section_path: tuple[str, ...] = (),
    table_rows: tuple[tuple[str, ...], ...] = (),
) -> DocumentBlock:
    return DocumentBlock(
        block_id=f"{source.source_id}-B{ordinal + 1:03d}",
        source_id=source.source_id,
        company=source.company,
        period=source.period,
        document_type=source.document_type,
        language=source.language,
        source_url=source.source_url,
        ordinal=ordinal,
        block_type=block_type,
        text=_clean_text(text),
        page_number=page_number,
        section_path=section_path,
        table_rows=table_rows,
    )


def parse_html(path: Path, source: DocumentSource) -> list[DocumentBlock]:
    """Parse ordered headings, paragraphs, lists and tables from HTML."""

    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    root = soup.find("main") or soup.body or soup
    blocks: list[DocumentBlock] = []
    heading_stack: list[str] = []

    for element in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "table", "li"]):
        tag = element.name
        if tag == "li" and element.find_parent("li") is not None:
            continue
        if tag and tag.startswith("h"):
            level = int(tag[1])
            heading = _clean_text(element.get_text(" ", strip=True))
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading)
            block_type: BlockType = "heading"
            text = heading
            table_rows: tuple[tuple[str, ...], ...] = ()
        elif tag == "table":
            rows = []
            for row in element.find_all("tr"):
                cells = tuple(
                    _clean_text(cell.get_text(" ", strip=True))
                    for cell in row.find_all(["th", "td"])
                )
                if cells:
                    rows.append(cells)
            table_rows = tuple(rows)
            if not table_rows:
                continue
            block_type = "table"
            text = "\n".join(" | ".join(row) for row in table_rows)
        elif tag == "li":
            block_type = "list"
            text = element.get_text(" ", strip=True)
            table_rows = ()
        else:
            block_type = "paragraph"
            text = element.get_text(" ", strip=True)
            table_rows = ()

        if not _clean_text(text):
            continue
        blocks.append(
            _new_block(
                source,
                ordinal=len(blocks),
                block_type=block_type,
                text=text,
                section_path=tuple(heading_stack),
                table_rows=table_rows,
            )
        )
    return blocks


def parse_pdf(path: Path, source: DocumentSource) -> list[DocumentBlock]:
    """Parse pages and tables from a machine-generated PDF with pdfplumber."""

    try:
        import pdfplumber
    except ImportError as error:  # pragma: no cover - exercised by setup diagnostics
        raise RuntimeError("PDF parsing requires `uv sync --extra rag`.") from error

    blocks: list[DocumentBlock] = []
    with pdfplumber.open(path) as document:
        for page_number, page in enumerate(document.pages, start=1):
            raw_text = page.extract_text() or ""
            lines = [_clean_text(line) for line in raw_text.splitlines() if _clean_text(line)]
            tables = []
            for raw_table in page.extract_tables() or []:
                cleaned_rows = []
                for raw_row in raw_table:
                    row = tuple(_clean_text(cell or "") for cell in raw_row)
                    if any(row):
                        cleaned_rows.append(row)
                if cleaned_rows:
                    tables.append(tuple(cleaned_rows))

            heading = lines[0] if lines else f"Page {page_number}"
            section_path = (heading,)
            blocks.append(
                _new_block(
                    source,
                    ordinal=len(blocks),
                    block_type="heading",
                    text=heading,
                    page_number=page_number,
                    section_path=section_path,
                )
            )

            table_cell_text = {
                cell for table in tables for row in table for cell in row if cell
            }
            paragraph_lines = [
                line
                for line in lines[1:]
                if line not in table_cell_text
                and not all(cell in line for cell in table_cell_text if cell)
            ]
            if paragraph_lines:
                blocks.append(
                    _new_block(
                        source,
                        ordinal=len(blocks),
                        block_type="paragraph",
                        text=" ".join(paragraph_lines),
                        page_number=page_number,
                        section_path=section_path,
                    )
                )

            for table_rows in tables:
                blocks.append(
                    _new_block(
                        source,
                        ordinal=len(blocks),
                        block_type="table",
                        text="\n".join(" | ".join(row) for row in table_rows),
                        page_number=page_number,
                        section_path=section_path,
                        table_rows=table_rows,
                    )
                )
    return blocks


def load_source_manifest(path: Path) -> list[DocumentSource]:
    """Load versioned source metadata from the course-data manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [DocumentSource(**record) for record in payload["sources"]]


def normalize_for_hash(value: str) -> str:
    """Expose stable whitespace normalization for fixture tooling."""

    return re.sub(r"\s+", " ", value).strip()
