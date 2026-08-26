"""Native-text parsing and evidence rendering for certified financial PDFs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

import pymupdf
import pymupdf4llm

from finai_academy.capstone.document_assets import verify_source_asset
from finai_academy.capstone.document_models import (
    BoundingBox,
    DocumentElement,
    ExtractionDiagnostic,
    FinancialDocumentSource,
    ParsedDocument,
    TableMatrix,
)

_WHITESPACE = re.compile(r"\s+")
_PRINTED_PAGE = re.compile(r"\s*(\d{1,4})\s*")
_EMPTY_CELL = "\u200b"


class FinancialDocumentParser(Protocol):
    """Parses a certified local document into public, immutable elements."""

    def parse(
        self,
        source: FinancialDocumentSource,
        *,
        project_root: Path,
        pages: tuple[int, ...] | None = None,
    ) -> ParsedDocument: ...


class OcrAdapter(Protocol):
    """Optional OCR metadata provider; OCR engines remain application-injected."""

    engine: str
    language: str
    confidence: float | None


class PyMuPDF4LLMParser:
    """Normalize PyMuPDF4LLM JSON without exposing parser-local metadata."""

    parser_name = "pymupdf4llm"
    parser_version = "1.28.2"

    def __init__(self, ocr_adapter: OcrAdapter | None = None) -> None:
        self._ocr_adapter = ocr_adapter

    def parse(
        self,
        source: FinancialDocumentSource,
        *,
        project_root: Path,
        pages: tuple[int, ...] | None = None,
    ) -> ParsedDocument:
        _validate_pages(pages, page_count=source.page_count)
        verify_source_asset(source, project_root)
        selected = None if pages is None else [page - 1 for page in pages]
        raw = json.loads(
            pymupdf4llm.to_json(
                str(project_root / source.local_asset_key),
                pages=selected,
                use_ocr=False,
            )
        )
        return _normalize_document(source, raw, ocr_adapter=self._ocr_adapter)


def render_evidence_crop(
    source: FinancialDocumentSource,
    *,
    project_root: Path,
    page_number: int,
    bbox: BoundingBox | None,
    destination: Path,
    scale: float = 2.0,
) -> Path:
    """Render a deterministic RGB evidence crop using public one-based pages."""

    if page_number < 1 or page_number > source.page_count:
        raise ValueError("page_number must be within the certified document")
    if scale <= 0:
        raise ValueError("scale must be positive")
    verify_source_asset(source, project_root)
    document = pymupdf.open(project_root / source.local_asset_key)
    try:
        page = document[page_number - 1]
        clip = None if bbox is None else pymupdf.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        pixmap.save(destination)
    finally:
        document.close()
    return destination


def _validate_pages(pages: tuple[int, ...] | None, *, page_count: int) -> None:
    if pages is None:
        return
    if not pages:
        raise ValueError("pages must not be empty")
    if any(page < 1 or page > page_count for page in pages):
        raise ValueError("pages must use one-based numbers within the certified document")
    if len(set(pages)) != len(pages):
        raise ValueError("pages must not contain duplicates")


def _normalize_document(
    source: FinancialDocumentSource,
    raw: dict[str, Any],
    *,
    ocr_adapter: OcrAdapter | None,
) -> ParsedDocument:
    pending: list[dict[str, Any]] = []
    diagnostics: list[ExtractionDiagnostic] = []
    heading_path: tuple[str, ...] = ()

    for page in raw.get("pages", []):
        physical_page = int(page["page_number"])
        boxes = page.get("boxes", [])
        table_boxes = [box for box in boxes if box.get("boxclass") == "table"]
        if not _page_text(page.get("fulltext")).strip() and not table_boxes:
            diagnostics.append(_ocr_diagnostic(physical_page, ocr_adapter))

        printed_page = _printed_page(boxes)
        for box in boxes:
            element = _normalize_box(
                source,
                box,
                physical_page=physical_page,
                printed_page=printed_page,
                heading_path=heading_path,
            )
            if element is None:
                continue
            if element["element_type"] == "heading":
                level = max(1, int(box.get("header_level") or 1))
                heading_path = heading_path[: level - 1] + (element["original_text"],)
                element["heading_path"] = heading_path
            pending.append(element)

    elements = _finalize_elements(source, pending)
    return ParsedDocument(
        source=source,
        parser_name=PyMuPDF4LLMParser.parser_name,
        parser_version=PyMuPDF4LLMParser.parser_version,
        elements=elements,
        diagnostics=tuple(diagnostics),
    )


def _normalize_box(
    source: FinancialDocumentSource,
    box: dict[str, Any],
    *,
    physical_page: int,
    printed_page: int | None,
    heading_path: tuple[str, ...],
) -> dict[str, Any] | None:
    boxclass = str(box.get("boxclass", ""))
    if boxclass in {"page-header", "page-footer"}:
        return None
    bbox = _bbox(box)
    if bbox is None:
        return None
    if boxclass == "table":
        table_payload = box.get("table") or {}
        table = _table_matrix(table_payload)
        original_markdown = str(table_payload.get("markdown") or "").strip()
        if not original_markdown:
            return None
        return {
            "physical_page": physical_page,
            "printed_page": printed_page,
            "element_type": "table",
            "bbox": bbox,
            "original_text": _table_text(table.rows),
            "original_markdown": original_markdown,
            "table": table,
            "heading_path": heading_path,
        }

    text = _box_text(box.get("textlines"))
    if not text:
        return None
    element_type = _element_type(boxclass, int(box.get("header_level") or 0))
    return {
        "physical_page": physical_page,
        "printed_page": printed_page,
        "element_type": element_type,
        "bbox": bbox,
        "original_text": text,
        "original_markdown": None,
        "table": None,
        "heading_path": heading_path,
    }


def _bbox(box: dict[str, Any]) -> BoundingBox | None:
    try:
        return BoundingBox(
            x0=float(box["x0"]),
            y0=float(box["y0"]),
            x1=float(box["x1"]),
            y1=float(box["y1"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _table_matrix(table: dict[str, Any]) -> TableMatrix:
    rows = tuple(
        tuple(_cell_text(cell) for cell in row)
        for row in table.get("extract", [])
    )
    row_count = int(table.get("row_count", len(rows)))
    column_count = int(table.get("col_count", len(rows[0]) if rows else 0))
    markdown = str(table.get("markdown") or "").strip()
    return TableMatrix(
        rows=rows,
        row_count=row_count,
        column_count=column_count,
        markdown=markdown,
    )


def _cell_text(value: Any) -> str:
    normalized = _collapse_whitespace(str(value or ""))
    return normalized or _EMPTY_CELL


def _table_text(rows: Iterable[Iterable[str]]) -> str:
    return "\n".join(" | ".join(row) for row in rows)


def _box_text(textlines: Any) -> str:
    if not isinstance(textlines, list):
        return ""
    lines = []
    for line in textlines:
        spans = line.get("spans", []) if isinstance(line, dict) else []
        text = _collapse_whitespace("".join(str(span.get("text", "")) for span in spans))
        if text:
            lines.append(text)
    return "\n".join(lines)


def _page_text(fulltext: Any) -> str:
    if isinstance(fulltext, str):
        return fulltext
    if not isinstance(fulltext, list):
        return ""
    lines = []
    for block in fulltext:
        if not isinstance(block, dict):
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", []) if isinstance(line, dict) else []
            lines.extend(str(span.get("text", "")) for span in spans)
    return "".join(lines)


def _printed_page(boxes: list[dict[str, Any]]) -> int | None:
    for box in reversed(boxes):
        if box.get("boxclass") != "page-footer":
            continue
        match = _PRINTED_PAGE.fullmatch(_box_text(box.get("textlines")))
        if match:
            return int(match.group(1))
    return None


def _element_type(boxclass: str, header_level: int) -> str:
    if header_level > 0 or boxclass in {"title", "heading"}:
        return "heading"
    if boxclass in {"list-item", "list"}:
        return "list"
    if boxclass in {"figure-caption", "caption"}:
        return "figure_caption"
    if boxclass in {"footnote", "footer"}:
        return "footnote"
    return "paragraph"


def _collapse_whitespace(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def _finalize_elements(
    source: FinancialDocumentSource, pending: list[dict[str, Any]]
) -> tuple[DocumentElement, ...]:
    ids = [
        _element_id(source, ordinal=index, **element)
        for index, element in enumerate(pending)
    ]
    return tuple(
        DocumentElement(
            element_id=element_id,
            document_id=source.document_id,
            ordinal=index,
            physical_page=element["physical_page"],
            printed_page=element["printed_page"],
            element_type=element["element_type"],
            bbox=element["bbox"],
            original_text=element["original_text"],
            original_markdown=element["original_markdown"],
            table=element["table"],
            heading_path=element["heading_path"],
            previous_element_id=ids[index - 1] if index else None,
            next_element_id=ids[index + 1] if index + 1 < len(ids) else None,
        )
        for index, (element_id, element) in enumerate(zip(ids, pending, strict=True))
    )


def _element_id(
    source: FinancialDocumentSource,
    *,
    ordinal: int,
    physical_page: int,
    element_type: str,
    bbox: BoundingBox,
    original_text: str,
    original_markdown: str | None,
    table: TableMatrix | None,
    **_: Any,
) -> str:
    payload = {
        "document_sha256": source.sha256,
        "physical_page": physical_page,
        "ordinal": ordinal,
        "element_type": element_type,
        "bbox": [round(value, 6) for value in (bbox.x0, bbox.y0, bbox.x1, bbox.y1)],
        "original_text": original_text,
        "original_markdown": original_markdown,
        "table": table.model_dump(mode="json") if table else None,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _ocr_diagnostic(
    physical_page: int, ocr_adapter: OcrAdapter | None
) -> ExtractionDiagnostic:
    if ocr_adapter is None:
        return ExtractionDiagnostic(
            code="ocr_required",
            severity="error",
            physical_page=physical_page,
            message="Page has no usable native text layer.",
            extraction_method="native_text",
        )
    confidence = "unknown" if ocr_adapter.confidence is None else str(ocr_adapter.confidence)
    return ExtractionDiagnostic(
        code="ocr_required",
        severity="warning",
        physical_page=physical_page,
        message=(
            "Page has no usable native text layer; "
            f"OCR engine={ocr_adapter.engine}, language={ocr_adapter.language}, confidence={confidence}."
        ),
        extraction_method="ocr",
    )
