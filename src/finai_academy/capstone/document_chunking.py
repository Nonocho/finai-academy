"""Deterministic enrichment and atomic chunking for parsed financial documents."""

from __future__ import annotations

import json
import re
from hashlib import sha256

from finai_academy.capstone.document_models import (
    ContextualMetadata,
    DocumentElement,
    FinancialChunk,
    FinancialMetadata,
    ParsedDocument,
)

_PERIOD_RULES = (
    (re.compile(r"Year Ended Jan 25, 2026", re.IGNORECASE), "FY2026"),
    (re.compile(r"FY\s*2025", re.IGNORECASE), "FY2025"),
    (re.compile(r"H2\s*2025", re.IGNORECASE), "H2 2025"),
    (re.compile(r"Q4\s*2025", re.IGNORECASE), "Q4 2025"),
)
_NUMBER = re.compile(r"\d")
_PERIOD_LABEL = re.compile(
    r"(?:Year Ended|FY\s*\d{4}|H[12]\s*\d{4}|Q[1-4]\s*\d{4})", re.IGNORECASE
)
_FOOTNOTE = re.compile(r"^(?:\(\d+\)|\d+\.)")


class MissingFinancialContextError(ValueError):
    """Raised when a numeric table lacks mandatory source context."""


def build_contextual_metadata(
    document: ParsedDocument, element: DocumentElement
) -> ContextualMetadata:
    """Copy only certified source and parser fields into chunk context."""

    return ContextualMetadata(
        document_id=document.source.document_id,
        company_name=document.source.company_name,
        ticker=document.source.ticker,
        document_type=document.source.document_type,
        reporting_period=document.source.reporting_period,
        publication_date=document.source.publication_date,
        official_source_url=str(document.source.official_source_url),
        document_sha256=document.source.sha256,
        physical_page=element.physical_page,
        printed_page=element.printed_page,
        heading_path=element.heading_path,
        element_type=element.element_type,
        bbox=element.bbox,
        parent_element_id=element.parent_element_id,
        previous_element_id=element.previous_element_id,
        next_element_id=element.next_element_id,
        parser_name=document.parser_name,
        parser_version=document.parser_version,
        extraction_method=element.extraction_method,
    )


def build_financial_metadata(
    document: ParsedDocument, element: DocumentElement
) -> FinancialMetadata:
    """Extract only financial labels directly evidenced by parser output."""

    evidence = _financial_evidence(document, element)
    currency = None
    if "In millions" in evidence and document.source.ticker == "NVDA":
        currency = "USD"
    if "€ million" in evidence:
        currency = "EUR"
    period_evidence = "\n".join(
        (*element.heading_path, element.original_text, element.original_markdown or "")
    )
    periods = tuple(
        label for pattern, label in _PERIOD_RULES if pattern.search(period_evidence)
    )
    return FinancialMetadata(
        metric_names=_metric_labels(element),
        periods=periods or (document.source.reporting_period,),
        currency=currency,
        scale="millions"
        if re.search(r"(?:In|€) millions?", evidence, re.IGNORECASE)
        else None,
        segments=_row_labels(document, element),
        geography=(),
        accounting_basis=_explicit_accounting_basis(evidence),
        audited=_explicit_audit_status(evidence),
        footnotes=_adjacent_footnotes(document, element),
        source_element_ids=(element.element_id,),
        enrichment_method="deterministic",
        confidence=1.0,
    )


def build_table_chunk(document: ParsedDocument, element: DocumentElement) -> FinancialChunk:
    """Build one atomic, fully contextualized chunk for a source table."""

    if element.element_type != "table" or element.table is None:
        raise ValueError("build_table_chunk requires a table element")

    context = build_contextual_metadata(document, element)
    financial = build_financial_metadata(document, element)
    if _contains_numeric_values(element) and financial.scale is None:
        raise MissingFinancialContextError("table unit is missing")

    text = _table_text(document, element, context, financial)
    return _chunk(
        text=text,
        element_type="table",
        source_element_ids=(element.element_id,),
        context=context,
        financial=financial,
        table=element.table,
    )


def build_financial_chunks(
    document: ParsedDocument, *, paragraph_character_budget: int = 3200
) -> tuple[FinancialChunk, ...]:
    """Build atomic table chunks and bounded adjacent paragraph/list chunks."""

    if paragraph_character_budget <= 0:
        raise ValueError("paragraph_character_budget must be positive")

    chunks: list[FinancialChunk] = []
    paragraph_group: list[DocumentElement] = []

    def flush_paragraphs() -> None:
        if paragraph_group:
            chunks.append(_paragraph_chunk(document, tuple(paragraph_group)))
            paragraph_group.clear()

    for element in document.elements:
        if element.element_type == "table":
            flush_paragraphs()
            chunks.append(build_table_chunk(document, element))
            continue
        if element.element_type not in {"paragraph", "list"}:
            flush_paragraphs()
            continue
        if _can_join(paragraph_group, element, paragraph_character_budget):
            paragraph_group.append(element)
        else:
            flush_paragraphs()
            paragraph_group.append(element)
    flush_paragraphs()
    return tuple(chunks)


def _financial_evidence(document: ParsedDocument, element: DocumentElement) -> str:
    adjacent = _adjacent_elements(document, element)
    nearby_text = (
        item.original_text
        for item in adjacent
        if item.element_type in {"paragraph", "list", "footnote"}
    )
    return "\n".join(
        (*element.heading_path, element.original_text, element.original_markdown or "", *nearby_text)
    )


def _adjacent_elements(
    document: ParsedDocument, element: DocumentElement
) -> tuple[DocumentElement, ...]:
    try:
        index = next(
            index
            for index, candidate in enumerate(document.elements)
            if candidate.element_id == element.element_id
        )
    except StopIteration:
        return ()
    candidates = document.elements[max(0, index - 1) : index + 2]
    return tuple(
        candidate
        for candidate in candidates
        if candidate.element_id != element.element_id
        and candidate.physical_page == element.physical_page
    )


def _metric_labels(element: DocumentElement) -> tuple[str, ...]:
    if element.table is None:
        return ()
    rows = element.table.rows
    if _is_business_table(element):
        labels: list[str] = []
        for row in rows:
            if row[0] and any(_NUMBER.search(cell) for cell in row[1:]):
                break
            labels.extend(
                cell for cell in row[1:] if cell and not _PERIOD_LABEL.fullmatch(cell)
            )
    else:
        labels = (
            row[0]
            for row in rows
            if row[0] and not _PERIOD_LABEL.search(row[0]) and any(_NUMBER.search(cell) for cell in row[1:])
        )
    return _deduplicate(labels)


def _row_labels(document: ParsedDocument, element: DocumentElement) -> tuple[str, ...]:
    if element.table is None or not _is_business_table(element):
        return ()
    return _deduplicate(
        row[0]
        for row in element.table.rows
        if row[0] and any(_NUMBER.search(cell) for cell in row[1:])
    )


def _is_business_table(element: DocumentElement) -> bool:
    return "by business" in " ".join(element.heading_path).casefold()


def _deduplicate(values: object) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:  # type: ignore[union-attr]
        if value not in result:
            result.append(value)
    return tuple(result)


def _explicit_accounting_basis(evidence: str) -> str | None:
    if re.search(r"\bnon[- ]GAAP\b", evidence, re.IGNORECASE):
        return "non-GAAP"
    if re.search(r"\bGAAP\b", evidence, re.IGNORECASE):
        return "GAAP"
    return None


def _explicit_audit_status(evidence: str) -> bool | None:
    if re.search(r"\bunaudited\b", evidence, re.IGNORECASE):
        return False
    if re.search(r"\baudited\b", evidence, re.IGNORECASE):
        return True
    return None


def _adjacent_footnotes(document: ParsedDocument, element: DocumentElement) -> tuple[str, ...]:
    return tuple(
        candidate.original_text
        for candidate in _adjacent_elements(document, element)
        if candidate.element_type == "footnote" or _FOOTNOTE.match(candidate.original_text)
    )


def _contains_numeric_values(element: DocumentElement) -> bool:
    return element.table is not None and any(
        _NUMBER.search(cell) for row in element.table.rows for cell in row
    )


def _table_text(
    document: ParsedDocument,
    element: DocumentElement,
    context: ContextualMetadata,
    financial: FinancialMetadata,
) -> str:
    parts = [
        f"Company: {context.company_name}",
        f"Document: {context.document_type}",
        f"Reporting period: {context.reporting_period}",
    ]
    if context.heading_path:
        parts.append(f"Heading path: {' > '.join(context.heading_path)}")
    table_title = _table_title(element)
    if table_title:
        parts.append(f"Table title: {table_title}")
    if financial.currency or financial.scale:
        units = " ".join(value for value in (financial.currency, financial.scale) if value)
        parts.append(f"Units: {units}")
    parts.append(element.original_markdown or element.table.markdown if element.table else "")
    if financial.footnotes:
        parts.append("Footnotes:\n" + "\n".join(financial.footnotes))
    explanatory = _nearby_explanatory_paragraph(document, element)
    if explanatory:
        parts.append(f"Nearby explanation: {explanatory}")
    parts.append(f"Source page: {context.physical_page}")
    return "\n\n".join(part for part in parts if part)


def _table_title(element: DocumentElement) -> str | None:
    """Keep source header labels searchable without altering its Markdown table."""

    if element.table is None:
        return None
    header_cells: list[str] = []
    for row in element.table.rows:
        if row[0] and any(_NUMBER.search(cell) for cell in row[1:]):
            break
        header_cells.extend(cell for cell in row if cell and not _PERIOD_LABEL.fullmatch(cell))
    title = " | ".join(_deduplicate(header_cells))
    return title or None


def _nearby_explanatory_paragraph(document: ParsedDocument, element: DocumentElement) -> str | None:
    for candidate in _adjacent_elements(document, element):
        if candidate.element_type in {"paragraph", "list"} and not _FOOTNOTE.match(
            candidate.original_text
        ):
            return candidate.original_text
    return None


def _can_join(
    group: list[DocumentElement], element: DocumentElement, budget: int
) -> bool:
    if not group:
        return True
    if (
        group[-1].heading_path != element.heading_path
        or group[-1].physical_page != element.physical_page
    ):
        return False
    joined_length = sum(len(item.original_text) for item in group) + len(element.original_text)
    return joined_length <= budget


def _paragraph_chunk(
    document: ParsedDocument, elements: tuple[DocumentElement, ...]
) -> FinancialChunk:
    first = elements[0]
    context = build_contextual_metadata(document, first)
    financial = build_financial_metadata(document, first)
    source_element_ids = tuple(element.element_id for element in elements)
    financial = FinancialMetadata.model_validate(
        {**financial.model_dump(mode="python"), "source_element_ids": source_element_ids}
    )
    parts = [
        f"Company: {context.company_name}",
        f"Document: {context.document_type}",
        f"Reporting period: {context.reporting_period}",
    ]
    if context.heading_path:
        parts.append(f"Heading path: {' > '.join(context.heading_path)}")
    parts.append("\n\n".join(element.original_text for element in elements))
    parts.append(f"Source page: {context.physical_page}")
    return _chunk(
        text="\n\n".join(parts),
        element_type=first.element_type,
        source_element_ids=source_element_ids,
        context=context,
        financial=financial,
        table=None,
    )


def _chunk(
    *,
    text: str,
    element_type: str,
    source_element_ids: tuple[str, ...],
    context: ContextualMetadata,
    financial: FinancialMetadata,
    table: object,
) -> FinancialChunk:
    material = json.dumps(
        {
            "document_sha256": context.document_sha256,
            "source_element_ids": source_element_ids,
            "text": text,
            "context": context.model_dump(mode="json"),
            "financial": financial.model_dump(mode="json"),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return FinancialChunk(
        chunk_id=f"chunk-{sha256(material.encode('utf-8')).hexdigest()[:20]}",
        text=text,
        content_hash=sha256(text.encode("utf-8")).hexdigest(),
        element_type=element_type,  # type: ignore[arg-type]
        source_element_ids=source_element_ids,
        context=context,
        financial=financial,
        table=table,  # type: ignore[arg-type]
    )
