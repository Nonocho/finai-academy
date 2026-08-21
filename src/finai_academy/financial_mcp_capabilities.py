"""Pure, deterministic financial capabilities for the Lesson 10 MCP server."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from finai_academy.self_correcting_agent import (
    MetricRegistry,
    MetricRequest,
    build_metric_registry,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = ROOT / "assets/course-data/market/lesson09_metrics_snapshot_v1.json"
DEFAULT_EVIDENCE = ROOT / "assets/course-data/mcp/lesson10_evidence_catalog_v1.json"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class CapabilityError(BaseModel):
    status: Literal["error"] = "error"
    error_code: str
    message: str
    rejected_value: str | int | None = None
    valid_values: tuple[str, ...] = ()
    retryable: bool = False


class CapabilityValidationError(ValueError):
    def __init__(self, error: CapabilityError) -> None:
        self.error = error
        super().__init__(error.model_dump_json())


class CompanyCoverage(BaseModel):
    ticker: str
    company: str


class CoverageSnapshot(BaseModel):
    dataset_id: str
    as_of: str
    companies: tuple[CompanyCoverage, ...]
    tickers: tuple[str, ...]
    supported_metrics: tuple[str, ...]
    document_ids: tuple[str, ...]
    source_notice: str


class MetricResult(BaseModel):
    status: Literal["ok"] = "ok"
    ticker: str
    company: str
    metric: str
    value: float
    unit: str
    as_of: str
    source: str


class DocumentHit(BaseModel):
    evidence_id: str
    text: str
    document_id: str
    section: str
    period: str
    source: str


class DocumentSearchResult(BaseModel):
    status: Literal["ok"] = "ok"
    query: str
    company: str
    hits: tuple[DocumentHit, ...]
    trace_id: str


class FinancialCapabilityRegistry:
    """Read controlled course fixtures through stable financial capability contracts."""

    def __init__(self, metric_payload: Mapping[str, Any], evidence_payload: Mapping[str, Any]) -> None:
        self._metric_registry: MetricRegistry = build_metric_registry(metric_payload)
        self._evidence_dataset_id = _required_string(evidence_payload, "dataset_id")
        self._evidence_as_of = _required_string(evidence_payload, "as_of")
        self._source_notice = _required_string(evidence_payload, "notice")
        documents = evidence_payload.get("documents")
        if not isinstance(documents, list) or not documents:
            raise ValueError("evidence catalog requires non-empty documents")
        self._documents = tuple(_validated_document(document) for document in documents)
        self._company_by_ticker = self._build_company_map(metric_payload)

    def coverage(self) -> CoverageSnapshot:
        companies = tuple(
            CompanyCoverage(ticker=ticker, company=company)
            for ticker, company in sorted(self._company_by_ticker.items())
        )
        return CoverageSnapshot(
            dataset_id=self._evidence_dataset_id,
            as_of=self._evidence_as_of,
            companies=companies,
            tickers=tuple(item.ticker for item in companies),
            supported_metrics=self._metric_registry.metric_names,
            document_ids=tuple(sorted({document["document_id"] for document in self._documents})),
            source_notice=self._source_notice,
        )

    def get_company_metric(self, ticker: str, metric: str) -> MetricResult:
        normalized_ticker = ticker.upper().strip()
        normalized_metric = metric.strip()
        observation = self._metric_registry.invoke(
            MetricRequest(ticker=normalized_ticker, metric=normalized_metric)
        )
        if observation.status == "error":
            valid_values = (
                self._metric_registry.tickers
                if observation.error_code == "unsupported_ticker"
                else self._metric_registry.metric_names
            )
            raise CapabilityValidationError(
                CapabilityError(
                    error_code=observation.error_code or "invalid_metric_request",
                    message=observation.message,
                    rejected_value=ticker if observation.error_code == "unsupported_ticker" else metric,
                    valid_values=valid_values,
                    retryable=observation.retryable,
                )
            )

        payload = observation.payload
        return MetricResult(
            ticker=str(payload["ticker"]),
            company=str(payload["company"]),
            metric=str(payload["metric"]),
            value=float(payload["value"]),
            unit=_metric_unit(str(payload["metric"])),
            as_of=str(payload["as_of"]),
            source=str(payload["source"]),
        )

    def search_financial_documents(
        self, company: str, query: str, top_k: int = 2
    ) -> DocumentSearchResult:
        if not 1 <= top_k <= 3:
            raise CapabilityValidationError(
                CapabilityError(
                    error_code="invalid_top_k",
                    message="top_k must be between 1 and 3.",
                    rejected_value=top_k,
                    valid_values=("1", "2", "3"),
                    retryable=True,
                )
            )

        normalized_company = self._normalize_company(company)
        normalized_query = query.strip()
        if not normalized_query:
            raise CapabilityValidationError(
                CapabilityError(
                    error_code="invalid_query",
                    message="query must not be blank.",
                    rejected_value=query,
                    retryable=True,
                )
            )

        query_tokens = _tokens(normalized_query)
        ranked_documents = sorted(
            (
                document
                for document in self._documents
                if document["company"] == normalized_company
            ),
            key=lambda document: (
                -len(query_tokens & _tokens(document["text"])),
                document["evidence_id"],
            ),
        )
        hits = tuple(
            DocumentHit(
                evidence_id=document["evidence_id"],
                text=document["text"],
                document_id=document["document_id"],
                section=document["section"],
                period=document["period"],
                source=document["source"],
            )
            for document in ranked_documents[:top_k]
        )
        trace_material = f"{normalized_company}|{normalized_query.casefold()}|{top_k}"
        return DocumentSearchResult(
            query=normalized_query,
            company=normalized_company,
            hits=hits,
            trace_id=f"lesson10-{sha256(trace_material.encode()).hexdigest()[:12]}",
        )

    def _build_company_map(self, metric_payload: Mapping[str, Any]) -> dict[str, str]:
        metrics = metric_payload.get("metrics")
        if not isinstance(metrics, Mapping):
            raise TypeError("metric snapshot requires metrics")
        company_by_ticker: dict[str, str] = {}
        for ticker, record in metrics.items():
            if not isinstance(record, Mapping) or not isinstance(record.get("company"), str):
                raise TypeError("metric records require company names")
            company_by_ticker[str(ticker)] = str(record["company"])
        return company_by_ticker

    def _normalize_company(self, company: str) -> str:
        aliases = {
            "nvidia": "NVIDIA",
            "nvda": "NVIDIA",
            "schneider": "Schneider Electric",
            "schneider electric": "Schneider Electric",
            "su.pa": "Schneider Electric",
        }
        normalized_company = aliases.get(company.casefold().strip())
        if normalized_company is None:
            raise CapabilityValidationError(
                CapabilityError(
                    error_code="unsupported_company",
                    message="Unknown company. Valid companies: NVIDIA, Schneider Electric.",
                    rejected_value=company,
                    valid_values=("NVIDIA", "Schneider Electric"),
                    retryable=True,
                )
            )
        return normalized_company


def build_financial_capability_registry(
    *,
    metric_snapshot_path: Path | None = None,
    evidence_catalog_path: Path | None = None,
) -> FinancialCapabilityRegistry:
    metric_payload = json.loads((metric_snapshot_path or DEFAULT_METRICS).read_text(encoding="utf-8"))
    evidence_payload = json.loads(
        (evidence_catalog_path or DEFAULT_EVIDENCE).read_text(encoding="utf-8")
    )
    return FinancialCapabilityRegistry(metric_payload, evidence_payload)


def _metric_unit(metric: str) -> str:
    return "x" if metric == "P/E" else "reported currency per share"


def _required_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"evidence catalog requires non-empty {field}")
    return value


def _validated_document(document: Any) -> dict[str, str]:
    if not isinstance(document, Mapping):
        raise TypeError("evidence catalog documents must be objects")
    required_fields = (
        "evidence_id",
        "company",
        "ticker",
        "document_id",
        "period",
        "section",
        "text",
        "source",
    )
    validated = {field: document.get(field) for field in required_fields}
    if not all(isinstance(value, str) and value.strip() for value in validated.values()):
        raise ValueError("evidence catalog documents require complete string metadata")
    return {field: str(value) for field, value in validated.items()}


def _tokens(text: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(text.casefold()))
