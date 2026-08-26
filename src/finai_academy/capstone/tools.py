"""Certified document retrieval and fail-closed capstone tool access."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from finai_academy.capstone.document_tools import (
    DocumentCapabilityRegistry,
    DocumentEvidenceOutcome,
    DocumentSearchOutcome,
    ReportedValue,
    ReportedValueComparison,
    build_document_capability_registry,
    compare_reported_values,
)
from finai_academy.capstone.models import CapstoneEvidenceHit

MANDATORY_ANALYST_TOOLS = frozenset(
    {
        "search_financial_documents",
        "inspect_document_evidence",
        "compare_reported_values",
    }
)
_INVALID_ARGUMENTS_MESSAGE = "Tool arguments must match the approved schema."
_REPORTING_PERIOD_BY_COMPANY = {
    "nvidia": "FY2026",
    "schneider electric": "FY2025",
}


class ToolOutcome(BaseModel):
    """A public success or typed validation failure from an approved analyst tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok", "error"]
    payload: DocumentSearchOutcome | DocumentEvidenceOutcome | ReportedValueComparison | None = None
    error_code: str | None = None
    message: str | None = None
    retryable: bool = False


class CertifiedRetriever:
    """Compatibility retriever backed by the certified document capability registry."""

    def __init__(self, capability_registry: DocumentCapabilityRegistry) -> None:
        self._capability_registry = capability_registry

    def search(self, company: str, query: str, top_k: int = 2) -> tuple[CapstoneEvidenceHit, ...]:
        """Return document-backed evidence without rebuilding ranking outside the index."""

        reporting_period = _REPORTING_PERIOD_BY_COMPANY.get(company.casefold().strip())
        if reporting_period is None:
            return ()
        outcome = self._capability_registry.search_financial_documents(
            company=company,
            reporting_period=reporting_period,
            query=query,
            top_k=top_k,
        )
        return tuple(_to_capstone_evidence_hit(hit) for hit in outcome.hits)


def build_certified_retriever(root: Path | None = None) -> CertifiedRetriever:
    """Build the legacy retriever facade from the verified full-document index."""

    return CertifiedRetriever(build_document_capability_registry(root))


class AnalystToolRegistry:
    """Expose exactly the three deterministic document-research capabilities."""

    def __init__(
        self,
        discovered: Sequence[str],
        capability_registry: DocumentCapabilityRegistry | None = None,
    ) -> None:
        self._runtime_discovered = frozenset(discovered)
        self._capability_registry = capability_registry or build_document_capability_registry()

    def discover(self) -> tuple[str, ...]:
        """Return the fixed policy intersection of runtime-discovered capabilities."""

        return tuple(sorted(self._runtime_discovered & MANDATORY_ANALYST_TOOLS))

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
        """Run one approved tool after exact schema validation without echoing rejected input."""

        if name not in MANDATORY_ANALYST_TOOLS:
            raise ValueError("Tool is not allowlisted.")
        if name not in self.discover():
            raise ValueError("Tool was not discovered.")
        validated_arguments = _validated_arguments(name, arguments)
        if validated_arguments is None:
            return _invalid_arguments_outcome()
        try:
            if name == "search_financial_documents":
                result = self._capability_registry.search_financial_documents(**validated_arguments)
            elif name == "inspect_document_evidence":
                result = self._capability_registry.inspect_document_evidence(**validated_arguments)
            else:
                result = compare_reported_values(**validated_arguments)
        except (TypeError, ValidationError, ValueError):
            return _invalid_arguments_outcome()
        return ToolOutcome(status="ok", payload=result)


def _to_capstone_evidence_hit(hit) -> CapstoneEvidenceHit:
    chunk = hit.retrieval.chunk
    return CapstoneEvidenceHit(
        company=chunk.context.company_name,
        text=chunk.text,
        evidence_id=chunk.chunk_id,
        document_id=chunk.context.document_id,
        section=" > ".join(chunk.context.heading_path) or chunk.element_type,
        period=chunk.context.reporting_period,
        source_reference=chunk.context.official_source_url,
    )


def _validated_arguments(name: str, arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    """Accept only complete approved schemas before forwarding document capability calls."""

    if not isinstance(arguments, Mapping):
        return None
    if name == "search_financial_documents":
        allowed_fields = {"company", "reporting_period", "query", "element_type", "top_k"}
        required_fields = {"company", "reporting_period", "query"}
        if not required_fields <= set(arguments) or not set(arguments) <= allowed_fields:
            return None
        values = {
            "company": arguments.get("company"),
            "reporting_period": arguments.get("reporting_period"),
            "query": arguments.get("query"),
            "element_type": arguments.get("element_type"),
            "top_k": arguments.get("top_k", 3),
        }
        if not all(isinstance(values[field], str) for field in required_fields):
            return None
        if values["element_type"] is not None and not isinstance(values["element_type"], str):
            return None
        if not isinstance(values["top_k"], int) or isinstance(values["top_k"], bool):
            return None
        return values
    if name == "inspect_document_evidence":
        if set(arguments) != {"chunk_id"} or not isinstance(arguments.get("chunk_id"), str):
            return None
        return {"chunk_id": arguments["chunk_id"]}
    if set(arguments) != {"left", "right"}:
        return None
    try:
        return {
            "left": ReportedValue.model_validate(arguments["left"]),
            "right": ReportedValue.model_validate(arguments["right"]),
        }
    except (TypeError, ValidationError, ValueError):
        return None


def _invalid_arguments_outcome() -> ToolOutcome:
    """Return the stable public schema failure without returning untrusted input."""

    return ToolOutcome(
        status="error",
        error_code="invalid_arguments",
        message=_INVALID_ARGUMENTS_MESSAGE,
        retryable=True,
    )
