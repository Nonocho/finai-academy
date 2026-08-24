"""Certified retrieval and fail-closed financial tool access for the capstone."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from finai_academy.capstone.models import CapstoneEvidenceHit, _clean_public_value
from finai_academy.financial_mcp_capabilities import (
    CapabilityValidationError,
    DocumentSearchResult,
    FinancialCapabilityRegistry,
    MetricResult,
    build_financial_capability_registry,
)
from finai_academy.financial_mcp_client import ALLOWED_TOOLS
from finai_academy.hybrid_retrieval import (
    DenseIndex,
    DeterministicTeachingEmbeddings,
    IndexedPassage,
    KeywordIndex,
    RetrievalFilters,
    reciprocal_rank_fusion,
)

MANDATORY_ANALYST_TOOLS = frozenset({"get_company_metric", "search_financial_documents"})
_EVIDENCE_CATALOG = (
    Path(__file__).resolve().parents[3] / "assets/course-data/mcp/lesson10_evidence_catalog_v1.json"
)
_INVALID_ARGUMENTS_MESSAGE = "Tool arguments must match the approved schema."


class ToolOutcome(BaseModel):
    """A public success or typed validation failure from an approved analyst tool."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok", "error"]
    payload: MetricResult | DocumentSearchResult | None = None
    error_code: str | None = None
    message: str | None = None
    retryable: bool = False


class CertifiedRetriever:
    """Search the versioned classroom catalog without crossing company boundaries."""

    def __init__(
        self,
        passages: Sequence[IndexedPassage],
        evidence_by_id: Mapping[str, CapstoneEvidenceHit],
        capability_registry: FinancialCapabilityRegistry,
    ) -> None:
        self._passages = tuple(passages)
        self._evidence_by_id = dict(evidence_by_id)
        self._capability_registry = capability_registry
        self._keyword_index = KeywordIndex(self._passages)
        embeddings = DeterministicTeachingEmbeddings()
        self._dense_index = DenseIndex(
            self._passages,
            embeddings,
            provider="certified-fixture",
            model=embeddings.model_name,
            chunking_strategy="evidence-catalog-v1",
        )

    def search(self, company: str, query: str, top_k: int = 2) -> tuple[CapstoneEvidenceHit, ...]:
        """Return no more than ``top_k`` hits after fixture-backed validation."""

        validated = self._capability_registry.search_financial_documents(company, query, top_k)
        company_filter = RetrievalFilters(company=validated.company)
        rankings = {
            "keyword": self._keyword_index.search(query, top_k=top_k, filters=company_filter),
            "dense": self._dense_index.search(query, top_k=top_k, filters=company_filter),
        }
        fused_hits = reciprocal_rank_fusion(rankings)
        return tuple(
            self._evidence_by_id[hit.passage.passage_id] for hit in fused_hits[:top_k]
        )


def build_certified_retriever() -> CertifiedRetriever:
    """Load the tracked Lesson 10 evidence catalog into a deterministic hybrid retriever."""

    payload = json.loads(_EVIDENCE_CATALOG.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("certified evidence catalog requires documents")

    passages: list[IndexedPassage] = []
    evidence_by_id: dict[str, CapstoneEvidenceHit] = {}
    for document in documents:
        if not isinstance(document, Mapping):
            raise TypeError("certified evidence catalog documents must be objects")
        evidence = CapstoneEvidenceHit(
            company=_required_catalog_string(document, "company"),
            text=_required_catalog_string(document, "text"),
            evidence_id=_required_catalog_string(document, "evidence_id"),
            document_id=_required_catalog_string(document, "document_id"),
            section=_required_catalog_string(document, "section"),
            period=_required_catalog_string(document, "period"),
            source_reference=_required_catalog_string(document, "source"),
        )
        if evidence.evidence_id in evidence_by_id:
            raise ValueError("certified evidence IDs must be unique")
        evidence_by_id[evidence.evidence_id] = evidence
        passages.append(
            IndexedPassage(
                passage_id=evidence.evidence_id,
                company=evidence.company,
                period=evidence.period,
                section=evidence.section,
                text=evidence.text,
                source_url=evidence.source_reference,
            )
        )

    return CertifiedRetriever(passages, evidence_by_id, build_financial_capability_registry())


class AnalystToolRegistry:
    """Expose only the two approved, deterministic financial read tools."""

    def __init__(
        self,
        discovered: Sequence[str],
        capability_registry: FinancialCapabilityRegistry | None = None,
    ) -> None:
        self._runtime_discovered = frozenset(discovered)
        self._capability_registry = capability_registry or build_financial_capability_registry()

    def discover(self) -> tuple[str, ...]:
        """Return the static-policy intersection of runtime capability discovery."""

        return tuple(sorted(self._runtime_discovered & MANDATORY_ANALYST_TOOLS & ALLOWED_TOOLS))

    def invoke(self, name: str, arguments: Mapping[str, Any]) -> ToolOutcome:
        """Run an approved, discovered tool or turn validation failures into public outcomes."""

        if name not in MANDATORY_ANALYST_TOOLS or name not in ALLOWED_TOOLS:
            raise ValueError("Tool is not allowlisted.")
        if name not in self.discover():
            raise ValueError("Tool was not discovered.")

        validated_arguments = _validated_arguments(name, arguments)
        if validated_arguments is None:
            return _invalid_arguments_outcome()

        try:
            if name == "get_company_metric":
                result = self._capability_registry.get_company_metric(**validated_arguments)
            else:
                result = self._capability_registry.search_financial_documents(**validated_arguments)
        except CapabilityValidationError as error:
            return ToolOutcome(
                status="error",
                error_code=error.error.error_code,
                message=error.error.message,
                retryable=error.error.retryable,
            )
        except (AttributeError, TypeError):
            return _invalid_arguments_outcome()
        return ToolOutcome(status="ok", payload=result)


def _required_catalog_string(document: Mapping[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"certified evidence catalog requires non-empty {field}")
    return value


def _validated_arguments(name: str, arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    """Accept only safe, complete schemas before forwarding calls to capability code."""

    if not isinstance(arguments, Mapping):
        return None
    if name == "get_company_metric":
        expected_fields = ("ticker", "metric")
        if set(arguments) != set(expected_fields):
            return None
    else:
        expected_fields = ("company", "query", "top_k")
        if set(arguments) not in ({"company", "query"}, set(expected_fields)):
            return None

    validated: dict[str, Any] = {}
    for field in expected_fields:
        if field == "top_k" and field not in arguments:
            validated[field] = 2
            continue
        value = arguments.get(field)
        if field == "top_k":
            if not isinstance(value, int) or isinstance(value, bool):
                return None
            validated[field] = value
            continue
        if not isinstance(value, str):
            return None
        try:
            validated[field] = _clean_public_value(value)
        except ValueError:
            if value.strip():
                return None
            validated[field] = value
    return validated


def _invalid_arguments_outcome() -> ToolOutcome:
    """Return the stable public schema failure without returning untrusted input."""

    return ToolOutcome(
        status="error",
        error_code="invalid_arguments",
        message=_INVALID_ARGUMENTS_MESSAGE,
        retryable=True,
    )
