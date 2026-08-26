"""Bounded document-search, evidence-inspection, and cited-value capabilities."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from finai_academy.capstone.document_index import (
    CertifiedDocumentIndex,
    CertifiedDocumentIndexError,
    load_certified_document_index,
)
from finai_academy.capstone.document_models import (
    DocumentFilters,
    DocumentRetrievalHit,
    FinancialChunk,
    FrozenDocumentModel,
)

_MANIFEST_PATH = Path("assets/course-data/manifest.json")
_CROP_RECORD_BY_LOCATION = {
    ("NVIDIA", 165): "nvidia_crop",
    ("Schneider Electric", 16): "schneider_crop",
}


class DocumentSearchHit(FrozenDocumentModel):
    """One stable public retrieval handle and its certified rank lineage."""

    chunk_id: str
    retrieval: DocumentRetrievalHit

    @model_validator(mode="after")
    def require_matching_chunk_id(self) -> Self:
        if self.chunk_id != self.retrieval.chunk.chunk_id:
            raise ValueError("chunk_id must match the certified retrieval hit")
        return self


class DocumentSearchOutcome(FrozenDocumentModel):
    """A safe, ranked result from constrained document retrieval."""

    status: Literal["ok"] = "ok"
    query: str
    hits: tuple[DocumentSearchHit, ...]


class DocumentEvidenceOutcome(FrozenDocumentModel):
    """The immutable certified chunk selected for evidence inspection."""

    status: Literal["ok"] = "ok"
    chunk: FinancialChunk
    crop_asset_key: str | None = None

    @model_validator(mode="after")
    def require_repository_relative_crop_key(self) -> Self:
        if self.crop_asset_key is None:
            return self
        crop_path = PurePosixPath(self.crop_asset_key)
        if (
            crop_path.is_absolute()
            or ".." in crop_path.parts
            or "\\" in self.crop_asset_key
            or not self.crop_asset_key.startswith("assets/")
        ):
            raise ValueError("crop_asset_key must be repository-relative")
        return self

    @property
    def chunk_id(self) -> str:
        """Expose the selected stable chunk ID without duplicating immutable chunk state."""

        return self.chunk.chunk_id

    @property
    def physical_page(self) -> int:
        """Expose the source PDF page for direct evidence rendering."""

        return self.chunk.context.physical_page


class ReportedValue(FrozenDocumentModel):
    """A displayed numeric value tied to one already selected evidence chunk."""

    label: str
    value: float
    unit: str
    chunk_id: str

    @field_validator("value")
    @classmethod
    def require_finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("reported values must be finite")
        return value


class ReportedValueComparison(FrozenDocumentModel):
    """A host-side calculation that retains both cited input values."""

    left: ReportedValue
    right: ReportedValue
    comparable: bool
    absolute_difference: float | None = None
    formula: str
    reason: str

    @model_validator(mode="after")
    def require_difference_only_for_comparable_values(self) -> Self:
        if self.comparable != (self.absolute_difference is not None):
            raise ValueError("absolute_difference is present exactly for comparable values")
        return self


class _SearchRequest(FrozenDocumentModel):
    """Internal input boundary that validates all public search fields before ranking."""

    company: str
    reporting_period: str
    query: str
    element_type: str | None = None
    top_k: int = Field(default=3, ge=1, le=5)


class _InspectionRequest(FrozenDocumentModel):
    """Internal input boundary for evidence IDs supplied by a host or MCP caller."""

    chunk_id: str


class DocumentCapabilityRegistry:
    """Use the certified index through its bounded public research operations."""

    def __init__(
        self,
        index: CertifiedDocumentIndex,
        *,
        crop_asset_keys: Mapping[tuple[str, int], str] | None = None,
    ) -> None:
        self._index = index
        self._crop_asset_keys = dict(crop_asset_keys or {})

    def search_financial_documents(
        self,
        company: str,
        reporting_period: str,
        query: str,
        element_type: str | None = None,
        top_k: int = 3,
    ) -> DocumentSearchOutcome:
        """Search only the filtered certified corpus; ranking stays inside the index."""

        request = _SearchRequest(
            company=company,
            reporting_period=reporting_period,
            query=query,
            element_type=element_type,
            top_k=top_k,
        )
        hits = self._index.search(
            request.query,
            filters=DocumentFilters(
                company_name=request.company,
                reporting_period=request.reporting_period,
                element_type=request.element_type,
            ),
            top_k=request.top_k,
        )
        return DocumentSearchOutcome(
            query=request.query,
            hits=tuple(
                DocumentSearchHit(chunk_id=hit.chunk.chunk_id, retrieval=hit) for hit in hits
            ),
        )

    def inspect_document_evidence(self, chunk_id: str) -> DocumentEvidenceOutcome:
        """Return the exact certified chunk referenced by a prior search hit."""

        request = _InspectionRequest(chunk_id=chunk_id)
        chunk = self._index.inspect(request.chunk_id)
        crop_asset_key = self._crop_asset_keys.get(
            (chunk.context.company_name, chunk.context.physical_page)
        )
        if chunk.element_type != "table":
            crop_asset_key = None
        return DocumentEvidenceOutcome(chunk=chunk, crop_asset_key=crop_asset_key)


def build_document_capability_registry(root: Path | None = None) -> DocumentCapabilityRegistry:
    """Build capabilities from the verified artifact index and manifest crop keys."""

    project_root = _project_root(root)
    return DocumentCapabilityRegistry(
        load_certified_document_index(project_root),
        crop_asset_keys=_load_crop_asset_keys(project_root),
    )


def compare_reported_values(left: ReportedValue, right: ReportedValue) -> ReportedValueComparison:
    """Compare only cited values whose explicitly displayed currency and scale match."""

    left_unit = _unit_parts(left.unit)
    right_unit = _unit_parts(right.unit)
    incomplete_formula = (
        f"{_display_number(left.value)} {left.unit} - {_display_number(right.value)} {right.unit}"
    )
    if left_unit[0] != right_unit[0]:
        return ReportedValueComparison(
            left=left,
            right=right,
            comparable=False,
            formula=incomplete_formula,
            reason="Currencies differ; no FX rate was supplied.",
        )
    if left_unit[1] != right_unit[1]:
        return ReportedValueComparison(
            left=left,
            right=right,
            comparable=False,
            formula=incomplete_formula,
            reason="Scales differ; no conversion factor was supplied.",
        )
    difference = left.value - right.value
    return ReportedValueComparison(
        left=left,
        right=right,
        comparable=True,
        absolute_difference=difference,
        formula=f"{incomplete_formula} = {_display_number(difference)} {left.unit}",
        reason="Values use the same currency and scale.",
    )


def _project_root(root: Path | None) -> Path:
    if root is None:
        return Path(__file__).resolve().parents[3]
    if not isinstance(root, Path):
        raise CertifiedDocumentIndexError("root must be a Path")
    return root


def _load_crop_asset_keys(project_root: Path) -> dict[tuple[str, int], str]:
    try:
        manifest = json.loads((project_root / _MANIFEST_PATH).read_text(encoding="utf-8"))
        records = manifest["capstone_derived_artifacts"]
        record = records[0]
    except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise CertifiedDocumentIndexError("certified crop manifest could not be loaded") from error
    if not isinstance(record, dict):
        raise CertifiedDocumentIndexError("certified crop manifest must contain an object")

    crop_keys: dict[tuple[str, int], str] = {}
    for location, record_name in _CROP_RECORD_BY_LOCATION.items():
        crop_record = record.get(record_name)
        asset_key = crop_record.get("path") if isinstance(crop_record, dict) else None
        if not isinstance(asset_key, str) or not _is_repository_relative_asset_key(asset_key):
            raise CertifiedDocumentIndexError("certified crop path must be repository-relative")
        crop_keys[location] = asset_key
    return crop_keys


def _is_repository_relative_asset_key(asset_key: str) -> bool:
    path = PurePosixPath(asset_key)
    return (
        bool(asset_key.strip())
        and asset_key.startswith("assets/")
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in asset_key
    )


def _unit_parts(unit: str) -> tuple[str, str]:
    normalized = " ".join(unit.casefold().split())
    currency, _, scale = normalized.partition(" ")
    return currency, scale


def _display_number(value: float) -> str:
    return format(value, ".15g")
