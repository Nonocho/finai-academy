"""Public, immutable contracts for certified financial document processing."""

from __future__ import annotations

import math
import re
from datetime import date
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, HttpUrl, model_validator

_SECRET_PATTERN = re.compile(
    r"""(?ix)(
        api[_-]?key\s*(?:=|:)\s*\S+
        | authorization\s*(?:=|:)\s*\S+
        | bearer\s+[a-z0-9._-]+
        | sk-[a-z0-9]{12,}
        | \b(?:password|secret|token|client[_-]?secret|access[_-]?token|private[_-]?key)\b
          \s*["']?\s*(?:=|:)\s*\S+
        | -----BEGIN(?:[A-Z ]+)?PRIVATE KEY-----
    )"""
)
_PERSONAL_PATH_PATTERN = re.compile(r"(?i)(?:^|[^A-Za-z0-9])/(?:Users|home)/")
_WINDOWS_PERSONAL_PATH_PATTERN = re.compile(
    r"(?i)(?:^|[^A-Za-z0-9])(?:[A-Za-z]:)?\\+(?:Users|home)(?:\\+|$)"
)
_WINDOWS_DRIVE_PATH_PATTERN = re.compile(r"(?i)(?:^|[^A-Za-z0-9])[A-Za-z]:(?=[\\/])")
_WINDOWS_ROOTED_PATH_PATTERN = re.compile(
    r"\\+(?=(?:[A-Za-z0-9][A-Za-z0-9._ -]*\\+)+"
    r"[A-Za-z0-9][A-Za-z0-9._ -]*)"
)


def _require_url_without_userinfo(value: str | AnyUrl) -> None:
    if isinstance(value, AnyUrl):
        username = value.username
        password = value.password
    else:
        parsed = urlsplit(value)
        username = parsed.username
        password = parsed.password
    if username is not None or password is not None:
        raise ValueError("public fields must not contain URL userinfo")


def _require_safe_public_string(value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("public text values must not be blank")
    if _SECRET_PATTERN.search(cleaned):
        raise ValueError("public fields must not contain credential-shaped text")
    _require_url_without_userinfo(cleaned)
    if _PERSONAL_PATH_PATTERN.search(cleaned) or _WINDOWS_PERSONAL_PATH_PATTERN.search(
        cleaned
    ):
        raise ValueError("public fields must not contain personal filesystem paths")
    if _WINDOWS_DRIVE_PATH_PATTERN.search(cleaned) or _WINDOWS_ROOTED_PATH_PATTERN.search(
        cleaned
    ):
        raise ValueError("public fields must not contain drive-qualified or rooted filesystem paths")
    return cleaned


def _clean_public_value(value: Any) -> Any:
    """Reject unsafe values before public contracts retain them."""

    if isinstance(value, str):
        return _require_safe_public_string(value)
    if isinstance(value, BaseModel):
        if isinstance(value, TableMatrix):
            return _clean_table_matrix_state(value.model_dump(mode="python"))
        return {
            field_name: _clean_public_value(getattr(value, field_name))
            for field_name in type(value).model_fields
        }
    if isinstance(value, AnyUrl):
        _require_url_without_userinfo(value)
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("public fields must contain JSON-compatible values")
        if _is_table_matrix_state(value):
            return _clean_table_matrix_state(value)
        return {key: _clean_public_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clean_public_value(item) for item in value)
    if isinstance(value, list):
        return [_clean_public_value(item) for item in value]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("public numeric values must be finite")
        return value
    if isinstance(value, date):
        return value
    raise ValueError("public fields must contain JSON-compatible values")


class FrozenDocumentModel(BaseModel):
    """Shared policy for immutable, JSON-safe document contracts.

    This remains local to the document subsystem so ``capstone.models`` can
    import document types later without creating a circular dependency.
    """

    model_config = ConfigDict(
        extra="forbid", frozen=True, str_strip_whitespace=True, allow_inf_nan=False
    )

    @model_validator(mode="before")
    @classmethod
    def clean_public_state(cls, value: Any) -> Any:
        if isinstance(value, dict) and "local_asset_key" in value:
            value = dict(value)
            local_asset_key = value.pop("local_asset_key")
            if not isinstance(local_asset_key, str):
                raise ValueError("local_asset_key must be repository-relative")
            cleaned_key = local_asset_key.strip()
            if not cleaned_key or _SECRET_PATTERN.search(cleaned_key):
                _clean_public_value(local_asset_key)
            _require_url_without_userinfo(cleaned_key)
            cleaned_state = _clean_public_value(value)
            cleaned_state["local_asset_key"] = cleaned_key
            return cleaned_state
        return _clean_public_value(value)


ElementType = Literal[
    "heading", "paragraph", "list", "table", "figure_caption", "footnote"
]


class FinancialDocumentSource(FrozenDocumentModel):
    document_id: str
    company_name: str
    ticker: str
    document_type: str
    reporting_period: str
    publication_date: date
    official_source_url: HttpUrl
    local_asset_key: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(gt=0)
    page_count: int = Field(gt=0)

    @model_validator(mode="after")
    def require_relative_asset_key(self) -> Self:
        posix_path = PurePosixPath(self.local_asset_key)
        windows_path = PureWindowsPath(self.local_asset_key)
        if (
            posix_path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or bool(windows_path.root)
            or ".." in posix_path.parts
            or ".." in windows_path.parts
        ):
            raise ValueError("local_asset_key must be repository-relative")
        return self


class BoundingBox(FrozenDocumentModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def require_positive_area(self) -> Self:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bounding box must have positive area")
        return self


class TableMatrix(FrozenDocumentModel):
    rows: tuple[tuple[str, ...], ...] = Field(min_length=1)
    row_count: int = Field(gt=0)
    column_count: int = Field(gt=0)
    markdown: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def clean_public_state(cls, value: Any) -> Any:
        """Preserve source-empty cells while enforcing safety everywhere else."""

        if not isinstance(value, dict):
            return _clean_public_value(value)
        return _clean_table_matrix_state(value)

    @model_validator(mode="after")
    def require_consistent_dimensions(self) -> Self:
        if len(self.rows) != self.row_count:
            raise ValueError("row_count must match rows")
        if any(len(row) != self.column_count for row in self.rows):
            raise ValueError("column_count must match every row")
        return self


def _clean_table_cell(value: Any) -> str:
    if not isinstance(value, str):
        _clean_public_value(value)
        raise TypeError("table cells must be strings")
    if value == "":
        return value
    return _require_safe_public_string(value)


def _clean_table_matrix_state(value: dict[str, Any]) -> dict[str, Any]:
    cleaned = {
        key: _clean_public_value(item)
        for key, item in value.items()
        if key != "rows"
    }
    rows = value.get("rows")
    if not isinstance(rows, (tuple, list)):
        cleaned["rows"] = _clean_public_value(rows)
        return cleaned
    cleaned["rows"] = tuple(
        tuple(_clean_table_cell(cell) for cell in row)
        if isinstance(row, (tuple, list))
        else _clean_public_value(row)
        for row in rows
    )
    return cleaned


def _is_table_matrix_state(value: dict[str, Any]) -> bool:
    return {"rows", "row_count", "column_count", "markdown"}.issubset(value)


class ExtractionDiagnostic(FrozenDocumentModel):
    code: str
    severity: Literal["warning", "error"]
    physical_page: int = Field(gt=0)
    message: str
    extraction_method: Literal["native_text", "ocr"]


class ContextualMetadata(FrozenDocumentModel):
    document_id: str
    company_name: str
    ticker: str
    document_type: str
    reporting_period: str
    publication_date: date
    official_source_url: str
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    physical_page: int = Field(gt=0)
    printed_page: int | None = Field(default=None, gt=0)
    heading_path: tuple[str, ...] = ()
    element_type: ElementType
    bbox: BoundingBox
    parent_element_id: str | None = None
    previous_element_id: str | None = None
    next_element_id: str | None = None
    parser_name: str
    parser_version: str
    extraction_method: Literal["native_text", "ocr"]


class FinancialMetadata(FrozenDocumentModel):
    metric_names: tuple[str, ...] = ()
    periods: tuple[str, ...] = Field(min_length=1)
    currency: str | None = None
    scale: str | None = None
    segments: tuple[str, ...] = ()
    geography: tuple[str, ...] = ()
    accounting_basis: Literal["GAAP", "non-GAAP"] | None = None
    audited: bool | None = None
    footnotes: tuple[str, ...] = ()
    source_element_ids: tuple[str, ...] = Field(min_length=1)
    enrichment_method: Literal["deterministic", "luna_structured"]
    confidence: float = Field(ge=0, le=1)


class DocumentElement(FrozenDocumentModel):
    element_id: str
    document_id: str
    ordinal: int = Field(ge=0)
    physical_page: int = Field(gt=0)
    printed_page: int | None = Field(default=None, gt=0)
    element_type: ElementType
    bbox: BoundingBox
    original_text: str
    original_markdown: str | None = None
    table: TableMatrix | None = None
    heading_path: tuple[str, ...] = ()
    parent_element_id: str | None = None
    previous_element_id: str | None = None
    next_element_id: str | None = None
    extraction_method: Literal["native_text", "ocr"] = "native_text"

    @model_validator(mode="after")
    def require_table_for_table_elements(self) -> Self:
        has_table = self.table is not None
        if (self.element_type == "table") != has_table:
            raise ValueError("table must be present exactly when element_type is 'table'")
        return self


class ParsedDocument(FrozenDocumentModel):
    source: FinancialDocumentSource
    parser_name: str
    parser_version: str
    extraction_schema_version: int = 2
    elements: tuple[DocumentElement, ...]
    diagnostics: tuple[ExtractionDiagnostic, ...] = ()


class FinancialChunk(FrozenDocumentModel):
    chunk_id: str
    text: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    element_type: ElementType
    source_element_ids: tuple[str, ...] = Field(min_length=1)
    context: ContextualMetadata
    financial: FinancialMetadata
    table: TableMatrix | None = None

    @model_validator(mode="after")
    def require_matching_source_element_ids(self) -> Self:
        if self.source_element_ids != self.financial.source_element_ids:
            raise ValueError("source_element_ids must match FinancialMetadata")
        return self


class DocumentFilters(FrozenDocumentModel):
    company_name: str | None = None
    reporting_period: str | None = None
    document_type: str | None = None
    element_type: ElementType | None = None

    def matches(self, chunk: FinancialChunk) -> bool:
        pairs = (
            (self.company_name, chunk.context.company_name),
            (self.reporting_period, chunk.context.reporting_period),
            (self.document_type, chunk.context.document_type),
            (self.element_type, chunk.element_type),
        )
        return all(
            expected is None or str(expected).casefold().strip() == actual.casefold().strip()
            for expected, actual in pairs
        )


class DocumentRetrievalHit(FrozenDocumentModel):
    chunk: FinancialChunk
    fused_score: float = Field(ge=0)
    channel_ranks: tuple[tuple[Literal["bm25", "dense"], int], ...]
    index_version: str
    selection_reason: str
