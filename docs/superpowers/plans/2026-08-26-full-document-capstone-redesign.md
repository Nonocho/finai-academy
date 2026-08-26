# Full-Document Financial Analyst Capstone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a simple, inspectable capstone that turns complete official NVIDIA and Schneider Electric PDFs into contextualized, table-aware evidence and cited financial answers.

**Architecture:** Versioned official PDFs pass through an isolated PyMuPDF4LLM parser into immutable typed elements, contextual metadata, table-aware chunks, and a deterministic hybrid index. A bounded service exposes document search and evidence inspection through MCP-shaped tools, uses recorded or `gpt-5.6-luna` structured synthesis, and renders Answer, Evidence, and How it worked in a simplified Streamlit workspace.

**Tech Stack:** Python 3.11+, Pydantic 2, PyMuPDF 1.28.2, PyMuPDF4LLM 1.28.2, existing BM25/dense retrieval primitives, MCP 2.x, OpenAI Python SDK 2.x Responses API, Streamlit 1.40+, MLflow 3.15+, pytest.

**Spec:** `docs/superpowers/specs/2026-08-26-full-document-capstone-redesign.md`

## Global Constraints

- Keep the mandatory route offline and deterministic from versioned source and derived artifacts.
- Use the complete NVIDIA FY2026 Annual Report and Schneider Electric FY2025 Full Year Results PDFs.
- Verify NVIDIA SHA-256 `0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c` and Schneider SHA-256 `5b17e6e8e5b8c47f697f5d352ed9639b98b261d9ecf886d3e8ce147493ebb00a` before parsing.
- Isolate PyMuPDF4LLM behind `FinancialDocumentParser`; document its AGPL/commercial dual-license requirement.
- Use native PDF text first. OCR is opt-in only for a page that fails the text-quality rule, and every OCR attempt is recorded.
- Never persist or display credentials, raw provider errors, or personal filesystem paths.
- Keep original parser text, Markdown tables, coordinates, and page numbers immutable; enrichment adds fields without replacing source extraction.
- Keep normal financial tables atomic. Never split values from headers, units, periods, footnotes, or source location.
- Apply company, period, document type, and element type filters before ranking.
- Default the OpenAI route to `gpt-5.6-luna` with Responses API structured output and `reasoning.effort="medium"`.
- Do not let a model parse geometry, perform displayed arithmetic, invent metadata, or decide deterministic release status.
- The first UI screen title is `Financial Document Analyst` and its introductory text is `Ask a financial question and see the exact report page and table behind the answer.`
- Result tabs appear in this exact order: `Answer`, `Evidence`, `How it worked`.
- Provider, model, readiness, data mode, and reset controls remain collapsed under `Advanced settings` by default.
- Keep the primary UI in plain language; technical names may appear only in advanced diagnostics.
- Keep the existing recorded route, privacy boundary, bounded step/replan policy, and MLflow audit discipline.
- Do not mix the separate Lessons 01–12 release-cleanup failures into this implementation.

---

## File responsibility map

### New focused modules

- `src/finai_academy/capstone/document_models.py` — immutable source, element, metadata, chunk, and retrieval contracts.
- `src/finai_academy/capstone/document_assets.py` — load and verify certified source and derived-artifact manifest records.
- `src/finai_academy/capstone/document_ingestion.py` — parser protocol, PyMuPDF4LLM adapter, typed diagnostics, and evidence crop rendering.
- `src/finai_academy/capstone/document_chunking.py` — deterministic context enrichment and table-aware chunk construction.
- `src/finai_academy/capstone/document_index.py` — pre-filtered BM25+dense retrieval, fusion, and exact evidence lookup.
- `src/finai_academy/capstone/document_tools.py` — pure typed search, inspection, and deterministic comparison capabilities.
- `src/finai_academy/capstone/mcp_server.py` — MCP exposure for the two document capabilities without duplicating business logic.
- `scripts/build_capstone_document_assets.py` — reproducible extraction, chunk, crop, and manifest artifact build.
- `scripts/smoke_capstone_openai.py` — explicit live Luna smoke test that prints only safe status data.

### Existing modules to evolve

- `src/finai_academy/capstone/models.py` — carry page-, chunk-, element-, and metadata-level provenance through the run.
- `src/finai_academy/capstone/tools.py` — replace the hand-written evidence catalog with the certified document index and fail-closed tool registry.
- `src/finai_academy/capstone/service.py` — execute the document-first mission and release only element-backed answers.
- `src/finai_academy/capstone/model_gateway.py` — add the native OpenAI Responses API adapter.
- `src/finai_academy/capstone/views.py` — create answer, evidence-comparison, and diagnostics view models.
- `src/finai_academy/capstone/streamlit_ui.py` — implement the simple question-first interface.
- `src/finai_academy/capstone/persistence.py` — record document/index identities and retrieval lineage in MLflow.
- `src/finai_academy/settings.py` — add validated reasoning effort while retaining Luna as the OpenAI default.
- `final-project/*` — replace assembly-only student seams with document-context tasks and update course instructions.
- `scripts/certify_capstone.py` — certify extraction, retrieval, UI, privacy, MLflow, and student outcomes.

---

### Task 1: Certify the source PDFs and base document contracts

**Files:**
- Create: `src/finai_academy/capstone/document_models.py`
- Create: `src/finai_academy/capstone/document_assets.py`
- Create: `tests/test_capstone_document_assets.py`
- Add: `assets/course-data/downloads/nvidia_fy2026_annual_report.pdf`
- Add: `assets/course-data/downloads/schneider_fy2025_full_year_results.pdf`
- Modify: `assets/course-data/manifest.json`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `LICENSE.md`

**Interfaces:**
- Produces: `FinancialDocumentSource`, `BoundingBox`, `TableMatrix`, `ExtractionDiagnostic`, `ContextualMetadata`, `FinancialMetadata`, `DocumentElement`, `ParsedDocument`, `FinancialChunk`, `DocumentFilters`, and `DocumentRetrievalHit`.
- Produces: `load_certified_document_sources(manifest_path: Path) -> tuple[FinancialDocumentSource, ...]`.
- Produces: `verify_source_asset(source: FinancialDocumentSource, root: Path) -> None`.

- [ ] **Step 1: Write failing source and privacy contract tests**

```python
from pathlib import Path

import pytest

from finai_academy.capstone.document_assets import (
    load_certified_document_sources,
    verify_source_asset,
)
from finai_academy.capstone.document_models import BoundingBox, FinancialDocumentSource

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets/course-data/manifest.json"


def test_capstone_manifest_certifies_both_complete_official_pdfs() -> None:
    sources = load_certified_document_sources(MANIFEST)
    assert [(source.company_name, source.page_count) for source in sources] == [
        ("NVIDIA", 175),
        ("Schneider Electric", 19),
    ]
    assert [source.sha256 for source in sources] == [
        "0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c",
        "5b17e6e8e5b8c47f697f5d352ed9639b98b261d9ecf886d3e8ce147493ebb00a",
    ]
    for source in sources:
        verify_source_asset(source, ROOT)


def test_public_source_contract_rejects_absolute_personal_paths() -> None:
    with pytest.raises(ValueError, match="local_asset_key"):
        FinancialDocumentSource(
            document_id="NVDA-FY2026-AR",
            company_name="NVIDIA",
            ticker="NVDA",
            document_type="Annual Report",
            reporting_period="FY2026",
            publication_date="2026-02-25",
            official_source_url="https://investor.nvidia.com/",
            local_asset_key="/Users/example/report.pdf",
            sha256="0" * 64,
            byte_size=1,
            page_count=1,
        )


def test_bounding_box_requires_positive_area() -> None:
    with pytest.raises(ValueError, match="positive area"):
        BoundingBox(x0=10, y0=10, x1=10, y1=12)
```

- [ ] **Step 2: Run the focused tests and verify the missing-module failure**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run pytest tests/test_capstone_document_assets.py -q`

Expected: FAIL during collection because `finai_academy.capstone.document_assets` does not exist.

- [ ] **Step 3: Add exact dependency and license records**

Add the following to the `capstone` optional dependency group and regenerate the lock:

```toml
capstone = [
  "streamlit>=1.40,<2",
  "mlflow>=3.15,<4",
  "pymupdf==1.28.2",
  "pymupdf4llm==1.28.2",
]
```

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv lock`

Add a `PyMuPDF / PyMuPDF4LLM` section to `LICENSE.md` that states version `1.28.2`, AGPL/commercial dual licensing, the official documentation URL, and that the course owner must confirm an AGPL-compatible distribution or obtain a commercial license before commercial release.

- [ ] **Step 4: Add and verify the exact official files**

Run:

```bash
curl -L --fail --output assets/course-data/downloads/nvidia_fy2026_annual_report.pdf https://s201.q4cdn.com/141608511/files/doc_financials/2026/ar/2026-Annual-Report-Web.pdf
cp assets/lesson-05/schneider-fy2025-results.pdf assets/course-data/downloads/schneider_fy2025_full_year_results.pdf
shasum -a 256 assets/course-data/downloads/nvidia_fy2026_annual_report.pdf assets/course-data/downloads/schneider_fy2025_full_year_results.pdf
```

Expected hashes, in order:

```text
0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c
5b17e6e8e5b8c47f697f5d352ed9639b98b261d9ecf886d3e8ce147493ebb00a
```

- [ ] **Step 5: Add the certified manifest records and typed models**

Add a `capstone_documents` array to `assets/course-data/manifest.json` with these exact records:

```json
[
  {
    "document_id": "NVDA-FY2026-ANNUAL-REPORT",
    "company_name": "NVIDIA",
    "ticker": "NVDA",
    "document_type": "Annual Report and Form 10-K",
    "reporting_period": "FY2026",
    "publication_date": "2026-02-25",
    "official_source_url": "https://s201.q4cdn.com/141608511/files/doc_financials/2026/ar/2026-Annual-Report-Web.pdf",
    "local_asset_key": "assets/course-data/downloads/nvidia_fy2026_annual_report.pdf",
    "sha256": "0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c",
    "byte_size": 15850437,
    "page_count": 175
  },
  {
    "document_id": "SU-FY2025-FULL-YEAR-RESULTS",
    "company_name": "Schneider Electric",
    "ticker": "SU.PA",
    "document_type": "Full Year Results",
    "reporting_period": "FY2025",
    "publication_date": "2026-02-26",
    "official_source_url": "https://www.se.com/ww/en/assets/564/document/528237/release-fy-results-2025.pdf",
    "local_asset_key": "assets/course-data/downloads/schneider_fy2025_full_year_results.pdf",
    "sha256": "5b17e6e8e5b8c47f697f5d352ed9639b98b261d9ecf886d3e8ce147493ebb00a",
    "byte_size": 781628,
    "page_count": 19
  }
]
```

Implement frozen Pydantic contracts. The stable public fields must use asset keys, never `Path` objects:

```python
class FinancialDocumentSource(_FrozenPublicModel):
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

    @field_validator("local_asset_key")
    @classmethod
    def require_relative_asset_key(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("local_asset_key must be repository-relative")
        return value
```

Define the remaining contracts with these fields and names so every later task shares one schema:

```python
ElementType = Literal[
    "heading", "paragraph", "list", "table", "figure_caption", "footnote"
]


class BoundingBox(_FrozenPublicModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def require_positive_area(self) -> Self:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("bounding box must have positive area")
        return self


class TableMatrix(_FrozenPublicModel):
    rows: tuple[tuple[str, ...], ...] = Field(min_length=1)
    row_count: int = Field(gt=0)
    column_count: int = Field(gt=0)
    markdown: str = Field(min_length=1)


class ExtractionDiagnostic(_FrozenPublicModel):
    code: str
    severity: Literal["warning", "error"]
    physical_page: int = Field(gt=0)
    message: str
    extraction_method: Literal["native_text", "ocr"]


class ContextualMetadata(_FrozenPublicModel):
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


class FinancialMetadata(_FrozenPublicModel):
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


class DocumentElement(_FrozenPublicModel):
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


class ParsedDocument(_FrozenPublicModel):
    source: FinancialDocumentSource
    parser_name: str
    parser_version: str
    extraction_schema_version: int = 2
    elements: tuple[DocumentElement, ...]
    diagnostics: tuple[ExtractionDiagnostic, ...] = ()


class FinancialChunk(_FrozenPublicModel):
    chunk_id: str
    text: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    element_type: ElementType
    source_element_ids: tuple[str, ...] = Field(min_length=1)
    context: ContextualMetadata
    financial: FinancialMetadata
    table: TableMatrix | None = None


class DocumentFilters(_FrozenPublicModel):
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


class DocumentRetrievalHit(_FrozenPublicModel):
    chunk: FinancialChunk
    fused_score: float = Field(ge=0)
    channel_ranks: tuple[tuple[Literal["bm25", "dense"], int], ...]
    index_version: str
    selection_reason: str
```

Add model validators requiring `DocumentElement.table` exactly when `element_type == "table"`, consistent `TableMatrix` dimensions, nonblank text fields, finite coordinates/scores, and matching `source_element_ids` between `FinancialChunk` and `FinancialMetadata`.

- [ ] **Step 6: Implement manifest loading and fail-closed verification**

```python
def load_certified_document_sources(manifest_path: Path) -> tuple[FinancialDocumentSource, ...]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return tuple(
        FinancialDocumentSource.model_validate(item)
        for item in payload["capstone_documents"]
    )


def verify_source_asset(source: FinancialDocumentSource, root: Path) -> None:
    path = root / source.local_asset_key
    raw = path.read_bytes()
    if len(raw) != source.byte_size:
        raise SourceAssetError("certified document byte size mismatch")
    if sha256(raw).hexdigest() != source.sha256:
        raise SourceAssetError("certified document SHA-256 mismatch")
```

- [ ] **Step 7: Run contract tests and the existing document suite**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run pytest tests/test_capstone_document_assets.py tests/test_documents.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the certified source foundation**

```bash
git add pyproject.toml uv.lock LICENSE.md assets/course-data/manifest.json assets/course-data/downloads/nvidia_fy2026_annual_report.pdf assets/course-data/downloads/schneider_fy2025_full_year_results.pdf src/finai_academy/capstone/document_models.py src/finai_academy/capstone/document_assets.py tests/test_capstone_document_assets.py
git commit -m "feat: certify capstone financial documents"
```

---

### Task 2: Parse real pages into immutable elements and evidence crops

**Files:**
- Create: `src/finai_academy/capstone/document_ingestion.py`
- Create: `tests/test_capstone_document_ingestion.py`

**Interfaces:**
- Consumes: `FinancialDocumentSource`, `BoundingBox`, `TableMatrix`, `DocumentElement`, `ExtractionDiagnostic`, and `ParsedDocument` from Task 1.
- Produces: `FinancialDocumentParser.parse(source: FinancialDocumentSource, *, project_root: Path, pages: tuple[int, ...] | None = None) -> ParsedDocument`.
- Produces: `PyMuPDF4LLMParser(ocr_adapter: OcrAdapter | None = None)`.
- Produces: `render_evidence_crop(source, *, project_root, page_number, bbox, destination, scale=2.0) -> Path`.

- [ ] **Step 1: Write failing real-table extraction tests**

```python
def _source(company: str) -> FinancialDocumentSource:
    return next(
        item for item in load_certified_document_sources(MANIFEST)
        if item.company_name == company
    )


def test_nvidia_target_page_preserves_one_14_by_4_table() -> None:
    parsed = PyMuPDF4LLMParser().parse(
        _source("NVIDIA"), project_root=ROOT, pages=(165,)
    )
    tables = [item for item in parsed.elements if item.element_type == "table"]
    assert len(tables) == 1
    assert tables[0].table is not None
    assert (tables[0].table.row_count, tables[0].table.column_count) == (14, 4)
    assert tables[0].table.rows[3] == (
        "Revenue", "$ 193,479", "$ 22,459", "$ 215,938"
    )
    assert tables[0].physical_page == 165
    assert tables[0].printed_page == 77


def test_schneider_target_page_preserves_three_six_column_tables() -> None:
    parsed = PyMuPDF4LLMParser().parse(
        _source("Schneider Electric"), project_root=ROOT, pages=(16,)
    )
    tables = [item for item in parsed.elements if item.element_type == "table"]
    assert [(item.table.row_count, item.table.column_count) for item in tables] == [
        (5, 6), (4, 6), (4, 6)
    ]
    assert tables[2].table.rows[-1] == (
        "Group", "40,152", "+8.9%", "+0.8%", "-4.1%", "+5.2%"
    )


def test_parser_output_contains_no_local_filename() -> None:
    parsed = PyMuPDF4LLMParser().parse(
        _source("NVIDIA"), project_root=ROOT, pages=(165,)
    )
    payload = parsed.model_dump_json()
    assert "/Users/" not in payload
    assert str(ROOT) not in payload
```

- [ ] **Step 2: Run the target tests and verify the missing-parser failure**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_document_ingestion.py -q`

Expected: FAIL because `PyMuPDF4LLMParser` does not exist.

- [ ] **Step 3: Implement the parser protocol and JSON normalization**

The adapter must parse one-based public page numbers and convert them to PyMuPDF's zero-based list:

```python
class FinancialDocumentParser(Protocol):
    def parse(
        self,
        source: FinancialDocumentSource,
        *,
        project_root: Path,
        pages: tuple[int, ...] | None = None,
    ) -> ParsedDocument: ...


class PyMuPDF4LLMParser:
    parser_name = "pymupdf4llm"
    parser_version = "1.28.2"

    def parse(self, source, *, project_root, pages=None):
        verify_source_asset(source, project_root)
        selected = None if pages is None else [page - 1 for page in pages]
        raw = json.loads(
            pymupdf4llm.to_json(
                str(project_root / source.local_asset_key),
                pages=selected,
                use_ocr=False,
            )
        )
        return _normalize_document(source, raw, requested_pages=pages)
```

Normalize `boxclass`, `x0/y0/x1/y1`, `textlines`, and `table.extract/markdown/row_count/col_count`. Collapse whitespace inside cells, keep Markdown line structure, drop the parser's `filename` and metadata path, and derive stable `element_id` from document hash, page, ordinal, element type, bounding box, and original content.

- [ ] **Step 4: Add heading, neighbor, printed-page, and OCR diagnostics**

Track heading boxes while walking each page. Attach the active heading path plus previous/next element IDs after all IDs are known. Derive printed page from explicit footer text only; otherwise leave it null.

Add this fail-closed rule:

```python
if not page["fulltext"].strip() and not table_boxes:
    diagnostics.append(
        ExtractionDiagnostic(
            code="ocr_required",
            severity="error" if self._ocr_adapter is None else "warning",
            physical_page=physical_page,
            message="Page has no usable native text layer.",
            extraction_method="native_text",
        )
    )
```

If an OCR adapter is configured, record `extraction_method="ocr"`, engine, language, and confidence. Do not add a concrete OCR dependency because both certified PDFs have usable native text.

- [ ] **Step 5: Implement deterministic page and crop rendering**

```python
def render_evidence_crop(
    source: FinancialDocumentSource,
    *,
    project_root: Path,
    page_number: int,
    bbox: BoundingBox | None,
    destination: Path,
    scale: float = 2.0,
) -> Path:
    document = pymupdf.open(project_root / source.local_asset_key)
    page = document[page_number - 1]
    clip = None if bbox is None else pymupdf.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), clip=clip, alpha=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(destination)
    return destination
```

Add a test that opens the PNG with Pillow and asserts positive dimensions and RGB/RGBA mode.

- [ ] **Step 6: Run extraction tests twice to confirm determinism**

Run twice: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_document_ingestion.py -q`

Expected: PASS both times with identical element IDs.

- [ ] **Step 7: Commit the parser boundary**

```bash
git add src/finai_academy/capstone/document_ingestion.py tests/test_capstone_document_ingestion.py
git commit -m "feat: parse capstone PDFs with table geometry"
```

---

### Task 3: Build contextual financial metadata and table-aware chunks

**Files:**
- Create: `src/finai_academy/capstone/document_chunking.py`
- Create: `tests/test_capstone_document_chunking.py`

**Interfaces:**
- Consumes: `ParsedDocument` from Task 2.
- Produces: `build_contextual_metadata(document: ParsedDocument, element: DocumentElement) -> ContextualMetadata`.
- Produces: `build_financial_metadata(document: ParsedDocument, element: DocumentElement) -> FinancialMetadata`.
- Produces: `build_table_chunk(document: ParsedDocument, element: DocumentElement) -> FinancialChunk`.
- Produces: `build_financial_chunks(document: ParsedDocument, *, paragraph_character_budget: int = 3200) -> tuple[FinancialChunk, ...]`.

- [ ] **Step 1: Write failing table-context and stable-ID tests**

```python
def test_nvidia_table_chunk_keeps_value_headers_period_unit_and_lineage(nvidia_page_165) -> None:
    chunks = build_financial_chunks(nvidia_page_165)
    chunk = next(item for item in chunks if "$ 193,479" in item.text)
    assert chunk.element_type == "table"
    assert chunk.financial.currency == "USD"
    assert chunk.financial.scale == "millions"
    assert "FY2026" in chunk.financial.periods
    assert chunk.context.company_name == "NVIDIA"
    assert chunk.context.physical_page == 165
    assert chunk.context.printed_page == 77
    assert chunk.table is not None and chunk.table.column_count == 4
    assert chunk.source_element_ids
    assert chunk.content_hash == sha256(chunk.text.encode("utf-8")).hexdigest()


def test_schneider_fy_table_is_atomic_and_keeps_financial_scope(schneider_page_16) -> None:
    chunks = build_financial_chunks(schneider_page_16)
    chunk = next(item for item in chunks if "33,130" in item.text and "40,152" in item.text)
    assert chunk.element_type == "table"
    assert chunk.financial.currency == "EUR"
    assert chunk.financial.scale == "millions"
    assert chunk.financial.periods == ("FY2025",)
    assert chunk.financial.segments == (
        "Energy Management", "Industrial Automation", "Group"
    )
    assert "Organic growth" in chunk.text
    assert chunk.table.row_count == 4


def test_identical_extraction_produces_identical_chunk_ids(nvidia_page_165) -> None:
    first = build_financial_chunks(nvidia_page_165)
    second = build_financial_chunks(nvidia_page_165)
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]
```

- [ ] **Step 2: Run the tests and verify the missing-function failure**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_document_chunking.py -q`

Expected: FAIL because the chunk builder is absent.

- [ ] **Step 3: Implement deterministic contextual metadata**

Build source, structural, and lineage context only from trusted source records and parser output:

```python
def build_contextual_metadata(document, element):
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
```

- [ ] **Step 4: Implement evidence-bound financial normalization**

Normalize only labels explicitly present in the table, active heading, document record, or adjacent text. Use deterministic rules for the certified documents:

```python
_PERIOD_RULES = (
    (re.compile(r"Year Ended Jan 25, 2026", re.I), "FY2026"),
    (re.compile(r"FY\s*2025", re.I), "FY2025"),
    (re.compile(r"H2\s*2025", re.I), "H2 2025"),
    (re.compile(r"Q4\s*2025", re.I), "Q4 2025"),
)


def build_financial_metadata(document, element):
    evidence = "\n".join((*element.heading_path, element.original_text))
    currency = "USD" if "In millions" in evidence and document.source.ticker == "NVDA" else None
    if "€ million" in evidence:
        currency = "EUR"
    periods = tuple(label for pattern, label in _PERIOD_RULES if pattern.search(evidence))
    return FinancialMetadata(
        metric_names=_metric_labels(element),
        periods=periods or (document.source.reporting_period,),
        currency=currency,
        scale="millions" if re.search(r"(?:In|€) millions?", evidence, re.I) else None,
        segments=_row_labels(element),
        geography=(),
        accounting_basis=_explicit_accounting_basis(evidence),
        audited=_explicit_audit_status(evidence),
        footnotes=_adjacent_footnotes(document, element),
        source_element_ids=(element.element_id,),
        enrichment_method="deterministic",
        confidence=1.0,
    )
```

Null is the correct result for unsupported metadata. Do not infer GAAP status, audit status, currency, or units from general background knowledge.

- [ ] **Step 5: Implement separate paragraph and table chunk policies**

For table chunks, compose text in this order: company/document/period, heading path and table title, units, complete Markdown table, footnotes, nearby explanatory paragraph, and source page. Preserve `TableMatrix` unchanged.

For paragraphs, group adjacent paragraph/list elements under the same heading until the character budget would be exceeded. Repeat full contextual metadata in every chunk.

Derive IDs from canonical JSON:

```python
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
chunk_id = f"chunk-{sha256(material.encode('utf-8')).hexdigest()[:20]}"
```

- [ ] **Step 6: Add fail-closed metadata validation tests**

```python
def test_table_value_without_the_explicit_unit_fails_closed(schneider_page_16) -> None:
    element = next(
        item for item in schneider_page_16.elements
        if item.table is not None and "40,152" in item.original_text
    )
    broken = element.model_copy(
        update={
            "original_text": element.original_text.replace("€ million", ""),
            "original_markdown": element.original_markdown.replace("€ million", ""),
        }
    )
    with pytest.raises(MissingFinancialContextError, match="table unit is missing"):
        build_table_chunk(
            schneider_page_16.model_copy(update={"elements": (broken,)}),
            broken,
        )


@pytest.mark.parametrize(
    "field",
    ["company_name", "reporting_period", "physical_page", "source_element_ids"],
)
def test_numeric_table_chunk_requires_source_context(valid_table_chunk, field) -> None:
    payload = valid_table_chunk.model_dump(mode="python")
    target = payload["context"] if field in payload["context"] else payload
    target[field] = None if field != "source_element_ids" else ()
    with pytest.raises(ValueError):
        FinancialChunk.model_validate(payload)
```

- [ ] **Step 7: Run chunking and parser tests**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_document_chunking.py tests/test_capstone_document_ingestion.py -q`

Expected: PASS.

- [ ] **Step 8: Commit contextual chunking**

```bash
git add src/finai_academy/capstone/document_chunking.py tests/test_capstone_document_chunking.py
git commit -m "feat: build contextual financial chunks"
```

---

### Task 4: Generate versioned offline extraction and crop artifacts

**Files:**
- Create: `scripts/build_capstone_document_assets.py`
- Create: `tests/test_capstone_document_artifacts.py`
- Add: `assets/course-data/capstone/document_elements_v2.json`
- Add: `assets/course-data/capstone/financial_chunks_v2.json`
- Add: `assets/course-data/capstone/crops/nvidia_segment_table_page_165.png`
- Add: `assets/course-data/capstone/crops/schneider_revenue_tables_page_16.png`
- Modify: `assets/course-data/manifest.json`

**Interfaces:**
- Consumes: source loader, parser, crop renderer, and chunk builder from Tasks 1–3.
- Produces: `build_capstone_document_assets(root: Path) -> CapstoneArtifactBuild`.
- Produces: a `capstone_derived_artifacts` manifest entry with schema version, parser version, source hashes, artifact hashes, and chunking strategy.

- [ ] **Step 1: Write failing artifact reproducibility tests**

```python
def test_committed_capstone_artifacts_match_manifest_hashes() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record = manifest["capstone_derived_artifacts"][0]
    for field in ("elements", "chunks", "nvidia_crop", "schneider_crop"):
        artifact = record[field]
        raw = (ROOT / artifact["path"]).read_bytes()
        assert sha256(raw).hexdigest() == artifact["sha256"]
    assert record["source_sha256s"] == [
        "0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c",
        "5b17e6e8e5b8c47f697f5d352ed9639b98b261d9ecf886d3e8ce147493ebb00a",
    ]


def test_committed_chunks_contain_no_personal_paths_or_parser_filenames() -> None:
    text = (ROOT / "assets/course-data/capstone/financial_chunks_v2.json").read_text()
    assert "/Users/" not in text
    assert '"filename"' not in text
```

- [ ] **Step 2: Run tests and verify missing-artifact failure**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_document_artifacts.py -q`

Expected: FAIL because the v2 artifacts and manifest entry do not exist.

- [ ] **Step 3: Implement the deterministic build script**

Define the build result and builder exactly:

```python
@dataclass(frozen=True)
class CapstoneArtifactBuild:
    document_count: int
    page_count: int
    nvidia_target_table_count: int
    schneider_target_table_count: int
    artifact_sha256s: dict[str, str]


def build_capstone_document_assets(root: Path) -> CapstoneArtifactBuild:
    sources = load_certified_document_sources(root / "assets/course-data/manifest.json")
    parser = PyMuPDF4LLMParser()
    documents = tuple(parser.parse(source, project_root=root) for source in sources)
    chunks = tuple(chunk for document in documents for chunk in build_financial_chunks(document))
    _write_canonical_json(ELEMENTS_PATH, [doc.model_dump(mode="json") for doc in documents])
    _write_canonical_json(CHUNKS_PATH, [chunk.model_dump(mode="json") for chunk in chunks])
    _render_target_crops(sources, documents, root)
    return _build_manifest_record(sources, documents, chunks, root)
```

Use this canonical writer:

```python
def _write_canonical_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
```

Write only repository-relative asset keys. Raise `ArtifactContractError` unless the NVIDIA count is one and the Schneider count is three.

- [ ] **Step 4: Generate all complete-document artifacts**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone python scripts/build_capstone_document_assets.py`

Expected safe summary:

```text
documents=2 pages=194 nvidia_target_tables=1 schneider_target_tables=3
```

The script may print counts and hashes. It must not print absolute paths or document text.

- [ ] **Step 5: Add artifact hashes to the manifest and rerun the builder**

Record schema version `2`, parser `pymupdf4llm`, parser version `1.28.2`, chunking strategy `financial-context-v2`, both source hashes, and the SHA-256 for each JSON/PNG artifact. Rerun the builder and assert the generated files are byte-identical.

- [ ] **Step 6: Run artifact, extraction, and chunk tests**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_document_artifacts.py tests/test_capstone_document_ingestion.py tests/test_capstone_document_chunking.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the offline document corpus**

```bash
git add scripts/build_capstone_document_assets.py tests/test_capstone_document_artifacts.py assets/course-data/manifest.json assets/course-data/capstone/document_elements_v2.json assets/course-data/capstone/financial_chunks_v2.json assets/course-data/capstone/crops
git commit -m "feat: build offline capstone document corpus"
```

---

### Task 5: Build metadata-filtered hybrid retrieval and evidence inspection

**Files:**
- Create: `src/finai_academy/capstone/document_index.py`
- Create: `tests/test_capstone_document_index.py`

**Interfaces:**
- Consumes: `FinancialChunk`, `DocumentFilters`, and `DocumentRetrievalHit` from Task 1.
- Consumes: `BM25Index`, `DenseIndex`, `DeterministicTeachingEmbeddings`, `IndexedPassage`, and `reciprocal_rank_fusion` from `hybrid_retrieval.py`.
- Produces: `CertifiedDocumentIndex.search(query: str, *, filters: DocumentFilters, top_k: int = 3) -> tuple[DocumentRetrievalHit, ...]`.
- Produces: `CertifiedDocumentIndex.inspect(chunk_id: str) -> FinancialChunk`.
- Produces: `load_certified_document_index(root: Path | None = None) -> CertifiedDocumentIndex`.

- [ ] **Step 1: Write failing pre-filter and rank-lineage tests**

```python
def test_company_period_and_table_filters_run_before_ranking() -> None:
    index = load_certified_document_index(ROOT)
    hits = index.search(
        "reported revenue organic growth",
        filters=DocumentFilters(
            company_name="Schneider Electric",
            reporting_period="FY2025",
            element_type="table",
        ),
        top_k=3,
    )
    assert hits
    assert all(hit.chunk.context.company_name == "Schneider Electric" for hit in hits)
    assert all(hit.chunk.context.reporting_period == "FY2025" for hit in hits)
    assert all(hit.chunk.element_type == "table" for hit in hits)
    assert "40,152" in hits[0].chunk.text
    assert {name for name, _rank in hits[0].channel_ranks} == {"bm25", "dense"}
    assert hits[0].fused_score > 0


def test_exact_nvidia_figure_retrieves_the_atomic_segment_table() -> None:
    index = load_certified_document_index(ROOT)
    hit = index.search(
        "Which NVIDIA business generated 193,479 million?",
        filters=DocumentFilters(company_name="NVIDIA", element_type="table"),
        top_k=1,
    )[0]
    assert "$ 193,479" in hit.chunk.text
    assert "Compute" in hit.chunk.table.rows[0][1]
    assert index.inspect(hit.chunk.chunk_id) == hit.chunk
```

- [ ] **Step 2: Run retrieval tests and verify the missing-index failure**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_document_index.py -q`

Expected: FAIL because the certified document index is absent.

- [ ] **Step 3: Load and verify chunks before index creation**

Read `financial_chunks_v2.json`, validate every record as `FinancialChunk`, verify the artifact hash from `capstone_derived_artifacts`, and reject duplicate chunk IDs or mismatched source hashes.

- [ ] **Step 4: Apply eligibility before constructing rankers**

```python
def search(self, query, *, filters, top_k=3):
    eligible = tuple(chunk for chunk in self._chunks if filters.matches(chunk))
    if not eligible:
        return ()
    passages = tuple(_to_indexed_passage(chunk) for chunk in eligible)
    bm25 = BM25Index(passages).search(query, top_k=len(passages))
    embeddings = DeterministicTeachingEmbeddings()
    dense = DenseIndex(
        passages,
        embeddings,
        provider="certified-fixture",
        model=embeddings.model_name,
        chunking_strategy="financial-context-v2",
    ).search(query, top_k=len(passages))
    fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense})
    return tuple(_to_public_hit(item, self._chunks_by_id) for item in fused[:top_k])
```

The filter object must match company, period, document type, and element type by case-folded exact equality. Reject blank queries and `top_k` outside 1–5 with stable typed errors.

- [ ] **Step 5: Preserve both rank channels and exact evidence lookup**

`DocumentRetrievalHit` stores the immutable chunk, `fused_score`, `channel_ranks`, index version, and a short deterministic selection reason such as `Matched exact financial terms and related document meaning.` The UI wording must not expose implementation jargon until the advanced diagnostics view.

- [ ] **Step 6: Run retrieval plus existing hybrid tests**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_document_index.py tests/test_hybrid_retrieval.py -q`

Expected: PASS.

- [ ] **Step 7: Commit the document index**

```bash
git add src/finai_academy/capstone/document_index.py tests/test_capstone_document_index.py
git commit -m "feat: retrieve contextual document evidence"
```

---

### Task 6: Replace catalog tools with document search, inspection, and comparison

**Files:**
- Create: `src/finai_academy/capstone/document_tools.py`
- Create: `src/finai_academy/capstone/mcp_server.py`
- Create: `tests/test_capstone_document_tools.py`
- Modify: `src/finai_academy/capstone/tools.py`
- Modify: `tests/test_capstone_tools.py`

**Interfaces:**
- Consumes: `CertifiedDocumentIndex` from Task 5.
- Produces: `DocumentSearchOutcome`, `DocumentEvidenceOutcome`, and `ReportedValueComparison`.
- Produces: `DocumentCapabilityRegistry.search_financial_documents(company: str, reporting_period: str, query: str, element_type: str | None = None, top_k: int = 3) -> DocumentSearchOutcome`.
- Produces: `DocumentCapabilityRegistry.inspect_document_evidence(chunk_id: str) -> DocumentEvidenceOutcome`.
- Produces: `compare_reported_values(left: ReportedValue, right: ReportedValue) -> ReportedValueComparison`.
- Produces: allowlist `{"search_financial_documents", "inspect_document_evidence", "compare_reported_values"}`.
- Produces: `build_capstone_mcp_server(registry: DocumentCapabilityRegistry | None = None) -> MCPServer` exposing the first two document tools.

- [ ] **Step 1: Write failing capability and allowlist tests**

```python
def test_registry_exposes_only_document_research_capabilities() -> None:
    registry = AnalystToolRegistry(
        discovered=(
            "search_financial_documents",
            "inspect_document_evidence",
            "compare_reported_values",
            "place_order",
        )
    )
    assert registry.discover() == (
        "compare_reported_values",
        "inspect_document_evidence",
        "search_financial_documents",
    )


def test_search_then_inspect_returns_the_same_stable_chunk() -> None:
    registry = build_document_capability_registry(ROOT)
    search = registry.search_financial_documents(
        company="NVIDIA",
        reporting_period="FY2026",
        query="segment revenue 193,479",
        element_type="table",
        top_k=1,
    )
    inspected = registry.inspect_document_evidence(search.hits[0].chunk_id)
    assert inspected.chunk_id == search.hits[0].chunk_id
    assert inspected.physical_page == 165
    assert inspected.crop_asset_key.endswith("nvidia_segment_table_page_165.png")


def test_reported_value_comparison_uses_only_cited_inputs() -> None:
    result = compare_reported_values(
        left=ReportedValue(label="NVIDIA total", value=215938, unit="USD millions", chunk_id="a"),
        right=ReportedValue(label="Schneider total", value=40152, unit="EUR millions", chunk_id="b"),
    )
    assert result.absolute_difference is None
    assert result.comparable is False
    assert result.reason == "Currencies differ; no FX rate was supplied."
```

Define `ReportedValue` with `label`, finite numeric `value`, `unit`, and `chunk_id`. Define `ReportedValueComparison` with `left`, `right`, `comparable`, optional `absolute_difference`, optional `formula`, and `reason`. Define `DocumentSearchOutcome` and `DocumentEvidenceOutcome` as frozen Pydantic models that contain only the Task 1 public contracts and repository-relative crop keys.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_document_tools.py tests/test_capstone_tools.py -q`

Expected: FAIL because the new capability registry and allowlist do not exist.

- [ ] **Step 3: Implement pure capabilities before MCP wrapping**

```python
class DocumentCapabilityRegistry:
    def __init__(self, index: CertifiedDocumentIndex) -> None:
        self._index = index

    def search_financial_documents(self, company, reporting_period, query, element_type=None, top_k=3):
        hits = self._index.search(
            query,
            filters=DocumentFilters(
                company_name=company,
                reporting_period=reporting_period,
                element_type=element_type,
            ),
            top_k=top_k,
        )
        return DocumentSearchOutcome(status="ok", query=query, hits=hits)

    def inspect_document_evidence(self, chunk_id):
        chunk = self._index.inspect(chunk_id)
        return _to_evidence_outcome(chunk)
```

The comparison function returns a numeric difference only when currency and scale match. It always retains both cited chunk IDs and the displayed formula.

- [ ] **Step 4: Replace the old capstone catalog retriever without changing Lesson 10 fixtures**

Remove `_EVIDENCE_CATALOG` usage from `capstone/tools.py`. Keep Lesson 10's `financial_mcp_capabilities.py` and teaching catalog untouched. `build_certified_retriever()` becomes a compatibility constructor around `load_certified_document_index()` and returns metadata-rich `CapstoneEvidenceHit` objects.

Validate each tool's exact argument schema before invocation and never echo rejected input. Keep trading, code execution, arbitrary file access, and unrestricted browsing outside the allowlist.

- [ ] **Step 5: Add the capstone MCP adapter**

```python
@server.tool()
def search_financial_documents(
    company: str,
    reporting_period: str,
    query: str,
    element_type: str | None = None,
    top_k: Annotated[int, Field(ge=1, le=5)] = 3,
) -> dict[str, object]:
    return registry.search_financial_documents(
        company, reporting_period, query, element_type, top_k
    ).model_dump(mode="json")


@server.tool()
def inspect_document_evidence(chunk_id: str) -> dict[str, object]:
    return registry.inspect_document_evidence(chunk_id).model_dump(mode="json")
```

Expose comparison only as a host-side deterministic tool because it operates on already selected typed inputs and needs no external protocol boundary.

- [ ] **Step 6: Run tool, MCP, and safety tests**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_document_tools.py tests/test_capstone_tools.py tests/test_financial_mcp_server.py tests/test_financial_mcp_capabilities.py tests/test_financial_mcp_client.py -q`

Expected: PASS, with existing Lesson 10 behavior unchanged.

- [ ] **Step 7: Commit the document capability layer**

```bash
git add src/finai_academy/capstone/document_tools.py src/finai_academy/capstone/mcp_server.py src/finai_academy/capstone/tools.py tests/test_capstone_document_tools.py tests/test_capstone_tools.py
git commit -m "feat: expose capstone document evidence tools"
```

---

### Task 7: Make the service document-first and enforce element-level citations

**Files:**
- Modify: `final-project/shared/reference_mission.json`
- Modify: `src/finai_academy/capstone/models.py`
- Modify: `src/finai_academy/capstone/service.py`
- Modify: `src/finai_academy/capstone/briefing.py`
- Modify: `src/finai_academy/capstone/persistence.py`
- Modify: `tests/test_capstone_models.py`
- Modify: `tests/test_capstone_service.py`
- Modify: `tests/test_capstone_briefing.py`
- Modify: `tests/test_capstone_persistence.py`

**Interfaces:**
- Consumes: the three capability contracts from Task 6.
- Produces: document-rich `CapstoneEvidenceHit` and `CitedFact` contracts with `chunk_id`, element IDs, page, units, crop key, and source hash.
- Produces: a recorded reference mission that uses only official-document evidence.
- Produces: deterministic evidence-gate rules that verify chunk/document identities before release.

- [ ] **Step 1: Replace the fixture mission and write failing end-to-end service assertions**

Set the mission text to:

```text
Using the official FY2026 NVIDIA annual report and Schneider Electric FY2025 results, compare the reported revenue evidence. Cite the exact table or passage, preserve periods and units, and explain why the figures are not directly comparable.
```

Write tests:

```python
def test_recorded_mission_releases_only_real_document_evidence() -> None:
    request = ResearchRequest.reference()
    result = build_reference_copilot(run_id_factory=lambda: "document-run-001").run(request)
    assert result.status == "completed"
    assert result.evidence_gate.passed
    assert result.replan_count <= 1
    assert {hit.company for hit in result.evidence_gate.evidence_hits} == {
        "NVIDIA", "Schneider Electric"
    }
    assert all(hit.chunk_id and hit.element_ids and hit.physical_page for hit in result.evidence_gate.evidence_hits)
    assert any("193,479" in hit.text for hit in result.evidence_gate.evidence_hits)
    assert any("40,152" in hit.text for hit in result.evidence_gate.evidence_hits)
    assert all(fact.chunk_id for fact in result.briefing.cited_facts)


def test_evidence_gate_rejects_value_without_unit_or_source_hash() -> None:
    forged = valid_schneider_hit.model_copy(update={"unit": None, "document_sha256": "0" * 64})
    decision = evaluate_evidence_gate((valid_nvidia_hit, forged))
    assert not decision.passed
    assert "Schneider Electric contextual table evidence" in decision.missing_requirements
```

- [ ] **Step 2: Run service tests and verify schema failures**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_models.py tests/test_capstone_service.py tests/test_capstone_briefing.py -q`

Expected: FAIL because existing evidence hits carry only catalog text.

- [ ] **Step 3: Extend public evidence and citation contracts**

Add required fields:

```python
class CapstoneEvidenceHit(_FrozenPublicModel):
    company: str
    text: str
    chunk_id: str
    element_ids: tuple[str, ...]
    document_id: str
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    section: str
    period: str
    unit: str | None
    physical_page: int = Field(gt=0)
    printed_page: int | None = Field(default=None, gt=0)
    element_type: Literal["heading", "paragraph", "list", "table", "figure_caption", "footnote"]
    bbox: BoundingBox
    source_reference: str
    crop_asset_key: str | None
    original_markdown: str | None
    selection_reason: str
    channel_ranks: tuple[tuple[str, int], ...]
    fused_score: float = Field(ge=0)


class CitedFact(_FrozenPublicModel):
    claim: str
    company: str
    provenance_kind: Literal["document", "calculation"]
    source_reference: str
    chunk_id: str
    element_ids: tuple[str, ...]
    physical_page: int
```

Update all fake builders and recorded fixtures to construct these contracts from certified chunks rather than hand-written IDs.

- [ ] **Step 4: Replace the metric-first plan with a clean document plan**

The default plan contains at most five steps:

```python
_DOCUMENT_PLAN = (
    search_step(1, company="NVIDIA", period="FY2026", query="reported segment revenue", element_type="table"),
    inspect_step(2, company="NVIDIA", depends_on=(1,)),
    search_step(3, company="Schneider Electric", period="FY2025", query="reported revenue organic growth", element_type="table"),
    inspect_step(4, company="Schneider Electric", depends_on=(3,)),
    compare_step(5, depends_on=(2, 4)),
)
```

The main route does not manufacture a failure. A bounded replan occurs only when search returns no contextual table or inspection detects missing metadata. Keep maximum six steps and one replan.

- [ ] **Step 5: Implement element-level evidence gating and briefing construction**

For every reported fact, verify:

```python
hit = hits_by_chunk_id[fact.chunk_id]
assert set(fact.element_ids) <= set(hit.element_ids)
assert fact.company == hit.company
assert fact.source_reference == hit.source_reference
assert fact.physical_page == hit.physical_page
assert hit.document_sha256 in certified_document_hashes
```

Separate reported facts, deterministic calculations, interpretations, and limitations. The cross-company section must explicitly state that NVIDIA and Schneider use different currencies, reporting scopes, and periods where applicable.

- [ ] **Step 6: Update MLflow identities and trace payloads**

Replace fixture evidence identity with the verified chunk-artifact hash:

```python
manifest = _read_json(_project_root() / "assets/course-data/manifest.json")
record = manifest["capstone_derived_artifacts"][0]
identities.append(
    {
        "kind": "document_index",
        "identity": "financial-context-v2",
        "sha256": str(record["chunks"]["sha256"]),
    }
)
```

Trace search filters, candidate chunk IDs, channel ranks, selected chunk IDs, and evidence-gate results. Store repository-relative crop keys and source URLs, not local paths.

- [ ] **Step 7: Run service, persistence, and privacy tests**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_models.py tests/test_capstone_service.py tests/test_capstone_briefing.py tests/test_capstone_persistence.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the document-first service**

```bash
git add final-project/shared/reference_mission.json src/finai_academy/capstone/models.py src/finai_academy/capstone/service.py src/finai_academy/capstone/briefing.py src/finai_academy/capstone/persistence.py tests/test_capstone_models.py tests/test_capstone_service.py tests/test_capstone_briefing.py tests/test_capstone_persistence.py
git commit -m "feat: ground capstone answers in document elements"
```

---

### Task 8: Standardize live synthesis on GPT-5.6 Luna Responses API

**Files:**
- Modify: `src/finai_academy/settings.py`
- Modify: `src/finai_academy/capstone/model_gateway.py`
- Create: `scripts/smoke_capstone_openai.py`
- Modify: `.env.example`
- Modify: `tests/test_settings.py`
- Modify: `tests/test_capstone_providers.py`

**Interfaces:**
- Consumes: existing `StructuredModel.generate(system_prompt: str, user_prompt: str, response_model: type[ResponseT]) -> ResponseT` protocol.
- Produces: `OpenAIResponsesStructuredModel(client: OpenAI, model: str, reasoning_effort: str = "medium")`.
- Produces: `Settings.reasoning_effort` constrained to `low`, `medium`, or `high`.
- Produces: an explicit, safe live smoke command using the existing `.env` without printing its key.

- [ ] **Step 1: Write failing Responses API adapter tests with a fake client**

```python
class ExpectedBrief(BaseModel):
    answer: str


class FakeResponses:
    def __init__(self, output_parsed: ExpectedBrief) -> None:
        self.output_parsed = output_parsed
        self.last_call: dict[str, object] | None = None

    def parse(self, **kwargs):
        self.last_call = kwargs
        return SimpleNamespace(output_parsed=self.output_parsed)


class FakeOpenAIClient:
    def __init__(self, output_parsed: ExpectedBrief) -> None:
        self.responses = FakeResponses(output_parsed)


def test_openai_adapter_uses_luna_medium_structured_responses() -> None:
    client = FakeOpenAIClient(output_parsed=ExpectedBrief(answer="Evidence is cited."))
    model = OpenAIResponsesStructuredModel(
        client=client,
        model="gpt-5.6-luna",
        reasoning_effort="medium",
    )
    result = model.generate(
        system_prompt="Use only cited evidence.",
        user_prompt="Evidence payload",
        response_model=ExpectedBrief,
    )
    assert isinstance(result, ExpectedBrief)
    assert client.responses.last_call == {
        "model": "gpt-5.6-luna",
        "reasoning": {"effort": "medium"},
        "instructions": "Use only cited evidence.",
        "input": "Evidence payload",
        "text_format": ExpectedBrief,
        "store": False,
    }


def test_settings_reject_unknown_reasoning_effort() -> None:
    with pytest.raises(ValueError, match="FINAI_REASONING_EFFORT"):
        Settings(provider="openai", reasoning_effort="extreme")
```

- [ ] **Step 2: Run provider tests and verify adapter failure**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra ai pytest tests/test_settings.py tests/test_capstone_providers.py -q`

Expected: FAIL because the Responses adapter and reasoning setting do not exist.

- [ ] **Step 3: Implement the native OpenAI Responses adapter**

```python
class OpenAIResponsesStructuredModel:
    def __init__(self, *, client: OpenAI, model: str, reasoning_effort: str = "medium") -> None:
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort

    def generate(self, *, system_prompt, user_prompt, response_model):
        response = self._client.responses.parse(
            model=self._model,
            reasoning={"effort": self._reasoning_effort},
            instructions=system_prompt,
            input=user_prompt,
            text_format=response_model,
            store=False,
        )
        if response.output_parsed is None:
            raise ModelOutputError("provider returned no structured output")
        return response_model.model_validate(response.output_parsed)
```

For `provider="openai"`, instantiate `OpenAI()` and this adapter directly. Keep LangChain only for the optional Ollama route. Convert provider exceptions to the existing generic public provider error without serializing raw messages.

- [ ] **Step 4: Add settings and safe example configuration**

Add `reasoning_effort: str = "medium"` and read `FINAI_REASONING_EFFORT`. Keep `CHAT_DEFAULTS["openai"] == "gpt-5.6-luna"`.

In `.env.example`, show names only:

```dotenv
OPENAI_API_KEY=
FINAI_MODEL_PROVIDER=openai
FINAI_CHAT_MODEL=gpt-5.6-luna
FINAI_REASONING_EFFORT=medium
```

Never modify or stage `.env`.

- [ ] **Step 5: Add the explicit live smoke script**

The script loads `Settings.from_environment()`, requires the OpenAI route, runs the fixed reference mission once, and prints only:

```text
provider=openai model=gpt-5.6-luna status=completed citations=2
```

It exits nonzero when the structured response, evidence gate, or citation validation fails. It never prints prompts, source text, response bodies, environment values, or raw exceptions.

- [ ] **Step 6: Run unit tests, then the authorized live smoke test**

Run unit tests:

`UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra ai pytest tests/test_settings.py tests/test_capstone_providers.py -q`

Expected: PASS without network.

Run live:

`UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra ai --extra capstone python scripts/smoke_capstone_openai.py`

Expected: safe one-line `status=completed`. Do not copy `.env` content into logs or artifacts.

- [ ] **Step 7: Commit the Luna route**

```bash
git add .env.example src/finai_academy/settings.py src/finai_academy/capstone/model_gateway.py scripts/smoke_capstone_openai.py tests/test_settings.py tests/test_capstone_providers.py
git commit -m "feat: use Luna Responses API in capstone"
```

---

### Task 9: Replace the process-heavy UI with a simple evidence-first workspace

**Files:**
- Modify: `src/finai_academy/capstone/views.py`
- Modify: `src/finai_academy/capstone/streamlit_ui.py`
- Modify: `tests/test_capstone_views.py`
- Modify: `tests/test_capstone_streamlit.py`

**Interfaces:**
- Consumes: document-rich `ResearchRunResult` from Task 7.
- Produces: `AnswerView`, `EvidenceComparisonView`, `HowItWorkedView`, and a simplified `CapstoneRunView`.
- Produces: first screen with one question, one primary action, one document-set sentence, and collapsed advanced settings.
- Produces: result tabs `Answer`, `Evidence`, and `How it worked` in that order.

- [ ] **Step 1: Write failing view and Streamlit hierarchy tests**

```python
def test_initial_screen_leads_with_the_learning_job() -> None:
    app = AppTest.from_function(_app, args=(_successful_factory, readiness)).run()
    text = _rendered_text(app)
    assert app.title[0].value == "Financial Document Analyst"
    assert "Ask a financial question and see the exact report page and table behind the answer." in text
    assert next(item for item in app.button if item.key == "analyze_reports").label == "Analyze the reports"
    assert [item.label for item in app.expander] == ["Advanced settings"]
    assert not app.tabs


def test_completed_result_orders_answer_evidence_then_process() -> None:
    app = _run_analysis(AppTest.from_function(_app, args=(_successful_factory, readiness)).run())
    assert [tab.label for tab in app.tabs] == ["Answer", "Evidence", "How it worked"]
    text = _rendered_text(app)
    assert "Original report" in text
    assert "Extracted table" in text
    assert "Why this evidence was selected" in text
    assert "Source details" in text
    assert "Research plan" not in app.tabs[0].value
```

- [ ] **Step 2: Run UI tests and verify the old hierarchy fails**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_views.py tests/test_capstone_streamlit.py -q`

Expected: FAIL because the old UI begins with provider controls and Reference mission/Ask the analyst tabs.

- [ ] **Step 3: Build plain-language public view models**

```python
class EvidenceComparisonView(_FrozenPublicModel):
    company: str
    page_label: str
    crop_asset_key: str
    extracted_markdown: str
    retrieved_chunk: str
    selection_reason: str
    source_details: tuple[tuple[str, str], ...]


class CapstoneRunView(_FrozenPublicModel):
    run_id: str
    question: str
    answer: AnswerView | None
    evidence: tuple[EvidenceComparisonView, ...]
    how_it_worked: HowItWorkedView
    release: ReleaseView
    outcome: OutcomeView
```

Map technical fields to learner copy. Keep exact channel ranks, tool attempts, trace events, MLflow IDs, and deterministic scores inside `HowItWorkedView`.

- [ ] **Step 4: Simplify the first screen**

Render in this order:

```python
st.title("Financial Document Analyst")
st.write("Ask a financial question and see the exact report page and table behind the answer.")
question = st.text_area("Question", value=REFERENCE_MISSION, height=110)
st.caption("Evidence comes from NVIDIA's FY2026 annual report and Schneider Electric's FY2025 results.")
run_clicked = st.button("Analyze the reports", key="analyze_reports", type="primary")
with st.expander("Advanced settings", expanded=False):
    provider_label = st.selectbox("Provider", tuple(_PROVIDERS))
    model = st.text_input("Model", value=_DEFAULT_MODELS[provider_label])
    st.caption(f"{provider_label} · {model} · certified document data")
    st.button("Reset analysis", key="reset_capstone")
```

Use recorded mode by default. Set `_DEFAULT_MODELS["OpenAI"] = "gpt-5.6-luna"`.

- [ ] **Step 5: Render results in the approved order**

`Answer` starts with the conclusion, then company evidence, comparison limits, and citations. `Evidence` uses two columns for each selected item:

```python
left, right = st.columns((1, 1))
with left:
    st.subheader("Original report")
    st.image(str(ROOT / evidence.crop_asset_key), caption=evidence.page_label)
with right:
    st.subheader("Extracted table")
    st.markdown(evidence.extracted_markdown)
st.subheader("Why this evidence was selected")
st.write(evidence.selection_reason)
with st.expander("Source details"):
    for label, value in evidence.source_details:
        st.markdown(f"**{label}:** {value}")
```

`How it worked` begins with the five-step pipeline in plain language. Put retrieval scores, tool activity, trace, evaluation, model route, and MLflow identifiers in collapsed expanders.

- [ ] **Step 6: Add blocked and error copy tests**

Assert these exact safe messages:

```text
The reports did not provide enough contextual evidence to release an answer.
Check the missing evidence below, then run the certified analysis again.
```

Never show `typed stop`, `trajectory`, `RRF`, raw exceptions, or evidence-gate jargon outside advanced diagnostics.

- [ ] **Step 7: Run UI and privacy tests**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_views.py tests/test_capstone_streamlit.py tests/test_capstone_models.py -q`

Expected: PASS.

- [ ] **Step 8: Commit the simple interface**

```bash
git add src/finai_academy/capstone/views.py src/finai_academy/capstone/streamlit_ui.py tests/test_capstone_views.py tests/test_capstone_streamlit.py
git commit -m "feat: simplify capstone evidence workspace"
```

---

### Task 10: Replace assembly-only student seams with document-context tasks

**Files:**
- Modify: `final-project/student/integration.py`
- Modify: `final-project/student/verify.py`
- Modify: `final-project/student/diagnostic_case.json`
- Modify: `final-project/student/README.md`
- Modify: `final-project/student/CHECKLIST.md`
- Modify: `final-project/STUDENT_BRIEF.md`
- Modify: `final-project/INSTRUCTOR_GUIDE.md`
- Modify: `final-project/README.md`
- Modify: `final-project/PRODUCT_SPEC.md`
- Modify: `final-project/reference/student_integration_solution.py`
- Modify: `tests/test_capstone_student.py`
- Modify: `tests/test_capstone_docs.py`

**Interfaces:**
- Consumes: certified parsed elements, chunk builder, index, and public evidence contracts.
- Produces four student seams: `preserve_table_context`, `attach_required_metadata`, `search_with_metadata_filters`, and `map_citation_to_evidence`.
- Produces a broken-table diagnostic where a financial value is missing its unit/header context.
- Produces: `load_diagnostic_case() -> BrokenTableCase`, `diagnose_context(case: BrokenTableCase) -> DiagnosticResult`, and `CitationMapping` with chunk, element, page, and source fields.

- [ ] **Step 1: Write failing student-contract tests**

```python
EXPECTED_SEAMS = (
    "preserve_table_context",
    "attach_required_metadata",
    "search_with_metadata_filters",
    "map_citation_to_evidence",
)


def test_student_solution_preserves_the_schneider_table_context() -> None:
    chunk = solution.preserve_table_context(schneider_fy_table_element, schneider_document)
    assert "€ million" in chunk.text
    assert "Organic growth" in chunk.text
    assert "40,152" in chunk.text
    assert chunk.table.column_count == 6


def test_student_diagnostic_rejects_a_number_without_unit_and_headers() -> None:
    case = load_diagnostic_case()
    result = diagnose_context(case)
    assert result.passed is False
    assert result.missing_fields == ("unit", "column_headers")
    assert result.release_safe is False
```

- [ ] **Step 2: Run student tests and verify the old seam names fail**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_student.py tests/test_capstone_docs.py -q`

Expected: FAIL because the scaffold still contains assembly seams.

- [ ] **Step 3: Implement the new incomplete scaffold and reference solution**

Each starter function raises `StudentIntegrationIncomplete` with one short hint. The reference solution delegates to the production boundaries:

```python
def preserve_table_context(element, document):
    return build_table_chunk(document, element)


def attach_required_metadata(chunk, document, element):
    return chunk.model_copy(
        update={
            "context": build_contextual_metadata(document, element),
            "financial": build_financial_metadata(document, element),
        }
    )


def search_with_metadata_filters(index, company, period, query):
    return index.search(
        query,
        filters=DocumentFilters(company_name=company, reporting_period=period),
        top_k=2,
    )


def map_citation_to_evidence(hit):
    return CitationMapping(
        chunk_id=hit.chunk.chunk_id,
        element_ids=hit.chunk.source_element_ids,
        physical_page=hit.chunk.context.physical_page,
        source_reference=str(hit.chunk.context.official_source_url),
    )
```

- [ ] **Step 4: Make the verifier deterministic and bounded**

Keep subprocess isolation, output limits, timeouts, and public-path screening. Change the four incomplete groups and retriever expectations to the new seams and stable v2 chunks. The starter must fail exactly five checks: four seams plus the diagnostic. The solved reference must print exactly one `CAPSTONE_PASS` marker and exit zero.

- [ ] **Step 5: Rewrite instructions around one visible learning journey**

The student brief opens with:

```text
Your job: make one real financial table safe for AI retrieval.

You will preserve its headers and units, attach contextual metadata, retrieve it with company and period filters, and map one cited answer back to the original report page.
```

Give a 60-minute schedule: 5 minutes inspect, 15 table context, 10 metadata, 10 filtered search, 10 citation mapping, 10 verification. Explain the exact run commands and what a safe result looks like. Keep provider and MLflow material in an optional post-lab section.

- [ ] **Step 6: Run student verification and documentation tests**

Run:

```bash
UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone python final-project/student/verify.py
UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_student.py tests/test_capstone_docs.py -q
```

Expected: the starter verifier reports the five intended failures; pytest passes because it validates both starter and injected reference solution behavior.

- [ ] **Step 7: Commit the document-learning lab**

```bash
git add final-project/student/integration.py final-project/student/verify.py final-project/student/diagnostic_case.json final-project/student/README.md final-project/student/CHECKLIST.md final-project/STUDENT_BRIEF.md final-project/INSTRUCTOR_GUIDE.md final-project/README.md final-project/PRODUCT_SPEC.md final-project/reference/student_integration_solution.py tests/test_capstone_student.py tests/test_capstone_docs.py
git commit -m "docs: teach contextual document evidence in capstone"
```

---

### Task 11: Certify extraction, retrieval, UI, MLflow, and privacy

**Files:**
- Modify: `scripts/certify_capstone.py`
- Modify: `tests/test_capstone_certification.py`
- Modify: `artifacts/capstone/visual-inspection.json`
- Replace: `artifacts/capstone/*.png`
- Modify: `artifacts/capstone/certification.json`
- Modify: `artifacts/capstone/readiness.md`

**Interfaces:**
- Consumes: all production and student boundaries from Tasks 1–10.
- Produces: deterministic certification schema version `2` and five bound 1440x1000 visual captures.
- Produces: an evidence-backed release result for the capstone workstream.

- [ ] **Step 1: Write failing certification-v2 assertions**

```python
def test_certification_records_document_extraction_and_context_gates(certification_runs) -> None:
    payload = json.loads((artifact_directory / "certification.json").read_text())
    assert payload["schema_version"] == 2
    assert payload["documents"] == {
        "source_hashes_valid": True,
        "page_count": 194,
        "nvidia_target_table_shape": [14, 4],
        "schneider_target_table_shapes": [[5, 6], [4, 6], [4, 6]],
        "contextual_metadata_valid": True,
        "personal_paths_absent": True,
    }
    assert payload["retrieval"]["filters_before_ranking"] is True
    assert payload["retrieval"]["target_tables_retrieved"] is True
    assert payload["streamlit"]["result_tabs"] == ["Answer", "Evidence", "How it worked"]
    assert payload["reference_mission"]["citation_pairs_valid"] is True


def test_visual_manifest_covers_the_simple_interface() -> None:
    assert set(visual["covered_elements"]) == {
        "clear_title_and_intro",
        "single_primary_action",
        "advanced_settings_collapsed",
        "answer_first",
        "original_and_extracted_evidence",
        "plain_language_process",
        "no_clipping",
        "blocked_state_guidance",
        "exact_footer",
    }
```

- [ ] **Step 2: Run certification tests and verify schema-v1 failure**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone pytest tests/test_capstone_certification.py -q`

Expected: FAIL because the committed certification still describes the catalog-based UI.

- [ ] **Step 3: Update deterministic certification gates**

Implement and call these exact helpers:

```python
def source_hashes_valid() -> bool:
    try:
        for source in load_certified_document_sources(_PROJECT_ROOT / "assets/course-data/manifest.json"):
            verify_source_asset(source, _PROJECT_ROOT)
    except (OSError, SourceAssetError, ValueError):
        return False
    return True


def target_table_contracts() -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    sources = load_certified_document_sources(_PROJECT_ROOT / "assets/course-data/manifest.json")
    parser = PyMuPDF4LLMParser()
    nvidia = parser.parse(sources[0], project_root=_PROJECT_ROOT, pages=(165,))
    schneider = parser.parse(sources[1], project_root=_PROJECT_ROOT, pages=(16,))
    nvidia_shapes = tuple(
        (item.table.row_count, item.table.column_count)
        for item in nvidia.elements if item.table is not None
    )
    schneider_shapes = tuple(
        (item.table.row_count, item.table.column_count)
        for item in schneider.elements if item.table is not None
    )
    return nvidia_shapes[0], schneider_shapes


def all_required_context_present() -> bool:
    chunks = load_certified_chunks(_PROJECT_ROOT)
    return all(
        chunk.context.company_name
        and chunk.context.reporting_period
        and chunk.context.physical_page > 0
        and chunk.context.document_sha256
        and chunk.source_element_ids
        and (chunk.element_type != "table" or chunk.table is not None)
        for chunk in chunks
    )


def target_queries_retrieve_target_tables() -> bool:
    index = load_certified_document_index(_PROJECT_ROOT)
    nvidia = index.search(
        "segment revenue 193,479",
        filters=DocumentFilters(company_name="NVIDIA", element_type="table"),
        top_k=1,
    )
    schneider = index.search(
        "reported revenue organic growth 40,152",
        filters=DocumentFilters(company_name="Schneider Electric", element_type="table"),
        top_k=1,
    )
    return bool(nvidia and schneider and "193,479" in nvidia[0].chunk.text and "40,152" in schneider[0].chunk.text)


def all_citations_resolve_to_elements(result: ResearchRunResult) -> bool:
    if result.briefing is None:
        return False
    hits = {hit.chunk_id: hit for hit in result.evidence_gate.evidence_hits}
    return all(
        fact.chunk_id in hits
        and set(fact.element_ids) <= set(hits[fact.chunk_id].element_ids)
        and fact.physical_page == hits[fact.chunk_id].physical_page
        for fact in result.briefing.cited_facts
    )


def mlflow_document_index_identity_present(identities: Mapping[str, object]) -> bool:
    return any(
        item.get("kind") == "document_index"
        and item.get("identity") == "financial-context-v2"
        for item in identities.get("identities", [])
        if isinstance(item, Mapping)
    )


def public_artifact_scan_passed(paths: Sequence[Path]) -> bool:
    return all(_safe_public_text(path.read_text(encoding="utf-8", errors="ignore")) for path in paths)


_require(source_hashes_valid())
_require(target_table_contracts() == ((14, 4), ((5, 6), (4, 6), (4, 6))))
_require(all_required_context_present())
_require(target_queries_retrieve_target_tables())
_require(reference_result.evidence_gate.passed)
_require(all_citations_resolve_to_elements(reference_result))
_require(mlflow_document_index_identity_present(dataset_identities))
_require(public_artifact_scan_passed(public_text_artifacts))
```

Do not require an artificial replan. Require `replan_count <= 1`, `observation_count <= max_steps`, and no duplicate successful tool signature.

- [ ] **Step 4: Capture and inspect five real browser states**

Run the reference app:

`UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone streamlit run final-project/reference/streamlit_app.py`

At 1440x1000 capture:

1. `01-initial-simple-workspace.png`
2. `02-answer-tab.png`
3. `03-evidence-tab.png`
4. `04-how-it-worked-tab.png`
5. `05-blocked-evidence-state.png`

Inspect every image for readable hierarchy, complete tables, correct page crops, no clipping, no personal paths, and no secret-shaped strings. Bind filename, SHA-256, dimensions, state, and covered elements in `visual-inspection.json`.

- [ ] **Step 5: Regenerate certification artifacts**

Run: `UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone python scripts/certify_capstone.py --artifact-dir artifacts/capstone`

Expected: `offline_release_passed=true`, two certified documents, target tables valid, reference mission completed, Streamlit journey passed, student solution passed, MLflow persisted, and public artifact scan passed.

- [ ] **Step 6: Run the capstone and document test groups**

Run:

```bash
UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra ai --extra capstone pytest tests/test_capstone_document_assets.py tests/test_capstone_document_ingestion.py tests/test_capstone_document_chunking.py tests/test_capstone_document_artifacts.py tests/test_capstone_document_index.py tests/test_capstone_document_tools.py -q
UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra ai --extra capstone pytest tests/test_capstone_models.py tests/test_capstone_tools.py tests/test_capstone_service.py tests/test_capstone_briefing.py tests/test_capstone_providers.py tests/test_capstone_views.py tests/test_capstone_streamlit.py tests/test_capstone_student.py tests/test_capstone_persistence.py tests/test_capstone_certification.py tests/test_capstone_docs.py -q
```

Expected: PASS.

- [ ] **Step 7: Run lint, offline mission, and full-suite regression checks**

Run:

```bash
UV_CACHE_DIR=/tmp/finai-uv-cache uv run ruff check src/finai_academy/capstone scripts/build_capstone_document_assets.py scripts/smoke_capstone_openai.py scripts/certify_capstone.py tests/test_capstone_*.py
UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra capstone python scripts/certify_capstone.py --artifact-dir /tmp/finai-capstone-certification
UV_CACHE_DIR=/tmp/finai-uv-cache uv run --extra ai --extra capstone pytest -q --junitxml=/tmp/finai-capstone-full-suite.xml
```

Expected for the capstone workstream: no capstone or document failures and byte-stable offline certification. The previously audited lesson-heading, deck-contract, Lesson 06 assertion, and notebook-render test-order failures belong to the separate course-release cleanup workstream; do not mask, delete, or weaken them here.

- [ ] **Step 8: Review the final diff for scope and privacy**

Run:

```bash
git diff --check
git status --short
rg -n 'sk-[A-Za-z0-9_-]{12,}|/Users/arnauddemes|OPENAI_API_KEY=' src final-project assets/course-data/capstone artifacts/capstone scripts tests
```

Expected: no whitespace errors; only intended capstone files plus pre-existing user changes; no real key or personal path in public files. The literal empty example `OPENAI_API_KEY=` is allowed only in `.env.example`.

- [ ] **Step 9: Commit certification evidence**

```bash
git add scripts/certify_capstone.py tests/test_capstone_certification.py artifacts/capstone
git commit -m "test: certify full-document capstone"
```

---

## Completion gate

The capstone workstream is complete only when all of the following are evidenced:

- both official PDFs match their manifest hashes;
- the NVIDIA and Schneider target tables match the exact shape/value contracts;
- contextual metadata and table-aware chunks are reproducible;
- hybrid retrieval applies metadata eligibility before rank scoring;
- every reported claim resolves to a chunk, source elements, page, source URL, and document hash;
- the recorded route passes without network access;
- the live Luna smoke route passes without exposing secrets;
- the initial UI is simple and the result order is Answer, Evidence, How it worked;
- the student exercise completes the four document-context seams;
- MLflow records sanitized source/index lineage and evaluation;
- all capstone tests pass in isolation and within the full repository run; and
- the separate course-release cleanup workstream is tracked before the entire course is called release-ready.
