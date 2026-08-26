from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from finai_academy.capstone.document_assets import load_certified_document_sources
from finai_academy.capstone.document_ingestion import (
    PyMuPDF4LLMParser,
    render_evidence_crop,
)
from finai_academy.capstone.document_models import BoundingBox, FinancialDocumentSource

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets/course-data/manifest.json"


def _source(company: str) -> FinancialDocumentSource:
    return next(
        item for item in load_certified_document_sources(MANIFEST) if item.company_name == company
    )


def _blank_source(tmp_path: Path) -> FinancialDocumentSource:
    path = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()
    raw = path.read_bytes()
    return FinancialDocumentSource(
        document_id="BLANK-OCR-TEST",
        company_name="Blank Test Company",
        ticker="BLNK",
        document_type="Test Document",
        reporting_period="FY2026",
        publication_date=date(2026, 1, 1),
        official_source_url="https://example.com/blank.pdf",
        local_asset_key="blank.pdf",
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_size=len(raw),
        page_count=1,
    )


@dataclass(frozen=True)
class _OcrResult:
    text: str
    bbox: BoundingBox
    engine: str
    language: str
    confidence: float


@dataclass
class _RecordingOcrAdapter:
    calls: list[tuple[Path, int]] = field(default_factory=list)

    def extract(self, *, asset_path: Path, page_number: int) -> _OcrResult:
        self.calls.append((asset_path, page_number))
        return _OcrResult(
            text="OCR-derived revenue evidence",
            bbox=BoundingBox(x0=10, y0=20, x1=200, y1=40),
            engine="test-ocr",
            language="en",
            confidence=0.98,
        )


class _FailingOcrAdapter:
    def extract(self, *, asset_path: Path, page_number: int) -> _OcrResult:
        raise RuntimeError(f"private failure at {asset_path}")


def test_nvidia_target_page_preserves_one_14_by_4_table() -> None:
    parsed = PyMuPDF4LLMParser().parse(_source("NVIDIA"), project_root=ROOT, pages=(165,))

    tables = [item for item in parsed.elements if item.element_type == "table"]
    assert len(tables) == 1
    assert tables[0].table is not None
    assert (tables[0].table.row_count, tables[0].table.column_count) == (14, 4)
    assert tables[0].table.rows[3] == (
        "Revenue",
        "$ 193,479",
        "$ 22,459",
        "$ 215,938",
    )
    assert tables[0].table.rows[0][0] == ""
    assert tables[0].physical_page == 165
    assert tables[0].printed_page == 77


def test_schneider_target_page_preserves_three_six_column_tables() -> None:
    parsed = PyMuPDF4LLMParser().parse(
        _source("Schneider Electric"), project_root=ROOT, pages=(16,)
    )

    tables = [item for item in parsed.elements if item.element_type == "table"]
    assert [
        (item.table.row_count, item.table.column_count) for item in tables if item.table is not None
    ] == [(5, 6), (4, 6), (4, 6)]
    assert tables[2].table is not None
    assert tables[2].table.rows[-1] == (
        "Group",
        "40,152",
        "+8.9%",
        "+0.8%",
        "-4.1%",
        "+5.2%",
    )
    assert tables[0].table is not None
    assert tables[0].table.rows[0][:3] == ("", "", "")


def test_parser_output_contains_no_local_filename() -> None:
    parsed = PyMuPDF4LLMParser().parse(_source("NVIDIA"), project_root=ROOT, pages=(165,))

    payload = parsed.model_dump_json()
    assert "/Users/" not in payload
    assert str(ROOT) not in payload
    assert "\u200b" not in payload


def test_parser_uses_deterministic_element_ids_and_neighbors() -> None:
    source = _source("Schneider Electric")
    first = PyMuPDF4LLMParser().parse(source, project_root=ROOT, pages=(16,))
    second = PyMuPDF4LLMParser().parse(source, project_root=ROOT, pages=(16,))

    assert [element.element_id for element in first.elements] == [
        element.element_id for element in second.elements
    ]
    assert first.elements[0].previous_element_id is None
    assert first.elements[-1].next_element_id is None
    assert all(
        element.next_element_id == first.elements[index + 1].element_id
        for index, element in enumerate(first.elements[:-1])
    )


def test_render_evidence_crop_creates_rgb_png(tmp_path: Path) -> None:
    destination = tmp_path / "evidence" / "nvidia-revenue.png"

    result = render_evidence_crop(
        _source("NVIDIA"),
        project_root=ROOT,
        page_number=165,
        bbox=BoundingBox(x0=36, y0=138, x1=576, y1=495),
        destination=destination,
    )

    assert result == destination
    with Image.open(result) as image:
        assert image.width > 0
        assert image.height > 0
        assert image.mode in {"RGB", "RGBA"}


@pytest.mark.parametrize("scale", (float("nan"), float("inf"), float("-inf"), 0, -1))
def test_render_evidence_crop_rejects_non_finite_or_non_positive_scale(
    tmp_path: Path, scale: float
) -> None:
    with pytest.raises(ValueError, match="scale must be a finite positive number"):
        render_evidence_crop(
            _source("NVIDIA"),
            project_root=ROOT,
            page_number=165,
            bbox=None,
            destination=tmp_path / "crop.png",
            scale=scale,
        )


def test_parser_uses_injected_ocr_for_native_text_empty_page(tmp_path: Path) -> None:
    source = _blank_source(tmp_path)
    adapter = _RecordingOcrAdapter()

    parsed = PyMuPDF4LLMParser(adapter).parse(source, project_root=tmp_path, pages=(1,))

    assert adapter.calls == [(tmp_path / "blank.pdf", 1)]
    assert [(element.original_text, element.extraction_method) for element in parsed.elements] == [
        ("OCR-derived revenue evidence", "ocr")
    ]
    assert parsed.diagnostics[0].model_dump() == {
        "code": "ocr_used",
        "severity": "warning",
        "physical_page": 1,
        "message": "OCR engine=test-ocr, language=en, confidence=0.98.",
        "extraction_method": "ocr",
    }


def test_parser_requires_ocr_for_native_text_empty_page_without_adapter(tmp_path: Path) -> None:
    parsed = PyMuPDF4LLMParser().parse(_blank_source(tmp_path), project_root=tmp_path, pages=(1,))

    assert parsed.elements == ()
    assert parsed.diagnostics[0].model_dump() == {
        "code": "ocr_required",
        "severity": "error",
        "physical_page": 1,
        "message": "Page has no usable native text layer.",
        "extraction_method": "native_text",
    }


def test_parser_reports_ocr_failure_without_exposing_exception_details(tmp_path: Path) -> None:
    source = _blank_source(tmp_path)

    parsed = PyMuPDF4LLMParser(_FailingOcrAdapter()).parse(
        source, project_root=tmp_path, pages=(1,)
    )

    assert parsed.elements == ()
    assert parsed.diagnostics[0].code == "ocr_failed"
    assert parsed.diagnostics[0].severity == "error"
    assert parsed.diagnostics[0].extraction_method == "ocr"
    assert "private failure" not in parsed.diagnostics[0].message
    assert str(tmp_path) not in parsed.diagnostics[0].message
