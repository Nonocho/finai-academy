from __future__ import annotations

from pathlib import Path

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


def test_parser_output_contains_no_local_filename() -> None:
    parsed = PyMuPDF4LLMParser().parse(_source("NVIDIA"), project_root=ROOT, pages=(165,))

    payload = parsed.model_dump_json()
    assert "/Users/" not in payload
    assert str(ROOT) not in payload


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
