from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "final-project" / "simple_app.py"


def _load_app():
    spec = importlib.util.spec_from_file_location("simple_student_app", APP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_retrieval_returns_the_most_relevant_company_scoped_chunk() -> None:
    app = _load_app()
    chunks = (
        app.Chunk("NVIDIA", 10, "Data center revenue grew 78% year over year."),
        app.Chunk("Schneider Electric", 16, "Organic growth was reported for Energy Management."),
    )

    hits = app.retrieve_chunks("NVIDIA data center revenue", chunks, top_k=1)

    assert len(hits) == 1
    assert hits[0].company == "NVIDIA"
    assert "Data center" in hits[0].text


def test_comparison_retrieval_keeps_both_companies_in_view() -> None:
    app = _load_app()
    chunks = (
        app.Chunk("NVIDIA", 165, "Data Center revenue increased 68% year over year."),
        app.Chunk("Schneider Electric", 5, "Organic growth was reported for Energy Management."),
    )

    hits = app.retrieve_chunks(
        "How did operating growth differ between NVIDIA and Schneider Electric?",
        chunks,
        top_k=4,
    )

    assert {hit.company for hit in hits} == {"NVIDIA", "Schneider Electric"}


def test_offline_preview_is_compact_and_cites_each_company() -> None:
    app = _load_app()
    hits = (
        app.Chunk("NVIDIA", 23, "Data Center revenue grew 59% and total revenue grew 65% year on year."),
        app.Chunk("Schneider Electric", 1, "FY25 revenues grew 9% organically; Energy Management grew 10%."),
    )

    preview = app.offline_preview(hits)

    assert "NVIDIA · page 23" in preview
    assert "Schneider Electric · page 1" in preview
    assert len(preview) < 500
