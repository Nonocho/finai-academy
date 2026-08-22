"""Contract tests for the Lesson 11 plan-and-execute notebook."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from xml.etree import ElementTree

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "11_plan_and_execute_analyst.ipynb"
BUILDER = ROOT / "scripts" / "build_lesson11_notebook.py"
EXECUTOR = ROOT / "scripts" / "execute_notebooks.py"
CHAPTER = ROOT / "chapters" / "11-plan-and-execute-analyst.md"
DECK = ROOT / "decks" / "11-plan-and-execute-analyst.pptx"
PRESENTATION_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def _shape_text(shape: ElementTree.Element) -> str:
    return " ".join(
        " ".join(
            node.text or ""
            for node in shape.iter(f"{{{DRAWING_NS}}}t")
        ).split()
    )


def _build_notebook():
    spec = spec_from_file_location("lesson11_notebook_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_notebook()


def _png_output_count(notebook) -> int:
    return sum(
        "image/png" in output.get("data", {})
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"}
    )


def _stream_text(notebook) -> str:
    return "".join(
        output.get("text", "")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def test_lesson11_notebook_is_output_free_and_contains_the_teaching_contract() -> None:
    assert NOTEBOOK.is_file()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert notebook.metadata["finai"]["expected_runtime_minutes"] == 40
    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
    assert all(
        cell.get("execution_count") is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    assert 24 <= len(notebook.cells) <= 28
    assert nbformat.writes(_build_notebook()) == nbformat.writes(notebook)

    for heading in (
        "## Learning objectives",
        "## Where this fits",
        "## Failure lab",
        "## Verification",
        "## Knowledge check",
        "## Challenge",
        "## Capstone integration",
        "## Recap",
    ):
        assert heading in source
    for marker in (
        "ResearchPlan",
        "ReplanDecision",
        "FinancialMcpPlanningExecutor",
        "unsupported_metric",
        "FINAI_LIVE_MODE",
        "Ollama",
        "OpenAI",
        "Lesson 12",
        "LESSON_11_PASS",
        "offline fixture · deterministic planner and replanner · real local MCP execution",
    ):
        assert marker in source
    assert "async def run_lesson11(*, live_mode: bool)" in source


def test_lesson11_notebook_executes_offline_with_visual_evidence(tmp_path: Path) -> None:
    assert NOTEBOOK.is_file()
    output_dir = tmp_path / "executed"
    result = subprocess.run(
        [
            sys.executable,
            str(EXECUTOR),
            str(NOTEBOOK),
            "--mode",
            "offline",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    executed = nbformat.read(output_dir / NOTEBOOK.name, as_version=4)
    text = _stream_text(executed)
    assert _png_output_count(executed) >= 6
    assert "Real MCP server:" in text
    assert "offline fixture · deterministic planner and replanner · real local MCP execution" in text
    assert "Plan revisions: 1" in text
    assert "Evidence gate passed: True" in text
    assert text.count("LESSON_11_PASS") == 1


def test_lesson11_chapter_defines_the_complete_instructor_route() -> None:
    """Catch loss of a teachable route, recovery path, or Lesson 12 handoff."""

    assert CHAPTER.is_file()
    chapter = CHAPTER.read_text(encoding="utf-8")
    normalized_chapter = chapter.lower()

    for marker in (
        "13:30-14:30",
        "12-minute concept deck",
        "40-minute notebook",
        "8-minute verification and debrief",
        "get_company_metric",
        "search_financial_documents",
        "unsupported_metric",
        "replace_remaining",
        "evidence gate",
        "Ollama",
        "OpenAI",
        "No-network fallback",
        "Skip if late",
        "LESSON_11_PASS",
        "Lesson 12",
        "read-only",
        "investment advice",
        "missing MCP SDK",
        "subprocess",
        "empty discovery",
        "invalid live output",
        "insufficient evidence",
        "same-tool recovery",
        "strategy revision",
        "lesson11-000",
        "lesson11-027",
        "1, 2, 3, 5, and 6",
        "initial and final plans",
        "capability names and arguments",
        "replan count",
        "latency per stage",
        "full Lesson 11 route is ready for an instructor-led test class",
        "| 6 | `search_financial_documents` | `Schneider Electric`, `energy management`, `top_k=2` |",
        "reported_facts",
        "cross_company_observations",
        "interpretation",
        "limitations",
        "source_references",
    ):
        assert marker.lower() in normalized_chapter
    assert "—" not in chapter
    assert "[Lesson 11 concept deck](../decks/11-plan-and-execute-analyst.pptx)" in chapter


def test_lesson11_final_assets_and_indexes_report_truthful_readiness() -> None:
    """Catch missing delivery assets, stale planned copy, or a hidden final deck."""

    for asset in (CHAPTER, NOTEBOOK, DECK):
        assert asset.is_file()

    indexes = {
        "chapters/README.md": "[Plan-and-execute financial analyst](11-plan-and-execute-analyst.md)",
        "notebooks/README.md": "[Plan-and-execute financial analyst](11_plan_and_execute_analyst.ipynb)",
        "README.md": "[Lesson 11 instructor chapter](chapters/11-plan-and-execute-analyst.md)",
    }
    for relative_path, expected_link in indexes.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        normalized_text = " ".join(text.split())
        assert expected_link in text
        assert "Lessons 11–12 remain planned" not in text
        assert "Lessons 08-11" in normalized_text
        assert "Lesson 12 remains planned" in normalized_text

    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[Lesson 11 concept deck](decks/11-plan-and-execute-analyst.pptx)" in root_readme

    deck_index = (ROOT / "decks" / "README.md").read_text(encoding="utf-8")
    normalized_deck_index = " ".join(deck_index.split())
    assert "[Plan-and-execute financial analyst](11-plan-and-execute-analyst.pptx)" in deck_index
    assert "Ready for instructor delivery" in deck_index
    assert "Planned, Task 7" not in deck_index
    assert "Lessons 08-11" in normalized_deck_index
    assert "Lesson 12 remains planned" in normalized_deck_index


def test_lesson11_deck_has_the_complete_sourced_concept_route() -> None:
    """Catch a missing, partial, unsourced, or visibly non-compliant deck."""

    assert DECK.is_file()
    with zipfile.ZipFile(DECK) as archive:
        names = archive.namelist()
        slide_names = sorted(
            name
            for name in names
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        notes_names = sorted(
            name
            for name in names
            if name.startswith("ppt/notesSlides/notesSlide") and name.endswith(".xml")
        )
        assert len(slide_names) == 9
        assert len(notes_names) == 9

        visible_text = "\n".join(
            "".join(
                node.text or ""
                for node in ElementTree.fromstring(archive.read(name)).iter(
                    "{http://schemas.openxmlformats.org/drawingml/2006/main}t"
                )
            )
            for name in slide_names
        )
        notes_text = "\n".join(
            "".join(
                node.text or ""
                for node in ElementTree.fromstring(archive.read(name)).iter(
                    "{http://schemas.openxmlformats.org/drawingml/2006/main}t"
                )
            )
            for name in notes_names
        )

    assert visible_text.count("First Finance - Arnaud Demes") == 9
    assert "—" not in visible_text
    for marker in (
        "Plan-and-Execute Financial Analyst",
        "REACT",
        "PLAN",
        "EXECUTE",
        "REPLAN",
        "REPORT",
        "HOST POLICY",
        "MCP DISCOVERY",
        "unsupported_metric",
        "EVIDENCE GATE",
        "LESSON 12",
    ):
        assert marker.casefold() in visible_text.casefold()
    assert notes_text.count("[Sources]") == 9
    assert notes_text.count("[/Sources]") == 9
    assert notes_text.count("chapters/11-plan-and-execute-analyst.md") == 9


def test_lesson11_deck_shares_central_host_state_across_four_roles() -> None:
    """Catch a linear fifth STATE role or loss of the four graph roles."""

    with zipfile.ZipFile(DECK) as archive:
        slide = ElementTree.fromstring(archive.read("ppt/slides/slide5.xml"))
        presentation = ElementTree.fromstring(archive.read("ppt/presentation.xml"))

    shapes = slide.findall(f".//{{{PRESENTATION_NS}}}sp")
    shape_texts = {_shape_text(shape): shape for shape in shapes if _shape_text(shape)}
    visible_text = " ".join(shape_texts).casefold()

    for role in ("PLANNER", "EXECUTOR", "REPLANNER", "REPORT WRITER"):
        assert role.casefold() in visible_text
    assert "STATE" not in shape_texts
    assert "host owned" not in visible_text

    host_state = shape_texts["HOST STATE"]
    transform = host_state.find(
        f"./{{{PRESENTATION_NS}}}spPr/{{{DRAWING_NS}}}xfrm"
    )
    assert transform is not None
    offset = transform.find(f"{{{DRAWING_NS}}}off")
    extent = transform.find(f"{{{DRAWING_NS}}}ext")
    assert offset is not None and extent is not None

    slide_size = presentation.find(f"{{{PRESENTATION_NS}}}sldSz")
    assert slide_size is not None
    slide_width = int(slide_size.attrib["cx"])
    host_state_center = int(offset.attrib["x"]) + int(extent.attrib["cx"]) / 2
    assert slide_width * 0.4 <= host_state_center <= slide_width * 0.6


def test_lesson11_evaluation_slides_cite_direct_primary_sources() -> None:
    """Catch generic inspiration links replacing slide-specific evaluation sources."""

    with zipfile.ZipFile(DECK) as archive:
        notes = {
            slide: "".join(
                node.text or ""
                for node in ElementTree.fromstring(
                    archive.read(f"ppt/notesSlides/notesSlide{slide}.xml")
                ).iter(f"{{{DRAWING_NS}}}t")
            )
            for slide in (8, 9)
        }

    assert "https://docs.langchain.com/langsmith/evaluation-concepts" in notes[8]
    assert "https://docs.langchain.com/langsmith/evaluate-complex-agent" in notes[9]
    assert "MLOps-Basics" not in notes[8]
    assert "MLOps-Basics" not in notes[9]
