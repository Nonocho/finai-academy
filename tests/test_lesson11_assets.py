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


def _cell_output_text(notebook, cell_id: str) -> str:
    cell = next(item for item in notebook.cells if item.id == cell_id)
    rendered: list[str] = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            rendered.append(output.get("text", ""))
            continue
        data = output.get("data", {})
        for mime_type in ("text/plain", "text/markdown", "text/html"):
            value = data.get(mime_type)
            if isinstance(value, str):
                rendered.append(value)
    return "\n".join(rendered)


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
    assert 16 <= len(notebook.cells) <= 20
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
    briefing_cell = next(cell for cell in notebook.cells if "CITED FACTS" in cell.source)
    for marker in (
        "fact.claim",
        "fact.provenance_kind",
        "fact.source_references",
        "fact.evidence_ids",
        "cross_company_observations",
        "interpretation",
        "limitations",
        "source_references",
    ):
        assert marker in briefing_cell.source
    assert "len(result.briefing" not in briefing_cell.source


def test_lesson11_notebook_approves_the_plan_before_executing_the_mission() -> None:
    """Catch a regression to running the complete mission before learners see its plan."""

    notebook = _build_notebook()
    sources = [cell.source for cell in notebook.cells]
    plan_index = next(
        index for index, source in enumerate(sources) if "Plan approved before execution" in source
    )
    execution_index = next(
        index for index, source in enumerate(sources) if "result = await run_plan_execute" in source
    )

    assert plan_index < execution_index
    assert "result = await run_lesson11" not in "\n".join(sources)
    assert sum("plt.subplots" in source for source in sources) <= 3


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
    briefing_text = _cell_output_text(executed, "lesson11-013")
    assert _png_output_count(executed) >= 3
    assert "Real MCP server:" in text
    assert "offline fixture · deterministic planner and replanner · real local MCP execution" in text
    assert "Plan revisions: 1" in text
    assert "Evidence gate passed: True" in text
    assert "Plan approved before execution: True" in text
    assert "Learner decision: replan the unfinished tail" in text
    for marker in (
        "CITED FACTS",
        "Kind: metric",
        "Kind: document",
        "NVIDIA P/E was 52.4 x as of 2026-08-20.",
        "Sources: First Finance controlled classroom fixture",
        "Evidence IDs: NVDA-FY2026-DATA-CENTER-001",
        "CROSS-COMPANY OBSERVATIONS",
        "INTERPRETATION",
        "LIMITATIONS",
        "Aggregate sources:",
        "assets/course-data/fixtures/nvidia_fy2026_excerpt.html",
    ):
        assert marker in briefing_text
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
        "lesson11-017",
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
        assert (
            "Lessons 08-12 are ready for an instructor-led offline test class"
            in normalized_text
        )
        assert "Lesson 12 remains planned" not in normalized_text

    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[Lesson 11 concept deck](decks/11-plan-and-execute-analyst.pptx)" in root_readme

    deck_index = (ROOT / "decks" / "README.md").read_text(encoding="utf-8")
    normalized_deck_index = " ".join(deck_index.split())
    assert "[Plan-and-execute financial analyst](11-plan-and-execute-analyst.pptx)" in deck_index
    assert "Ready for instructor delivery" in deck_index
    assert "Planned, Task 7" not in deck_index
    assert (
        "Lessons 08-12 are ready for an instructor-led offline test class"
        in normalized_deck_index
    )
    assert "Lesson 12 remains planned" not in normalized_deck_index


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
        assert len(slide_names) == 11
        assert len(notes_names) == 11

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

    assert visible_text.count("First Finance - Arnaud Demes") == 11
    assert "—" not in visible_text
    for marker in (
        "Plan-and-Execute Financial Analyst",
        "PLAN",
        "REPLAN",
        "APPROVED PLAN",
        "HOST POLICY",
        "MCP DISCOVERY",
        "TYPED FAILURE",
        "unsupported_metric",
        "EVIDENCE GATE",
        "LESSON 12",
    ):
        assert marker.casefold() in visible_text.casefold()
    assert notes_text.count("[Sources]") == 11
    assert notes_text.count("[/Sources]") == 11
    assert notes_text.count("chapters/11-plan-and-execute-analyst.md") == 11

    for marker in (
        "One mission, several evidence gaps",
        "The plan is visible before any tool runs",
        "One failed metric changes the remaining strategy",
        "Every claim must point back to evidence",
    ):
        assert marker in visible_text

    assert "https://www.anthropic.com/engineering/building-effective-agents" in notes_text
    assert "https://openai.github.io/openai-agents-python/visualization/" in notes_text


def test_lesson11_deck_contains_real_visual_evidence() -> None:
    """Catch a return to a shapes-only deck with no sourced or run-derived imagery."""

    with zipfile.ZipFile(DECK) as archive:
        media = [
            name
            for name in archive.namelist()
            if name.startswith("ppt/media/")
            and name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ]

    assert len(media) >= 4


def test_lesson11_deck_makes_mcp_discovery_and_host_filters_visible() -> None:
    """Catch loss of the concrete catalog-to-permission teaching backbone."""

    with zipfile.ZipFile(DECK) as archive:
        slide = ElementTree.fromstring(archive.read("ppt/slides/slide5.xml"))

    shapes = slide.findall(f".//{{{PRESENTATION_NS}}}sp")
    visible_text = " ".join(_shape_text(shape) for shape in shapes).casefold()

    for marker in (
        "discovered",
        "allowlisted",
        "arguments validated",
        "get_company_metric",
        "search_financial_documents",
        "host policy turns them into permission",
    ):
        assert marker.casefold() in visible_text


def test_lesson11_evidence_slides_cite_the_run_and_primary_report() -> None:
    """Catch generic inspiration links replacing run-derived and primary evidence."""

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

    assert "notebooks/11_plan_and_execute_analyst.ipynb" in notes[8]
    assert "assets/lesson-11/evidence-gate.png" in notes[8]
    assert "assets/lesson-05/schneider-fy2025-results.pdf" in notes[9]
    assert "assets/course-data/mcp/lesson10_evidence_catalog_v1.json" in notes[9]
