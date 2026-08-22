"""Contract tests for the Lesson 11 plan-and-execute notebook."""

from __future__ import annotations

import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "11_plan_and_execute_analyst.ipynb"
BUILDER = ROOT / "scripts" / "build_lesson11_notebook.py"
EXECUTOR = ROOT / "scripts" / "execute_notebooks.py"
CHAPTER = ROOT / "chapters" / "11-plan-and-execute-analyst.md"


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
        "deck remains planned",
    ):
        assert marker.lower() in normalized_chapter
    assert "—" not in chapter
    assert "](../decks/11-plan-and-execute-analyst.pptx)" not in chapter


def test_lesson11_chapter_and_notebook_are_discoverable_while_deck_is_planned() -> None:
    """Catch an index that hides usable materials or falsely activates the deck."""

    indexes = {
        "chapters/README.md": "[Plan-and-execute financial analyst](11-plan-and-execute-analyst.md)",
        "notebooks/README.md": "[Plan-and-execute financial analyst](11_plan_and_execute_analyst.ipynb)",
        "README.md": "[Lesson 11 instructor chapter](chapters/11-plan-and-execute-analyst.md)",
    }
    for relative_path, expected_link in indexes.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert expected_link in text
        assert "Lessons 11–12 remain planned" not in text

    deck_index = (ROOT / "decks" / "README.md").read_text(encoding="utf-8")
    assert "`11-plan-and-execute-analyst.pptx`" in deck_index
    assert "Planned, Task 7" in deck_index
    assert "](11-plan-and-execute-analyst.pptx)" not in deck_index
