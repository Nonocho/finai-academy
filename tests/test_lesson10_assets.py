"""Contract tests for the Lesson 10 financial MCP notebook."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from xml.etree import ElementTree

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "10_financial_mcp.ipynb"
EXECUTOR = ROOT / "scripts" / "execute_notebooks.py"
BUILDER = ROOT / "scripts" / "build_lesson10_notebook.py"
CHAPTER = ROOT / "chapters" / "10-financial-mcp.md"
DECK = ROOT / "decks" / "10-financial-mcp.pptx"


def _build_notebook():
    spec = spec_from_file_location("lesson10_notebook_builder", BUILDER)
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


def test_lesson10_notebook_is_output_free_and_contains_the_teaching_contract() -> None:
    assert NOTEBOOK.is_file()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert notebook.metadata["finai"]["expected_runtime_minutes"] == 30
    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
    assert all(
        cell.get("execution_count") is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    assert 22 <= len(notebook.cells) <= 26
    assert nbformat.writes(_build_notebook()) == nbformat.writes(notebook)
    for heading in (
        "## Learning objectives",
        "## Where this fits",
        "## Failure lab",
        "## Verification",
        "## Challenge",
        "## Capstone integration",
        "## Recap",
    ):
        assert heading in source
    for marker in (
        "MCPServer",
        "finance://coverage",
        "get_company_metric",
        "search_financial_documents",
        "compare_companies",
        "stdio",
        "FINAI_LIVE_MODE",
        "create_chat_model",
        "Ollama",
        "OpenAI",
        "LESSON_10_PASS",
        "DiscoveredToolSpec",
        "FinancialCapabilityRegistry",
        "registry_source",
        "server_source",
        "Knowledge check",
    ):
        assert marker in source
    assert "tool_catalog = [" not in source
    assert "mcp_run.tool_specs" in source


def test_lesson10_notebook_executes_offline_with_visual_evidence(tmp_path: Path) -> None:
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
    assert _png_output_count(executed) >= 5
    assert "LESSON_10_PASS" in _stream_text(executed)


def test_lesson10_chapter_and_discoverable_indexes_expose_the_classroom_contract() -> None:
    assert CHAPTER.is_file()
    chapter = CHAPTER.read_text(encoding="utf-8")

    for marker in (
        "11:15–12:00",
        "10-minute concept deck",
        "30-minute notebook",
        "5-minute verification and debrief",
        "finance://coverage",
        "get_company_metric",
        "search_financial_documents",
        "compare_companies",
        "MCPServer",
        "FastMCP",
        "stdio",
        "Streamable HTTP",
        "Ollama",
        "OpenAI",
        "No-network fallback",
        "Skip if late",
        "Lesson 11",
        "LESSON_10_PASS",
        "lesson10-003",
        "lesson10-010",
        "lesson10-016",
        "lesson10-018",
        "lesson10-022",
    ):
        assert marker in chapter

    assert "notebooks/10_financial_mcp.ipynb" in chapter
    assert "decks/10-financial-mcp.pptx" in chapter
    assert "full Lesson 10 route is ready for an instructor-led test class" in chapter
    assert "pending Task 6" not in chapter
    assert "Static recovery catalog" in chapter
    for capability in (
        "| Resource | Application | `finance://coverage` |",
        "| Tool | Model + host approval | `get_company_metric` |",
        "| Tool | Model + host approval | `search_financial_documents` |",
        "| Prompt | User | `compare_companies` |",
    ):
        assert capability in chapter

    chapter_index = (ROOT / "chapters" / "README.md").read_text(encoding="utf-8")
    notebook_index = (ROOT / "notebooks" / "README.md").read_text(encoding="utf-8")
    deck_index = (ROOT / "decks" / "README.md").read_text(encoding="utf-8")
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[Financial MCP](10-financial-mcp.md)" in chapter_index
    assert "[Financial MCP](10_financial_mcp.ipynb)" in notebook_index
    for index in (chapter_index, notebook_index, deck_index, root_readme):
        assert "full Lesson 10 route is ready for an instructor-led test class" in index
        assert "pending Task 6" not in index
        assert "when available" not in index
    assert "[Build a financial MCP](10-financial-mcp.pptx)" in deck_index


def test_lesson10_deck_has_the_complete_sourced_concept_route() -> None:
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
        "Build a Financial MCP",
        "Direct import",
        "HOST",
        "CLIENT",
        "SERVER",
        "RESOURCES",
        "TOOLS",
        "PROMPTS",
        "finance://coverage",
        "get_company_metric",
        "search_financial_documents",
        "compare_companies",
        "DISCOVERY",
        "stdio",
        "DISCOVERY IS NOT PERMISSION",
        "LESSON 11",
    ):
        assert marker.casefold() in visible_text.casefold()
    assert notes_text.count("[Sources]") == 9
    assert notes_text.count("[/Sources]") == 9
    assert "chapters/10-financial-mcp.md" in notes_text
    assert "https://modelcontextprotocol.io" in notes_text
