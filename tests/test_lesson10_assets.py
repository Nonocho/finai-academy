"""Contract tests for the Lesson 10 financial MCP notebook."""

from __future__ import annotations

import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "10_financial_mcp.ipynb"
EXECUTOR = ROOT / "scripts" / "execute_notebooks.py"
BUILDER = ROOT / "scripts" / "build_lesson10_notebook.py"
CHAPTER = ROOT / "chapters" / "10-financial-mcp.md"


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
    assert "Task 6" in chapter
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
    assert "[Financial MCP](10-financial-mcp.md)" in chapter_index
    assert "[Financial MCP](10_financial_mcp.ipynb)" in notebook_index
    assert "full 45-minute delivery route is pending Task 6 deck" in chapter_index
    assert "full 45-minute delivery route is pending Task 6 deck" in notebook_index
    assert "ready for instructor delivery" not in chapter_index
    assert "ready for instructor delivery" not in notebook_index
    assert "[Build a financial MCP](10-financial-mcp.pptx)" in (
        ROOT / "decks" / "README.md"
    ).read_text(encoding="utf-8")
