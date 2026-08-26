from __future__ import annotations

import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "09_self_correcting_agent.ipynb"
CHAPTER = ROOT / "chapters" / "09-self-correcting-agent.md"
DECK = ROOT / "decks" / "09-self-correcting-agent.pptx"
FIXTURE = ROOT / "assets/course-data/market/lesson09_metrics_snapshot_v1.json"
EXECUTOR = ROOT / "scripts" / "execute_notebooks.py"


def _pptx_part_texts(prefix: str) -> list[str]:
    assert DECK.is_file()
    with zipfile.ZipFile(DECK) as archive:
        parts = sorted(
            (
                name
                for name in archive.namelist()
                if name.startswith(prefix) and name.endswith(".xml")
            ),
            key=lambda name: int(re.search(r"(\d+)\.xml$", name).group(1)),
        )
        return [
            " ".join(
                node.text or ""
                for node in ElementTree.fromstring(archive.read(part)).iter()
                if node.tag.endswith("}t")
            )
            for part in parts
        ]


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


def test_lesson09_fixture_is_explicitly_controlled_and_provenance_rich() -> None:
    assert FIXTURE.is_file()
    snapshot = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert snapshot["dataset_id"] == "lesson09-metrics-snapshot-v1"
    assert "not live" in snapshot["notice"].casefold()
    assert snapshot["as_of"]
    assert snapshot["source"]
    assert set(snapshot["metrics"]) == {"NVDA", "SU.PA"}
    assert all("P/E" in record and "EPS" in record for record in snapshot["metrics"].values())


def test_lesson09_notebook_is_output_free_and_contains_the_teaching_contract() -> None:
    assert NOTEBOOK.is_file()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
    assert all(
        cell.get("execution_count") is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    assert notebook.metadata["finai"]["expected_runtime_minutes"] == 30
    assert len(notebook.cells) <= 17
    assert sum(cell.cell_type == "code" for cell in notebook.cells) <= 8
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
        "StateGraph",
        "unsupported_metric",
        "MAX_RETRIES",
        "MAX_TOOL_CALLS",
        "FINAI_LIVE_MODE",
        "create_chat_model",
        "with_structured_output",
        "Ollama",
        "OpenAI",
        "LESSON_09_PASS",
        "ModelAgentAction",
        "model-correctable",
        "transient",
    ):
        assert marker in source


def test_lesson09_notebook_executes_offline_with_visual_evidence(tmp_path: Path) -> None:
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
    assert _png_output_count(executed) >= 4
    stream = _stream_text(executed)
    assert "success_path=completed" in stream
    assert "failure_path=retry_budget_exhausted" in stream
    assert "LESSON_09_PASS" in stream


def test_lesson09_chapter_and_indexes_are_discoverable() -> None:
    assert CHAPTER.is_file()
    chapter = CHAPTER.read_text(encoding="utf-8")
    notebook_index = (ROOT / "notebooks/README.md").read_text(encoding="utf-8")
    chapter_index = (ROOT / "chapters/README.md").read_text(encoding="utf-8")
    deck_index = (ROOT / "decks/README.md").read_text(encoding="utf-8")

    for marker in (
        "10:30–11:15",
        "10-minute concept deck",
        "30-minute notebook",
        "MAX_RETRIES",
        "MAX_TOOL_CALLS",
        "unsupported_metric",
        "Ollama",
        "OpenAI",
        "No-network fallback",
        "Lesson 10",
    ):
        assert marker in chapter
    assert "09_self_correcting_agent.ipynb" in notebook_index
    assert "09-self-correcting-agent.md" in chapter_index
    assert "09-self-correcting-agent.pptx" in deck_index


def test_lesson09_deck_is_visual_simple_and_fully_sourced() -> None:
    slide_texts = _pptx_part_texts("ppt/slides/slide")
    note_texts = _pptx_part_texts("ppt/notesSlides/notesSlide")
    joined = " ".join(slide_texts)

    assert len(slide_texts) == 11
    assert len(note_texts) == 11
    assert all("First Finance - Arnaud Demes" in text for text in slide_texts)
    assert all("[Sources]" in text and "[/Sources]" in text for text in note_texts)
    assert "—" not in joined
    for marker in (
        "A TOOL ERROR CAN BECOME THE NEXT INPUT",
        "Errors need different recovery strategies",
        "External feedback changes the next action",
        "MODEL-CORRECTABLE",
        "TRANSIENT",
        "USER-FIXABLE",
        "UNEXPECTED",
        "LANGGRAPH MAKES THE RECOVERY ROUTE EXPLICIT",
        "agent",
        "tools",
        "unsupported_metric",
        "Valid metrics: EPS, P/E",
        "PE",
        "P/E",
        "MAX_RETRIES = 1",
        "MAX_TOOL_CALLS = 4",
        "Self-correction does not guarantee truth",
        "LESSON 10",
        "MCP",
    ):
        assert marker.casefold() in joined.casefold()
    assert "Self-correction needs feedback, state and limits" not in joined
    notes_joined = " ".join(note_texts)
    assert "docs.langchain.com/oss/python/langgraph/thinking-in-langgraph" in notes_joined
    assert "arxiv.org/abs/2310.01798" in notes_joined
