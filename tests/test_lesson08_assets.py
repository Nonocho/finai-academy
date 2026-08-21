from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "08_workflows_vs_agents.ipynb"
CHAPTER = ROOT / "chapters" / "08-workflows-vs-agents.md"
DECK = ROOT / "decks" / "08-workflows-vs-agents.pptx"


def _pptx_part_texts(prefix: str) -> list[str]:
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
            " ".join(node.text or "" for node in ElementTree.fromstring(archive.read(part)).iter() if node.tag.endswith("}t"))
            for part in parts
        ]


def test_lesson08_notebook_is_output_free_and_contains_the_teaching_contract() -> None:
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
        "unsupported_dependency",
        "MAX_STEPS",
        "get_market_price",
        "convert_currency",
        "LESSON_08_PASS",
    ):
        assert marker in source


def test_lesson08_notebook_declares_the_provider_and_runtime_contract() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert notebook.metadata["finai"]["expected_runtime_minutes"] == 30
    assert "FINAI_LIVE_MODE" in source
    assert "create_chat_model" in source
    assert "with_structured_output" in source
    assert "offline fixture" in source
    assert "Ollama" in source
    assert "OpenAI" in source


def test_lesson08_chapter_and_asset_indexes_are_discoverable() -> None:
    assert CHAPTER.is_file()
    chapter = CHAPTER.read_text(encoding="utf-8")
    notebook_index = (ROOT / "notebooks/README.md").read_text(encoding="utf-8")
    chapter_index = (ROOT / "chapters/README.md").read_text(encoding="utf-8")
    deck_index = (ROOT / "decks/README.md").read_text(encoding="utf-8")

    for marker in (
        "09:30–10:15",
        "10-minute concept deck",
        "30-minute notebook",
        "MAX_STEPS",
        "unsupported_dependency",
        "Ollama",
        "OpenAI",
        "No-network fallback",
        "Lesson 09",
    ):
        assert marker in chapter
    assert "08_workflows_vs_agents.ipynb" in notebook_index
    assert "08-workflows-vs-agents.md" in chapter_index
    assert "08-workflows-vs-agents.pptx" in deck_index


def test_lesson08_deck_teaches_the_required_decision_and_is_fully_sourced() -> None:
    slide_texts = _pptx_part_texts("ppt/slides/slide")
    note_texts = _pptx_part_texts("ppt/notesSlides/notesSlide")
    joined = " ".join(slide_texts)

    assert len(slide_texts) == 9
    assert len(note_texts) == 9
    assert all("First Finance - Arnaud Demes" in text for text in slide_texts)
    assert all("[Sources]" in text and "[/Sources]" in text for text in note_texts)
    assert "—" not in joined
    for jargon in (
        "failure surface",
        "task prestige",
        "orchestration policy",
        "trajectory",
        "grounded response",
    ):
        assert jargon not in joined.casefold()
    for marker in (
        "THREE SYSTEMS AT A GLANCE",
        "Who chooses the next step?",
        "Python code",
        "One LLM",
        "Several LLMs",
        "WORKFLOW OR AGENT?",
        "Is the path known?",
        "Coded by you",
        "Chosen after each result",
        "TYPED TOOL BOUNDARY",
        "get_market_price",
        "convert_currency",
        "unsupported_dependency",
        "MAX_STEPS",
        "If you can draw every step before the run, use a workflow.",
        "If the next step depends on a tool result, use an agent.",
        "LESSON 09",
    ):
        assert marker in joined
