from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "08_workflows_vs_agents.ipynb"


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
