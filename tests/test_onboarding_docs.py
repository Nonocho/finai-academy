from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DAY_ONE_NOTEBOOKS = [
    "notebooks/01_model_gateway.ipynb",
    "notebooks/02_prompts_and_structured_outputs.ipynb",
    "notebooks/03_cag_financial_document.ipynb",
    "notebooks/04_rag_from_scratch.ipynb",
    "notebooks/05_document_and_chunking_lab.ipynb",
    "notebooks/06_hybrid_retrieval.ipynb",
    "notebooks/07_rag_evaluation.ipynb",
]


def test_readme_exposes_one_four_command_quick_start() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for command in (
        "uv sync --extra ai --extra rag --extra evaluation --extra dev",
        "uv run python scripts/setup_check.py --offline",
        "uv run python scripts/setup_check.py --provider ollama",
        "uv run jupyter lab",
    ):
        assert command in text


def test_day_one_guide_lists_canonical_notebooks_in_order() -> None:
    text = (ROOT / "docs" / "day-1-student-guide.md").read_text(encoding="utf-8")
    positions = [text.index(path) for path in DAY_ONE_NOTEBOOKS]

    assert positions == sorted(positions)


def test_env_is_ignored_and_example_contains_no_key_value() -> None:
    ignore_lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert ".env" in ignore_lines
    assert "OPENAI_API_KEY=" in example
    assert not re.search(r"OPENAI_API_KEY=\S+", example)


def test_local_markdown_links_resolve() -> None:
    documents = (
        ROOT / "README.md",
        ROOT / "docs" / "getting-started.md",
        ROOT / "docs" / "day-1-student-guide.md",
        ROOT / "docs" / "troubleshooting.md",
        ROOT / "notebooks" / "README.md",
        ROOT / "chapters" / "README.md",
        ROOT / "decks" / "README.md",
    )
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    missing: list[str] = []

    for document in documents:
        for raw_target in link_pattern.findall(document.read_text(encoding="utf-8")):
            target = raw_target.split("#", maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (document.parent / target).resolve().exists():
                missing.append(f"{document.relative_to(ROOT)} -> {raw_target}")

    assert missing == []
