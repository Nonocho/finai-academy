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
LEGACY_PATHS = (
    "notebooks/00-product-demo-and-system-map.ipynb",
    "notebooks/01-ai-and-llm-foundations.ipynb",
    "notebooks/02-prompting-and-structured-outputs.ipynb",
    "notebooks/03-retrieval-from-first-principles.ipynb",
    "notebooks/04-document-ingestion-and-chunking.ipynb",
    "notebooks/05-embeddings-and-advanced-retrieval.ipynb",
    "notebooks/06-rag-with-evidence.ipynb",
    "notebooks/07-tools-and-deterministic-workflows.ipynb",
    "notebooks/08-langgraph-agents-and-self-correction.ipynb",
    "notebooks/09-multi-agent-financial-research.ipynb",
    "notebooks/10-evaluation-observability-and-llmops.ipynb",
    "chapters/00-product-demo-and-system-map.md",
    "chapters/01-ai-and-llm-foundations.md",
    "chapters/02-prompting-and-structured-outputs.md",
    "chapters/03-retrieval-from-first-principles.md",
    "chapters/04-document-ingestion-and-chunking.md",
    "chapters/05-embeddings-and-advanced-retrieval.md",
    "chapters/06-rag-with-evidence.md",
    "chapters/07-tools-and-deterministic-workflows.md",
    "chapters/08-langgraph-agents-and-self-correction.md",
    "chapters/09-multi-agent-financial-research.md",
    "chapters/10-evaluation-observability-and-llmops.md",
)


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


def test_legacy_seed_assets_are_absent_from_student_paths() -> None:
    assert [path for path in LEGACY_PATHS if (ROOT / path).exists()] == []
