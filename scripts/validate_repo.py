"""Validate the public course skeleton without third-party dependencies."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHAPTER_IDS = [f"{index:02d}" for index in range(11)]
SIGNATURE = "FinAI Academy"


def validate_notebooks() -> list[str]:
    errors: list[str] = []
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))

    if len(notebooks) != len(CHAPTER_IDS):
        errors.append(f"Expected {len(CHAPTER_IDS)} notebooks, found {len(notebooks)}")

    for chapter_id in CHAPTER_IDS:
        matches = list((ROOT / "notebooks").glob(f"{chapter_id}-*.ipynb"))
        if len(matches) != 1:
            errors.append(f"Chapter {chapter_id} must have exactly one notebook")
            continue

        notebook = json.loads(matches[0].read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4:
            errors.append(f"{matches[0].name}: expected nbformat 4")

        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        if SIGNATURE not in source:
            errors.append(f"{matches[0].name}: missing FinAI Academy signature")

    return errors


def validate_chapters() -> list[str]:
    errors: list[str] = []
    for chapter_id in CHAPTER_IDS:
        if len(list((ROOT / "chapters").glob(f"{chapter_id}-*.md"))) != 1:
            errors.append(f"Chapter {chapter_id} must have exactly one brief")
    return errors


def main() -> None:
    errors = validate_notebooks() + validate_chapters()
    if errors:
        raise SystemExit("\n".join(f"- {error}" for error in errors))
    print("FinAI Academy repository structure is valid.")


if __name__ == "__main__":
    main()
