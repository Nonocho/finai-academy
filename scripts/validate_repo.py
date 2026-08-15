"""Validate the implemented prefix of the canonical course manifest."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COURSE_MANIFEST = ROOT / "course.yml"
COURSE_SIGNATURES = ("First Finance - Arnaud Demes", "FinAI Academy")


def load_notebook_lessons() -> list[dict[str, str]]:
    manifest = yaml.safe_load(COURSE_MANIFEST.read_text(encoding="utf-8"))
    return [lesson for lesson in manifest["lessons"] if lesson.get("notebook")]


def validate_notebooks() -> list[str]:
    errors: list[str] = []
    encountered_missing_lesson = False

    for lesson in load_notebook_lessons():
        notebook_path = ROOT / lesson["notebook"]
        if not notebook_path.exists():
            encountered_missing_lesson = True
            continue
        if encountered_missing_lesson:
            errors.append(
                f"Lesson {lesson['id']} is implemented after a missing canonical lesson"
            )
            continue

        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        if notebook.get("nbformat") != 4:
            errors.append(f"{notebook_path.name}: expected nbformat 4")

        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        if not any(signature in source for signature in COURSE_SIGNATURES):
            errors.append(f"{notebook_path.name}: missing an approved course signature")

    return errors


def validate_chapters() -> list[str]:
    errors: list[str] = []
    for lesson in load_notebook_lessons():
        notebook_path = ROOT / lesson["notebook"]
        if not notebook_path.exists():
            continue

        for artifact_key in ("chapter", "deck"):
            artifact_path = ROOT / lesson[artifact_key]
            if not artifact_path.exists():
                errors.append(
                    f"Lesson {lesson['id']} is missing {artifact_key}: "
                    f"{lesson[artifact_key]}"
                )
    return errors


def main() -> None:
    errors = validate_notebooks() + validate_chapters()
    if errors:
        raise SystemExit("\n".join(f"- {error}" for error in errors))
    print("FinAI Academy repository structure is valid.")


if __name__ == "__main__":
    main()
