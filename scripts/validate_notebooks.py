"""Validate learner-facing notebook structure and repository hygiene."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import nbformat

REQUIRED_HEADINGS = (
    "Learning objectives",
    "Where this fits",
    "Failure lab",
    "Verification",
    "Challenge",
    "Capstone integration",
    "Recap",
)

ABSOLUTE_USER_PATH = re.compile(r"(?:/Users/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)")
SECRET_PATTERN = re.compile(r"(?:sk-[A-Za-z0-9_-]{16,}|tvly-[A-Za-z0-9_-]{16,})")
COURSE_SIGNATURES = ("First Finance - Arnaud Demes", "FinAI Academy")


def validate_notebook(path: Path) -> list[str]:
    """Return human-readable contract violations for one notebook."""

    notebook = nbformat.read(path, as_version=4)
    errors: list[str] = []
    source = "\n".join(cell.source for cell in notebook.cells)

    if not any(signature in source for signature in COURSE_SIGNATURES):
        errors.append("missing an approved course signature")

    for heading in REQUIRED_HEADINGS:
        if f"## {heading}" not in source:
            errors.append(f"missing required heading: {heading}")

    if not notebook.metadata.get("kernelspec"):
        errors.append("missing kernelspec metadata")

    if not notebook.metadata.get("finai", {}).get("expected_runtime_minutes"):
        errors.append("missing finai.expected_runtime_minutes metadata")

    cell_ids = [cell.get("id") for cell in notebook.cells]
    if any(not cell_id for cell_id in cell_ids) or len(cell_ids) != len(set(cell_ids)):
        errors.append("cell IDs must be present and unique")

    if any(cell.cell_type == "code" and cell.get("outputs") for cell in notebook.cells):
        errors.append("contains stored outputs")

    if any(cell.cell_type == "code" and cell.get("execution_count") is not None for cell in notebook.cells):
        errors.append("contains stored execution counts")

    if ABSOLUTE_USER_PATH.search(source):
        errors.append("contains an absolute user path")

    if SECRET_PATTERN.search(source):
        errors.append("contains a value that looks like an API key")

    return errors


def collect_paths(raw_paths: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.ipynb")))
        else:
            paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Notebook files or directories to validate")
    args = parser.parse_args()

    paths = collect_paths(args.paths)
    failures: list[str] = []
    for path in paths:
        for error in validate_notebook(path):
            failures.append(f"{path}: {error}")

    if failures:
        print("\n".join(f"- {failure}" for failure in failures))
        raise SystemExit(1)

    noun = "notebook" if len(paths) == 1 else "notebooks"
    print(f"{len(paths)} {noun} passed the course notebook contract.")


if __name__ == "__main__":
    main()
