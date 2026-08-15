from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_notebooks.py"
EXECUTOR = ROOT / "scripts" / "execute_notebooks.py"

REQUIRED_HEADINGS = (
    "Learning objectives",
    "Where this fits",
    "Failure lab",
    "Verification",
    "Challenge",
    "Capstone integration",
    "Recap",
)


def write_notebook(path: Path, *, body: str, with_output: bool = False) -> None:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    notebook.metadata.finai = {"expected_runtime_minutes": 10}
    notebook.cells = [
        nbformat.v4.new_markdown_cell(
            "# 01 — Test lesson\n\n**FinAI Academy — Arnaud Demes**\n\n"
            + "\n\n".join(f"## {heading}\n\n{body}" for heading in REQUIRED_HEADINGS)
        ),
        nbformat.v4.new_code_cell("result = 2 + 2"),
    ]
    if with_output:
        notebook.cells[1].outputs = [
            nbformat.v4.new_output("execute_result", data={"text/plain": "4"}, execution_count=1)
        ]
        notebook.cells[1].execution_count = 1
    nbformat.write(notebook, path)


def run_validator(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_valid_notebook_passes_the_teaching_contract(tmp_path: Path) -> None:
    notebook_path = tmp_path / "01_valid.ipynb"
    write_notebook(notebook_path, body="Clear learner-facing explanation.")

    result = run_validator(notebook_path)

    assert result.returncode == 0
    assert "1 notebook passed" in result.stdout


def test_notebook_with_outputs_and_local_paths_is_rejected(tmp_path: Path) -> None:
    notebook_path = tmp_path / "01_invalid.ipynb"
    write_notebook(
        notebook_path,
        body="Open /Users/example/private-file.pdf before continuing.",
        with_output=True,
    )

    result = run_validator(notebook_path)

    assert result.returncode == 1
    assert "contains stored outputs" in result.stdout
    assert "contains an absolute user path" in result.stdout


def test_offline_executor_runs_a_notebook_and_saves_the_evidence(tmp_path: Path) -> None:
    notebook_path = tmp_path / "01_execute.ipynb"
    write_notebook(notebook_path, body="Execution contract.")
    notebook = nbformat.read(notebook_path, as_version=4)
    notebook.cells[1].source = 'print("execution complete")'
    nbformat.write(notebook, notebook_path)
    output_dir = tmp_path / "executed"

    result = subprocess.run(
        [
            sys.executable,
            str(EXECUTOR),
            str(notebook_path),
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

    assert result.returncode == 0
    executed = nbformat.read(output_dir / notebook_path.name, as_version=4)
    assert executed.cells[1].outputs[0]["text"] == "execution complete\n"
