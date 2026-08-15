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


def write_notebook(
    path: Path,
    *,
    body: str,
    with_output: bool = False,
    signature: str = "FinAI Academy — Arnaud Demes",
) -> None:
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
            f"# 01 — Test lesson\n\n**{signature}**\n\n"
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


def count_png_outputs(notebook) -> int:
    return sum(
        "image/png" in output.get("data", {})
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"}
    )


def stream_text(notebook) -> str:
    return "".join(
        output.get("text", "")
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )


def test_valid_notebook_passes_the_teaching_contract(tmp_path: Path) -> None:
    notebook_path = tmp_path / "01_valid.ipynb"
    write_notebook(notebook_path, body="Clear learner-facing explanation.")

    result = run_validator(notebook_path)

    assert result.returncode == 0
    assert "1 notebook passed" in result.stdout


def test_first_finance_signature_passes_the_teaching_contract(tmp_path: Path) -> None:
    notebook_path = tmp_path / "01_first_finance.ipynb"
    write_notebook(
        notebook_path,
        body="Clear learner-facing explanation.",
        signature="First Finance - Arnaud Demes",
    )

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


def test_model_gateway_offline_run_reaches_the_grounding_target(tmp_path: Path) -> None:
    notebook_path = ROOT / "notebooks" / "01_model_gateway.ipynb"
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

    assert result.returncode == 0, result.stderr
    executed = nbformat.read(output_dir / notebook_path.name, as_version=4)
    stream_text = "".join(
        output.get("text", "")
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    assert "Grounding score: 4/4" in stream_text
    assert "PASS — provider-neutral model gateway verified" in stream_text


def test_structured_outputs_offline_run_reaches_the_validation_target(
    tmp_path: Path,
) -> None:
    notebook_path = ROOT / "notebooks" / "02_prompts_and_structured_outputs.ipynb"
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

    assert result.returncode == 0, result.stderr
    executed = nbformat.read(output_dir / notebook_path.name, as_version=4)
    stream_text = "".join(
        output.get("text", "")
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    assert "Validation caught the unsupported candidate" in stream_text
    assert "PASS — structured financial brief verified" in stream_text


def test_cag_notebook_offline_run_produces_visual_evidence_and_a_decision(
    tmp_path: Path,
) -> None:
    notebook_path = ROOT / "notebooks" / "03_cag_financial_document.ipynb"
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

    assert result.returncode == 0, result.stderr
    executed = nbformat.read(output_dir / notebook_path.name, as_version=4)
    visual_outputs = [
        output
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"}
        and "image/png" in output.get("data", {})
    ]
    stream_text = "".join(
        output.get("text", "")
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )

    assert len(visual_outputs) >= 3
    assert "Decision: RAG required" in stream_text
    assert "PASS — CAG boundary verified" in stream_text


def test_naive_rag_notebook_offline_run_visualizes_and_verifies_baseline(
    tmp_path: Path,
) -> None:
    notebook_path = ROOT / "notebooks" / "04_rag_from_scratch.ipynb"
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

    assert result.returncode == 0, result.stderr
    executed = nbformat.read(output_dir / notebook_path.name, as_version=4)
    visual_outputs = [
        output
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"}
        and "image/png" in output.get("data", {})
    ]
    stream_text = "".join(
        output.get("text", "")
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )

    assert len(visual_outputs) >= 5
    assert "Retrieval check:" in stream_text
    assert "Grounding check:" in stream_text
    assert "PASS — naive RAG baseline verified" in stream_text


def test_document_chunking_notebook_offline_run_is_visual_and_verified(
    tmp_path: Path,
) -> None:
    notebook_path = ROOT / "notebooks" / "05_document_and_chunking_lab.ipynb"
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

    assert result.returncode == 0, result.stderr
    executed = nbformat.read(output_dir / notebook_path.name, as_version=4)
    visual_outputs = [
        output
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"}
        and "image/png" in output.get("data", {})
    ]
    stream_text = "".join(
        output.get("text", "")
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )

    assert len(visual_outputs) >= 8
    assert "Table integrity failure reproduced" in stream_text
    assert "Seven strategies compared" in stream_text
    assert "PASS — document and chunking laboratory verified" in stream_text


def test_hybrid_retrieval_notebook_offline_run_is_visual_and_verified(tmp_path):
    notebook_path = ROOT / "notebooks" / "06_hybrid_retrieval.ipynb"
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

    assert result.returncode == 0, result.stderr
    executed = nbformat.read(output_dir / notebook_path.name, as_version=4)
    assert count_png_outputs(executed) >= 8
    output_text = stream_text(executed)
    assert "Dense exact-term failure reproduced" in output_text
    assert "Cross-company leakage blocked" in output_text
    assert "Hybrid retrieval improves maintained recall" in output_text
    assert "PASS — hybrid retrieval laboratory verified" in output_text


def test_hybrid_retrieval_notebook_separates_live_and_controlled_outcomes():
    notebook = nbformat.read(ROOT / "notebooks" / "06_hybrid_retrieval.ipynb", as_version=4)
    code_by_id = {
        cell.id: cell.source for cell in notebook.cells if cell.cell_type == "code"
    }

    assert "controlled_dense_index" in code_by_id["lesson06-012"]
    assert "controlled_dense_index" in code_by_id["lesson06-014"]
    assert "observed_cosine_min" in code_by_id["lesson06-009"]
    assert (
        'if not live_mode:\n    assert maintained_recall["Reranked hybrid"] '
        '> maintained_recall["Dense"]'
        in code_by_id["lesson06-019"]
    )
    assert '"provider vectors are finite"' in code_by_id["lesson06-021"]
    assert "if not live_mode:\n    assert moved_rankings" in code_by_id["lesson06-023"]


def test_hybrid_retrieval_notebook_qualifies_offline_success_in_markdown():
    notebook = nbformat.read(ROOT / "notebooks" / "06_hybrid_retrieval.ipynb", as_version=4)
    learning_objectives = next(cell for cell in notebook.cells if cell.id == "lesson06-001")

    assert "deterministic offline laboratory success condition" in learning_objectives.source
    assert "Live OpenAI and Ollama runs report observed recall" in learning_objectives.source
    assert "provider-invariant structural behavior" in learning_objectives.source
