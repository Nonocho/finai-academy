"""Contract tests for the Lesson 12 MLflow agent-evaluation notebook."""

from __future__ import annotations

import re
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "12_evaluating_agentic_systems.ipynb"
BUILDER = ROOT / "scripts" / "build_lesson12_notebook.py"
EXECUTOR = ROOT / "scripts" / "execute_notebooks.py"
DATASET_SHA256 = "c8f81fc59b182df8b2044c70d759fcb1fdac1fa90faead4bb70812b409ba0131"
METRIC_NAMES = (
    "tool_call_correctness",
    "tool_call_efficiency",
    "answer_relevance",
    "answer_completeness",
    "citation_integrity",
)


def _build_notebook():
    spec = spec_from_file_location("lesson12_notebook_builder", BUILDER)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_notebook()


def executable_source(notebook) -> str:
    return "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )


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


def _cell_output_text(notebook, cell_id: str) -> str:
    cell = next(item for item in notebook.cells if item.id == cell_id)
    rendered: list[str] = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            rendered.append(output.get("text", ""))
            continue
        data = output.get("data", {})
        for mime_type in ("text/plain", "text/markdown", "text/html"):
            value = data.get(mime_type)
            if isinstance(value, str):
                rendered.append(value)
    return "\n".join(rendered)


def test_lesson12_notebook_is_output_free_stable_and_contains_the_teaching_contract() -> None:
    assert BUILDER.is_file()
    assert NOTEBOOK.is_file()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert notebook.metadata["finai"]["expected_runtime_minutes"] == 40
    assert len(notebook.cells) == 27
    assert [cell.id for cell in notebook.cells] == [
        f"lesson12-{index:03d}" for index in range(27)
    ]
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    assert all(
        not cell.get("outputs")
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    assert all(
        cell.get("execution_count") is None
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    assert nbformat.writes(_build_notebook()) == nbformat.writes(notebook)
    assert source.count("LESSON_12_PASS") == 1
    assert "—" not in source
    for marker in (
        "bounded-agent-v1",
        "regressed-agent-v0",
        "agent-cases-v1",
        *METRIC_NAMES,
        "FINAI_EVAL_JUDGE_MODEL",
        "openai:/",
        "ollama_chat:/",
        "mlflow ui --backend-store-uri sqlite:///",
        "FinancialMcpPlanningExecutor",
        "reference_completed",
        "NVIDIA",
        "Schneider Electric",
        "## Failure lab",
        "## Verification",
        "## Knowledge check",
        "## Challenge",
        "## Capstone integration",
        "## Recap",
    ):
        assert marker in source
    assert "docker" not in executable_source(notebook).casefold()


def test_lesson12_notebook_executes_offline_with_persisted_visual_evidence(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "executed"
    mlflow_dir = tmp_path / "mlflow"
    command = [
        sys.executable,
        str(EXECUTOR),
        str(NOTEBOOK),
        "--mode",
        "offline",
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env={**__import__("os").environ, "FINAI_MLFLOW_DIR": str(mlflow_dir)},
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    executed = nbformat.read(output_dir / NOTEBOOK.name, as_version=4)
    stream_text = _stream_text(executed)
    all_output_text = "\n".join(
        _cell_output_text(executed, cell.id) for cell in executed.cells
    )

    assert _png_output_count(executed) >= 6
    assert "Reference public signature: MATCH" in stream_text
    assert "Dataset: agent-cases-v1" in stream_text
    assert f"Dataset SHA-256: {DATASET_SHA256}" in stream_text
    assert "Configurations: bounded-agent-v1, regressed-agent-v0" in stream_text
    assert "Cases per configuration: 6" in stream_text
    assert "Total traces: 12" in stream_text
    run_ids = re.findall(r"Run ID \((?:bounded-agent-v1|regressed-agent-v0)\): ([0-9a-f]+)", stream_text)
    assert len(run_ids) == 2
    assert len(set(run_ids)) == 2
    trace_ids = re.findall(r"Trace ID \([^\n]+\): (tr-[0-9a-f]+)", stream_text)
    assert len(trace_ids) == 12
    assert len(set(trace_ids)) == 12
    for metric_name in METRIC_NAMES:
        assert metric_name in _cell_output_text(executed, "lesson12-017")
    assert "failure_stage" in _cell_output_text(executed, "lesson12-017")
    assert "unsupported_metric" in _cell_output_text(executed, "lesson12-008")
    assert "expected signature" in _cell_output_text(executed, "lesson12-009")
    assert "observed signature" in _cell_output_text(executed, "lesson12-009")
    assert "phase" in _cell_output_text(executed, "lesson12-012")
    assert "latency_ms" in _cell_output_text(executed, "lesson12-012")
    assert "Execution revisions: [0, 0, 0, 1, 1]" in _cell_output_text(
        executed, "lesson12-012"
    )
    assert "NOT RUN" in _cell_output_text(executed, "lesson12-022")
    assert "openai:/<model>" in _cell_output_text(executed, "lesson12-022")
    assert "ollama_chat:/<model>" in _cell_output_text(executed, "lesson12-022")
    expected_database = (mlflow_dir / "mlflow.db").resolve()
    expected_ui_command = f"mlflow ui --backend-store-uri sqlite:///{expected_database}"
    assert str(expected_database) in _cell_output_text(executed, "lesson12-023")
    assert expected_ui_command in _cell_output_text(executed, "lesson12-023")
    assert "http://127.0.0.1:5000" in _cell_output_text(executed, "lesson12-023")
    assert all_output_text.count("LESSON_12_PASS") == 1
