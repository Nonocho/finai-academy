from __future__ import annotations

import os
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


def visible_output_text(notebook) -> str:
    """Return learner-visible stream and rich-text output from an executed notebook."""
    parts: list[str] = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                parts.append(output.get("text", ""))
                continue
            data = output.get("data", {})
            for mime_type in ("text/markdown", "text/plain"):
                if mime_type in data:
                    parts.append(data[mime_type])
                    break
    return "\n".join(parts)


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


def test_live_executor_provider_overrides_stale_model_and_embedding_environment(
    tmp_path: Path,
) -> None:
    """Explicit live provider selection must replace both stale provider variables."""

    notebook_path = tmp_path / "01_provider_override.ipynb"
    write_notebook(notebook_path, body="Provider override contract.")
    notebook = nbformat.read(notebook_path, as_version=4)
    notebook.cells[1].source = (
        "import os\n"
        'print(os.environ["FINAI_MODEL_PROVIDER"] + "/" + '
        'os.environ["FINAI_EMBEDDING_PROVIDER"])'
    )
    nbformat.write(notebook, notebook_path)
    output_dir = tmp_path / "executed"
    hostile_environment = os.environ.copy()
    hostile_environment.update(
        {
            "FINAI_MODEL_PROVIDER": "openai",
            "FINAI_EMBEDDING_PROVIDER": "openai",
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            str(EXECUTOR),
            str(notebook_path),
            "--mode",
            "live",
            "--provider",
            "ollama",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        env=hostile_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    executed = nbformat.read(output_dir / notebook_path.name, as_version=4)
    assert stream_text(executed) == "ollama/ollama\n"


def test_lesson06_offline_execution_ignores_hostile_provider_environment(
    tmp_path: Path,
) -> None:
    """Offline deterministic embeddings must be selected before provider validation."""

    notebook_path = ROOT / "notebooks" / "06_hybrid_retrieval.ipynb"
    output_dir = tmp_path / "executed"
    hostile_environment = os.environ.copy()
    hostile_environment.update(
        {
            "FINAI_MODEL_PROVIDER": "hostile-model-provider",
            "FINAI_EMBEDDING_PROVIDER": "hostile-embedding-provider",
        }
    )

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
        env=hostile_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    executed = nbformat.read(output_dir / notebook_path.name, as_version=4)
    assert "Embedding runtime: offline / financial-concepts-v1" in stream_text(executed)


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
    output_text = visible_output_text(executed)
    assert "### First gateway run" in output_text
    assert "Unavailable" in output_text
    assert "**Streaming demo**" in output_text
    assert "### Curated SEC evidence card" in output_text
    assert "### Grounding score: 4/4" in output_text
    assert output_text.count("PASS — provider-neutral model gateway verified") == 1


def test_model_gateway_keeps_mlflow_out_of_executable_lesson_code() -> None:
    notebook = nbformat.read(ROOT / "notebooks" / "01_model_gateway.ipynb", as_version=4)
    executable_code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "import mlflow" not in executable_code.casefold()
    assert "from mlflow" not in executable_code.casefold()


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
    output_text = visible_output_text(executed)
    stream_text = "".join(
        output.get("text", "")
        for cell in executed.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    assert "Validation caught the unsupported candidate" in stream_text
    assert "Prompt comparison: zero-shot -> few-shot -> schema-bound" in stream_text
    assert "Prompt injection remains source data: PASS" in stream_text
    assert stream_text.count("PASS — structured financial brief verified") == 1
    assert "### Five-layer prompt contract" in output_text
    assert "### Zero-shot vs few-shot" in output_text
    assert "### Why the candidate failed" in output_text
    assert "### Accepted AnalystBrief" in output_text
    assert "### Three reliability layers" in output_text
    assert "### Verification: 6/6" in output_text
    assert len(executed.cells) <= 16


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
    assert "Decision: CAG for the bounded official context pack" in stream_text
    assert "Decision: RAG for the complete official filing" in stream_text
    assert "PASS — real-document CAG/RAG boundary verified" in stream_text


def test_cag_notebook_distinguishes_context_cache_memory_grounding_and_rag() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks" / "03_cag_financial_document.ipynb",
        as_version=4,
    )
    source = "\n".join(cell.source for cell in notebook.cells)
    executable = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    for concept in ("Context", "Cache", "Memory", "Grounding", "RAG"):
        assert concept in source
    assert "ContextDecision" in source
    assert "Use only F1 and F2" in source
    assert "square brackets [F1] and [F2]" in source
    assert "round monetary values to one decimal place" in source
    assert "Live grounding remains an observation" in source
    assert "if not live_mode:" in executable
    assert "assert grounding_result.passed" in executable
    assert "import mlflow" not in executable
    assert "assets/course-data/downloads/nvidia_fy2026_form_10k.html" in source
    assert "bounded official context pack" in source.casefold()
    assert "complete official filing" in source.casefold()
    assert "synthetic neutral appendix" not in source.casefold()
    assert len(notebook.cells) <= 16


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

    assert len(visual_outputs) >= 4
    assert "Official filing verified: PASS" in stream_text
    assert "Retrieval check: PASS" in stream_text
    assert "Grounding check:" in stream_text
    assert "Failure diagnosis: RETRIEVAL" in stream_text
    assert "PASS — real-document naive RAG boundary verified" in stream_text


def test_naive_rag_notebook_uses_the_official_filing_and_stays_compact() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks" / "04_rag_from_scratch.ipynb",
        as_version=4,
    )
    source = "\n".join(cell.source for cell in notebook.cells)
    executable = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    setup_cell = notebook.cells[1]

    assert "assets/course-data/downloads/nvidia_fy2026_form_10k.html" in source
    assert "Why RAG?" in source
    assert "naive HTML parsing" in source
    assert "naive character windows" in source
    assert "Parsing → chunking → retrieval → generation" in source
    assert "Instructor setup — run once" in source
    assert "This is everything the model can see" in source
    assert "Schneider Electric" not in source
    assert "Live answer grounding remains an observation" in source
    assert "naive_parse_html" in executable
    assert "naive_fixed_windows" in executable
    assert "if not live_mode:" in executable
    assert "assert retrieval_check.passed" in executable
    assert "import mlflow" not in executable
    assert setup_cell.metadata["jupyter"]["source_hidden"] is True
    assert len(notebook.cells) == 12


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
    assert "Provider-aware semantic boundaries verified" in stream_text
    assert "Generated contextual enrichment verified" in stream_text
    assert "Raw evidence preserved" in stream_text
    assert "Token inflation measured" in stream_text
    assert "Retrieval comparison complete" in stream_text
    assert "Table integrity failure reproduced" in stream_text
    assert "Seven strategies compared" in stream_text
    assert "PASS — document and chunking laboratory verified" in stream_text


def test_document_chunking_notebook_teaches_the_contextual_progression() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks" / "05_document_and_chunking_lab.ipynb",
        as_version=4,
    )
    source = "\n".join(cell.source for cell in notebook.cells)
    executable = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert "Parser ladder" in source
    assert "Provider-aware semantic boundaries" in source
    assert "LLM contextual enrichment" in source
    assert "LLM contextual enrichment is not agentic chunking" in source
    assert "Token inflation" in source
    assert "Retrieval comparison" in source
    assert "Optional extension: proposition chunking" in source
    assert "embedding_similarity_profile" in executable
    assert "contextual_enrich_chunks" in executable
    assert "generated_context" in executable
    assert "raw_text" in executable


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
    assert count_png_outputs(executed) >= 6
    output_text = stream_text(executed)
    assert "BM25 exact-term recovery reproduced" in output_text
    assert "Dense exact-term failure reproduced" in output_text
    assert "Cross-company leakage blocked" in output_text
    assert "Hybrid retrieval improves maintained recall" in output_text
    assert "PASS — hybrid retrieval laboratory verified" in output_text


def test_hybrid_retrieval_notebook_separates_live_and_controlled_outcomes():
    notebook = nbformat.read(ROOT / "notebooks" / "06_hybrid_retrieval.ipynb", as_version=4)
    executable = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )

    assert "controlled_dense_index" in executable
    assert "if not live_mode:" in executable
    assert 'maintained_recall["Reranked hybrid"]' in executable
    assert '"provider vectors are finite"' in executable


def test_hybrid_retrieval_notebook_qualifies_offline_success_in_markdown():
    notebook = nbformat.read(ROOT / "notebooks" / "06_hybrid_retrieval.ipynb", as_version=4)
    learning_objectives = next(
        cell for cell in notebook.cells
        if cell.cell_type == "markdown" and "success condition" in cell.source
    )

    assert "deterministic offline laboratory success condition" in learning_objectives.source
    assert "Live OpenAI and Ollama runs report observed recall" in learning_objectives.source
    assert "provider-invariant structural behavior" in learning_objectives.source


def test_hybrid_retrieval_notebook_is_compact_and_bm25_focused() -> None:
    notebook = nbformat.read(
        ROOT / "notebooks" / "06_hybrid_retrieval.ipynb",
        as_version=4,
    )
    source = "\n".join(cell.source for cell in notebook.cells)
    executable = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")

    assert len(notebook.cells) <= 18
    assert sum(cell.cell_type == "code" for cell in notebook.cells) <= 8
    assert sum(len(cell.source.splitlines()) for cell in notebook.cells if cell.cell_type == "code") <= 320
    assert "BM25" in source
    assert "BM25Index" in executable
    assert "Reciprocal-rank fusion" in source
    assert "Stage timings" not in source
    assert "Local index versus pgvector/HNSW" not in source
    assert "stage_measurements" not in executable
    assert "import mlflow" not in executable


def test_rag_evaluation_notebook_is_output_free_and_complete() -> None:
    notebook_path = ROOT / "notebooks" / "07_rag_evaluation.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)

    assert all(not cell.get("outputs") for cell in notebook.cells if cell.cell_type == "code")
    assert all(
        cell.get("execution_count") is None for cell in notebook.cells if cell.cell_type == "code"
    )
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    assert len(notebook.cells) <= 18
    assert sum(cell.cell_type == "code" for cell in notebook.cells) <= 8
    assert sum(
        len(cell.source.splitlines())
        for cell in notebook.cells
        if cell.cell_type == "code"
    ) <= 340
    executable = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, f"{notebook_path.name}:{cell.id}", "exec")
    assert "BM25Index" in executable
    assert "KeywordIndex" not in executable
    assert "retrieval_weights" in executable
    for marker in (
        "Versioned golden set",
        "Retrieval metrics and answer metrics",
        "MLflow trace",
        "Compare two configurations",
        "Failure analysis",
        "Optional Ragas comparison",
        "Knowledge check",
        "Capstone integration",
        "PASS — RAG evaluation and tracing verified",
    ):
        assert marker in source


def test_rag_evaluation_notebook_offline_run_is_visual_and_verified(tmp_path: Path) -> None:
    notebook_path = ROOT / "notebooks" / "07_rag_evaluation.ipynb"
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
    assert 6 <= count_png_outputs(executed) <= 7
    output_text = stream_text(executed)
    assert output_text.count("PASS — RAG evaluation and tracing verified") == 1
    assert "MLflow traces recorded: 8" in output_text


def test_lesson08_notebook_offline_run_is_visual_and_verified(tmp_path: Path) -> None:
    notebook_path = ROOT / "notebooks" / "08_workflows_vs_agents.ipynb"
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
    assert count_png_outputs(executed) >= 4
    output_text = stream_text(executed)
    assert output_text.count("LESSON_08_PASS") == 1
    assert "get_market_price" in output_text
    assert "convert_currency" in output_text
    assert "workflow_compound_status=completed" in output_text
    assert "preferred_architecture=workflow" in output_text
