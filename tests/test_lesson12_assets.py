"""Contract tests for the Lesson 12 MLflow agent-evaluation notebook."""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from xml.etree import ElementTree

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "12_evaluating_agentic_systems.ipynb"
BUILDER = ROOT / "scripts" / "build_lesson12_notebook.py"
EXECUTOR = ROOT / "scripts" / "execute_notebooks.py"
CHAPTER = ROOT / "chapters" / "12-evaluating-agentic-systems.md"
GETTING_STARTED = ROOT / "docs" / "getting-started.md"
DECK = ROOT / "decks" / "12-evaluating-agentic-systems.pptx"
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PRESENTATION_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
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


def _slide_root(archive: zipfile.ZipFile, slide_number: int):
    return ElementTree.fromstring(
        archive.read(f"ppt/slides/slide{slide_number}.xml")
    )


def _visible_text(root) -> str:
    return "\n".join(
        node.text or "" for node in root.iter(f"{{{DRAWING_NS}}}t")
    )


def _shape_context_colors(root, marker: str) -> set[str]:
    shapes = list(root.iter(f"{{{PRESENTATION_NS}}}sp"))
    for index, shape in enumerate(shapes):
        text = " ".join(
            node.text or "" for node in shape.iter(f"{{{DRAWING_NS}}}t")
        )
        if marker in text:
            return {
                node.get("val", "")
                for context_shape in shapes[max(0, index - 1) : index + 1]
                for node in context_shape.iter(f"{{{DRAWING_NS}}}srgbClr")
            }
    raise AssertionError(f"Slide shape containing {marker!r} was not found")


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


def test_lesson12_notebook_source_loads_one_persisted_failed_root_trace() -> None:
    """Catch replacement of the offline trace drill with IDs or aggregate rows only."""

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    failure_lab = next(cell for cell in notebook.cells if cell.id == "lesson12-020")
    for marker in (
        "mlflow.search_traces",
        "bounded-agent-v1",
        "unsupported_metric_not_recovered",
        'trace_metadata["mlflow.sourceRun"]',
        "bounded_summary.trace_ids",
        "parent_id is None",
        "start_time_ns",
        "span_type",
        "attempt_id",
        "plan_revision",
        "error_code",
        "guardrail",
        "failure_stage",
    ):
        assert marker in failure_lab.source


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
    trace_matches = re.findall(
        r"Trace ID \(([^/]+)/([^\n)]+)\): (tr-[0-9a-f]+)", stream_text
    )
    trace_ids = [trace_id for _, _, trace_id in trace_matches]
    assert len(trace_ids) == 12
    assert len(set(trace_ids)) == 12
    trace_ids_by_case = {
        (configuration_id, case_id): trace_id
        for configuration_id, case_id, trace_id in trace_matches
    }
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
    failure_lab_text = _cell_output_text(executed, "lesson12-020")
    assert "Selected failed trace configuration: bounded-agent-v1" in failure_lab_text
    assert "Selected failed trace case: unsupported_metric_not_recovered" in (
        failure_lab_text
    )
    assert f"Associated run ID: {run_ids[0]}" in failure_lab_text
    assert (
        "Trace ID: "
        + trace_ids_by_case[
            ("bounded-agent-v1", "unsupported_metric_not_recovered")
        ]
        in failure_lab_text
    )
    assert re.search(r"Root span ID: [0-9a-f]+", failure_lab_text)
    assert (
        "Persisted child order: planning -> plan_gate -> execution:1 -> "
        "replanning -> execution:2 -> replanning -> execution:3 -> "
        "evidence_gate -> report"
    ) in failure_lab_text
    for marker in (
        "span_type",
        "phase",
        "public_status",
        "attempt_id",
        "plan_revision",
        "typed_error",
        "guardrail_evidence",
        "unsupported_metric",
        "blocked | Execution stopped after the unsupported metric was not recovered.",
        "Failure owner: evidence_gate",
    ):
        assert marker in failure_lab_text
    assert "NOT RUN" in _cell_output_text(executed, "lesson12-022")
    assert "openai:/<model>" in _cell_output_text(executed, "lesson12-022")
    assert "ollama_chat:/<model>" in _cell_output_text(executed, "lesson12-022")
    expected_database = (mlflow_dir / "mlflow.db").resolve()
    expected_ui_command = f"mlflow ui --backend-store-uri sqlite:///{expected_database}"
    assert str(expected_database) in _cell_output_text(executed, "lesson12-023")
    assert expected_ui_command in _cell_output_text(executed, "lesson12-023")
    assert "http://127.0.0.1:5000" in _cell_output_text(executed, "lesson12-023")
    assert all_output_text.count("LESSON_12_PASS") == 1


def test_lesson12_chapter_defines_the_complete_instructor_route() -> None:
    """Catch loss of the exact route, evaluation contracts, or recovery guidance."""

    assert CHAPTER.is_file()
    chapter = CHAPTER.read_text(encoding="utf-8")
    normalized_chapter = " ".join(chapter.split())

    for marker in (
        "14:30-15:30",
        "12-minute concept deck",
        "40-minute notebook",
        "8-minute verification and debrief",
        "uv sync --extra ai --extra evaluation --extra dev",
        "No-network fallback",
        "Skip if late",
        "LESSON_12_PASS",
        "OpenAI",
        "Ollama",
        "NOT RUN",
        "mlflow ui --backend-store-uri sqlite:////absolute/path/to/mlflow.db",
        "reference_completed",
        "unsupported_metric_not_recovered",
        "redundant_metric_call",
        "missing_schneider_document",
        "document_fact_without_evidence_id",
        "wrong_source_evidence_pair",
        "bounded-agent-v1",
        "regressed-agent-v0",
        *METRIC_NAMES,
        "metric fact",
        "document fact",
        "aggregate sources",
        "dataset/hash mismatch",
        "local SQLite failure",
        "trace/run association failure",
        "judge timeout or disagreement",
        "suspected secret or private-data exposure",
        "public serializable state",
        "no trading",
        "portfolio mutation",
        "price target",
        "investment recommendation",
        "full Lesson 12 route is ready for an instructor-led offline test class",
    ):
        assert marker.casefold() in normalized_chapter.casefold()
    for cell_index in range(27):
        assert f"lesson12-{cell_index:03d}" in chapter
    for figure_purpose in (
        "Versioned expectations evaluate trajectory and answer separately",
        "Expected and observed dependency-aware call signatures align",
        "One public trace retains phase, attempt, revision, status, and latency",
        "Per-case metrics reveal failures hidden by configuration means",
        "Aligned configurations compare all five means on one dataset hash",
        "Failure diagnosis assigns the earliest public owner",
    ):
        assert figure_purpose in chapter
    assert chapter.count("Answer:") >= 5
    assert "—" not in chapter


def test_lesson12_chapter_defines_exact_optional_judge_outcome_taxonomy() -> None:
    chapter = " ".join(CHAPTER.read_text(encoding="utf-8").split())

    for contract in (
        (
            "Missing configuration or an unavailable explicit provider, adapter, "
            "client, or service is `NOT RUN`"
        ),
        "A completed scorer, including a low or disagreeing score, is `COMPLETED`",
        "A timeout or ordinary runtime invocation failure is `ERROR`",
        (
            "All three outcomes are observational and never change deterministic "
            "metrics or `release_passed`"
        ),
    ):
        assert contract in chapter
    assert "**Answer:** Use the exact three-way taxonomy:" in chapter


def test_lesson12_onboarding_documents_local_mlflow_and_explicit_judges() -> None:
    """Catch accidental Docker, browser, or implicit-provider requirements."""

    onboarding = GETTING_STARTED.read_text(encoding="utf-8")
    for marker in (
        "Lesson 12",
        "FINAI_MLFLOW_DIR",
        "local SQLite",
        "local artifacts",
        "http://127.0.0.1:5000",
        "Docker is not required for Lesson 12",
        "browser UI is not required for Lesson 12",
        "FINAI_EVAL_JUDGE_MODEL=openai:/<model>",
        "FINAI_EVAL_JUDGE_MODEL=ollama_chat:/<model>",
        "explicit judge URIs",
    ):
        assert marker in onboarding


def test_lesson12_indexes_expose_completed_course_links_without_stale_status() -> None:
    """Catch a hidden Lesson 12 route or obsolete planned-course status."""

    expected_links = {
        "chapters/README.md": (
            "[Evaluating agentic systems with MLflow](12-evaluating-agentic-systems.md)",
        ),
        "notebooks/README.md": (
            "[Evaluating agentic systems with MLflow](12_evaluating_agentic_systems.ipynb)",
        ),
        "decks/README.md": (
            "[Evaluating agentic systems with MLflow](12-evaluating-agentic-systems.pptx)",
        ),
        "README.md": (
            "[Lesson 12 instructor chapter](chapters/12-evaluating-agentic-systems.md)",
            "[Lesson 12 notebook](notebooks/12_evaluating_agentic_systems.ipynb)",
            "[Lesson 12 concept deck](decks/12-evaluating-agentic-systems.pptx)",
        ),
    }
    for relative_path, links in expected_links.items():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        normalized_text = " ".join(text.split())
        for link in links:
            assert link in text
        assert "Lesson 12 remains planned" not in normalized_text
        assert "Lessons 08-12 are ready for an instructor-led offline test class" in (
            normalized_text
        )


def test_lesson12_deck_has_the_complete_sourced_concept_route() -> None:
    """Catch a missing, partial, unsourced, or visibly non-compliant deck."""

    assert DECK.is_file()
    with zipfile.ZipFile(DECK) as archive:
        names = archive.namelist()
        slide_names = sorted(
            name
            for name in names
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        notes_names = sorted(
            name
            for name in names
            if name.startswith("ppt/notesSlides/notesSlide")
            and name.endswith(".xml")
        )
        assert len(slide_names) == 11
        assert len(notes_names) == 11

        visible_text = "\n".join(
            "".join(
                node.text or ""
                for node in ElementTree.fromstring(archive.read(name)).iter(
                    f"{{{DRAWING_NS}}}t"
                )
            )
            for name in slide_names
        )
        notes_text = "\n".join(
            "".join(
                node.text or ""
                for node in ElementTree.fromstring(archive.read(name)).iter(
                    f"{{{DRAWING_NS}}}t"
                )
            )
            for name in notes_names
        )

    assert visible_text.count("First Finance - Arnaud Demes") == 11
    assert "—" not in visible_text
    for marker in (
        "Evaluating Agentic Systems with MLflow",
        "SAME ANSWER",
        "DIFFERENT PATH",
        "TRAJECTORY",
        "ANSWER",
        "agent-cases-v1",
        "MLFLOW RUN",
        "ROOT TRACE",
        "TOOL",
        *METRIC_NAMES,
        "DETERMINISTIC RELEASE GATE",
        "LLM JUDGE",
        "CITATION INTEGRITY",
        "CAPSTONE",
    ):
        assert marker.casefold() in visible_text.casefold()
    assert notes_text.count("Instructor purpose:") == 11
    assert notes_text.count("Planned timing:") == 11
    assert notes_text.count("[Sources]") == 11
    assert notes_text.count("[/Sources]") == 11
    assert notes_text.count("chapters/12-evaluating-agentic-systems.md") == 11
    for source_url in (
        "https://mlflow.org/docs/latest/genai/tracing/",
        "https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/",
        "https://mlflow.org/docs/latest/genai/eval-monitor/scorers/index.html",
    ):
        assert source_url in notes_text


def test_lesson12_deck_uses_native_tables_for_comparison_slides() -> None:
    """Catch flattened or shape-only substitutes for the three required tables."""

    with zipfile.ZipFile(DECK) as archive:
        for slide_number in (3, 7, 8):
            slide = ElementTree.fromstring(
                archive.read(f"ppt/slides/slide{slide_number}.xml")
            )
            assert slide.find(f".//{{{DRAWING_NS}}}tbl") is not None


def test_lesson12_deck_versioned_case_slide_keeps_dataset_case_and_budget_distinct() -> None:
    """Catch dataset/case identity swaps or the wrong reference-case budget."""

    with zipfile.ZipFile(DECK) as archive:
        slide_text = _visible_text(_slide_root(archive, 4))

    normalized = " ".join(slide_text.split())
    assert re.search(r"dataset_version:\s*agent-cases-v1", normalized)
    assert re.search(r"case_id:\s*reference_completed", normalized)
    assert "max_tool_calls: 5" in normalized


def test_lesson12_deck_trace_slide_shows_recovery_before_the_evidence_gate() -> None:
    """Catch collapsed replan/gate stages or missing post-error tool work."""

    with zipfile.ZipFile(DECK) as archive:
        slide_text = " ".join(_visible_text(_slide_root(archive, 6)).split())

    cursor = 0
    for marker in (
        "PLAN",
        "PLAN GATE",
        "TOOL ATTEMPTS",
        "REPLAN",
        "TOOL ATTEMPTS",
        "EVIDENCE GATE",
        "REPORT",
    ):
        marker_position = slide_text.find(marker, cursor)
        assert marker_position >= cursor, f"Missing or out-of-order stage: {marker}"
        cursor = marker_position + len(marker)
    assert "unsupported_metric" in slide_text
    assert "post-error success" in slide_text


def test_lesson12_deck_trace_connectors_point_toward_the_next_stage() -> None:
    """Catch leftward arrowheads that reverse the trace's reading direction."""

    with zipfile.ZipFile(DECK) as archive:
        slide = _slide_root(archive, 6)

    connectors = list(slide.iter(f"{{{PRESENTATION_NS}}}cxnSp"))
    assert len(connectors) == 6
    for connector in connectors:
        assert connector.find(f".//{{{DRAWING_NS}}}headEnd") is None
        tail = connector.find(f".//{{{DRAWING_NS}}}tailEnd")
        assert tail is not None and tail.get("type") == "triangle"


def test_lesson12_deck_failure_path_uses_warning_not_pass_color() -> None:
    """Catch a failure path rendered with the deck's positive green cue."""

    with zipfile.ZipFile(DECK) as archive:
        slide = _slide_root(archive, 2)

    for marker in ("extra call after gate", "FAIL"):
        colors = _shape_context_colors(slide, marker)
        assert "F07D00" in colors
        assert "2E8B57" not in colors
