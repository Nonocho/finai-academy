"""Run the deterministic Financial Analyst Copilot capstone certification."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image
from streamlit.testing.v1 import AppTest

from finai_academy.capstone import ResearchRequest, build_reference_copilot
from finai_academy.capstone.models import ResearchRunResult
from finai_academy.capstone.persistence import CapstoneRunStore, audit_capstone_mlflow

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ARTIFACT_DIRECTORY = _PROJECT_ROOT / "artifacts" / "capstone"
_REFERENCE_APP = _PROJECT_ROOT / "final-project" / "reference" / "streamlit_app.py"
_STUDENT_DIRECTORY = _PROJECT_ROOT / "final-project" / "student"
_SOLUTION_PATH = _PROJECT_ROOT / "final-project" / "reference" / "student_integration_solution.py"
_METRIC_NAMES = (
    "tool_call_correctness",
    "tool_call_efficiency",
    "answer_relevance",
    "answer_completeness",
    "citation_integrity",
)
_INCOMPLETE_GROUPS = (
    "wire_retriever",
    "register_analyst_capabilities",
    "evaluate_student_evidence_gate",
    "assemble_public_briefing_view",
)
_EXPECTED_STARTER_FAILURES = (
    *(f"FAIL {name}" for name in _INCOMPLETE_GROUPS),
    "FAIL diagnostic_case",
)
_MLFLOW_ARTIFACTS = (
    "evidence/briefing.json",
    "evidence/dataset_identities.json",
    "evidence/trajectory.json",
)
_VISUAL_CHECKS = (
    "readable_hierarchy",
    "no_clipping",
    "provider_data_labels",
    "plan_replan_tool_states",
    "briefing_readability",
    "evidence_citation_readability",
    "trace_readability",
    "status_distinctions",
    "release_judge_separation",
    "exact_footer",
)
_FORBIDDEN_PUBLIC_TEXT = (
    "/Users/",
    "/home/",
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
    "Authorization: Bearer",
    "client_secret",
    "private_key",
)


class CertificationFailure(RuntimeError):
    """A mandatory gate failed without exposing private diagnostic content."""


def _require(condition: bool) -> None:
    if not condition:
        raise CertificationFailure("mandatory certification gate failed")


def _rendered_text(app: AppTest) -> str:
    element_types = (
        "title",
        "header",
        "subheader",
        "markdown",
        "caption",
        "info",
        "warning",
        "success",
        "error",
        "text",
        "write",
    )
    return "\n".join(
        str(element.value)
        for element_type in element_types
        for element in app.get(element_type)
        if getattr(element, "value", None) is not None
    )


def _safe_public_text(value: str) -> bool:
    lowered = value.casefold()
    return not any(item.casefold() in lowered for item in _FORBIDDEN_PUBLIC_TEXT)


def _validate_reference_mission(result: ResearchRunResult) -> dict[str, object]:
    request = ResearchRequest.reference()
    _require(result.request == request)
    _require(result.status == "completed")
    _require(result.provider == "recorded")
    _require(result.data_mode == "certified")
    _require(result.replan_count == 1)
    _require(result.replan_count <= request.max_replans)
    _require(len(result.observations) == 5)
    _require(len(result.observations) <= request.max_steps)
    _require(result.initial_plan != result.final_plan)
    _require(result.evidence_gate.passed)
    _require(result.briefing is not None)

    briefing = result.briefing
    assert briefing is not None
    companies = tuple(request.companies)
    _require(tuple(briefing.company_evidence) == companies)
    _require(all(briefing.company_evidence[company] for company in companies))
    _require({fact.company for fact in briefing.cited_facts} == set(companies))

    hits_by_id = {hit.evidence_id: hit for hit in result.evidence_gate.evidence_hits}
    _require(len(hits_by_id) == len(result.evidence_gate.evidence_hits))
    collected_sources = {
        source for observation in result.observations for source in observation.source_references
    }
    _require(bool(collected_sources))
    for fact in briefing.cited_facts:
        _require(bool(fact.source_reference))
        _require(fact.source_reference in collected_sources)
        if fact.provenance_kind == "document":
            hit = hits_by_id.get(fact.evidence_id or "")
            _require(hit is not None)
            assert hit is not None
            _require(hit.company == fact.company)
            _require(hit.source_reference == fact.source_reference)
    expected_sources = tuple(dict.fromkeys(fact.source_reference for fact in briefing.cited_facts))
    _require(briefing.aggregate_sources == expected_sources)

    metrics = result.deterministic_evaluation.metrics
    metric_names = tuple(metric.name for metric in metrics)
    metric_values = tuple(metric.value for metric in metrics)
    _require(metric_names == _METRIC_NAMES)
    _require(metric_values == (1.0,) * 5)
    _require(result.deterministic_evaluation.release_passed)
    citation_integrity = next(
        metric.value for metric in metrics if metric.name == "citation_integrity"
    )
    _require(citation_integrity == 1.0)

    return {
        "status": "completed",
        "provider": "recorded",
        "data_mode": "certified",
        "bounded_replan": True,
        "replan_count": result.replan_count,
        "observation_count": len(result.observations),
        "companies_evidenced": list(companies),
        "citation_pairs_valid": True,
        "citation_integrity": citation_integrity,
        "metric_names": list(metric_names),
        "metric_values": list(metric_values),
        "release_passed": True,
    }


def _certify_reference_app() -> dict[str, object]:
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        app = AppTest.from_file(_REFERENCE_APP).run(timeout=30)
        _require(not app.exception)
        _require(app.title and app.title[0].value == "Financial Document Analyst")
        _require(not app.tabs)
        button = next(item for item in app.button if item.key == "analyze_reports")
        app = button.click().run(timeout=30)
    _require(not app.exception)
    text = _rendered_text(app)
    expander_labels = {item.label for item in app.expander}
    footer = next(
        item for item in app.main.markdown if 'data-testid="capstone-footer"' in item.value
    )
    provider_data_labels_visible = "certified document data" in text
    plan_replan_tool_states_visible = {
        "Retrieval details",
        "Tool activity",
        "Technical trace",
        "Evaluation and run details",
    } <= expander_labels
    evidence_citations_visible = "Original report" in text and "Source details" in expander_labels
    trace_visible = "How the analysis works" in text
    release_judge_separated = "Conclusion" in text and "Company evidence" in text
    footer_exact = (
        footer.value
        == '<footer data-testid="capstone-footer"><hr>First Finance - Arnaud Demes</footer>'
    )
    _require(provider_data_labels_visible)
    _require(plan_replan_tool_states_visible)
    _require(evidence_citations_visible)
    _require(trace_visible)
    _require(release_judge_separated)
    _require(footer_exact)

    return {
        "app_test_passed": True,
        "reference_journey_passed": True,
        "provider_data_labels_visible": True,
        "plan_replan_tool_states_visible": True,
        "evidence_citations_visible": True,
        "trace_visible": True,
        "release_judge_separated": True,
        "footer_exact": True,
    }


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    python_path = [str(_PROJECT_ROOT / "src")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    for name in (
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "FINAI_MODEL_PROVIDER",
        "FINAI_CHAT_MODEL",
    ):
        environment.pop(name, None)
    return environment


def _run_verifier(student_directory: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(student_directory / "verify.py")],
        cwd=_PROJECT_ROOT,
        env=_subprocess_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


def _certify_student() -> dict[str, object]:
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        app = AppTest.from_file(_STUDENT_DIRECTORY / "streamlit_app.py").run(timeout=30)
    _require(not app.exception)
    starter_text = _rendered_text(app)
    _require(starter_text.count("Incomplete:") == 4)
    _require(all(name in starter_text for name in _INCOMPLETE_GROUPS))

    starter = _run_verifier(_STUDENT_DIRECTORY)
    starter_failures = [
        line.split(":", 1)[0] for line in starter.stdout.splitlines() if line.startswith("FAIL ")
    ]
    _require(starter.returncode != 0)
    _require(tuple(starter_failures) == _EXPECTED_STARTER_FAILURES)
    _require(starter.stdout.splitlines().count("CAPSTONE_PASS") == 0)
    _require(not starter.stderr)
    _require(_safe_public_text(starter.stdout))

    with tempfile.TemporaryDirectory(prefix="finai-capstone-solved-") as directory:
        solved_directory = Path(directory) / "student"
        shutil.copytree(_STUDENT_DIRECTORY, solved_directory)
        shutil.copyfile(_SOLUTION_PATH, solved_directory / "integration.py")
        shutil.copyfile(
            _PROJECT_ROOT / "final-project/reference/student_diagnostic_solution.json",
            solved_directory / "diagnostic_case.json",
        )
        solved = _run_verifier(solved_directory)
    marker_count = solved.stdout.splitlines().count("CAPSTONE_PASS")
    _require(solved.returncode == 0)
    _require(marker_count == 1)
    _require(not solved.stderr)
    _require(_safe_public_text(solved.stdout))

    return {
        "starter_launches": True,
        "incomplete_groups": list(_INCOMPLETE_GROUPS),
        "incomplete_group_count": 4,
        "starter_diagnostic_failed": True,
        "starter_marker_count": 0,
        "solved_exit_zero": True,
        "solved_diagnostic_passed": True,
        "solved_marker_count": marker_count,
    }


def _certify_mlflow(
    result: ResearchRunResult,
    tracking_directory: Path,
) -> dict[str, object]:
    captured_output = io.StringIO()
    with (
        contextlib.redirect_stdout(captured_output),
        contextlib.redirect_stderr(captured_output),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        references = CapstoneRunStore(tracking_directory).persist(result)
        mlflow = importlib.import_module("mlflow")
        client_class = importlib.import_module("mlflow.tracking").MlflowClient
        internal_uri = f"sqlite:///{(tracking_directory / 'mlflow.db').resolve()}"
        client = client_class(tracking_uri=internal_uri)
        _require(references.tracking_status == "persisted")
        _require(bool(references.run_id))
        _require(bool(references.trace_id))
        _require(references.analysis_run_id == result.run_id)
        run = client.get_run(references.run_id)
        artifact_names = tuple(
            artifact
            for artifact in _MLFLOW_ARTIFACTS
            if (
                tracking_directory / "artifacts" / references.run_id / "artifacts" / artifact
            ).is_file()
        )
        artifact_paths = [
            tracking_directory / "artifacts" / references.run_id / "artifacts" / artifact
            for artifact in _MLFLOW_ARTIFACTS
        ]
        artifact_payloads = [
            json.loads(path.read_text(encoding="utf-8")) for path in artifact_paths
        ]
        traces = mlflow.search_traces(
            run_id=references.run_id,
            locations=[run.info.experiment_id],
            return_type="list",
            flush=True,
        )
        mlflow.flush_trace_async_logging(terminate=True)

    _require(artifact_names == _MLFLOW_ARTIFACTS)
    _require(set(run.data.metrics) >= {*_METRIC_NAMES, "release_passed"})
    _require(all(run.data.metrics[name] == 1.0 for name in _METRIC_NAMES))
    _require(run.data.metrics["release_passed"] == 1.0)
    _require(run.data.params["analysis_run_id"] == result.run_id)
    _require(run.data.params["release_decision"] == "passed")
    _require(len(traces) == 1)
    root_span = next(span for span in traces[0].data.spans if span.parent_id is None)
    trace_id = getattr(traces[0].info, "trace_id", root_span.trace_id)
    _require(trace_id == references.trace_id)
    _require(root_span.inputs["analysis_run_id"] == result.run_id)
    public_payload = json.dumps(
        {
            "references": references.model_dump(mode="json"),
            "artifacts": artifact_payloads,
        },
        sort_keys=True,
    )
    public_artifacts_validated = _safe_public_text(public_payload)
    _require(public_artifacts_validated)
    privacy_audit = _public_privacy_audit(
        audit_capstone_mlflow(
            tracking_directory / "mlflow.db", tracking_directory / "artifacts"
        )
    )

    return {
        "persisted": True,
        "run_present": True,
        "trace_present": True,
        "run_trace_linked": True,
        "metric_names": list(_METRIC_NAMES),
        "release_persisted": True,
        "artifact_names": list(_MLFLOW_ARTIFACTS),
        "public_artifacts_validated": public_artifacts_validated,
        "privacy_audit": privacy_audit,
    }


def _public_privacy_audit(audit: Mapping[str, int]) -> dict[str, bool]:
    """Publish stable coverage without MLflow's asynchronous artifact count."""

    sqlite_metadata_scanned = audit["sqlite_text_values_scanned"] > 0
    required_artifacts_scanned = audit["artifact_files_scanned"] >= len(_MLFLOW_ARTIFACTS)
    _require(sqlite_metadata_scanned)
    _require(required_artifacts_scanned)
    return {
        "sqlite_metadata_scanned": sqlite_metadata_scanned,
        "required_artifacts_scanned": required_artifacts_scanned,
    }


def validate_visual_evidence(artifact_directory: Path) -> dict[str, object]:
    """Validate hash-bound, fully decoded browser captures and coverage."""

    inspection_path = artifact_directory / "visual-inspection.json"
    if not inspection_path.exists():
        return {
            "status": "NOT RUN",
            "captures": [],
            "covered_elements": [],
            "limitation": "No real browser capture and inspection result was recorded.",
        }

    inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
    _require(isinstance(inspection, dict))
    _require(inspection.get("schema_version") == 2)
    status = inspection.get("status")
    _require(status in {"NOT RUN", "PASS"})
    limitation = inspection.get("limitation")
    _require(isinstance(limitation, str) and bool(limitation.strip()))
    _require(_safe_public_text(limitation))
    captures = inspection.get("captures")
    _require(isinstance(captures, list))
    covered: set[str] = set()
    filenames: set[str] = set()
    hashes: set[str] = set()
    completed_state_present = False
    typed_stop_state_present = False
    validated: list[dict[str, object]] = []
    for capture in captures:
        _require(isinstance(capture, dict))
        _require(
            set(capture)
            == {
                "file",
                "sha256",
                "width",
                "height",
                "browser",
                "route",
                "state",
                "captured_at",
                "visible_elements",
            }
        )
        filename = capture["file"]
        _require(isinstance(filename, str) and Path(filename).name == filename)
        _require(filename not in filenames)
        filenames.add(filename)
        sha256 = capture["sha256"]
        _require(isinstance(sha256, str) and sha256 not in hashes)
        hashes.add(sha256)
        _require(_safe_public_text(json.dumps(capture, sort_keys=True)))
        _require(capture["browser"] == "Codex in-app Browser")
        _require(isinstance(capture["route"], str) and bool(capture["route"]))
        state = capture["state"]
        _require(isinstance(state, str) and bool(state))
        completed_state_present |= state.startswith("completed mission;")
        typed_stop_state_present |= state.startswith("typed ") and " stop" in state
        captured_at = capture["captured_at"]
        _require(
            isinstance(captured_at, str) and ("+" in captured_at[10:] or captured_at.endswith("Z"))
        )
        visible = capture["visible_elements"]
        _require(isinstance(visible, list) and all(item in _VISUAL_CHECKS for item in visible))
        path = artifact_directory / filename
        raw = path.read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == sha256)
        with Image.open(path) as image:
            _require(image.format == "PNG")
            image.verify()
        with Image.open(path) as image:
            image.load()
            _require(image.size == (capture["width"], capture["height"]) == (1440, 1000))
            _require(_safe_public_text(json.dumps(image.info, default=str, sort_keys=True)))
        covered.update(visible)
        validated.append(capture)
    if status == "PASS":
        _require(len(validated) >= 4)
        _require(covered == set(_VISUAL_CHECKS))
        _require(completed_state_present)
        _require(typed_stop_state_present)
    if status == "NOT RUN":
        _require(not validated)

    return {
        "status": status,
        "captures": validated,
        "covered_elements": sorted(covered),
        "limitation": limitation,
    }


_visual_evidence = validate_visual_evidence


def _readiness_markdown(payload: Mapping[str, Any]) -> str:
    visual = payload["streamlit"]["visual_evidence"]
    optional = payload["optional_providers"]
    lines = [
        "# Financial Analyst Copilot capstone readiness",
        "",
        "Offline certification: **PASS**",
        "",
        "| Mandatory gate | Status | Evidence |",
        "| --- | --- | --- |",
        "| Recorded reference mission | PASS | Completed with one bounded replan, both-company evidence, valid citation pairs, five 100% metrics, and deterministic release. |",
        "| Streamlit AppTest journey | PASS | Recorded mission renders plan, tools, evidence, citations, trace, release, optional judge, and the exact footer. |",
        "| Student starter | PASS | Launches with four intended incomplete groups plus one regressed diagnostic and no success marker. |",
        "| Solved student copy | PASS | Four seams and the diagnostic correction pass; verifier prints exactly one `CAPSTONE_PASS`. |",
        "| MLflow | PASS | Run, linked trace, five metrics, release decision, and three sanitized public artifacts persisted. |",
        "| Public artifact scan | PASS | Certification outputs contain no credentials or personal filesystem paths. |",
        "",
        "## Visual evidence",
        "",
        f"Status: **{visual['status']}**",
        "",
        f"Limitation: {visual['limitation']}",
    ]
    if visual["status"] == "PASS":
        lines.extend(
            [
                "",
                "The hash-bound 1440×1000 browser captures were fully decoded and inspected for readable hierarchy, no clipping, provider/data labels, plan/replan/tool states, briefing, evidence and citation readability, trace readability, status distinctions, release/judge separation, and the exact footer.",
            ]
        )
    lines.extend(
        [
            "",
            "## Optional coverage",
            "",
            "Optional checks do not affect the offline release gate.",
            "",
            "| Check | Status |",
            "| --- | --- |",
            f"| OpenAI | {optional['openai']} |",
            f"| Ollama | {optional['ollama']} |",
            f"| Tavily | {optional['tavily']} |",
            f"| Timed classroom rehearsal | {optional['timed_classroom_rehearsal']} |",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_payload(artifact_directory: Path) -> dict[str, object]:
    result = build_reference_copilot(run_id_factory=lambda: "capstone-certification-run").run(
        ResearchRequest.reference()
    )
    reference_mission = _validate_reference_mission(result)
    streamlit = _certify_reference_app()
    streamlit["visual_evidence"] = validate_visual_evidence(artifact_directory)
    student = _certify_student()
    mlflow = _certify_mlflow(result, artifact_directory / "mlflow")
    public_scan_passed = all(
        _safe_public_text(json.dumps(section, sort_keys=True))
        for section in (reference_mission, streamlit, student, mlflow)
    )
    _require(public_scan_passed)
    return {
        "schema_version": 1,
        "offline_release_passed": True,
        "reference_mission": reference_mission,
        "streamlit": streamlit,
        "student": student,
        "mlflow": mlflow,
        "repository": {
            "public_artifact_scan_passed": public_scan_passed,
            "credential_scan_passed": public_scan_passed,
            "personal_path_scan_passed": public_scan_passed,
        },
        "optional_providers": {
            "openai": "NOT RUN",
            "ollama": "NOT RUN",
            "tavily": "NOT RUN",
            "timed_classroom_rehearsal": "NOT RUN",
        },
    }


def certify(artifact_directory: Path) -> None:
    artifact_directory.mkdir(parents=True, exist_ok=True)
    payload = _artifact_payload(artifact_directory)
    encoded = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    readiness = _readiness_markdown(payload)
    _require(_safe_public_text(encoded))
    _require(_safe_public_text(readiness))
    (artifact_directory / "certification.json").write_text(encoded, encoding="utf-8")
    (artifact_directory / "readiness.md").write_text(readiness, encoding="utf-8")


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=_DEFAULT_ARTIFACT_DIRECTORY,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        certify(arguments.artifact_dir)
    except Exception:  # noqa: BLE001 - certification output must remain public and path-free
        print("CAPSTONE_CERTIFICATION_FAILED")
        return 1
    print("CAPSTONE_CERTIFICATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
