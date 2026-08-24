from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _PROJECT_ROOT / "scripts" / "certify_capstone.py"
_METRIC_NAMES = [
    "tool_call_correctness",
    "tool_call_efficiency",
    "answer_relevance",
    "answer_completeness",
    "citation_integrity",
]
_INCOMPLETE_GROUPS = [
    "wire_retriever",
    "register_analyst_capabilities",
    "evaluate_student_evidence_gate",
    "assemble_public_briefing_view",
]
_OPTIONAL_STATUSES = {"NOT RUN", "AVAILABLE", "PASS", "ERROR"}


def _run_certification(
    artifact_directory: Path,
    *,
    optional_environment: bool,
) -> subprocess.CompletedProcess[str]:
    committed = _PROJECT_ROOT / "artifacts" / "capstone"
    artifact_directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        committed / "visual-inspection.json", artifact_directory / "visual-inspection.json"
    )
    manifest = json.loads((committed / "visual-inspection.json").read_text())
    for capture in manifest.get("captures", []):
        shutil.copyfile(committed / capture["file"], artifact_directory / capture["file"])
    environment = os.environ.copy()
    for name in ("OPENAI_API_KEY", "TAVILY_API_KEY", "FINAI_MODEL_PROVIDER"):
        environment.pop(name, None)
    if optional_environment:
        environment.update(
            {
                "OPENAI_API_KEY": "sk-test-value-that-must-not-be-read",
                "TAVILY_API_KEY": "tvly-test-value-that-must-not-be-read",
                "FINAI_MODEL_PROVIDER": "openai",
            }
        )
    return subprocess.run(
        [sys.executable, str(_SCRIPT), "--artifact-dir", str(artifact_directory)],
        cwd=_PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )


@pytest.fixture(scope="module")
def certification_runs(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("capstone-certification")
    first_directory = root / "without-optional-environment"
    second_directory = root / "with-optional-environment"
    first = _run_certification(first_directory, optional_environment=False)
    second = _run_certification(second_directory, optional_environment=True)
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    return first_directory, second_directory, first, second


def test_certification_artifact_records_all_mandatory_gates(certification_runs) -> None:
    artifact_directory, _, _, _ = certification_runs
    payload = json.loads((artifact_directory / "certification.json").read_text())

    assert payload["schema_version"] == 1
    assert payload["offline_release_passed"] is True
    assert set(payload) == {
        "schema_version",
        "offline_release_passed",
        "reference_mission",
        "streamlit",
        "student",
        "mlflow",
        "repository",
        "optional_providers",
    }
    assert payload["reference_mission"] == {
        "status": "completed",
        "provider": "recorded",
        "data_mode": "certified",
        "bounded_replan": True,
        "replan_count": 1,
        "observation_count": 5,
        "companies_evidenced": ["NVIDIA", "Schneider Electric"],
        "citation_pairs_valid": True,
        "citation_integrity": 1.0,
        "metric_names": _METRIC_NAMES,
        "metric_values": [1.0, 1.0, 1.0, 1.0, 1.0],
        "release_passed": True,
    }
    assert payload["streamlit"]["app_test_passed"] is True
    assert payload["streamlit"]["reference_journey_passed"] is True
    assert payload["student"]["starter_launches"] is True
    assert payload["student"]["incomplete_groups"] == _INCOMPLETE_GROUPS
    assert payload["student"]["incomplete_group_count"] == 4
    assert payload["student"]["solved_exit_zero"] is True
    assert payload["student"]["solved_marker_count"] == 1
    assert payload["mlflow"]["persisted"] is True
    assert payload["mlflow"]["run_present"] is True
    assert payload["mlflow"]["trace_present"] is True
    assert payload["mlflow"]["public_artifacts_validated"] is True
    assert payload["repository"]["public_artifact_scan_passed"] is True


def test_certification_is_stable_and_ignores_optional_provider_environment(
    certification_runs,
) -> None:
    first_directory, second_directory, _, _ = certification_runs
    first_json = (first_directory / "certification.json").read_bytes()
    second_json = (second_directory / "certification.json").read_bytes()
    first_readiness = (first_directory / "readiness.md").read_bytes()
    second_readiness = (second_directory / "readiness.md").read_bytes()
    payload = json.loads(first_json)

    assert first_json == second_json
    assert first_readiness == second_readiness
    assert payload["offline_release_passed"] is True
    assert set(payload["optional_providers"].values()) <= _OPTIONAL_STATUSES
    assert payload["optional_providers"] == {
        "openai": "NOT RUN",
        "ollama": "NOT RUN",
        "tavily": "NOT RUN",
        "timed_classroom_rehearsal": "NOT RUN",
    }


def test_public_certification_outputs_contain_no_secrets_or_personal_paths(
    certification_runs,
) -> None:
    first_directory, second_directory, first, second = certification_runs
    public_text = "\n".join(
        [
            (first_directory / "certification.json").read_text(),
            (first_directory / "readiness.md").read_text(),
            (second_directory / "certification.json").read_text(),
            (second_directory / "readiness.md").read_text(),
            first.stdout,
            first.stderr,
            second.stdout,
            second.stderr,
        ]
    )

    for forbidden in (
        "/Users/",
        "/home/",
        "arnauddemes",
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "sk-test-value",
        "tvly-test-value",
        "Authorization: Bearer",
    ):
        assert forbidden not in public_text


def test_visual_evidence_truthfully_records_missing_four_state_capture(
    certification_runs,
) -> None:
    artifact_directory, _, _, _ = certification_runs
    payload = json.loads((artifact_directory / "certification.json").read_text())
    visual = payload["streamlit"]["visual_evidence"]
    assert visual["status"] == "NOT RUN"
    assert visual["captures"] == []
    assert visual["limitation"]


@pytest.fixture(scope="module")
def certification_module():
    spec = importlib.util.spec_from_file_location("capstone_certification", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_visual_evidence(destination: Path) -> dict[str, object]:
    source = _PROJECT_ROOT / "artifacts" / "capstone" / "reference-mission.png"
    destination.mkdir()
    checks = [
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
    ]
    captures = []
    for index in range(4):
        filename = f"copied-browser-fixture-{index}.png"
        shutil.copyfile(source, destination / filename)
        captures.append(
            {
                "file": filename,
                "sha256": hashlib.sha256((destination / filename).read_bytes()).hexdigest(),
                "width": 1440,
                "height": 1000,
                "browser": "Codex in-app Browser",
                "route": "recorded/reference/test-fixture",
                "state": f"validator-fixture-{index}",
                "captured_at": "2026-08-24T10:00:00+02:00",
                "visible_elements": checks if index == 0 else [],
            }
        )
    manifest = {
        "schema_version": 2,
        "status": "PASS",
        "captures": captures,
        "limitation": "Validator fixture only.",
    }
    (destination / "visual-inspection.json").write_text(json.dumps(manifest))
    return manifest


def test_visual_pass_path_fully_decodes_copied_bound_images(
    certification_module, tmp_path: Path
) -> None:
    directory = tmp_path / "pass"
    _copy_visual_evidence(directory)
    result = certification_module.validate_visual_evidence(directory)
    assert result["status"] == "PASS"
    assert len(result["captures"]) == 4


def test_visual_evidence_rejects_tampered_hash(certification_module, tmp_path: Path) -> None:
    manifest = _copy_visual_evidence(tmp_path / "hash")
    manifest["captures"][0]["sha256"] = "0" * 64
    (tmp_path / "hash" / "visual-inspection.json").write_text(json.dumps(manifest))

    with pytest.raises(certification_module.CertificationFailure):
        certification_module.validate_visual_evidence(tmp_path / "hash")


def test_visual_evidence_rejects_truncated_png(certification_module, tmp_path: Path) -> None:
    manifest = _copy_visual_evidence(tmp_path / "truncated")
    image = tmp_path / "truncated" / manifest["captures"][0]["file"]
    image.write_bytes(image.read_bytes()[:100])
    manifest["captures"][0]["sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    (tmp_path / "truncated" / "visual-inspection.json").write_text(json.dumps(manifest))

    with pytest.raises((OSError, certification_module.CertificationFailure)):
        certification_module.validate_visual_evidence(tmp_path / "truncated")


def test_visual_evidence_rejects_missing_acceptance_coverage(
    certification_module, tmp_path: Path
) -> None:
    manifest = _copy_visual_evidence(tmp_path / "coverage")
    for capture in manifest["captures"]:
        capture["visible_elements"] = [
            item for item in capture["visible_elements"] if item != "exact_footer"
        ]
    (tmp_path / "coverage" / "visual-inspection.json").write_text(json.dumps(manifest))

    with pytest.raises(certification_module.CertificationFailure):
        certification_module.validate_visual_evidence(tmp_path / "coverage")


def test_committed_certification_artifacts_match_the_contract() -> None:
    artifact_directory = _PROJECT_ROOT / "artifacts" / "capstone"
    payload = json.loads((artifact_directory / "certification.json").read_text())
    manifest = json.loads((artifact_directory / "visual-inspection.json").read_text())

    assert payload["schema_version"] == 1
    assert payload["offline_release_passed"] is True
    assert payload["reference_mission"]["citation_integrity"] == 1.0
    assert payload["streamlit"]["app_test_passed"] is True
    assert payload["streamlit"]["visual_evidence"]["status"] == "NOT RUN"
    assert payload["student"]["solved_marker_count"] == 1
    assert payload["mlflow"]["persisted"] is True
    assert manifest["captures"] == []
