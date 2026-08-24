from __future__ import annotations

import importlib
import json
import sqlite3
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from finai_academy.capstone.models import ResearchRequest
from finai_academy.capstone.persistence import (
    CapstoneRunStore,
    audit_capstone_mlflow,
    iter_mlflow_text_values,
)
from finai_academy.capstone.service import build_reference_copilot


def completed_result():
    return build_reference_copilot(run_id_factory=lambda: "analysis-run-001").run(
        ResearchRequest.reference()
    )


def test_persisted_run_and_trace_share_the_analysis_identity(tmp_path: Path) -> None:
    result = completed_result()

    references = CapstoneRunStore(tmp_path / "capstone-mlflow").persist(result)

    assert references.tracking_status == "persisted"
    assert references.run_id
    assert references.trace_id
    assert references.analysis_run_id == result.run_id
    assert references.tracking_uri.startswith("sqlite:")


def test_sqlite_run_contains_metrics_release_trajectory_briefing_and_datasets(
    tmp_path: Path,
) -> None:
    tracking_directory = tmp_path / "capstone-mlflow"
    result = completed_result()
    references = CapstoneRunStore(tracking_directory).persist(result)
    internal_uri = f"sqlite:///{(tracking_directory / 'mlflow.db').resolve()}"
    client = MlflowClient(tracking_uri=internal_uri)

    run = client.get_run(references.run_id)
    artifacts = {
        path.relative_to(
            tracking_directory / "artifacts" / references.run_id / "artifacts"
        ).as_posix()
        for path in (
            tracking_directory / "artifacts" / references.run_id / "artifacts" / "evidence"
        ).glob("*.json")
    }
    traces = mlflow.search_traces(
        run_id=references.run_id,
        locations=[run.info.experiment_id],
        return_type="list",
        flush=True,
    )

    assert set(run.data.metrics) >= {
        "tool_call_correctness",
        "tool_call_efficiency",
        "answer_relevance",
        "answer_completeness",
        "citation_integrity",
        "release_passed",
    }
    assert run.data.params["release_decision"] == "passed"
    assert run.data.params["analysis_run_id"] == result.run_id
    assert artifacts == {
        "evidence/briefing.json",
        "evidence/dataset_identities.json",
        "evidence/trajectory.json",
    }
    assert len(traces) == 1
    root = next(span for span in traces[0].data.spans if span.parent_id is None)
    assert root.inputs["analysis_run_id"] == result.run_id
    assert getattr(traces[0].info, "trace_id", root.trace_id) == references.trace_id


def test_persistence_public_output_and_artifacts_contain_no_paths_or_credentials(
    tmp_path: Path,
) -> None:
    tracking_directory = tmp_path / "Users" / "private" / "capstone-mlflow"
    references = CapstoneRunStore(tracking_directory).persist(completed_result())
    public_payload = references.model_dump_json()
    internal_uri = f"sqlite:///{(tracking_directory / 'mlflow.db').resolve()}"
    client = MlflowClient(tracking_uri=internal_uri)
    run = client.get_run(references.run_id)
    traces = mlflow.search_traces(
        run_id=references.run_id,
        locations=[run.info.experiment_id],
        return_type="list",
        flush=True,
    )
    downloaded = [
        tracking_directory / "artifacts" / references.run_id / "artifacts" / artifact
        for artifact in (
            "evidence/trajectory.json",
            "evidence/briefing.json",
            "evidence/dataset_identities.json",
        )
    ]
    artifact_payload = json.dumps(
        [json.loads(Path(path).read_text(encoding="utf-8")) for path in downloaded],
        sort_keys=True,
    )
    trace_info = traces[0].to_dict()["info"]
    mlflow_public_metadata = json.dumps(
        {"run_tags": run.data.tags, "trace_info": trace_info},
        sort_keys=True,
    )

    assert "/Users/" not in public_payload
    assert "/Users/" not in artifact_payload
    assert "/Users/" not in mlflow_public_metadata
    assert "OPENAI_API_KEY" not in public_payload
    assert "sk-" not in public_payload
    assert "OPENAI_API_KEY" not in artifact_payload
    assert "sk-" not in artifact_payload
    audit = audit_capstone_mlflow(
        tracking_directory / "mlflow.db", tracking_directory / "artifacts"
    )
    assert audit["sqlite_text_values_scanned"] > 0
    persisted_text = "\n".join(
        value for _, _, value in iter_mlflow_text_values(tracking_directory / "mlflow.db")
    )
    assert "arnauddemes" not in persisted_text.casefold()
    assert "/Users/" not in persisted_text
    database_bytes = (tracking_directory / "mlflow.db").read_bytes()
    assert b"file:///private/" not in database_bytes
    assert b"/private/var/folders/" not in database_bytes


def test_privacy_audit_rejects_private_paths_retained_in_sqlite_pages(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    private_value = "file:///private/var/folders/course-secret/" + "x" * 1024
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA secure_delete = OFF")
        connection.execute("CREATE TABLE metadata (value TEXT)")
        connection.execute("INSERT INTO metadata VALUES (?)", (private_value,))
        connection.commit()
        connection.execute("DELETE FROM metadata")

    assert private_value.encode() in database.read_bytes()
    assert list(iter_mlflow_text_values(database)) == []
    with pytest.raises(ValueError, match="private MLflow database bytes"):
        audit_capstone_mlflow(database, tmp_path / "artifacts")


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("experiments", "artifact_location"),
        ("runs", "artifact_uri"),
        ("tags", "value"),
        ("trace_info", "request_preview"),
        ("trace_request_metadata", "value"),
        ("trace_tags", "value"),
        ("spans", "content"),
    ],
)
def test_privacy_audit_rejects_sensitive_text_in_every_mlflow_domain(
    tmp_path: Path, table: str, column: str
) -> None:
    database = tmp_path / "mlflow.db"
    with sqlite3.connect(database) as connection:
        connection.execute(f'CREATE TABLE "{table}" ("{column}" TEXT)')
        connection.execute(
            f'INSERT INTO "{table}" ("{column}") VALUES (?)',
            ("file:///Users/arnauddemes/private",),
        )

    with pytest.raises(ValueError, match="private MLflow metadata"):
        audit_capstone_mlflow(database, tmp_path / "artifacts")


def test_privacy_audit_rejects_sensitive_artifact_metadata_and_content(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE safe (value TEXT)")
        connection.execute("INSERT INTO safe VALUES ('recorded')")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "metadata.json").write_text(
        '{"credential":"Authorization: Bearer test-secret"}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="private MLflow artifact"):
        audit_capstone_mlflow(database, artifacts)


def test_missing_mlflow_is_typed_unavailable_and_does_not_change_analysis(
    monkeypatch,
    tmp_path: Path,
) -> None:
    result = completed_result()
    before = result.model_dump(mode="json")
    real_import_module = importlib.import_module

    def import_without_mlflow(name: str, package: str | None = None):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("missing optional package at /Users/private")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", import_without_mlflow)

    references = CapstoneRunStore(tmp_path / "capstone-mlflow").persist(result)

    assert references.tracking_status == "unavailable"
    assert references.run_id is None
    assert references.trace_id is None
    assert references.analysis_run_id == result.run_id
    assert result.status == "completed"
    assert result.model_dump(mode="json") == before
    assert "/Users/" not in references.model_dump_json()


@pytest.mark.parametrize("import_error", [OSError("loader failure"), RuntimeError("broken ABI")])
def test_non_importerror_mlflow_import_failures_return_typed_error(
    monkeypatch,
    tmp_path: Path,
    import_error: Exception,
) -> None:
    result = completed_result()
    before = result.model_dump(mode="json")
    real_import_module = importlib.import_module

    def fail_mlflow_import(name: str, package: str | None = None):
        del package
        if name == "mlflow":
            raise import_error
        return real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fail_mlflow_import)

    references = CapstoneRunStore(tmp_path / "capstone-mlflow").persist(result)

    assert references.tracking_status == "error"
    assert references.run_id is None
    assert references.trace_id is None
    assert result.model_dump(mode="json") == before
    assert "loader failure" not in references.model_dump_json()
    assert "broken ABI" not in references.model_dump_json()
