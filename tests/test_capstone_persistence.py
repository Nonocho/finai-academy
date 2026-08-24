from __future__ import annotations

import importlib
import json
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from finai_academy.capstone.models import ResearchRequest
from finai_academy.capstone.persistence import CapstoneRunStore
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
        item.path
        for item in client.list_artifacts(references.run_id, path="evidence")
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
        client.download_artifacts(references.run_id, artifact)
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
    trace_info["tags"].pop("mlflow.artifactLocation", None)
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
