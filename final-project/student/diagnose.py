"""Run and inspect the capstone's bounded MLflow diagnostic exercise."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import warnings
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from finai_academy.capstone import ResearchRequest, build_reference_copilot
from finai_academy.capstone.models import CapstoneEvidenceHit
from finai_academy.capstone.persistence import CapstoneRunStore
from finai_academy.capstone.tools import build_certified_retriever


class _DiagnosticRetriever:
    def __init__(self, drop_company: str | None) -> None:
        self._drop_company = drop_company
        self._wrapped = build_certified_retriever()

    def search(
        self, company: str, query: str, top_k: int = 2
    ) -> tuple[CapstoneEvidenceHit, ...]:
        if company == self._drop_company:
            return ()
        return self._wrapped.search(company, query, top_k)


def _load_case(path: Path) -> str | None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "drop_company"}:
        raise ValueError("invalid diagnostic case")
    if payload["schema_version"] != 1:
        raise ValueError("invalid diagnostic case")
    drop_company = payload["drop_company"]
    if drop_company not in {None, "Schneider Electric"}:
        raise ValueError("invalid diagnostic case")
    return drop_company


def _run(case_path: Path, artifact_directory: Path) -> int:
    drop_company = _load_case(case_path)
    result = build_reference_copilot(
        retriever=_DiagnosticRetriever(drop_company),
        run_id_factory=lambda: "student-diagnostic-analysis",
    ).run(ResearchRequest.reference())
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        references = CapstoneRunStore(artifact_directory / "mlflow").persist(result)
        mlflow.flush_trace_async_logging(terminate=True)
    if references.tracking_status != "persisted" or not references.run_id or not references.trace_id:
        print("DIAGNOSTIC_PERSISTENCE=unavailable")
        return 1
    latest = {
        "run_id": references.run_id,
        "trace_id": references.trace_id,
        "status": result.status.value,
        "release": "passed" if result.deterministic_evaluation.release_passed else "blocked",
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    (artifact_directory / "latest.json").write_text(
        json.dumps(latest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"DIAGNOSTIC_STATUS={latest['status']}")
    print(f"RELEASE={latest['release']}")
    print(f"MLFLOW_RUN_ID={latest['run_id']}")
    print(f"MLFLOW_TRACE_ID={latest['trace_id']}")
    return 0 if latest["release"] == "passed" else 1


def _inspect(artifact_directory: Path) -> int:
    latest: dict[str, Any] = json.loads(
        (artifact_directory / "latest.json").read_text(encoding="utf-8")
    )
    run_id = latest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("invalid diagnostic record")
    tracking_directory = artifact_directory / "mlflow"
    tracking_uri = f"sqlite:///{(tracking_directory / 'mlflow.db').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    traces = mlflow.search_traces(
        run_id=run_id,
        locations=[run.info.experiment_id],
        return_type="list",
        flush=True,
    )
    if len(traces) != 1:
        raise ValueError("diagnostic trace unavailable")
    root = next(span for span in traces[0].data.spans if span.parent_id is None)
    print(f"TRACE_STATUS={root.outputs['final_status']}")
    print(f"FAILURE_OWNER={root.outputs['failure_owner']}")
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "inspect"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "artifacts"
        / "capstone"
        / "student-diagnostic",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    try:
        if arguments.action == "run":
            return _run(Path(__file__).with_name("diagnostic_case.json"), arguments.artifact_dir)
        return _inspect(arguments.artifact_dir)
    except Exception:  # noqa: BLE001 - learner diagnostics remain path-free
        print("DIAGNOSTIC_ERROR")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
