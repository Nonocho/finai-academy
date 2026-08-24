"""Optional local MLflow persistence for public capstone evidence."""

from __future__ import annotations

import importlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from finai_academy.capstone.models import ResearchRunResult, _clean_public_value

_PUBLIC_TRACKING_URI = "sqlite:///mlflow.db"
_EXPERIMENT_NAME = "financial-analyst-copilot-capstone"
_NEUTRAL_USER = "local-capstone-user"
_PRIVATE_PATTERN = re.compile(
    r"(?i)(?:arnauddemes|file:/+(?:Users|home|private)/|"
    r"/(?:Users|home|private)/[^\s\"']+|"
    r"[a-z]:[\\/]+Users[\\/]+[^\s\"']+|authorization\s*:\s*(?:bearer|basic)|"
    r"\b(?:openai|tavily)?[_-]?api[_-]?key\b\s*[:=]|\bsk-[a-z0-9_-]{8,}\b|"
    r"client[_-]?secret|private[_-]?key)"
)


class PersistedRunReferences(BaseModel):
    """Path-free references suitable for UI state and certification output."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    run_id: str | None = None
    trace_id: str | None = None
    analysis_run_id: str = Field(min_length=1)
    tracking_status: Literal["persisted", "unavailable", "error"]
    tracking_uri: str = Field(min_length=1)
    guidance: str = Field(min_length=1)


class CapstoneRunStore:
    """Persist one immutable analysis result to an injected local SQLite store."""

    def __init__(self, tracking_directory: Path) -> None:
        self._tracking_directory = Path(tracking_directory)

    def persist(self, result: ResearchRunResult) -> PersistedRunReferences:
        """Persist public evidence or return a typed optional-dependency state."""

        try:
            mlflow = importlib.import_module("mlflow")
            helper = importlib.import_module("finai_academy.mlflow_agent_evaluation")
            store = helper.initialize_local_mlflow(self._tracking_directory)
            _prepare_experiment_storage(store.database_path, store.artifact_directory)
            client_class = importlib.import_module("mlflow.tracking").MlflowClient
            client = client_class(tracking_uri=store.tracking_uri)
            experiment = client.get_experiment_by_name(_EXPERIMENT_NAME)
            if experiment is None:
                experiment_id = client.create_experiment(
                    _EXPERIMENT_NAME,
                    artifact_location=store.artifact_directory.as_uri(),
                )
            else:
                experiment_id = experiment.experiment_id

            parameters = _parameters(result)
            metrics = _metrics(result)
            trajectory = _trajectory_payload(result)
            briefing = _briefing_payload(result)
            datasets = _certified_dataset_identities()
            for payload in (parameters, metrics, trajectory, datasets):
                _clean_public_value(payload)
            if briefing is not None:
                _clean_public_value(briefing)

            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f"capstone-{result.provider}",
                tags={
                    "mlflow.user": "local-capstone-user",
                    "mlflow.source.name": "finai-academy-capstone",
                    "mlflow.source.type": "LOCAL",
                },
            ) as active_run:
                run_id = active_run.info.run_id
                mlflow.log_params(parameters)
                mlflow.log_metrics(metrics)
                mlflow.log_dict(trajectory, "evidence/trajectory.json")
                mlflow.log_dict(datasets, "evidence/dataset_identities.json")
                if briefing is not None:
                    mlflow.log_dict(briefing, "evidence/briefing.json")
                with mlflow.start_span(
                    name="capstone-analysis",
                    span_type="CHAIN",
                    run_id=run_id,
                    attributes={"analysis_run_id": result.run_id},
                ) as root_span:
                    mlflow.update_current_trace(
                        user="local-capstone-user",
                        metadata={
                            "mlflow.source.name": "finai-academy-capstone",
                            "mlflow.source.type": "LOCAL",
                            "mlflow.source.git.repoURL": "not-recorded",
                            "mlflow.source.git.branch": "not-recorded",
                            "mlflow.source.git.commit": "not-recorded",
                        },
                    )
                    root_span.set_inputs(
                        {
                            "analysis_run_id": result.run_id,
                            "mission_id": parameters["mission_id"],
                            "provider": result.provider.value,
                            "model": result.model,
                            "data_mode": result.data_mode.value,
                        }
                    )
                    root_span.set_outputs(
                        {
                            "analysis_run_id": result.run_id,
                            "final_status": result.status.value,
                            "release_decision": parameters["release_decision"],
                        }
                    )
                    trace_id = root_span.trace_id
                mlflow.flush_trace_async_logging()

            if not run_id or not trace_id:
                raise RuntimeError("MLflow did not return stable identifiers")
            sanitize_capstone_mlflow(
                store.database_path,
                run_id=run_id,
                trace_id=trace_id,
            )
            audit_capstone_mlflow(store.database_path, store.artifact_directory)
            return self._references(
                result,
                run_id=run_id,
                trace_id=trace_id,
                tracking_status="persisted",
                guidance="MLflow evidence was persisted locally.",
            )
        except ImportError:
            return self._references(
                result,
                tracking_status="unavailable",
                guidance="Install the evaluation extra to persist MLflow evidence.",
            )
        except Exception:  # noqa: BLE001 - persistence errors are optional and sanitized
            return self._references(
                result,
                tracking_status="error",
                guidance="MLflow evidence persistence failed; analysis output is unchanged.",
            )

    @staticmethod
    def _references(
        result: ResearchRunResult,
        *,
        tracking_status: Literal["persisted", "unavailable", "error"],
        guidance: str,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> PersistedRunReferences:
        return PersistedRunReferences(
            run_id=run_id,
            trace_id=trace_id,
            analysis_run_id=result.run_id,
            tracking_status=tracking_status,
            tracking_uri=_PUBLIC_TRACKING_URI,
            guidance=guidance,
        )


def _parameters(result: ResearchRunResult) -> dict[str, str | int]:
    release_passed = result.deterministic_evaluation.release_passed
    return {
        "analysis_run_id": result.run_id,
        "provider": result.provider.value,
        "model": result.model,
        "data_mode": result.data_mode.value,
        "mission_id": _mission_id(result),
        "max_steps": result.request.max_steps,
        "max_replans": result.request.max_replans,
        "final_status": result.status.value,
        "release_decision": "passed" if release_passed else "failed",
    }


def _metrics(result: ResearchRunResult) -> dict[str, float]:
    values = {
        metric.name: float(metric.value) for metric in result.deterministic_evaluation.metrics
    }
    values["release_passed"] = float(result.deterministic_evaluation.release_passed)
    return values


def _mission_id(result: ResearchRunResult) -> str:
    if result.request.mode == "reference":
        fixture = _read_json(_project_root() / "final-project/shared/reference_mission.json")
        return str(fixture["mission_id"])
    return "custom-two-company-v1"


def _trajectory_payload(result: ResearchRunResult) -> dict[str, object]:
    return {
        "analysis_run_id": result.run_id,
        "events": [event.model_dump(mode="json") for event in result.trajectory],
    }


def _briefing_payload(result: ResearchRunResult) -> dict[str, object] | None:
    if result.briefing is None:
        return None
    return {
        "analysis_run_id": result.run_id,
        "briefing": result.briefing.model_dump(mode="json"),
    }


def _certified_dataset_identities() -> dict[str, object]:
    manifest = _read_json(_project_root() / "assets/course-data/manifest.json")
    identities: list[dict[str, str]] = []
    for source in manifest["sources"]:
        identities.append(
            {
                "kind": "document_source",
                "identity": str(source["source_id"]),
                "sha256": str(source["fixture_sha256"]),
            }
        )
    for market in manifest["market_datasets"]:
        if market["dataset_id"] == "lesson09-metrics-snapshot-v1":
            identities.append(
                {
                    "kind": "market_dataset",
                    "identity": str(market["dataset_id"]),
                    "sha256": str(market["sha256"]),
                }
            )
    for dataset in manifest["mcp_datasets"]:
        identities.append(
            {
                "kind": "mcp_dataset",
                "identity": str(dataset["dataset_id"]),
                "sha256": str(dataset["sha256"]),
            }
        )
    return {"schema_version": 1, "identities": identities}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_capstone_mlflow(
    database_path: Path,
    *,
    run_id: str,
    trace_id: str,
) -> None:
    """Replace MLflow-generated host identity and absolute artifact metadata."""

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA secure_delete = ON")
        connection.execute(
            "UPDATE experiments SET artifact_location = ?",
            ("artifacts/experiments",),
        )
        connection.execute(
            "UPDATE runs SET artifact_uri = 'artifacts/' || run_uuid || '/artifacts', "
            "user_id = ?, source_name = ?",
            (_NEUTRAL_USER, "finai-academy-capstone"),
        )
        connection.execute("UPDATE tags SET value = ? WHERE key = 'mlflow.user'", (_NEUTRAL_USER,))
        connection.execute(
            "UPDATE trace_request_metadata SET value = ? "
            "WHERE key IN ('mlflow.user', 'mlflow.trace.user')",
            (_NEUTRAL_USER,),
        )
        connection.execute(
            "UPDATE trace_tags SET value = 'artifacts/traces/' || request_id || '/artifacts' "
            "WHERE key = 'mlflow.artifactLocation'",
        )
        connection.commit()
        connection.execute("VACUUM")


def _prepare_experiment_storage(database_path: Path, artifact_directory: Path) -> None:
    """Temporarily restore the internal writable location for an existing store."""

    with sqlite3.connect(database_path) as connection:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'experiments'"
        ).fetchone()
        if table_exists is None:
            return
        connection.execute(
            "UPDATE experiments SET artifact_location = ? WHERE name = ?",
            (artifact_directory.as_uri(), _EXPERIMENT_NAME),
        )


def iter_mlflow_text_values(database_path: Path):
    """Yield every persisted SQLite string with its table and column provenance."""

    with sqlite3.connect(database_path) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        for table in tables:
            escaped_table = table.replace('"', '""')
            columns = [
                row[1] for row in connection.execute(f'PRAGMA table_info("{escaped_table}")')
            ]
            if not columns:
                continue
            selected = ", ".join(f'"{column.replace(chr(34), chr(34) * 2)}"' for column in columns)
            for row in connection.execute(f'SELECT {selected} FROM "{escaped_table}"'):
                for column, value in zip(columns, row, strict=True):
                    if isinstance(value, str):
                        yield table, column, value


def audit_capstone_mlflow(database_path: Path, artifact_directory: Path) -> dict[str, int]:
    """Fail closed if SQLite or artifact metadata/content contains private text."""

    sqlite_value_count = 0
    artifact_file_count = 0
    for _table, _column, value in iter_mlflow_text_values(database_path):
        sqlite_value_count += 1
        if _PRIVATE_PATTERN.search(value):
            raise ValueError("private MLflow metadata detected")
    database_text = database_path.read_bytes().decode("utf-8", errors="ignore")
    if _PRIVATE_PATTERN.search(database_text):
        raise ValueError("private MLflow database bytes detected")
    if artifact_directory.exists():
        for path in sorted(item for item in artifact_directory.rglob("*") if item.is_file()):
            artifact_file_count += 1
            relative = path.relative_to(artifact_directory).as_posix()
            if _PRIVATE_PATTERN.search(relative):
                raise ValueError("private MLflow artifact metadata detected")
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if _PRIVATE_PATTERN.search(content):
                raise ValueError("private MLflow artifact content detected")
    return {
        "sqlite_text_values_scanned": sqlite_value_count,
        "artifact_files_scanned": artifact_file_count,
    }
