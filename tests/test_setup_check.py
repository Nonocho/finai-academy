from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP_CHECK = ROOT / "scripts" / "setup_check.py"
sys.path.insert(0, str(ROOT))

from finai_academy import Settings
from scripts import setup_check


def run_setup_check(*arguments: str, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SETUP_CHECK), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_offline_setup_check_is_ready_without_external_services() -> None:
    result = run_setup_check("--offline")

    assert result.returncode == 0
    assert "PASS Python" in result.stdout
    assert "PASS Dependencies" in result.stdout
    assert "OPTIONAL Ollama" in result.stdout
    assert "OPTIONAL OpenAI" in result.stdout
    assert "OPTIONAL Docker" in result.stdout or "PASS Docker" in result.stdout
    assert "READY Course readiness" in result.stdout


def test_requested_openai_provider_is_not_ready_without_a_key() -> None:
    environment = os.environ.copy()
    environment["OPENAI_API_KEY"] = ""

    result = run_setup_check("--provider", "openai", environment=environment)

    assert result.returncode == 1
    assert "FAIL OpenAI" in result.stdout
    assert "Set OPENAI_API_KEY" in result.stdout
    assert "NOT READY Course readiness" in result.stdout


def test_missing_docker_is_optional(monkeypatch) -> None:
    monkeypatch.setattr(setup_check.shutil, "which", lambda _: None)

    result = setup_check.check_docker()

    assert result.status == "OPTIONAL"
    assert result.name == "Docker"


def test_ollama_reports_the_service_and_each_required_model(monkeypatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(
                {
                    "models": [
                        {"name": "qwen3:8b"},
                        {"name": "qwen3-embedding:0.6b"},
                    ]
                }
            ).encode()

    monkeypatch.setattr(setup_check, "urlopen", lambda *_args, **_kwargs: Response())

    results = setup_check.check_ollama(Settings())

    assert [(result.status, result.name) for result in results] == [
        ("PASS", "Ollama service"),
        ("PASS", "Chat model"),
        ("PASS", "Embedding model"),
    ]
