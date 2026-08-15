from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP_CHECK = ROOT / "scripts" / "setup_check.py"


def run_setup_check(*arguments: str, environment: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SETUP_CHECK), *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_offline_setup_check_verifies_core_environment() -> None:
    result = run_setup_check("--offline")

    assert result.returncode == 0
    assert "PASS Python" in result.stdout
    assert "PASS Core imports" in result.stdout
    assert "SKIP Live provider" in result.stdout


def test_openai_setup_check_reports_the_missing_key() -> None:
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment["FINAI_MODEL_PROVIDER"] = "openai"
    environment["FINAI_EMBEDDING_PROVIDER"] = "openai"

    result = run_setup_check("--provider", "openai", environment=environment)

    assert result.returncode == 1
    assert "FAIL OpenAI credentials" in result.stdout
    assert "Set OPENAI_API_KEY" in result.stdout
