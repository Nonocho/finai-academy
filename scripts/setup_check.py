"""Check the FinAI Academy environment before a live class."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finai_academy.settings import Settings


@dataclass(frozen=True)
class CheckResult:
    status: str
    name: str
    detail: str

    def render(self) -> str:
        return f"{self.status} {self.name} — {self.detail}"


def check_python() -> CheckResult:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return CheckResult("PASS", "Python", version)


def check_dependencies() -> CheckResult:
    try:
        import nbformat  # noqa: F401
        import pandas  # noqa: F401
        import pydantic  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as error:
        command = "uv sync --extra ai --extra rag --extra evaluation --extra dev"
        return CheckResult("FAIL", "Dependencies", f"{error}. Run `{command}`.")
    return CheckResult("PASS", "Dependencies", "notebooks, data, validation, and retrieval")


def check_openai_credentials() -> CheckResult:
    if not os.getenv("OPENAI_API_KEY"):
        return CheckResult(
            "FAIL",
            "OpenAI",
            "Set OPENAI_API_KEY before selecting the OpenAI provider.",
        )
    return CheckResult("PASS", "OpenAI", "OPENAI_API_KEY is configured")


def check_ollama(settings: Settings) -> list[CheckResult]:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        with urlopen(url, timeout=3) as response:
            payload = json.load(response)
    except (URLError, TimeoutError, OSError) as error:
        return [
            CheckResult(
                "FAIL",
                "Ollama service",
                f"Cannot reach {settings.ollama_base_url}: {error}. Start Ollama and retry.",
            )
        ]

    available = {
        model_name
        for item in payload.get("models", [])
        for model_name in (item.get("name"), item.get("model"))
        if model_name
    }

    def model_result(name: str, model: str) -> CheckResult:
        if model in available:
            return CheckResult("PASS", name, model)
        return CheckResult("FAIL", name, f"Missing {model}. Run `ollama pull {model}`.")

    return [
        CheckResult("PASS", "Ollama service", settings.ollama_base_url),
        model_result("Chat model", settings.chat_model),
        model_result("Embedding model", settings.embedding_model),
    ]


def check_docker() -> CheckResult:
    executable = shutil.which("docker")
    if executable is None:
        return CheckResult("OPTIONAL", "Docker", "Not required for Day 1.")
    return CheckResult("PASS", "Docker", executable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Skip provider reachability")
    parser.add_argument("--provider", choices=("ollama", "openai"))
    args = parser.parse_args()

    if args.provider:
        os.environ["FINAI_MODEL_PROVIDER"] = args.provider
        os.environ["FINAI_EMBEDDING_PROVIDER"] = args.provider
    settings = Settings.from_environment()

    results = [check_python(), check_dependencies()]
    if args.offline:
        results.extend(
            [
                CheckResult("OPTIONAL", "Ollama", "Skipped by offline setup check."),
                CheckResult("OPTIONAL", "OpenAI", "Not required for Day 1."),
            ]
        )
    elif settings.provider == "openai":
        results.extend(
            [
                CheckResult("OPTIONAL", "Ollama", "Not selected."),
                check_openai_credentials(),
            ]
        )
    else:
        results.extend(check_ollama(settings))
        results.append(CheckResult("OPTIONAL", "OpenAI", "Not required for Day 1."))

    results.append(check_docker())
    failed = any(result.status == "FAIL" for result in results)
    results.append(
        CheckResult(
            "NOT READY" if failed else "READY",
            "Course readiness",
            "Resolve failed checks before class." if failed else "Environment is ready for Day 1.",
        )
    )

    for result in results:
        print(result.render())

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
