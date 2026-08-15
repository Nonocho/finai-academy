"""Check the FinAI Academy environment before a live class."""

from __future__ import annotations

import argparse
import json
import os
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


def check_core_imports() -> CheckResult:
    try:
        import nbformat  # noqa: F401
        import pandas  # noqa: F401
        import pydantic  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as error:
        return CheckResult("FAIL", "Core imports", f"{error}. Run `uv sync --extra dev`.")
    return CheckResult("PASS", "Core imports", "notebooks, data, validation, and retrieval")


def check_openai_credentials() -> CheckResult:
    if not os.getenv("OPENAI_API_KEY"):
        return CheckResult(
            "FAIL",
            "OpenAI credentials",
            "Set OPENAI_API_KEY before selecting the OpenAI provider.",
        )
    return CheckResult("PASS", "OpenAI credentials", "OPENAI_API_KEY is configured")


def check_ollama(settings: Settings) -> CheckResult:
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        with urlopen(url, timeout=3) as response:
            payload = json.load(response)
    except (URLError, TimeoutError, OSError) as error:
        return CheckResult(
            "FAIL",
            "Ollama",
            f"Cannot reach {settings.ollama_base_url}: {error}. Start Ollama and retry.",
        )

    available = {item.get("name") for item in payload.get("models", [])}
    required = {settings.chat_model, settings.embedding_model}
    missing = sorted(required - available)
    if missing:
        commands = " && ".join(f"ollama pull {model}" for model in missing)
        return CheckResult("FAIL", "Ollama models", f"Missing {', '.join(missing)}. Run `{commands}`.")
    return CheckResult("PASS", "Ollama", f"chat={settings.chat_model}, embeddings={settings.embedding_model}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Skip provider reachability")
    parser.add_argument("--provider", choices=("ollama", "openai"))
    args = parser.parse_args()

    if args.provider:
        os.environ["FINAI_MODEL_PROVIDER"] = args.provider
        os.environ["FINAI_EMBEDDING_PROVIDER"] = args.provider
    settings = Settings.from_environment()

    results = [check_python(), check_core_imports()]
    if args.offline:
        results.append(CheckResult("SKIP", "Live provider", "offline setup check requested"))
    elif settings.provider == "openai":
        results.append(check_openai_credentials())
    else:
        results.append(check_ollama(settings))

    for result in results:
        print(result.render())

    if any(result.status == "FAIL" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
