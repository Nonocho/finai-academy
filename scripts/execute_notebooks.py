"""Execute course notebooks in offline, Ollama, or OpenAI mode."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


def configure_mode(mode: str, provider: str | None) -> None:
    if mode == "offline":
        os.environ["FINAI_LIVE_MODE"] = "0"
        return

    if provider is None:
        raise ValueError("--provider is required when --mode=live")
    os.environ["FINAI_LIVE_MODE"] = "1"
    os.environ["FINAI_MODEL_PROVIDER"] = provider
    os.environ.setdefault("FINAI_EMBEDDING_PROVIDER", provider)


def execute_notebook(path: Path, output_dir: Path, timeout: int) -> Path:
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / path.name
    nbformat.write(notebook, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--mode", choices=("offline", "live"), default="offline")
    parser.add_argument("--provider", choices=("ollama", "openai"))
    parser.add_argument("--output-dir", type=Path, default=Path(".artifacts/executed-notebooks"))
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    configure_mode(args.mode, args.provider)
    failures: list[str] = []
    for path in args.notebooks:
        try:
            output_path = execute_notebook(path, args.output_dir, args.timeout)
            print(f"PASS {path} -> {output_path}")
        except CellExecutionError as error:
            failures.append(f"FAIL {path}: {error}")

    if failures:
        print("\n".join(failures))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
