"""Optionally download complete official documents referenced by the course manifest."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from finai_academy.documents import load_source_manifest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "assets" / "course-data" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "assets" / "course-data" / "downloads"


def download_sources(manifest: Path, output_dir: Path) -> list[Path]:
    """Download each official source without replacing committed fixtures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for source in load_source_manifest(manifest):
        suffix = Path(urlparse(source.source_url).path).suffix or ".html"
        destination = output_dir / f"{source.source_id}{suffix}"
        request = urllib.request.Request(
            source.source_url,
            headers={"User-Agent": "FinAI Academy classroom material contact: instructor"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            destination.write_bytes(response.read())
        downloaded.append(destination)
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    for path in download_sources(arguments.manifest, arguments.output_dir):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
