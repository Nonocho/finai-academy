from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets/course-data/manifest.json"


def test_committed_capstone_artifacts_match_manifest_hashes() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    record = manifest["capstone_derived_artifacts"][0]

    for field in ("elements", "chunks", "nvidia_crop", "schneider_crop"):
        artifact = record[field]
        raw = (ROOT / artifact["path"]).read_bytes()
        assert sha256(raw).hexdigest() == artifact["sha256"]

    assert record["source_sha256s"] == [
        "0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c",
        "5b17e6e8e5b8c47f697f5d352ed9639b98b261d9ecf886d3e8ce147493ebb00a",
    ]


def test_committed_chunks_contain_no_personal_paths_or_parser_filenames() -> None:
    text = (ROOT / "assets/course-data/capstone/financial_chunks_v2.json").read_text(
        encoding="utf-8"
    )

    assert "/Users/" not in text
    assert '"filename"' not in text


def test_builder_certifies_complete_document_target_tables() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_capstone_document_assets.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "documents=2 pages=194 nvidia_target_tables=1 schneider_target_tables=3\n"


def test_builder_regenerates_byte_identical_artifacts() -> None:
    paths = (MANIFEST, *(ROOT / json.loads(MANIFEST.read_text())["capstone_derived_artifacts"][0][field]["path"] for field in ("elements", "chunks", "nvidia_crop", "schneider_crop")))
    before = {path: path.read_bytes() for path in paths}
    result = subprocess.run([sys.executable, "scripts/build_capstone_document_assets.py"], cwd=ROOT, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert {path: path.read_bytes() for path in paths} == before
