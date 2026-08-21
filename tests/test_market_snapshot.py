from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "refresh_lesson08_market_snapshot.py"
SPEC = importlib.util.spec_from_file_location("refresh_lesson08_market_snapshot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_snapshot = MODULE.build_snapshot
update_manifest = MODULE.update_manifest


OBSERVATIONS = {
    "NVDA": {"close": 180.12345, "date": "2026-08-20"},
    "SU.PA": {"close": 240.98765, "date": "2026-08-20"},
    "EURUSD=X": {"close": 1.162345, "date": "2026-08-20"},
}


def test_build_snapshot_rounds_prices_and_inverts_eurusd() -> None:
    snapshot = build_snapshot(OBSERVATIONS)

    assert snapshot["prices"]["NVDA"]["price"] == 180.1235
    assert snapshot["prices"]["SU.PA"]["price"] == 240.9877
    assert snapshot["fx"]["USD_EUR"]["rate"] == round(1 / 1.162345, 6)
    assert snapshot["notice"].startswith("Checked-in course snapshot")


def test_build_snapshot_rejects_observations_more_than_seven_days_apart() -> None:
    observations = {key: dict(value) for key, value in OBSERVATIONS.items()}
    observations["SU.PA"]["date"] = "2026-08-10"

    with pytest.raises(ValueError, match="seven calendar days"):
        build_snapshot(observations)


def test_update_manifest_registers_dataset_hash_and_sources() -> None:
    updated = update_manifest(
        {"schema_version": 1, "sources": [], "evaluation_datasets": []},
        snapshot_sha256="a" * 64,
        retrieval_date="2026-08-21",
    )

    entry = updated["market_datasets"][0]
    assert entry["dataset_id"] == "lesson08-market-snapshot-v1"
    assert entry["path"] == "assets/course-data/market/lesson08_market_snapshot_v1.json"
    assert entry["sha256"] == "a" * 64
    assert len(entry["source_urls"]) == 3
