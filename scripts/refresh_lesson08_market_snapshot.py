"""Refresh the checked-in market snapshot used by Lesson 08."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = ROOT / "assets/course-data/market/lesson08_market_snapshot_v1.json"
MANIFEST_PATH = ROOT / "assets/course-data/manifest.json"
DATASET_ID = "lesson08-market-snapshot-v1"

SOURCE_URLS = {
    "NVDA": "https://finance.yahoo.com/quote/NVDA/history/",
    "SU.PA": "https://finance.yahoo.com/quote/SU.PA/history/",
    "EURUSD=X": "https://finance.yahoo.com/quote/EURUSD%3DX/history/",
}


def _round_half_up(value: float, places: int) -> float:
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def _validated_close(observations: Mapping[str, Mapping[str, Any]], ticker: str) -> float:
    record = observations.get(ticker)
    if record is None:
        raise ValueError(f"Missing observation for {ticker}")
    close = float(record["close"])
    if not math.isfinite(close) or close <= 0:
        raise ValueError(f"Close for {ticker} must be a positive finite number")
    return close


def build_snapshot(observations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Create the versioned course schema from three last-close observations."""

    parsed_dates = {
        ticker: date.fromisoformat(str(observations[ticker]["date"]))
        for ticker in SOURCE_URLS
        if ticker in observations
    }
    if set(parsed_dates) != set(SOURCE_URLS):
        missing = ", ".join(sorted(set(SOURCE_URLS) - set(parsed_dates)))
        raise ValueError(f"Missing observation dates for: {missing}")
    if (max(parsed_dates.values()) - min(parsed_dates.values())).days > 7:
        raise ValueError("Market observations must be within seven calendar days")

    nvda_close = _validated_close(observations, "NVDA")
    schneider_close = _validated_close(observations, "SU.PA")
    eurusd_close = _validated_close(observations, "EURUSD=X")
    return {
        "dataset_id": DATASET_ID,
        "notice": (
            "Checked-in course snapshot; not a live quote or investment recommendation."
        ),
        "prices": {
            "NVDA": {
                "company": "NVIDIA",
                "price": _round_half_up(nvda_close, 4),
                "currency": "USD",
                "as_of": parsed_dates["NVDA"].isoformat(),
                "source": SOURCE_URLS["NVDA"],
            },
            "SU.PA": {
                "company": "Schneider Electric",
                "price": _round_half_up(schneider_close, 4),
                "currency": "EUR",
                "as_of": parsed_dates["SU.PA"].isoformat(),
                "source": SOURCE_URLS["SU.PA"],
            },
        },
        "fx": {
            "USD_EUR": {
                "rate": _round_half_up(1.0 / eurusd_close, 6),
                "as_of": parsed_dates["EURUSD=X"].isoformat(),
                "source": SOURCE_URLS["EURUSD=X"],
            }
        },
    }


def update_manifest(
    manifest: Mapping[str, Any],
    *,
    snapshot_sha256: str,
    retrieval_date: str,
) -> dict[str, Any]:
    """Register the snapshot without disturbing existing provenance entries."""

    updated = deepcopy(dict(manifest))
    entries = [
        entry
        for entry in updated.get("market_datasets", [])
        if entry.get("dataset_id") != DATASET_ID
    ]
    entries.append(
        {
            "dataset_id": DATASET_ID,
            "path": "assets/course-data/market/lesson08_market_snapshot_v1.json",
            "sha256": snapshot_sha256,
            "retrieval_date": retrieval_date,
            "source_urls": [SOURCE_URLS[ticker] for ticker in SOURCE_URLS],
        }
    )
    updated["market_datasets"] = entries
    return updated


def fetch_observations() -> dict[str, dict[str, Any]]:
    """Fetch the last complete daily close for the maintained symbols."""

    try:
        import yfinance as yf
    except ImportError as error:
        raise RuntimeError("Install the finance extra: uv sync --extra finance") from error

    observations: dict[str, dict[str, Any]] = {}
    for ticker in SOURCE_URLS:
        history = yf.Ticker(ticker).history(period="7d", interval="1d", auto_adjust=False)
        closes = history["Close"].dropna()
        if closes.empty:
            raise RuntimeError(f"Yahoo Finance returned no complete closes for {ticker}")
        observations[ticker] = {
            "close": float(closes.iloc[-1]),
            "date": closes.index[-1].date().isoformat(),
        }
    return observations


def main() -> None:
    observations = fetch_observations()
    snapshot = build_snapshot(observations)
    snapshot_text = json.dumps(snapshot, indent=2, sort_keys=True) + "\n"
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(snapshot_text, encoding="utf-8")

    snapshot_hash = hashlib.sha256(snapshot_text.encode("utf-8")).hexdigest()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    retrieval_date = max(record["date"] for record in observations.values())
    updated_manifest = update_manifest(
        manifest,
        snapshot_sha256=snapshot_hash,
        retrieval_date=retrieval_date,
    )
    MANIFEST_PATH.write_text(
        json.dumps(updated_manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {SNAPSHOT_PATH.relative_to(ROOT)} ({snapshot_hash})")


if __name__ == "__main__":
    main()
