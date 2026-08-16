from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from finai_academy.measurement import NullStageObserver, RunMeasurement, TokenUsage


def test_token_usage_requires_consistent_non_negative_counts() -> None:
    usage = TokenUsage(input_tokens=10, output_tokens=4, total_tokens=14)

    assert usage.total_tokens == usage.input_tokens + usage.output_tokens


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_tokens", -1),
        ("output_tokens", -1),
        ("total_tokens", -1),
        ("input_tokens", True),
    ],
)
def test_token_usage_rejects_invalid_counts(field: str, value: int) -> None:
    values = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    values[field] = value

    with pytest.raises(ValueError, match="non-negative integers"):
        TokenUsage(**values)


def test_token_usage_rejects_an_inconsistent_total() -> None:
    with pytest.raises(ValueError, match="total_tokens"):
        TokenUsage(input_tokens=10, output_tokens=4, total_tokens=15)


def test_run_measurement_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="duration_ms"):
        RunMeasurement(stage="generate", duration_ms=-0.1)


def test_run_measurement_requires_a_non_empty_stage() -> None:
    with pytest.raises(ValueError, match="stage"):
        RunMeasurement(stage="  ", duration_ms=0.0)


def test_run_measurement_copies_metadata_into_an_immutable_mapping() -> None:
    source = {"status": "completed"}
    measurement = RunMeasurement(stage="generate", duration_ms=2.5, metadata=source)
    source["status"] = "changed"

    assert measurement.metadata == {"status": "completed"}
    with pytest.raises(TypeError):
        measurement.metadata["status"] = "changed"


def test_null_stage_observer_preserves_inputs_without_side_effects() -> None:
    observer = NullStageObserver()

    with observer.span("keyword", inputs={"query": "data center revenue"}) as span:
        assert span.name == "keyword"
        assert span.inputs == {"query": "data center revenue"}

    with pytest.raises(FrozenInstanceError):
        span.name = "dense"
