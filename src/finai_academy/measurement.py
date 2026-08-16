"""Provider-neutral measurements and stage observation boundaries."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol

MeasurementValue = str | int | float | bool | None


@dataclass(frozen=True)
class TokenUsage:
    """Normalized input, output, and total token counts."""

    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        values = (self.input_tokens, self.output_tokens, self.total_tokens)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise ValueError("token counts must be non-negative integers")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("total_tokens must equal input_tokens + output_tokens")


@dataclass(frozen=True)
class RunMeasurement:
    """One observable application-stage measurement."""

    stage: str
    duration_ms: float
    token_usage: TokenUsage | None = None
    metadata: Mapping[str, MeasurementValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_stage = self.stage.strip()
        if not normalized_stage:
            raise ValueError("stage must not be empty")
        if (
            not isinstance(self.duration_ms, (int, float))
            or isinstance(self.duration_ms, bool)
            or not math.isfinite(self.duration_ms)
            or self.duration_ms < 0
        ):
            raise ValueError("duration_ms must be a finite non-negative number")
        object.__setattr__(self, "stage", normalized_stage)
        object.__setattr__(self, "duration_ms", float(self.duration_ms))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class StageSpan:
    """Provider-neutral description exposed while a stage is running."""

    name: str
    inputs: Mapping[str, MeasurementValue]

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()
        if not normalized_name:
            raise ValueError("span name must not be empty")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))


class StageObserver(Protocol):
    """Context-manager boundary implemented by no-op and tracing observers."""

    def span(
        self,
        name: str,
        *,
        inputs: Mapping[str, MeasurementValue],
    ) -> AbstractContextManager[StageSpan]:
        """Observe one named application stage."""


class NullStageObserver:
    """No-op observer used before tracing is introduced in Lesson 07."""

    @contextmanager
    def span(
        self,
        name: str,
        *,
        inputs: Mapping[str, MeasurementValue],
    ) -> Iterator[StageSpan]:
        yield StageSpan(name=name, inputs=inputs)
