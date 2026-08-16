"""Optional Ragas-shaped comparison that never chooses an implicit judge."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from types import MappingProxyType
from typing import Protocol


def _required_text(value: str, *, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


@dataclass(frozen=True)
class RagasEvaluationRow:
    """One single-turn RAG row before conversion to Ragas dataset columns."""

    case_id: str
    user_input: str
    retrieved_contexts: tuple[str, ...]
    response: str
    reference_answer: str

    def __post_init__(self) -> None:
        for field_name in ("case_id", "user_input", "response", "reference_answer"):
            object.__setattr__(
                self,
                field_name,
                _required_text(getattr(self, field_name), field_name=field_name),
            )
        contexts = tuple(context.strip() for context in self.retrieved_contexts)
        if any(not context for context in contexts):
            raise ValueError("retrieved_contexts must not contain empty text")
        object.__setattr__(self, "retrieved_contexts", contexts)


class RagasJudge(Protocol):
    """Explicit provider/model adapter responsible for invoking Ragas metrics."""

    provider: str
    model: str

    def evaluate(
        self,
        rows: Sequence[Mapping[str, object]],
    ) -> Mapping[str, Sequence[float]]: ...


@dataclass(frozen=True)
class RecordedRagasJudge:
    """Offline teaching adapter with recorded metrics or an explicit skipped result."""

    metrics: Mapping[str, Sequence[float]]
    provider: str = "recorded"
    model: str = "no-live-judge"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType(
                {name: tuple(values) for name, values in self.metrics.items()}
            ),
        )

    def evaluate(
        self,
        rows: Sequence[Mapping[str, object]],
    ) -> Mapping[str, Sequence[float]]:
        del rows
        return self.metrics


@dataclass(frozen=True)
class RagasEvaluationResult:
    """Only context recall and faithfulness, aligned to stable case IDs."""

    status: str
    judge_provider: str
    judge_model: str
    context_recall: Mapping[str, float] = field(default_factory=dict)
    faithfulness: Mapping[str, float] = field(default_factory=dict)
    mean_context_recall: float | None = None
    mean_faithfulness: float | None = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "recorded_or_skipped"}:
            raise ValueError("unsupported Ragas evaluation status")
        object.__setattr__(
            self,
            "context_recall",
            MappingProxyType(dict(self.context_recall)),
        )
        object.__setattr__(
            self,
            "faithfulness",
            MappingProxyType(dict(self.faithfulness)),
        )


def _ragas_dataset_rows(
    rows: Sequence[RagasEvaluationRow],
) -> tuple[Mapping[str, object], ...]:
    """Return columns accepted by current Ragas single-turn evaluation datasets."""

    return tuple(
        MappingProxyType(
            {
                "user_input": row.user_input,
                "retrieved_contexts": list(row.retrieved_contexts),
                "response": row.response,
                "reference": row.reference_answer,
            }
        )
        for row in rows
    )


def _validated_metric_values(
    metrics: Mapping[str, Sequence[float]],
    *,
    name: str,
    expected_count: int,
) -> tuple[float, ...]:
    if name not in metrics:
        raise ValueError(f"Ragas judge did not return {name}")
    values = tuple(float(value) for value in metrics[name])
    if len(values) != expected_count:
        raise ValueError(f"Ragas {name} count does not match evaluation rows")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Ragas {name} values must be finite and between 0 and 1")
    return values


def evaluate_with_ragas(
    rows: Sequence[RagasEvaluationRow],
    *,
    judge: RagasJudge | None,
) -> RagasEvaluationResult:
    """Run only context recall and faithfulness with an explicit judge adapter."""

    if judge is None:
        raise ValueError("an explicit Ragas judge is required")
    if not rows:
        raise ValueError("rows must not be empty")
    case_ids = [row.case_id for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("case_id values must be unique")
    provider = _required_text(judge.provider, field_name="judge provider")
    model = _required_text(judge.model, field_name="judge model")
    metrics = judge.evaluate(_ragas_dataset_rows(rows))
    if not metrics:
        return RagasEvaluationResult(
            status="recorded_or_skipped",
            judge_provider=provider,
            judge_model=model,
        )

    context_recall_values = _validated_metric_values(
        metrics,
        name="context_recall",
        expected_count=len(rows),
    )
    faithfulness_values = _validated_metric_values(
        metrics,
        name="faithfulness",
        expected_count=len(rows),
    )
    return RagasEvaluationResult(
        status="completed",
        judge_provider=provider,
        judge_model=model,
        context_recall=dict(zip(case_ids, context_recall_values, strict=True)),
        faithfulness=dict(zip(case_ids, faithfulness_values, strict=True)),
        mean_context_recall=fmean(context_recall_values),
        mean_faithfulness=fmean(faithfulness_values),
    )
