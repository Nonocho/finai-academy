"""Explicit-judge boundary for the optional Ragas comparison."""

from __future__ import annotations

import pytest

from finai_academy.ragas_evaluation import (
    RagasEvaluationRow,
    RecordedRagasJudge,
    evaluate_with_ragas,
)


def _rows():
    return (
        RagasEvaluationRow(
            case_id="nvda-fact",
            user_input="Which NVIDIA business generated $193.7 billion?",
            retrieved_contexts=("Data Center revenue was $193.7 billion.",),
            response="Data Center [NVDA-TABLE].",
            reference_answer="Data Center generated $193.7 billion.",
        ),
        RagasEvaluationRow(
            case_id="schneider-margin",
            user_input="What margin did Schneider adjusted EBITA reach?",
            retrieved_contexts=("Adjusted EBITA reached an 18.7% margin.",),
            response="Adjusted EBITA margin was 18.7% [SU-TABLE].",
            reference_answer="The margin was 18.7%.",
        ),
    )


def test_ragas_adapter_never_selects_a_default_judge():
    with pytest.raises(ValueError, match="judge"):
        evaluate_with_ragas(_rows(), judge=None)


def test_offline_mode_returns_recorded_or_skipped_metrics():
    result = evaluate_with_ragas(
        _rows(),
        judge=RecordedRagasJudge(metrics={}),
    )

    assert result.status == "recorded_or_skipped"
    assert result.judge_provider == "recorded"
    assert result.judge_model == "no-live-judge"
    assert result.mean_context_recall is None
    assert result.mean_faithfulness is None


def test_explicit_ragas_metrics_remain_aligned_by_case():
    result = evaluate_with_ragas(
        _rows(),
        judge=RecordedRagasJudge(
            metrics={
                "context_recall": (1.0, 0.5),
                "faithfulness": (0.75, 1.0),
            }
        ),
    )

    assert result.status == "completed"
    assert result.context_recall == {"nvda-fact": 1.0, "schneider-margin": 0.5}
    assert result.faithfulness == {"nvda-fact": 0.75, "schneider-margin": 1.0}
    assert result.mean_context_recall == 0.75
    assert result.mean_faithfulness == 0.875


def test_ragas_rows_use_current_single_turn_dataset_columns():
    captured = {}

    class CapturingJudge:
        provider = "ollama"
        model = "qwen3:8b"

        def evaluate(self, rows):
            captured["rows"] = rows
            return {
                "context_recall": (1.0, 1.0),
                "faithfulness": (1.0, 1.0),
            }

    evaluate_with_ragas(_rows(), judge=CapturingJudge())

    assert set(captured["rows"][0]) == {
        "user_input",
        "retrieved_contexts",
        "response",
        "reference",
    }
