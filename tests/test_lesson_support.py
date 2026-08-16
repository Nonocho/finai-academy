from __future__ import annotations

from itertools import pairwise

import numpy as np

from finai_academy.lesson_support import (
    RecordedContextualChunkingModel,
    compact_manifest_labels,
    normalize_rows,
    spread_label_positions,
)


def test_normalize_rows_scales_nonzero_queries_and_preserves_zero_rows() -> None:
    """Projection inputs need unit query rows without turning zero rows non-finite."""

    normalized = normalize_rows([[3.0, 4.0], [0.0, 0.0]])

    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.array_equal(normalized[1], [0.0, 0.0])


def test_compact_manifest_labels_keep_company_period_and_collision_safe_ids() -> None:
    """Abbreviated visual labels must expand IDs when the compact form collides."""

    passages = [
        type("Passage", (), {
            "company": "NVIDIA",
            "period": "FY2026",
            "passage_id": "NVDA-ONE-CONTEXTUAL-001",
        })(),
        type("Passage", (), {
            "company": "NVIDIA",
            "period": "FY2026",
            "passage_id": "NVDA-TWO-CONTEXTUAL-001",
        })(),
    ]

    labels = compact_manifest_labels(passages)

    assert len(set(labels.values())) == 2
    assert all("NVIDIA · FY2026" in label for label in labels.values())
    assert all(passage.passage_id in labels[passage.passage_id] for passage in passages)


def test_spread_label_positions_is_deterministic_bounded_and_non_overlapping() -> None:
    """Coincident annotations must receive stable non-colliding vertical positions."""

    first = spread_label_positions([50.0, 50.0, 51.0], lower=20.0, upper=100.0, minimum_gap=15.0)
    second = spread_label_positions([50.0, 50.0, 51.0], lower=20.0, upper=100.0, minimum_gap=15.0)

    assert first == second
    assert min(first) >= 20.0 and max(first) <= 100.0
    assert all(right - left >= 15.0 for left, right in pairwise(first))


def test_grounding_rubric_rewards_an_evidence_bounded_nvidia_answer() -> None:
    from finai_academy.lesson_support import evaluate_grounding

    answer = (
        "NVIDIA reported fiscal 2026 revenue of $215.9 billion, up 65% [F1]. "
        "Data Center revenue was $193.7 billion, up 68% [F2]. "
        "The filing does not establish valuation or a price target."
    )

    result = evaluate_grounding(answer)

    assert result.score == 4
    assert result.maximum == 4
    assert result.passed is True


def test_recorded_chat_model_returns_a_citable_grounded_answer() -> None:
    from finai_academy.lesson_support import RecordedChatModel

    model = RecordedChatModel()
    response = model.invoke(
        [
            ("system", "Use only the evidence card."),
            ("human", "Analyse NVIDIA fiscal 2026 using evidence F1 to F4."),
        ]
    )

    assert "NVIDIA" in response.content
    assert "[F1]" in response.content
    assert "[F2]" in response.content
    assert "cannot" in response.content.casefold()


def test_grounding_rubric_rejects_a_ratio_not_present_in_the_evidence() -> None:
    from finai_academy.lesson_support import evaluate_grounding

    answer = (
        "NVIDIA reported fiscal 2026 revenue of $215.9 billion, up 65% [F1]. "
        "Data Center revenue was $193.7 billion, up 68% and represented 89% of revenue [F2]. "
        "The filing does not establish valuation."
    )

    result = evaluate_grounding(answer)

    assert result.passed is False
    assert result.checks["at least two evidence-bounded metrics"] is False


def test_recorded_structured_model_returns_a_valid_financial_brief() -> None:
    from finai_academy.capstone import AnalystBrief, EvidenceType
    from finai_academy.lesson_support import RecordedStructuredModel

    model = RecordedStructuredModel()
    brief = model.generate(
        system_prompt="Use only the supplied evidence.",
        user_prompt="Create an NVIDIA fiscal 2026 analyst brief.",
        response_model=AnalystBrief,
    )

    assert isinstance(brief, AnalystBrief)
    assert brief.company == "NVIDIA"
    assert any(
        finding.evidence_type == EvidenceType.REPORTED_FACT and finding.source_excerpt
        for finding in brief.findings
    )
    assert brief.caveats


def test_recorded_rag_model_answers_only_from_labelled_evidence() -> None:
    from finai_academy.lesson_support import RecordedRagModel

    response = RecordedRagModel().invoke(
        [
            ("system", "Use only retrieved evidence."),
            (
                "human",
                "[NVDA-F2] Data Center revenue reached $193.7 billion, up 68%.",
            ),
        ]
    )

    assert "[NVDA-F2]" in response.content
    assert "193.7" in response.content
    assert "does not establish valuation" in response.content


def test_recorded_contextual_chunking_model_uses_the_stable_chunk_id() -> None:
    import json

    model = RecordedContextualChunkingModel(
        {"NVDA-STRUCTURE-001": "NVIDIA fiscal 2026 revenue context."}
    )

    response = model.invoke(
        [
            ("system", "Return JSON."),
            (
                "human",
                json.dumps(
                    {
                        "chunk_id": "NVDA-STRUCTURE-001",
                        "document": "NVIDIA fiscal 2026 filing.",
                        "chunk": "Revenue was $215.9 billion.",
                    }
                ),
            ),
        ]
    )

    assert json.loads(response.content) == {
        "context": "NVIDIA fiscal 2026 revenue context."
    }
