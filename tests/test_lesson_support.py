from __future__ import annotations


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
