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
