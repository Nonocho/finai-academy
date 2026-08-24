import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from finai_academy.capstone import (
    CapstoneBriefing,
    CapstoneEvidenceHit,
    CitedFact,
    DeterministicEvaluation,
    EvidenceGateDecision,
    JudgeEvaluation,
    MetricEvaluation,
    ResearchRequest,
    ResearchRunResult,
)


def reference_fixture() -> dict[str, object]:
    fixture_path = Path(__file__).parents[1] / "final-project/shared/reference_mission.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


def complete_evaluation(*, release_passed: bool = True) -> DeterministicEvaluation:
    return DeterministicEvaluation(
        metrics=(
            MetricEvaluation(
                name="tool_call_correctness", value=1.0, rationale="Expected calls match."
            ),
            MetricEvaluation(
                name="tool_call_efficiency", value=1.0, rationale="No redundant calls."
            ),
            MetricEvaluation(name="answer_relevance", value=1.0, rationale="Answers the mission."),
            MetricEvaluation(
                name="answer_completeness", value=1.0, rationale="All sections are present."
            ),
            MetricEvaluation(
                name="citation_integrity", value=1.0, rationale="Citations are supported."
            ),
        ),
        release_passed=release_passed,
    )


def reference_evidence() -> tuple[CapstoneEvidenceHit, CapstoneEvidenceHit]:
    return (
        CapstoneEvidenceHit(
            company="NVIDIA",
            text="NVIDIA reported operating growth.",
            evidence_id="nvidia-fy2026-growth",
            document_id="nvidia-fy2026-results",
            section="Operating results",
            period="FY2026",
            source_reference="NVIDIA FY2026 results",
        ),
        CapstoneEvidenceHit(
            company="Schneider Electric",
            text="Schneider Electric reported operating growth.",
            evidence_id="schneider-fy2025-growth",
            document_id="schneider-fy2025-results",
            section="Financial results",
            period="FY2025",
            source_reference="Schneider Electric FY2025 results",
        ),
    )


def sample_briefing(
    *,
    company_evidence: dict[str, tuple[CapstoneEvidenceHit, ...]] | None = None,
    cited_facts: tuple[CitedFact, ...] | None = None,
) -> CapstoneBriefing:
    nvidia_hit, schneider_hit = reference_evidence()
    facts = cited_facts or (
        CitedFact(
            claim="NVIDIA published an operating-growth statement.",
            company="NVIDIA",
            provenance_kind="document",
            source_reference=nvidia_hit.source_reference,
            evidence_id=nvidia_hit.evidence_id,
        ),
        CitedFact(
            claim="Schneider Electric published an operating-growth statement.",
            company="Schneider Electric",
            provenance_kind="document",
            source_reference=schneider_hit.source_reference,
            evidence_id=schneider_hit.evidence_id,
        ),
    )
    return CapstoneBriefing(
        executive_summary="Both companies have documented operating-growth evidence.",
        cited_facts=facts,
        company_evidence=company_evidence
        or {"NVIDIA": (nvidia_hit,), "Schneider Electric": (schneider_hit,)},
        cross_company_observations=("Direct comparison is limited by different businesses.",),
        interpretation=("The evidence supports a qualified comparison.",),
        limitations=("The reported periods and metrics are not fully comparable.",),
        open_questions=("Which aligned operating metric is most decision-useful?",),
        aggregate_sources=tuple(dict.fromkeys(fact.source_reference for fact in facts)),
    )


def result_payload(
    *,
    status: str = "insufficient_evidence",
    evidence_gate: EvidenceGateDecision | dict[str, object] | None = None,
    briefing: CapstoneBriefing | None = None,
    deterministic_evaluation: DeterministicEvaluation | None = None,
    **overrides: object,
) -> dict[str, object]:
    payload = {
        "run_id": "reference-run-001",
        "request": ResearchRequest.reference(),
        "provider": "recorded",
        "model": "recorded-capstone-v1",
        "data_mode": "certified",
        "status": status,
        "initial_plan": (),
        "final_plan": (),
        "observations": (),
        "trajectory": (),
        "evidence_gate": evidence_gate
        if evidence_gate is not None
        else {"passed": False, "coverage": {}, "missing_requirements": ("NVIDIA evidence",)},
        "briefing": briefing,
        "deterministic_evaluation": deterministic_evaluation or complete_evaluation(),
        "judge_evaluation": None,
        "mlflow_run_id": None,
        "mlflow_trace_id": None,
        "replan_count": 0,
        "total_duration_ms": 1.0,
    }
    payload.update(overrides)
    return payload


def test_reference_factory_matches_the_versioned_mission_fixture() -> None:
    fixture = reference_fixture()

    request = ResearchRequest.reference()

    assert request.question == fixture["mission"]
    assert request.companies == tuple(fixture["companies"])
    assert request.provider == fixture["provider"]
    assert request.data_mode == fixture["data_mode"]
    assert request.max_steps == fixture["max_steps"]
    assert request.max_replans == fixture["max_replans"]
    assert request.include_news is False


def test_reference_request_locks_company_universe_and_safety_limits() -> None:
    request = ResearchRequest.reference()

    assert request.companies == ("NVIDIA", "Schneider Electric")
    assert request.max_steps == 6
    assert request.max_replans == 1

    with pytest.raises(ValidationError):
        ResearchRequest.reference(max_steps=7)
    with pytest.raises(ValidationError):
        ResearchRequest.reference(max_replans=2)
    with pytest.raises(ValidationError):
        ResearchRequest.reference(companies=("NVIDIA", "Other"))
    with pytest.raises(ValidationError, match="reference mode requires the fixed mission"):
        ResearchRequest.reference(question="Compare a different company set.")


def test_reference_request_permits_explicit_live_provider_selection() -> None:
    request = ResearchRequest.reference(
        provider="openai",
        model="gpt-5-mini",
        data_mode="live_enrichment",
        include_news=True,
    )

    assert request.question == reference_fixture()["mission"]
    assert request.companies == ("NVIDIA", "Schneider Electric")
    assert request.provider == "openai"
    assert request.data_mode == "live_enrichment"
    assert request.include_news


@pytest.mark.parametrize("question", ["", "   "])
def test_request_rejects_a_blank_question(question: str) -> None:
    with pytest.raises(ValidationError):
        ResearchRequest(
            mode="custom",
            question=question,
            companies=("NVIDIA", "Schneider Electric"),
            provider="recorded",
            model="recorded-capstone-v1",
            data_mode="certified",
        )


def test_request_rejects_news_without_live_enrichment() -> None:
    with pytest.raises(ValidationError, match="live_enrichment"):
        ResearchRequest.reference(include_news=True)


def test_document_fact_requires_an_evidence_id() -> None:
    with pytest.raises(ValidationError, match="evidence_id"):
        CitedFact(
            claim="A document reports operating growth.",
            company="NVIDIA",
            provenance_kind="document",
            source_reference="NVIDIA FY2026 results",
        )


def test_metric_fact_forbids_an_evidence_id() -> None:
    with pytest.raises(ValidationError, match="evidence_id"):
        CitedFact(
            claim="A selected metric is available.",
            company="NVIDIA",
            provenance_kind="metric",
            source_reference="NVIDIA selected metric",
            evidence_id="not-a-document-hit",
        )


def test_completed_result_requires_a_passing_gate_and_briefing() -> None:
    with pytest.raises(ValidationError, match="completed run"):
        ResearchRunResult.model_validate(
            result_payload(status="completed", evidence_gate={"passed": False, "coverage": {}})
        )


def test_failed_evidence_gate_cannot_expose_a_briefing() -> None:
    with pytest.raises(ValidationError, match="failed evidence gate"):
        ResearchRunResult.model_validate(
            result_payload(
                evidence_gate={"passed": False, "coverage": {}}, briefing=sample_briefing()
            )
        )


def test_release_decision_must_match_all_five_metric_values() -> None:
    metrics = list(complete_evaluation().metrics)
    metrics[-1] = MetricEvaluation(
        name="citation_integrity", value=0.0, rationale="A citation is missing."
    )

    with pytest.raises(ValidationError, match="release decision"):
        ResearchRunResult.model_validate(
            result_payload(
                deterministic_evaluation=DeterministicEvaluation(
                    metrics=tuple(metrics), release_passed=True
                )
            )
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("provider", "openai", "provider must match request"),
        ("model", "different-recorded-model", "model must match request"),
        ("data_mode", "live_enrichment", "data_mode must match request"),
    ],
)
def test_run_result_requires_labels_to_match_its_request(
    field_name: str, value: str, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ResearchRunResult.model_validate(result_payload(**{field_name: value}))


def test_completed_reference_run_rejects_document_fact_without_matching_collected_evidence() -> (
    None
):
    nvidia_hit, schneider_hit = reference_evidence()
    briefing = sample_briefing(
        cited_facts=(
            CitedFact(
                claim="NVIDIA published an operating-growth statement.",
                company="NVIDIA",
                provenance_kind="document",
                source_reference="Different NVIDIA source",
                evidence_id=nvidia_hit.evidence_id,
            ),
            CitedFact(
                claim="Schneider Electric published an operating-growth statement.",
                company="Schneider Electric",
                provenance_kind="document",
                source_reference=schneider_hit.source_reference,
                evidence_id=schneider_hit.evidence_id,
            ),
        )
    )
    payload = result_payload(
        status="completed",
        evidence_gate={
            "passed": True,
            "coverage": {"NVIDIA": ("document",), "Schneider Electric": ("document",)},
            "evidence_hits": (nvidia_hit, schneider_hit),
        },
        briefing=briefing,
    )
    with pytest.raises(ValidationError, match="source_reference"):
        ResearchRunResult.model_validate(payload)


def test_completed_reference_run_rejects_unknown_document_evidence_id() -> None:
    nvidia_hit, schneider_hit = reference_evidence()
    briefing = sample_briefing(
        cited_facts=(
            CitedFact(
                claim="NVIDIA published an operating-growth statement.",
                company="NVIDIA",
                provenance_kind="document",
                source_reference=nvidia_hit.source_reference,
                evidence_id="unknown-evidence-id",
            ),
            CitedFact(
                claim="Schneider Electric published an operating-growth statement.",
                company="Schneider Electric",
                provenance_kind="document",
                source_reference=schneider_hit.source_reference,
                evidence_id=schneider_hit.evidence_id,
            ),
        )
    )

    with pytest.raises(ValidationError, match="evidence_id"):
        ResearchRunResult.model_validate(
            result_payload(
                status="completed",
                evidence_gate={
                    "passed": True,
                    "coverage": {
                        "NVIDIA": ("document",),
                        "Schneider Electric": ("document",),
                    },
                    "evidence_hits": (nvidia_hit, schneider_hit),
                },
                briefing=briefing,
            )
        )


def test_completed_reference_run_requires_nonempty_company_evidence_for_both_companies() -> None:
    nvidia_hit, schneider_hit = reference_evidence()
    payload = result_payload(
        status="completed",
        evidence_gate={
            "passed": True,
            "coverage": {"NVIDIA": ("document",), "Schneider Electric": ("document",)},
            "evidence_hits": (nvidia_hit, schneider_hit),
        },
        briefing=sample_briefing(company_evidence={"NVIDIA": (nvidia_hit,)}),
    )

    with pytest.raises(ValidationError, match="company evidence"):
        ResearchRunResult.model_validate(payload)


def test_judge_evaluation_defaults_to_not_run() -> None:
    assert JudgeEvaluation().status == "not_run"


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "Authorization: Bearer super-secret-token",
        "password=secret-value",
        "token=secret-value",
        "client_secret=secret-value",
        "access_token=secret-value",
        "private_key=secret-value",
        "/Users/analyst/private-notes.txt",
        "file:///Users/analyst/private-notes.txt",
    ],
)
def test_public_contracts_reject_credential_text_and_personal_paths(unsafe_text: str) -> None:
    with pytest.raises(ValidationError, match="public"):
        CitedFact(
            claim=unsafe_text,
            company="NVIDIA",
            provenance_kind="metric",
            source_reference="NVIDIA selected metric",
        )
