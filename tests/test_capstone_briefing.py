from typing import TypeVar

import pytest
from pydantic import BaseModel, ValidationError

from finai_academy.capstone import (
    AnalystBrief,
    AnalystBriefService,
    AnalystFinding,
    EvidenceType,
    FindingCategory,
)

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class FakeStructuredModel:
    def __init__(self, response: AnalystBrief) -> None:
        self.response = response
        self.system_prompt = ""
        self.user_prompt = ""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
    ) -> ResponseT:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return response_model.model_validate(self.response)


def sample_brief() -> AnalystBrief:
    return AnalystBrief(
        company="model-chosen company",
        reporting_period="model-chosen period",
        executive_summary="Demand increased, while supply remained a stated risk.",
        findings=[
            AnalystFinding(
                statement="Management reported stronger demand.",
                category=FindingCategory.KEY_RESULT,
                evidence_type=EvidenceType.MANAGEMENT_CLAIM,
                source_excerpt="we experienced stronger demand",
            )
        ],
        open_questions=["What was the quantified supply impact?"],
    )


def test_reported_fact_requires_a_source_excerpt() -> None:
    with pytest.raises(ValidationError, match="source_excerpt"):
        AnalystFinding(
            statement="Revenue increased.",
            category=FindingCategory.KEY_RESULT,
            evidence_type=EvidenceType.REPORTED_FACT,
        )


def test_interpretation_requires_a_rationale() -> None:
    with pytest.raises(ValidationError, match="rationale"):
        AnalystFinding(
            statement="Growth is concentrated.",
            category=FindingCategory.RISK,
            evidence_type=EvidenceType.INTERPRETATION,
        )


def test_capstone_package_exports_the_structured_model_factory() -> None:
    from finai_academy import capstone

    assert callable(capstone.create_structured_model)


def test_service_generates_validated_brief_and_preserves_trusted_inputs() -> None:
    model = FakeStructuredModel(sample_brief())
    service = AnalystBriefService(model)

    brief = service.generate(
        company="NVIDIA",
        reporting_period="FY2026",
        source_text="Management stated that we experienced stronger demand.",
    )

    assert brief.company == "NVIDIA"
    assert brief.reporting_period == "FY2026"
    assert brief.findings[0].evidence_type == EvidenceType.MANAGEMENT_CLAIM
    assert "<source_document>" in model.user_prompt
    assert "untrusted data" in model.system_prompt


@pytest.mark.parametrize(
    ("company", "period", "source"),
    [
        ("", "FY2026", "source"),
        ("NVIDIA", "", "source"),
        ("NVIDIA", "FY2026", ""),
    ],
)
def test_service_rejects_empty_inputs(company: str, period: str, source: str) -> None:
    service = AnalystBriefService(FakeStructuredModel(sample_brief()))

    with pytest.raises(ValueError):
        service.generate(company=company, reporting_period=period, source_text=source)
