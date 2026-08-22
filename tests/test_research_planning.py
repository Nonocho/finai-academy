from __future__ import annotations

import pytest

from finai_academy.research_planning import (
    AnalystBriefing,
    EvidenceGateResult,
    PlannerToolSpec,
    PlanStep,
    ResearchObservation,
    ResearchPlan,
    evaluate_evidence_gate,
    validate_plan,
    validate_replacement,
)


def tool_catalog() -> tuple[PlannerToolSpec, ...]:
    return (
        PlannerToolSpec(
            name="get_company_metric",
            description="Return one controlled company metric.",
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "metric": {"type": "string"},
                },
                "required": ["ticker", "metric"],
            },
        ),
        PlannerToolSpec(
            name="search_financial_documents",
            description="Search controlled financial evidence.",
            input_schema={
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 3},
                },
                "required": ["company", "query"],
            },
        ),
    )


def valid_plan() -> ResearchPlan:
    return ResearchPlan(
        goal="Compare available NVIDIA and Schneider Electric evidence.",
        steps=(
            PlanStep(
                step_id=1,
                capability="get_company_metric",
                arguments={"ticker": "NVDA", "metric": "P/E"},
                purpose="Collect NVIDIA valuation evidence.",
                expected_evidence=("NVDA P/E",),
            ),
            PlanStep(
                step_id=2,
                capability="get_company_metric",
                arguments={"ticker": "SU.PA", "metric": "P/E"},
                purpose="Collect Schneider Electric valuation evidence.",
                expected_evidence=("SU.PA P/E",),
            ),
        ),
    )


def test_valid_plan_accepts_discovered_allowlisted_tools() -> None:
    checked = validate_plan(valid_plan(), tool_catalog(), max_steps=6)
    assert checked.steps[0].step_id == 1
    assert checked.steps[1].depends_on == ()


def test_plan_rejects_unknown_capability_before_execution() -> None:
    plan = valid_plan().model_copy(
        update={
            "steps": (
                valid_plan().steps[0].model_copy(update={"capability": "delete_portfolio"}),
            )
        }
    )
    with pytest.raises(ValueError, match="capability_not_permitted"):
        validate_plan(plan, tool_catalog(), max_steps=6)


def test_plan_rejects_non_sequential_initial_ids() -> None:
    plan = valid_plan().model_copy(
        update={"steps": (valid_plan().steps[0].model_copy(update={"step_id": 2}),)}
    )
    with pytest.raises(ValueError, match="initial_step_ids"):
        validate_plan(plan, tool_catalog(), max_steps=6)


def test_replacement_requires_new_monotonic_ids() -> None:
    replacement = (
        PlanStep(
            step_id=3,
            capability="search_financial_documents",
            arguments={"company": "NVIDIA", "query": "revenue growth", "top_k": 2},
            purpose="Replace the unsupported revenue metric with document evidence.",
            expected_evidence=("NVIDIA revenue evidence",),
            depends_on=(1,),
        ),
    )
    checked = validate_replacement(
        replacement,
        catalog=tool_catalog(),
        prior_step_ids=(1, 2),
        successful_step_ids=(1,),
        max_total_steps=6,
    )
    assert checked[0].step_id == 3


def test_plan_rejects_missing_required_argument_and_unknown_argument() -> None:
    missing = valid_plan().model_copy(
        update={
            "steps": (
                valid_plan().steps[0].model_copy(update={"arguments": {"ticker": "NVDA"}}),
            )
        }
    )
    with pytest.raises(ValueError, match="arguments_not_accepted"):
        validate_plan(missing, tool_catalog(), max_steps=6)

    strict_catalog = (
        tool_catalog()[0].model_copy(
            update={
                "input_schema": {
                    **tool_catalog()[0].input_schema,
                    "additionalProperties": False,
                }
            }
        ),
        tool_catalog()[1],
    )
    unknown = valid_plan().model_copy(
        update={
            "steps": (
                valid_plan().steps[0].model_copy(
                    update={"arguments": {"ticker": "NVDA", "metric": "P/E", "extra": 1}}
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="arguments_not_accepted"):
        validate_plan(unknown, strict_catalog, max_steps=6)


def test_plan_accepts_schema_valued_additional_properties() -> None:
    schema_catalog = (
        tool_catalog()[0].model_copy(
            update={
                "input_schema": {
                    **tool_catalog()[0].input_schema,
                    "additionalProperties": {"type": "string"},
                }
            }
        ),
        tool_catalog()[1],
    )
    plan = valid_plan().model_copy(
        update={
            "steps": (
                valid_plan().steps[0].model_copy(
                    update={"arguments": {"ticker": "NVDA", "metric": "P/E", "note": "latest"}}
                ),
            )
        }
    )

    checked = validate_plan(plan, schema_catalog, max_steps=6)

    assert checked.steps[0].arguments["note"] == "latest"


def test_plan_rejects_schema_valued_additional_property_with_wrong_type() -> None:
    schema_catalog = (
        tool_catalog()[0].model_copy(
            update={
                "input_schema": {
                    **tool_catalog()[0].input_schema,
                    "additionalProperties": {"type": "string"},
                }
            }
        ),
        tool_catalog()[1],
    )
    plan = valid_plan().model_copy(
        update={
            "steps": (
                valid_plan().steps[0].model_copy(
                    update={"arguments": {"ticker": "NVDA", "metric": "P/E", "note": 7}}
                ),
            )
        }
    )

    with pytest.raises(ValueError, match="arguments_not_accepted"):
        validate_plan(plan, schema_catalog, max_steps=6)


def test_plan_rejects_invalid_schema_types_and_bounds() -> None:
    bad_type = valid_plan().model_copy(
        update={
            "steps": (
                PlanStep(
                    step_id=1,
                    capability="search_financial_documents",
                    arguments={"company": "NVIDIA", "query": "growth", "top_k": True},
                    purpose="Collect document evidence.",
                    expected_evidence=("document",),
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="arguments_not_accepted"):
        validate_plan(bad_type, tool_catalog(), max_steps=6)

    bounded = valid_plan().model_copy(
        update={
            "steps": (
                PlanStep(
                    step_id=1,
                    capability="search_financial_documents",
                    arguments={"company": "NVIDIA", "query": "growth", "top_k": 4},
                    purpose="Collect document evidence.",
                    expected_evidence=("document",),
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="arguments_not_accepted"):
        validate_plan(bounded, tool_catalog(), max_steps=6)


def test_observation_requires_result_for_success_and_code_for_error() -> None:
    with pytest.raises(ValueError, match="ok observation requires result"):
        ResearchObservation(
            attempt_id=1,
            step_id=1,
            plan_revision=0,
            capability="get_company_metric",
            arguments={},
            status="ok",
            duration_ms=0,
        )

    with pytest.raises(ValueError, match="error observation requires error_code"):
        ResearchObservation(
            attempt_id=1,
            step_id=1,
            plan_revision=0,
            capability="get_company_metric",
            arguments={},
            status="error",
            duration_ms=0,
        )


def test_briefing_requires_facts_limitations_and_sources() -> None:
    with pytest.raises(ValueError, match="reported_facts"):
        AnalystBriefing(
            reported_facts=(),
            cross_company_observations=("Different periods.",),
            interpretation=("No advice.",),
            limitations=("Periods differ.",),
            source_references=("metric-1",),
        )
    with pytest.raises(ValueError, match="limitations"):
        AnalystBriefing(
            reported_facts=("NVIDIA P/E is available.",),
            cross_company_observations=(),
            interpretation=(),
            limitations=(),
            source_references=("metric-1",),
        )
    with pytest.raises(ValueError, match="source_references"):
        AnalystBriefing(
            reported_facts=("NVIDIA P/E is available.",),
            cross_company_observations=(),
            interpretation=(),
            limitations=("Periods differ.",),
            source_references=(),
        )


def _observation(
    *, company: str, capability: str, evidence_ids: tuple[str, ...] = (), hits: tuple[str, ...] = ()
) -> ResearchObservation:
    result = {"company": company}
    if capability == "search_financial_documents":
        result["hits"] = hits
    return ResearchObservation(
        attempt_id=1,
        step_id=1,
        plan_revision=0,
        capability=capability,
        arguments={},
        status="ok",
        result=result,
        evidence_ids=evidence_ids,
        duration_ms=1,
    )


def test_evidence_gate_requires_metric_and_document_evidence_for_both_companies() -> None:
    observations = (
        _observation(company="NVIDIA", capability="get_company_metric", evidence_ids=("m-nvda",)),
        _observation(
            company="NVIDIA",
            capability="search_financial_documents",
            evidence_ids=("d-nvda",),
            hits=("hit-1",),
        ),
        _observation(
            company="Schneider Electric", capability="get_company_metric", evidence_ids=("m-su",)
        ),
        _observation(
            company="Schneider Electric",
            capability="search_financial_documents",
            evidence_ids=("d-su",),
            hits=("hit-2",),
        ),
    )

    gate = evaluate_evidence_gate(observations)

    assert gate.passed is True
    assert gate.missing_requirements == ()
    assert gate.coverage == {
        "NVIDIA": ("document", "metric"),
        "Schneider Electric": ("document", "metric"),
    }


def test_evidence_gate_reports_missing_document_evidence() -> None:
    observations = (
        _observation(company="NVIDIA", capability="get_company_metric"),
        _observation(
            company="NVIDIA", capability="search_financial_documents", hits=("hit-1",)
        ),
        _observation(company="Schneider Electric", capability="get_company_metric"),
    )

    gate = evaluate_evidence_gate(observations)

    assert gate.passed is False
    assert "Schneider Electric document evidence" in gate.missing_requirements


def test_evidence_gate_ignores_unsuccessful_observations() -> None:
    failed = ResearchObservation(
        attempt_id=1,
        step_id=1,
        plan_revision=0,
        capability="search_financial_documents",
        arguments={},
        status="error",
        error_code="upstream_error",
        result={"company": "NVIDIA", "hits": ("hit",)},
        duration_ms=1,
    )
    gate = evaluate_evidence_gate((failed,))
    assert gate.passed is False
    assert gate.coverage == {"NVIDIA": (), "Schneider Electric": ()}


def test_evidence_gate_result_is_a_typed_contract() -> None:
    result = EvidenceGateResult(passed=True, coverage={"NVIDIA": ("metric",)})
    assert result.coverage["NVIDIA"] == ("metric",)
