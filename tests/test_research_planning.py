from __future__ import annotations

import pytest
from pydantic import ValidationError

from finai_academy import research_planning
from finai_academy.research_planning import (
    AnalystBriefing,
    CitedFact,
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


def test_plan_rejects_explicit_null_additional_properties_schema() -> None:
    """Breaks if malformed explicit null is treated like an omitted schema keyword."""
    malformed_catalog = (
        tool_catalog()[0].model_copy(
            update={
                "input_schema": {
                    **tool_catalog()[0].input_schema,
                    "additionalProperties": None,
                }
            }
        ),
        tool_catalog()[1],
    )

    with pytest.raises(ValueError, match="invalid additionalProperties"):
        validate_plan(valid_plan(), malformed_catalog, max_steps=6)


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


def test_cited_fact_requires_a_claim_and_non_empty_provenance() -> None:
    """Breaks if a factual claim can exist without a usable source reference."""
    cited_fact_type = getattr(research_planning, "CitedFact", None)
    assert cited_fact_type is not None

    fact = cited_fact_type(
        claim="NVIDIA P/E was 47.2 x as of 2026-08-15.",
        provenance_kind="metric",
        source_references=("NVIDIA metrics snapshot",),
    )
    assert fact.evidence_ids == ()

    for payload in (
        {
            "claim": " ",
            "provenance_kind": "metric",
            "source_references": ("NVIDIA metrics snapshot",),
        },
        {
            "claim": "NVIDIA P/E was 47.2 x.",
            "provenance_kind": "metric",
            "source_references": (),
        },
        {
            "claim": "NVIDIA P/E was 47.2 x.",
            "provenance_kind": "metric",
            "source_references": ("  ",),
        },
        {
            "claim": "NVIDIA Data Center revenue grew 56% year over year.",
            "provenance_kind": "document",
            "source_references": ("NVIDIA public filing",),
            "evidence_ids": (" ",),
        },
    ):
        with pytest.raises(ValidationError):
            cited_fact_type(**payload)


def _fact(
    claim: str = "NVIDIA P/E is available.",
    *,
    source_references: tuple[str, ...] = ("metric-1",),
    evidence_ids: tuple[str, ...] = (),
    provenance_kind: str = "metric",
) -> CitedFact:
    return CitedFact(
        claim=claim,
        provenance_kind=provenance_kind,
        source_references=source_references,
        evidence_ids=evidence_ids,
    )


def test_briefing_requires_facts_limitations_and_exact_aggregate_sources() -> None:
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
            reported_facts=(_fact(),),
            cross_company_observations=(),
            interpretation=(),
            limitations=(),
            source_references=("metric-1",),
        )
    with pytest.raises(ValueError, match="source_references"):
        AnalystBriefing(
            reported_facts=(_fact(),),
            cross_company_observations=(),
            interpretation=(),
            limitations=("Periods differ.",),
            source_references=(),
        )

    with pytest.raises(ValueError, match="exactly match cited facts"):
        AnalystBriefing(
            reported_facts=(_fact(),),
            cross_company_observations=("Different periods.",),
            interpretation=("No advice.",),
            limitations=("Periods differ.",),
            source_references=("metric-1", "unused-source"),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cross_company_observations", (" ",)),
        ("interpretation", ("\t",)),
        ("limitations", ("\n",)),
        ("source_references", ("  ",)),
    ],
)
def test_briefing_rejects_whitespace_only_tuple_entries(
    field: str, value: tuple[str, ...]
) -> None:
    """Breaks if visible report sections can contain blank-looking content."""
    payload = {
        "reported_facts": (_fact(),),
        "cross_company_observations": ("Different periods.",),
        "interpretation": ("No advice.",),
        "limitations": ("Periods differ.",),
        "source_references": ("metric-1",),
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        AnalystBriefing(**payload)


def test_briefing_reported_facts_are_typed_cited_claims() -> None:
    """Breaks if provenance can drift away from the factual claim it supports."""
    briefing = AnalystBriefing(
        reported_facts=(_fact(),),
        cross_company_observations=("Different periods.",),
        interpretation=("No advice.",),
        limitations=("Periods differ.",),
        source_references=("metric-1",),
    )

    assert briefing.reported_facts[0].claim == "NVIDIA P/E is available."
    assert briefing.reported_facts[0].source_references == ("metric-1",)


def test_document_cited_fact_requires_one_exact_source_and_evidence_pair() -> None:
    """Breaks if one fact can cross-pair several sources and evidence IDs."""
    with pytest.raises(ValidationError, match="exactly one source reference and one evidence ID"):
        CitedFact(
            claim="Two document claims were incorrectly merged.",
            provenance_kind="document",
            source_references=("document-a", "document-b"),
            evidence_ids=("doc-1", "doc-2"),
        )

    metric_fact = CitedFact(
        claim="NVIDIA P/E is available.",
        provenance_kind="metric",
        source_references=("metric-source",),
    )
    assert metric_fact.evidence_ids == ()


def test_cited_fact_kind_enforces_metric_and_document_shapes() -> None:
    """Breaks if source-only document provenance can masquerade as a metric fact."""
    with pytest.raises(ValidationError, match="document fact requires exactly one source"):
        CitedFact(
            claim="NVIDIA revenue grew.",
            provenance_kind="document",
            source_references=("document-source",),
        )

    with pytest.raises(ValidationError, match="metric fact requires exactly one source"):
        CitedFact(
            claim="NVIDIA P/E is available.",
            provenance_kind="metric",
            source_references=("metric-source", "second-metric-source"),
        )

    schema = CitedFact.model_json_schema()
    assert schema["properties"]["provenance_kind"]["enum"] == ["metric", "document"]
    assert "provenance_kind" in schema["required"]


def test_validate_briefing_support_rejects_cross_capability_provenance() -> None:
    """Breaks if a fact kind can cite provenance produced by the wrong MCP capability."""
    observations = (
        ResearchObservation(
            attempt_id=1,
            step_id=1,
            plan_revision=0,
            capability="get_company_metric",
            arguments={"ticker": "NVDA", "metric": "P/E"},
            status="ok",
            result={"company": "NVIDIA"},
            source_references=("metric-source",),
            duration_ms=1,
        ),
        ResearchObservation(
            attempt_id=2,
            step_id=2,
            plan_revision=0,
            capability="search_financial_documents",
            arguments={"company": "NVIDIA", "query": "growth"},
            status="ok",
            result={
                "company": "NVIDIA",
                "hits": ({"source": "document-source", "evidence_id": "doc-1"},),
            },
            source_references=("document-source",),
            evidence_ids=("doc-1",),
            duration_ms=1,
        ),
    )
    document_source_as_metric = AnalystBriefing(
        reported_facts=(
            CitedFact(
                claim="NVIDIA revenue grew.",
                provenance_kind="metric",
                source_references=("document-source",),
            ),
        ),
        cross_company_observations=("Different periods.",),
        interpretation=("No advice.",),
        limitations=("Periods differ.",),
        source_references=("document-source",),
    )

    with pytest.raises(ValueError, match="unsupported metric source reference"):
        research_planning.validate_briefing_support(document_source_as_metric, observations)


def test_validate_briefing_support_rejects_unknown_sources_and_evidence_ids() -> None:
    """Breaks if a provider can cite provenance absent from successful observations."""
    observations = (
        ResearchObservation(
            attempt_id=1,
            step_id=1,
            plan_revision=0,
            capability="get_company_metric",
            arguments={"ticker": "NVDA", "metric": "P/E"},
            status="ok",
            result={"company": "NVIDIA"},
            source_references=("metric-source",),
            duration_ms=1,
        ),
        ResearchObservation(
            attempt_id=2,
            step_id=2,
            plan_revision=0,
            capability="search_financial_documents",
            arguments={"company": "NVIDIA", "query": "growth"},
            status="ok",
            result={
                "company": "NVIDIA",
                "hits": (
                    {"source": "document-source", "evidence_id": "doc-1"},
                    {"source": "second-document-source", "evidence_id": "doc-2"},
                ),
            },
            source_references=("document-source", "second-document-source"),
            evidence_ids=("doc-1", "doc-2"),
            duration_ms=1,
        ),
    )
    valid = AnalystBriefing(
        reported_facts=(
            _fact(source_references=("metric-source",)),
            _fact(
                "NVIDIA revenue grew.",
                provenance_kind="document",
                source_references=("document-source",),
                evidence_ids=("doc-1",),
            ),
        ),
        cross_company_observations=("Different periods.",),
        interpretation=("No advice.",),
        limitations=("Periods differ.",),
        source_references=("metric-source", "document-source"),
    )
    validate_support = getattr(research_planning, "validate_briefing_support", None)
    assert validate_support is not None
    assert validate_support(valid, observations) == valid

    unknown_source = AnalystBriefing(
        reported_facts=(_fact(source_references=("invented-source",)),),
        cross_company_observations=("Different periods.",),
        interpretation=("No advice.",),
        limitations=("Periods differ.",),
        source_references=("invented-source",),
    )
    with pytest.raises(ValueError, match="unsupported metric source reference"):
        validate_support(unknown_source, observations)

    unknown_evidence = AnalystBriefing(
        reported_facts=(
            _fact(
                "NVIDIA revenue grew.",
                provenance_kind="document",
                source_references=("document-source",),
                evidence_ids=("invented-evidence",),
            ),
        ),
        cross_company_observations=("Different periods.",),
        interpretation=("No advice.",),
        limitations=("Periods differ.",),
        source_references=("document-source",),
    )
    with pytest.raises(ValueError, match="unsupported document source/evidence pairing"):
        validate_support(unknown_evidence, observations)

    mismatched_pair = AnalystBriefing(
        reported_facts=(
            _fact(
                "NVIDIA revenue grew.",
                provenance_kind="document",
                source_references=("metric-source",),
                evidence_ids=("doc-1",),
            ),
        ),
        cross_company_observations=("Different periods.",),
        interpretation=("No advice.",),
        limitations=("Periods differ.",),
        source_references=("metric-source",),
    )
    with pytest.raises(ValueError, match="unsupported document source/evidence pairing"):
        validate_support(mismatched_pair, observations)

    mismatched_hit_pair = AnalystBriefing(
        reported_facts=(
            _fact(
                "NVIDIA revenue grew.",
                provenance_kind="document",
                source_references=("second-document-source",),
                evidence_ids=("doc-1",),
            ),
        ),
        cross_company_observations=("Different periods.",),
        interpretation=("No advice.",),
        limitations=("Periods differ.",),
        source_references=("second-document-source",),
    )
    with pytest.raises(ValueError, match="unsupported document source/evidence pairing"):
        validate_support(mismatched_hit_pair, observations)


def _observation(
    *,
    company: str,
    capability: str,
    evidence_ids: tuple[str, ...] = (),
    hits: tuple[dict[str, str], ...] = (),
    source_references: tuple[str, ...] | None = None,
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
        source_references=(
            (f"{company} public source",)
            if source_references is None
            else source_references
        ),
        duration_ms=1,
    )


def test_evidence_gate_requires_metric_and_document_evidence_for_both_companies() -> None:
    observations = (
        _observation(company="NVIDIA", capability="get_company_metric", evidence_ids=("m-nvda",)),
        _observation(
            company="NVIDIA",
            capability="search_financial_documents",
            evidence_ids=("d-nvda",),
            hits=({"source": "NVIDIA public source", "evidence_id": "d-nvda"},),
        ),
        _observation(
            company="Schneider Electric", capability="get_company_metric", evidence_ids=("m-su",)
        ),
        _observation(
            company="Schneider Electric",
            capability="search_financial_documents",
            evidence_ids=("d-su",),
            hits=(
                {
                    "source": "Schneider Electric public source",
                    "evidence_id": "d-su",
                },
            ),
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
            company="NVIDIA",
            capability="search_financial_documents",
            hits=({"source": "NVIDIA public source", "evidence_id": "d-nvda"},),
        ),
        _observation(company="Schneider Electric", capability="get_company_metric"),
    )

    gate = evaluate_evidence_gate(observations)

    assert gate.passed is False
    assert "Schneider Electric document evidence" in gate.missing_requirements


def test_evidence_gate_rejects_untraceable_metric_and_document_observations() -> None:
    """Breaks if result shape alone can satisfy coverage without reportable provenance."""
    untraceable = (
        _observation(
            company="NVIDIA",
            capability="get_company_metric",
            source_references=(),
        ),
        _observation(
            company="NVIDIA",
            capability="search_financial_documents",
            evidence_ids=("d-nvda",),
            hits=({"source": "NVIDIA public source", "evidence_id": "d-nvda"},),
            source_references=(),
        ),
        _observation(
            company="Schneider Electric",
            capability="get_company_metric",
            source_references=(),
        ),
        _observation(
            company="Schneider Electric",
            capability="search_financial_documents",
            hits=(
                {
                    "source": "Schneider Electric public source",
                    "evidence_id": "d-su",
                },
            ),
            source_references=(),
        ),
    )

    gate = evaluate_evidence_gate(untraceable)

    assert gate.passed is False
    assert gate.coverage == {"NVIDIA": (), "Schneider Electric": ()}
    assert set(gate.missing_requirements) == {
        "NVIDIA metric evidence",
        "NVIDIA document evidence",
        "Schneider Electric metric evidence",
        "Schneider Electric document evidence",
    }


def test_evidence_gate_rejects_document_provenance_absent_from_returned_hits() -> None:
    """Breaks if non-empty metadata can replace an exact returned citation pair."""
    mismatched = _observation(
        company="NVIDIA",
        capability="search_financial_documents",
        source_references=("declared-source",),
        evidence_ids=("declared-id",),
        hits=({"source": "returned-source", "evidence_id": "returned-id"},),
    )

    gate = evaluate_evidence_gate((mismatched,))

    assert "document" not in gate.coverage["NVIDIA"]
    assert "NVIDIA document evidence" in gate.missing_requirements


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
