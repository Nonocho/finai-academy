"""Safe, serializable presentation models for the capstone interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from finai_academy.capstone.models import (
    CapstoneProvider,
    DataMode,
    ResearchMode,
    ResearchRequest,
    ResearchRunResult,
    RunStatus,
    _FrozenPublicModel,
)
from finai_academy.research_planning import ResearchObservation

_PROVIDER_LABELS = {
    "recorded": "Recorded demo",
    "ollama": "Ollama",
    "openai": "OpenAI",
}
_DATA_MODE_LABELS = {
    "certified": "Certified snapshots",
    "live_enrichment": "Optional live enrichment",
}
_STATUS_LABELS = {
    "completed": "Completed",
    "plan_blocked": "Plan blocked",
    "execution_stopped": "Execution stopped",
    "replan_budget_exhausted": "Replan budget exhausted",
    "insufficient_evidence": "Insufficient evidence",
    "provider_error": "Provider error",
}
_SCORE_LABELS = {
    "tool_call_correctness": "Tool-call correctness",
    "tool_call_efficiency": "Tool-call efficiency",
    "answer_relevance": "Answer relevance",
    "answer_completeness": "Answer completeness",
    "citation_integrity": "Citation integrity",
}


class ReadinessView(_FrozenPublicModel):
    """Compact route and run readiness labels."""

    provider: str
    model: str
    data_mode: str
    run_status: str


class PlanRowView(_FrozenPublicModel):
    """One display-safe final plan row."""

    step: str
    capability: str
    purpose: str
    expected_evidence: str
    depends_on: str


class ToolRowView(_FrozenPublicModel):
    """One display-safe tool attempt without raw arguments or result objects."""

    attempt: str
    capability: str
    company: str
    status: str
    outcome: str
    provenance: str
    duration: str


class CitedFactRowView(_FrozenPublicModel):
    """One factual claim with company and provenance kept distinct."""

    company: str
    provenance: Literal["Document", "Metric"]
    claim: str
    source: str
    evidence_id: str


class EvidenceRowView(_FrozenPublicModel):
    """One collected, source-addressable document passage."""

    company: str
    provenance: Literal["Document"] = "Document"
    period: str
    section: str
    evidence: str
    evidence_id: str
    source: str


class TraceRowView(_FrozenPublicModel):
    """One public trace event with formatted timing and ownership."""

    index: str
    phase: str
    capability: str
    attempt: str
    revision: str
    status: str
    error: str
    duration: str
    failure_owner: str
    summary: str


class ScoreRowView(_FrozenPublicModel):
    """One deterministic metric rendered as a score and rationale."""

    metric: str
    score: str
    rationale: str


class CompanyEvidenceView(_FrozenPublicModel):
    """Public company evidence claims grouped for the briefing."""

    company: str
    claims: tuple[str, ...] = Field(min_length=1)


class BriefingSectionsView(_FrozenPublicModel):
    """The five briefing sections exposed only after a passing evidence gate."""

    executive_briefing: str
    company_evidence: tuple[CompanyEvidenceView, ...] = Field(min_length=1)
    cross_company_comparison: tuple[str, ...] = Field(min_length=1)
    limitations_and_open_questions: tuple[str, ...] = Field(min_length=1)
    sources_and_execution: tuple[str, ...] = Field(min_length=1)


class JudgeView(_FrozenPublicModel):
    """Optional model-judge result, separate from release evaluation."""

    status: str
    summary: str
    score: str


class ReleaseView(_FrozenPublicModel):
    """Deterministic evidence-gate and release decision."""

    evidence_gate: Literal["Evidence gate passed", "Evidence gate failed"]
    decision: Literal["Release passed", "Release blocked"]
    missing_requirements: tuple[str, ...] = ()


class OutcomeView(_FrozenPublicModel):
    """Host-derived public outcome copy and rendering status."""

    status: Literal["passed", "blocked", "error"]
    message: Literal["Release passed", "Release blocked"]
    assistant_message: str


class CapstoneRunView(_FrozenPublicModel):
    """Complete public, JSON-compatible state rendered by Streamlit."""

    run_id: str
    question: str
    companies: tuple[str, ...]
    readiness: ReadinessView
    plan: tuple[PlanRowView, ...]
    tool_activity: tuple[ToolRowView, ...]
    cited_facts: tuple[CitedFactRowView, ...]
    evidence: tuple[EvidenceRowView, ...]
    trace: tuple[TraceRowView, ...]
    scores: tuple[ScoreRowView, ...]
    briefing: BriefingSectionsView | None
    judge: JudgeView
    release: ReleaseView
    outcome: OutcomeView
    replan_count: int = Field(ge=0)
    total_duration: str


def build_capstone_request(
    *,
    mode: ResearchMode | Literal["reference", "custom"],
    question: str | None,
    provider: CapstoneProvider | Literal["recorded", "ollama", "openai"],
    model: str,
    data_mode: DataMode | Literal["certified", "live_enrichment"],
) -> ResearchRequest:
    """Map public route selections to one bounded, validated research request."""

    selected_mode = ResearchMode(mode)
    selected_provider = CapstoneProvider(provider)
    selected_data_mode = DataMode(data_mode)
    include_news = selected_data_mode == DataMode.LIVE_ENRICHMENT
    route = {
        "provider": selected_provider,
        "model": model,
        "data_mode": selected_data_mode,
        "include_news": include_news,
    }
    if selected_mode == ResearchMode.REFERENCE:
        if question is not None:
            raise ValueError("reference mode uses the fixed mission")
        return ResearchRequest.reference(**route)
    if question is None:
        raise ValueError("custom mode requires a question")
    return ResearchRequest(
        mode=selected_mode,
        question=question,
        companies=("NVIDIA", "Schneider Electric"),
        **route,
    )


def to_run_view(result: ResearchRunResult) -> CapstoneRunView:
    """Convert a domain result into a strict display-only view."""

    briefing = result.briefing
    company_claims: dict[str, list[str]] = {company: [] for company in result.request.companies}
    cited_facts: tuple[CitedFactRowView, ...] = ()
    briefing_view: BriefingSectionsView | None = None
    if briefing is not None:
        cited_facts = tuple(
            CitedFactRowView(
                company=fact.company,
                provenance="Document" if fact.provenance_kind == "document" else "Metric",
                claim=fact.claim,
                source=fact.source_reference,
                evidence_id=fact.evidence_id or "Not applicable",
            )
            for fact in briefing.cited_facts
        )
        for fact in briefing.cited_facts:
            company_claims.setdefault(fact.company, []).append(fact.claim)
        limitations = briefing.limitations + tuple(
            f"Open question: {question}" for question in briefing.open_questions
        )
        sources = tuple(f"Source: {source}" for source in briefing.aggregate_sources) + (
            f"Execution: {len(result.observations)} attempts, {result.replan_count} replans.",
        )
        briefing_view = BriefingSectionsView(
            executive_briefing=briefing.executive_summary,
            company_evidence=tuple(
                CompanyEvidenceView(company=company, claims=tuple(company_claims[company]))
                for company in result.request.companies
                if company_claims.get(company)
            ),
            cross_company_comparison=(
                briefing.cross_company_observations + briefing.interpretation
            ),
            limitations_and_open_questions=limitations,
            sources_and_execution=sources,
        )

    judge = result.judge_evaluation
    outcome = _outcome_view(result)
    return CapstoneRunView(
        run_id=result.run_id,
        question=result.request.question,
        companies=result.request.companies,
        readiness=ReadinessView(
            provider=_PROVIDER_LABELS[result.provider.value],
            model=result.model,
            data_mode=_DATA_MODE_LABELS[result.data_mode.value],
            run_status=_STATUS_LABELS[result.status.value],
        ),
        plan=tuple(
            PlanRowView(
                step=str(step.step_id),
                capability=_humanize(step.capability),
                purpose=step.purpose,
                expected_evidence=", ".join(step.expected_evidence) or "Not specified",
                depends_on=", ".join(str(item) for item in step.depends_on) or "None",
            )
            for step in result.final_plan
        ),
        tool_activity=tuple(_tool_row(observation) for observation in result.observations),
        cited_facts=cited_facts,
        evidence=tuple(
            EvidenceRowView(
                company=hit.company,
                period=hit.period,
                section=hit.section,
                evidence=hit.text,
                evidence_id=hit.evidence_id,
                source=hit.source_reference,
            )
            for hit in result.evidence_gate.evidence_hits
        ),
        trace=tuple(
            TraceRowView(
                index=str(event.index),
                phase=_humanize(event.phase),
                capability=_humanize(event.capability) if event.capability else "Not applicable",
                attempt=str(event.attempt_id) if event.attempt_id is not None else "Not applicable",
                revision=str(event.plan_revision),
                status=_humanize(event.status),
                error=event.error_code or "None",
                duration=_format_duration(event.duration_ms),
                failure_owner=_humanize(event.failure_owner)
                if event.failure_owner
                else "None",
                summary=event.summary,
            )
            for event in result.trajectory
        ),
        scores=tuple(
            ScoreRowView(
                metric=_SCORE_LABELS[metric.name],
                score=f"{metric.value:.0%}",
                rationale=metric.rationale,
            )
            for metric in result.deterministic_evaluation.metrics
        ),
        briefing=briefing_view,
        judge=JudgeView(
            status=_humanize(judge.status) if judge is not None else "Not run",
            summary=judge.summary if judge is not None else "No judge evaluation was run.",
            score=f"{judge.score:.0%}" if judge is not None and judge.score is not None else "Not scored",
        ),
        release=ReleaseView(
            evidence_gate=(
                "Evidence gate passed" if result.evidence_gate.passed else "Evidence gate failed"
            ),
            decision=(
                "Release passed"
                if result.deterministic_evaluation.release_passed
                else "Release blocked"
            ),
            missing_requirements=result.evidence_gate.missing_requirements,
        ),
        outcome=outcome,
        replan_count=result.replan_count,
        total_duration=_format_duration(result.total_duration_ms),
    )


def _outcome_view(result: ResearchRunResult) -> OutcomeView:
    if result.deterministic_evaluation.release_passed:
        return OutcomeView(
            status="passed",
            message="Release passed",
            assistant_message=(
                "The evidence-backed research run completed. Review the public result below."
            ),
        )
    if result.status == RunStatus.PROVIDER_ERROR:
        return OutcomeView(
            status="error",
            message="Release blocked",
            assistant_message="The selected route did not complete. No briefing was released.",
        )
    return OutcomeView(
        status="blocked",
        message="Release blocked",
        assistant_message=(
            "The release checks did not pass. Review the evidence gate and execution trace below."
        ),
    )


def _tool_row(observation: ResearchObservation) -> ToolRowView:
    result = observation.result if isinstance(observation.result, Mapping) else {}
    company = result.get("company")
    if not isinstance(company, str):
        ticker = observation.arguments.get("ticker")
        company = {"NVDA": "NVIDIA", "SU.PA": "Schneider Electric"}.get(
            ticker, "Not available"
        )
    outcome = _tool_outcome(observation, result)
    return ToolRowView(
        attempt=str(observation.attempt_id),
        capability=_humanize(observation.capability),
        company=company,
        status=_humanize(observation.status),
        outcome=outcome,
        provenance=", ".join(dict.fromkeys(observation.source_references)) or "None",
        duration=_format_duration(observation.duration_ms),
    )


def _tool_outcome(observation: ResearchObservation, result: Mapping[str, object]) -> str:
    if observation.status != "ok":
        return f"Typed error: {observation.error_code or 'unknown_error'}"
    if observation.capability == "get_company_metric":
        value = result.get("value")
        unit = result.get("unit")
        as_of = result.get("as_of")
        if isinstance(value, int | float) and not isinstance(value, bool):
            return f"{_format_unit(value, unit)} as of {as_of}"
    hits = result.get("hits")
    if isinstance(hits, list | tuple):
        return f"{len(hits)} document passages"
    return "Completed"


def _format_unit(value: float, unit: object) -> str:
    number = f"{value:,.2f}".rstrip("0").rstrip(".")
    if unit == "x":
        return f"{number}×"
    if unit == "%":
        return f"{number}%"
    if isinstance(unit, str):
        return f"{number} {unit}"
    return number


def _format_duration(duration_ms: float) -> str:
    if duration_ms == 0:
        return "0.0 ms"
    if duration_ms < 0.1:
        return "<0.1 ms"
    return f"{duration_ms:,.1f} ms"


def _humanize(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


__all__ = [
    "BriefingSectionsView",
    "CapstoneRunView",
    "CitedFactRowView",
    "CompanyEvidenceView",
    "EvidenceRowView",
    "JudgeView",
    "OutcomeView",
    "PlanRowView",
    "ReadinessView",
    "ReleaseView",
    "ScoreRowView",
    "ToolRowView",
    "TraceRowView",
    "build_capstone_request",
    "to_run_view",
]
