"""Safe, serializable presentation models for the capstone interface."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from finai_academy.capstone.models import (
    CapstoneEvidenceHit,
    CapstoneProvider,
    DataMode,
    ResearchMode,
    ResearchRequest,
    ResearchRunResult,
    RunStatus,
    _FrozenPublicModel,
)
from finai_academy.research_planning import ResearchObservation

_PROVIDER_LABELS = {"recorded": "Recorded demo", "ollama": "Ollama", "openai": "OpenAI"}
_SCORE_LABELS = {
    "tool_call_correctness": "Tool-call correctness",
    "tool_call_efficiency": "Tool-call efficiency",
    "answer_relevance": "Answer relevance",
    "answer_completeness": "Answer completeness",
    "citation_integrity": "Citation integrity",
}
_PIPELINE_STEPS = (
    "Find the relevant official report sections.",
    "Inspect the original tables and their context.",
    "Check that each company has enough supporting evidence.",
    "Write a qualified answer from the selected report evidence.",
    "Check the answer and citations before showing it.",
)


class ReadinessView(_FrozenPublicModel):
    """Legacy route labels retained for downstream compatibility."""

    provider: str
    model: str
    data_mode: str
    run_status: str


class PlanRowView(_FrozenPublicModel):
    """One display-safe final plan row for advanced diagnostics."""

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
    """One report-backed factual claim and its source."""

    company: str
    provenance: Literal["Document", "Metric"]
    claim: str
    source: str
    evidence_id: str


class TraceRowView(_FrozenPublicModel):
    """One public trace event for advanced diagnostics."""

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
    """One deterministic evaluation result."""

    metric: str
    score: str
    rationale: str


class CompanyEvidenceView(_FrozenPublicModel):
    """Learner-facing claims grouped by company."""

    company: str
    claims: tuple[str, ...] = Field(min_length=1)


class EvidenceRowView(_FrozenPublicModel):
    """Compatibility contract for earlier callers of the public view module."""

    company: str
    provenance: Literal["Document"] = "Document"
    period: str
    section: str
    evidence: str
    evidence_id: str
    source: str


class BriefingSectionsView(_FrozenPublicModel):
    """Compatibility contract for earlier callers of the public view module."""

    executive_briefing: str
    company_evidence: tuple[CompanyEvidenceView, ...] = Field(min_length=1)
    cross_company_comparison: tuple[str, ...] = Field(min_length=1)
    limitations_and_open_questions: tuple[str, ...] = Field(min_length=1)
    sources_and_execution: tuple[str, ...] = Field(min_length=1)


class AnswerView(_FrozenPublicModel):
    """The conclusion and its cited company evidence."""

    conclusion: str
    company_evidence: tuple[CompanyEvidenceView, ...] = Field(min_length=1)
    comparison_limits: tuple[str, ...] = Field(min_length=1)
    citations: tuple[CitedFactRowView, ...] = Field(min_length=1)


class EvidenceComparisonView(_FrozenPublicModel):
    """A certified report crop, extracted table, and supporting context."""

    company: str
    page_label: str
    crop_asset_key: str
    extracted_markdown: str
    retrieved_chunk: str
    selection_reason: str
    source_details: tuple[tuple[str, str], ...]


class RetrievalDetailView(_FrozenPublicModel):
    """Rank lineage shown only in advanced diagnostics."""

    company: str
    chunk_id: str
    channel_ranks: tuple[tuple[str, int], ...]
    fused_score: float


class HowItWorkedView(_FrozenPublicModel):
    """Plain-language process summary plus collapsed diagnostic detail."""

    pipeline_steps: tuple[str, ...] = Field(min_length=5, max_length=5)
    retrieval_details: tuple[RetrievalDetailView, ...]
    tool_activity: tuple[ToolRowView, ...]
    trace: tuple[TraceRowView, ...]
    scores: tuple[ScoreRowView, ...]
    model_route: str
    mlflow_run_id: str | None = None
    mlflow_trace_id: str | None = None
    total_duration: str


class JudgeView(_FrozenPublicModel):
    """Optional model-judge result, separate from release evaluation."""

    status: str
    summary: str
    score: str


class ReleaseView(_FrozenPublicModel):
    """Deterministic release decision for safe failure guidance."""

    evidence_gate: Literal["Evidence gate passed", "Evidence gate failed"]
    decision: Literal["Release passed", "Release blocked"]
    missing_requirements: tuple[str, ...] = ()


class OutcomeView(_FrozenPublicModel):
    """Host-derived public outcome copy and rendering status."""

    status: Literal["passed", "blocked", "error"]
    message: Literal["Release passed", "Release blocked"]
    assistant_message: str


class CapstoneRunView(_FrozenPublicModel):
    """Complete public state for the answer-first Streamlit workspace."""

    run_id: str
    question: str
    answer: AnswerView | None
    briefing: BriefingSectionsView | None = None
    evidence: tuple[EvidenceComparisonView, ...]
    how_it_worked: HowItWorkedView
    release: ReleaseView
    outcome: OutcomeView


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
    route = {
        "provider": selected_provider,
        "model": model,
        "data_mode": selected_data_mode,
        "include_news": selected_data_mode == DataMode.LIVE_ENRICHMENT,
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
    """Convert a domain result into a strict, display-only learner view."""

    citations = _citation_rows(result)
    return CapstoneRunView(
        run_id=result.run_id,
        question=result.request.question,
        answer=_answer_view(result, citations),
        briefing=_briefing_sections(result.briefing),
        evidence=tuple(
            _evidence_view(hit)
            for hit in result.evidence_gate.evidence_hits
            if _is_renderable_evidence(hit)
        ),
        how_it_worked=HowItWorkedView(
            pipeline_steps=_PIPELINE_STEPS,
            retrieval_details=tuple(
                RetrievalDetailView(
                    company=hit.company,
                    chunk_id=hit.chunk_id,
                    channel_ranks=hit.channel_ranks,
                    fused_score=hit.fused_score,
                )
                for hit in result.evidence_gate.evidence_hits
            ),
            tool_activity=tuple(_tool_row(observation) for observation in result.observations),
            trace=tuple(_trace_row(event) for event in result.trajectory),
            scores=tuple(
                ScoreRowView(
                    metric=_SCORE_LABELS[metric.name],
                    score=f"{metric.value:.0%}",
                    rationale=metric.rationale,
                )
                for metric in result.deterministic_evaluation.metrics
            ),
            model_route=f"{_PROVIDER_LABELS[result.provider.value]} · {result.model}",
            mlflow_run_id=result.mlflow_run_id,
            mlflow_trace_id=result.mlflow_trace_id,
            total_duration=_format_duration(result.total_duration_ms),
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
        outcome=_outcome_view(result),
    )


def _citation_rows(result: ResearchRunResult) -> tuple[CitedFactRowView, ...]:
    if result.briefing is None:
        return ()
    return tuple(
        CitedFactRowView(
            company=fact.company,
            provenance="Document" if fact.provenance_kind == "document" else "Metric",
            claim=fact.claim,
            source=fact.source_reference,
            evidence_id=fact.evidence_id or "Not applicable",
        )
        for fact in result.briefing.cited_facts
    )


def _briefing_sections(briefing: object) -> BriefingSectionsView | None:
    if briefing is None:
        return None
    if not hasattr(briefing, "executive_summary"):
        return None
    cited_facts = getattr(briefing, "cited_facts", ())
    if not cited_facts:
        return None
    company_claims: dict[str, list[str]] = {}
    for fact in cited_facts:
        company = getattr(fact, "company", None)
        claim = getattr(fact, "claim", None)
        if isinstance(company, str) and isinstance(claim, str):
            company_claims.setdefault(company, []).append(claim)
    if not company_claims:
        return None
    return BriefingSectionsView(
        executive_briefing=briefing.executive_summary,
        company_evidence=tuple(
            CompanyEvidenceView(company=company, claims=tuple(claims))
            for company, claims in company_claims.items()
        ),
        cross_company_comparison=tuple(briefing.cross_company_observations),
        limitations_and_open_questions=tuple(
            briefing.limitations
            + tuple(f"Open question: {question}" for question in briefing.open_questions)
        ),
        sources_and_execution=briefing.aggregate_sources,
    )


def _answer_view(
    result: ResearchRunResult, citations: tuple[CitedFactRowView, ...]
) -> AnswerView | None:
    briefing = result.briefing
    if briefing is None:
        return None
    company_claims: dict[str, list[str]] = {company: [] for company in result.request.companies}
    for fact in briefing.cited_facts:
        company_claims.setdefault(fact.company, []).append(fact.claim)
    return AnswerView(
        conclusion=briefing.executive_summary,
        company_evidence=tuple(
            CompanyEvidenceView(company=company, claims=tuple(company_claims[company]))
            for company in result.request.companies
            if company_claims.get(company)
        ),
        comparison_limits=briefing.limitations
        + tuple(f"Open question: {question}" for question in briefing.open_questions),
        citations=citations,
    )


def _evidence_view(hit: CapstoneEvidenceHit) -> EvidenceComparisonView:
    """Map one certified evidence hit; callers ensure its visual assets exist."""

    assert hit.crop_asset_key is not None and hit.original_markdown is not None
    report = (
        "NVIDIA FY2026 annual report"
        if hit.company == "NVIDIA"
        else "Schneider Electric FY2025 results"
    )
    return EvidenceComparisonView(
        company=hit.company,
        page_label=f"{hit.company} · {hit.period} · page {hit.physical_page}",
        crop_asset_key=hit.crop_asset_key,
        extracted_markdown=hit.original_markdown,
        retrieved_chunk=hit.text,
        selection_reason=hit.selection_reason,
        source_details=(
            ("Report", report),
            ("Section", hit.section),
            ("Reporting period", hit.period),
            ("Page", str(hit.physical_page)),
            ("Unit", hit.unit or "Not stated"),
            ("Source", hit.source_reference),
            ("Document hash", hit.document_sha256),
        ),
    )


def _is_renderable_evidence(hit: CapstoneEvidenceHit) -> bool:
    return hit.crop_asset_key is not None and hit.original_markdown is not None


def _trace_row(event: object) -> TraceRowView:
    return TraceRowView(
        index=str(event.index),
        phase=_humanize(event.phase),
        capability=_humanize(event.capability) if event.capability else "Not applicable",
        attempt=str(event.attempt_id) if event.attempt_id is not None else "Not applicable",
        revision=str(event.plan_revision),
        status=_humanize(event.status),
        error=event.error_code or "None",
        duration=_format_duration(event.duration_ms),
        failure_owner=_humanize(event.failure_owner) if event.failure_owner else "None",
        summary=event.summary,
    )


def _outcome_view(result: ResearchRunResult) -> OutcomeView:
    if result.deterministic_evaluation.release_passed:
        return OutcomeView(
            status="passed",
            message="Release passed",
            assistant_message="The evidence-backed analysis is ready to review.",
        )
    if result.status == RunStatus.PROVIDER_ERROR:
        return OutcomeView(
            status="error",
            message="Release blocked",
            assistant_message="The selected route could not complete the certified analysis.",
        )
    return OutcomeView(
        status="blocked",
        message="Release blocked",
        assistant_message="The reports did not provide enough contextual evidence to release an answer.",
    )


def _tool_row(observation: ResearchObservation) -> ToolRowView:
    result = observation.result if isinstance(observation.result, Mapping) else {}
    company = result.get("company")
    if not isinstance(company, str):
        ticker = observation.arguments.get("ticker")
        company = {"NVDA": "NVIDIA", "SU.PA": "Schneider Electric"}.get(
            ticker, "Not available"
        )
    return ToolRowView(
        attempt=str(observation.attempt_id),
        capability=_humanize(observation.capability),
        company=company,
        status=_humanize(observation.status),
        outcome=_tool_outcome(observation, result),
        provenance=", ".join(dict.fromkeys(observation.source_references)) or "None",
        duration=_format_duration(observation.duration_ms),
    )


def _tool_outcome(observation: ResearchObservation, result: Mapping[str, object]) -> str:
    if observation.status != "ok":
        return f"Typed error: {observation.error_code or 'unknown_error'}"
    hits = result.get("hits")
    if isinstance(hits, list | tuple):
        return f"{len(hits)} document passages"
    return "Completed"


def _format_duration(duration_ms: float) -> str:
    if duration_ms == 0:
        return "0.0 ms"
    if duration_ms < 0.1:
        return "<0.1 ms"
    return f"{duration_ms:,.1f} ms"


def _humanize(value: str) -> str:
    return value.replace("_", " ").strip().capitalize()


__all__ = [
    "AnswerView",
    "BriefingSectionsView",
    "CapstoneRunView",
    "CitedFactRowView",
    "CompanyEvidenceView",
    "EvidenceComparisonView",
    "EvidenceRowView",
    "HowItWorkedView",
    "JudgeView",
    "OutcomeView",
    "PlanRowView",
    "ReadinessView",
    "ReleaseView",
    "RetrievalDetailView",
    "ScoreRowView",
    "ToolRowView",
    "TraceRowView",
    "build_capstone_request",
    "to_run_view",
]
