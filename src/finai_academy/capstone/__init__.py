"""Domain and application code for the Financial Analyst Copilot."""

from collections.abc import Callable, Mapping

from finai_academy.capstone.briefing import (
    PROMPT_VERSION,
    AnalystBriefService,
    build_analyst_brief_prompt,
)
from finai_academy.capstone.model_gateway import (
    ProviderReadiness,
    create_structured_model,
    provider_readiness,
)
from finai_academy.capstone.models import (
    AnalystBrief,
    AnalystFinding,
    CapstoneBriefing,
    CapstoneEvidenceHit,
    CapstoneProvider,
    CitedFact,
    DataMode,
    DeterministicEvaluation,
    EvidenceGateDecision,
    EvidenceType,
    FindingCategory,
    JudgeEvaluation,
    MetricEvaluation,
    PublicTraceEvent,
    ResearchMode,
    ResearchRequest,
    ResearchRunResult,
    RunStatus,
)
from finai_academy.capstone.persistence import (
    CapstoneRunStore,
    PersistedRunReferences,
)
from finai_academy.capstone.service import (
    FinancialAnalystCopilot,
    build_copilot_for_request,
    build_reference_copilot,
)
from finai_academy.capstone.views import (
    BriefingSectionsView,
    CapstoneRunView,
    CitedFactRowView,
    CompanyEvidenceView,
    EvidenceRowView,
    JudgeView,
    PlanRowView,
    ReadinessView,
    ReleaseView,
    ScoreRowView,
    ToolRowView,
    TraceRowView,
    to_run_view,
)


def render_capstone(
    service_factory: Callable[[ResearchRequest], FinancialAnalystCopilot] | None = None,
    *,
    integration_status: Mapping[str, str] | None = None,
) -> None:
    """Load the optional Streamlit renderer only when the UI is launched."""

    from finai_academy.capstone.streamlit_ui import render_capstone as render

    render(service_factory, integration_status=integration_status)


__all__ = [
    "PROMPT_VERSION",
    "AnalystBrief",
    "AnalystBriefService",
    "AnalystFinding",
    "BriefingSectionsView",
    "CapstoneBriefing",
    "CapstoneEvidenceHit",
    "CapstoneProvider",
    "CapstoneRunStore",
    "CapstoneRunView",
    "CitedFact",
    "CitedFactRowView",
    "CompanyEvidenceView",
    "DataMode",
    "DeterministicEvaluation",
    "EvidenceGateDecision",
    "EvidenceRowView",
    "EvidenceType",
    "FinancialAnalystCopilot",
    "FindingCategory",
    "JudgeEvaluation",
    "JudgeView",
    "MetricEvaluation",
    "PersistedRunReferences",
    "PlanRowView",
    "ProviderReadiness",
    "PublicTraceEvent",
    "ReadinessView",
    "ReleaseView",
    "ResearchMode",
    "ResearchRequest",
    "ResearchRunResult",
    "RunStatus",
    "ScoreRowView",
    "ToolRowView",
    "TraceRowView",
    "build_analyst_brief_prompt",
    "build_copilot_for_request",
    "build_reference_copilot",
    "create_structured_model",
    "provider_readiness",
    "render_capstone",
    "to_run_view",
]
