"""Domain and application code for the Financial Analyst Copilot."""

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

__all__ = [
    "PROMPT_VERSION",
    "AnalystBrief",
    "AnalystBriefService",
    "AnalystFinding",
    "CapstoneBriefing",
    "CapstoneEvidenceHit",
    "CapstoneProvider",
    "CapstoneRunStore",
    "CitedFact",
    "DataMode",
    "DeterministicEvaluation",
    "EvidenceGateDecision",
    "EvidenceType",
    "FinancialAnalystCopilot",
    "FindingCategory",
    "JudgeEvaluation",
    "MetricEvaluation",
    "PersistedRunReferences",
    "ProviderReadiness",
    "PublicTraceEvent",
    "ResearchMode",
    "ResearchRequest",
    "ResearchRunResult",
    "RunStatus",
    "build_analyst_brief_prompt",
    "build_copilot_for_request",
    "build_reference_copilot",
    "create_structured_model",
    "provider_readiness",
]
