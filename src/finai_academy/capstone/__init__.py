"""Domain and application code for the Financial Analyst Copilot."""

from finai_academy.capstone.briefing import (
    PROMPT_VERSION,
    AnalystBriefService,
    build_analyst_brief_prompt,
)
from finai_academy.capstone.model_gateway import create_structured_model
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

__all__ = [
    "PROMPT_VERSION",
    "AnalystBrief",
    "AnalystBriefService",
    "AnalystFinding",
    "CapstoneBriefing",
    "CapstoneEvidenceHit",
    "CapstoneProvider",
    "CitedFact",
    "DataMode",
    "DeterministicEvaluation",
    "EvidenceGateDecision",
    "EvidenceType",
    "FindingCategory",
    "JudgeEvaluation",
    "MetricEvaluation",
    "PublicTraceEvent",
    "ResearchMode",
    "ResearchRequest",
    "ResearchRunResult",
    "RunStatus",
    "build_analyst_brief_prompt",
    "create_structured_model",
]
