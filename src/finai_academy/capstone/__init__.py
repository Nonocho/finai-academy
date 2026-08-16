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
    EvidenceType,
    FindingCategory,
)

__all__ = [
    "PROMPT_VERSION",
    "AnalystBrief",
    "AnalystBriefService",
    "AnalystFinding",
    "EvidenceType",
    "FindingCategory",
    "build_analyst_brief_prompt",
    "create_structured_model",
]
