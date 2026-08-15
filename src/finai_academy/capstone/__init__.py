"""Domain and application code for the Financial Analyst Copilot."""

from finai_academy.capstone.briefing import AnalystBriefService
from finai_academy.capstone.model_gateway import create_structured_model
from finai_academy.capstone.models import (
    AnalystBrief,
    AnalystFinding,
    EvidenceType,
    FindingCategory,
)

__all__ = [
    "AnalystBrief",
    "AnalystBriefService",
    "AnalystFinding",
    "EvidenceType",
    "FindingCategory",
    "create_structured_model",
]
