"""Four bounded integration seams for the Financial Analyst Copilot capstone."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from finai_academy.capstone.models import (
    CapstoneEvidenceHit,
    EvidenceGateDecision,
    ResearchRunResult,
)
from finai_academy.capstone.views import CapstoneRunView


@dataclasses.dataclass(frozen=True, slots=True)
class StudentIntegrationIncomplete(Exception):
    """A conceptual, path-free status for one unfinished student seam."""

    seam: str
    hint: str

    def __str__(self) -> str:
        return f"{self.seam}: {self.hint}"


def wire_retriever(company: str, query: str) -> tuple[CapstoneEvidenceHit, ...]:
    """Return certified document evidence for one company-scoped query."""

    raise StudentIntegrationIncomplete(
        seam="wire_retriever",
        hint="Use the certified retriever and preserve the company boundary.",
    )


def register_analyst_capabilities(discovered: Sequence[str]) -> tuple[str, ...]:
    """Return the safe subset of capabilities discovered at runtime."""

    raise StudentIntegrationIncomplete(
        seam="register_analyst_capabilities",
        hint="Intersect discovered tools with the certified analyst allowlist.",
    )


def evaluate_student_evidence_gate(
    hits: Sequence[CapstoneEvidenceHit],
) -> EvidenceGateDecision:
    """Require document evidence for both companies in the fixed mission."""

    raise StudentIntegrationIncomplete(
        seam="evaluate_student_evidence_gate",
        hint="Require document evidence for both companies before release.",
    )


def assemble_public_briefing_view(result: ResearchRunResult) -> CapstoneRunView:
    """Convert a completed domain result to the safe public presentation model."""

    raise StudentIntegrationIncomplete(
        seam="assemble_public_briefing_view",
        hint="Convert the run through the public presentation boundary.",
    )
