"""Capstone student integration seams for the document-evidence challenge."""

from __future__ import annotations

from collections.abc import Mapping
from finai_academy.capstone.models import CapstoneEvidenceHit, EvidenceGateDecision, ResearchRunResult

class StudentIntegrationIncomplete(Exception):
    """Raised whenever a student seam is intentionally left as a TODO."""

    seam: str
    hint: str

    def __init__(self, seam: str, hint: str) -> None:
        self.seam = seam
        self.hint = hint
        super().__init__(seam)


def wire_retriever(company: str, query: str) -> tuple[CapstoneEvidenceHit, ...]:
    """Return certified retrieval hits bounded by company and certified source."""

    raise StudentIntegrationIncomplete(
        seam="wire_retriever",
        hint="Connect the certified company-scoped retrieval boundary.",
    )


def register_analyst_capabilities(
    discovered: Mapping[str, object] | tuple[str, ...] | list[str] | set[str],
) -> tuple[str, ...]:
    """Return only discovered and allowed analyst capabilities."""

    raise StudentIntegrationIncomplete(
        seam="register_analyst_capabilities",
        hint="Apply discovery through the approved read-tool policy.",
    )


def evaluate_student_evidence_gate(hits: tuple[CapstoneEvidenceHit, ...]) -> EvidenceGateDecision:
    """Return a compact evidence-gate decision for the completed run."""

    raise StudentIntegrationIncomplete(
        seam="evaluate_student_evidence_gate",
        hint="Require document evidence for both companies.",
    )


def assemble_public_briefing_view(result: ResearchRunResult):
    """Build the safe public run view for the learner-facing output."""

    raise StudentIntegrationIncomplete(
        seam="assemble_public_briefing_view",
        hint="Use the display-safe public view boundary.",
    )
