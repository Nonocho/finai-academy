"""Instructor/reference correction for the four student integration seams.

This module is assessment material.  The student application never imports it;
tests copy it into an isolated workspace only to prove the challenge is solvable.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from finai_academy.capstone.models import (
    CapstoneEvidenceHit,
    EvidenceGateDecision,
    ResearchRunResult,
)
from finai_academy.capstone.tools import AnalystToolRegistry, build_certified_retriever
from finai_academy.capstone.views import CapstoneRunView, to_run_view

__all__ = [
    "StudentIntegrationIncomplete",
    "assemble_public_briefing_view",
    "evaluate_student_evidence_gate",
    "register_analyst_capabilities",
    "wire_retriever",
]


@dataclasses.dataclass(frozen=True, slots=True)
class StudentIntegrationIncomplete(Exception):
    """Keep the starter's public exception contract in solved copies."""

    seam: str
    hint: str


def wire_retriever(company: str, query: str) -> tuple[CapstoneEvidenceHit, ...]:
    """Return certified document evidence for one company-scoped query."""

    return build_certified_retriever().search(company, query)


def register_analyst_capabilities(discovered: Sequence[str]) -> tuple[str, ...]:
    """Apply runtime discovery through the certified analyst tool policy."""

    return AnalystToolRegistry(discovered=discovered).discover()


def evaluate_student_evidence_gate(
    hits: Sequence[CapstoneEvidenceHit],
) -> EvidenceGateDecision:
    """Require document evidence for both companies in the fixed mission."""

    companies = ("NVIDIA", "Schneider Electric")
    retriever = build_certified_retriever()
    certified_identities = {
        (hit.company, hit.evidence_id, hit.source_reference)
        for company in companies
        for hit in retriever.search(company, "operating growth")
    }
    covered = {
        hit.company
        for hit in hits
        if (hit.company, hit.evidence_id, hit.source_reference) in certified_identities
    }
    missing = tuple(
        f"{company} document evidence" for company in companies if company not in covered
    )
    return EvidenceGateDecision(
        passed=not missing,
        coverage={
            company: (("document",) if company in covered else ())
            for company in companies
        },
        missing_requirements=missing,
        evidence_hits=tuple(hits),
    )


def assemble_public_briefing_view(result: ResearchRunResult) -> CapstoneRunView:
    """Convert a domain result through the display-safe public boundary."""

    return to_run_view(result)
