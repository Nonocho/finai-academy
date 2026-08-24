"""Launchable student workspace for the four capstone integration seams."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st
from integration import (
    StudentIntegrationIncomplete,
    assemble_public_briefing_view,
    evaluate_student_evidence_gate,
    register_analyst_capabilities,
    wire_retriever,
)

from finai_academy.capstone import ResearchRequest, build_reference_copilot
from finai_academy.capstone.models import CapstoneEvidenceHit, EvidenceGateDecision
from finai_academy.capstone.tools import build_certified_retriever
from finai_academy.capstone.views import CapstoneRunView

_SEAM_ORDER = (
    "wire_retriever",
    "register_analyst_capabilities",
    "evaluate_student_evidence_gate",
    "assemble_public_briefing_view",
)
_INCOMPLETE_HINTS = {
    "wire_retriever": "Connect the certified company-scoped retrieval boundary.",
    "register_analyst_capabilities": "Apply discovery through the approved read-tool policy.",
    "evaluate_student_evidence_gate": "Require document evidence for both companies.",
    "assemble_public_briefing_view": "Use the display-safe public view boundary.",
}
_ERROR_MESSAGE = "This integration did not complete safely."


def _seam_checks() -> dict[str, tuple[Callable[[], object], Callable[[object], bool]]]:
    retriever = build_certified_retriever()
    evidence_hits = (
        *retriever.search("NVIDIA", "operating growth", top_k=1),
        *retriever.search("Schneider Electric", "operating growth", top_k=1),
    )
    result = build_reference_copilot(run_id_factory=lambda: "student-preview").run(
        ResearchRequest.reference()
    )
    return {
        "wire_retriever": (
            lambda: wire_retriever("NVIDIA", "operating growth"),
            lambda value: (
                isinstance(value, tuple)
                and bool(value)
                and all(
                    isinstance(hit, CapstoneEvidenceHit)
                    and hit.company == "NVIDIA"
                    and bool(hit.evidence_id)
                    and bool(hit.source_reference)
                    for hit in value
                )
            ),
        ),
        "register_analyst_capabilities": (
            lambda: register_analyst_capabilities(
                ("place_order", "search_financial_documents", "get_company_metric")
            ),
            lambda value: value == (
                "get_company_metric",
                "search_financial_documents",
            ),
        ),
        "evaluate_student_evidence_gate": (
            lambda: evaluate_student_evidence_gate(evidence_hits),
            lambda value: (
                isinstance(value, EvidenceGateDecision)
                and value.passed
                and value.evidence_hits == evidence_hits
                and value.coverage
                == {
                    "NVIDIA": ("document",),
                    "Schneider Electric": ("document",),
                }
            ),
        ),
        "assemble_public_briefing_view": (
            lambda: assemble_public_briefing_view(result),
            lambda value: (
                isinstance(value, CapstoneRunView)
                and value.run_id == result.run_id
                and value.release.decision == "Release passed"
                and value.briefing is not None
            ),
        ),
    }


def _render_statuses() -> None:
    checks = _seam_checks()
    for seam in _SEAM_ORDER:
        check, is_valid = checks[seam]
        try:
            value = check()
        except StudentIntegrationIncomplete:
            st.warning(f"{seam} — Incomplete: {_INCOMPLETE_HINTS[seam]}")
        except Exception:  # noqa: BLE001 - learner errors must remain sanitized per seam
            st.error(f"{seam} — Error: {_ERROR_MESSAGE}")
        else:
            if is_valid(value):
                st.success(f"{seam} — Ready")
            else:
                st.error(f"{seam} — Error: {_ERROR_MESSAGE}")


st.set_page_config(
    page_title="Financial Analyst Copilot · Student challenge",
    page_icon="🧩",
    layout="wide",
)
st.title("Student integration challenge")
st.caption("Financial Analyst Copilot · certified offline route")

st.header("Reference mission")
st.text_area(
    "Fixed mission",
    value=ResearchRequest.reference().question,
    disabled=True,
    height=120,
)
st.info(
    "The recorded route uses repository fixtures only. It needs no API key, network, "
    "Tavily, or Ollama service."
)

st.header("Four integration seams")
_render_statuses()

st.header("Reference output shape")
st.markdown(
    """
1. Executive briefing
2. Company evidence
3. Cross-company comparison
4. Limitations and open questions
5. Sources and execution
"""
)

st.header("How to work")
st.markdown(
    """
- Edit only the four function bodies in `integration.py`.
- Run the verifier after each seam to keep failures independent.
- Keep the fixed company boundary and use the certified public components.
- A complete solution ends with one standalone success marker from `verify.py`.
"""
)
st.code(".venv/bin/python final-project/student/verify.py", language="bash")
st.caption("Research support only. Not investment advice.")
