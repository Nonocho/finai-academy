"""Launchable student workspace for the capstone integration challenge."""

from __future__ import annotations

import streamlit as st

from collections.abc import Callable
from pathlib import Path

from integration import (
    StudentIntegrationIncomplete,
    assemble_public_briefing_view,
    evaluate_student_evidence_gate,
    register_analyst_capabilities,
    wire_retriever,
)
from finai_academy.capstone import build_reference_copilot, ResearchRequest

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

_REFERENCE_RESULT = None


def _reference_result():
    global _REFERENCE_RESULT
    if _REFERENCE_RESULT is None:
        _REFERENCE_RESULT = build_reference_copilot(run_id_factory=lambda: "student-streamlit-check").run(
            ResearchRequest.reference()
        )
    return _REFERENCE_RESULT


def _seam_checks() -> dict[str, tuple[Callable[[], object], Callable[[object], bool]]]:
    return {
        "wire_retriever": (
            lambda: wire_retriever("Schneider Electric", "organic growth"),
            lambda value: isinstance(value, tuple) and all(
                hasattr(item, "chunk_id") for item in value
            ),
        ),
        "register_analyst_capabilities": (
            lambda: register_analyst_capabilities(
                ("search_financial_documents", "place_order", "get_company_metric")
            ),
            lambda value: value == ("get_company_metric", "search_financial_documents"),
        ),
        "evaluate_student_evidence_gate": (
            lambda: evaluate_student_evidence_gate(_reference_hits()),
            lambda value: hasattr(value, "passed") and hasattr(value, "coverage"),
        ),
        "assemble_public_briefing_view": (
            lambda: assemble_public_briefing_view(_reference_result()),
            lambda value: value is not None,
        ),
    }


def _reference_hits():
    result = _reference_result()
    return tuple(result.evidence_gate.evidence_hits)


def _render_statuses() -> None:
    checks = _seam_checks()
    for seam in _SEAM_ORDER:
        check, is_valid = checks[seam]
        try:
            value = check()
            if not is_valid(value):
                raise ValueError("invalid integration result")
        except StudentIntegrationIncomplete:
            st.warning(f"{seam} — Incomplete: {_INCOMPLETE_HINTS[seam]}")
        except Exception:  # noqa: BLE001
            st.error(f"{seam} — Error: {_ERROR_MESSAGE}")
        else:
            st.success(f"{seam} — Ready")


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
    value="Compare NVIDIA and Schneider Electric using official documents and selected financial metrics.",
    disabled=True,
    height=96,
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
- Edit only the four function bodies in `final-project/student/integration.py`.
- Run the verifier after each seam to keep failures isolated.
- Keep the fixed company boundary and use the certified document-index components.
- A complete solution ends with one standalone success marker from `verify.py`.
"""
)
st.code("uv run python final-project/student/verify.py", language="bash")
st.caption("Research support only. Not investment advice.")
