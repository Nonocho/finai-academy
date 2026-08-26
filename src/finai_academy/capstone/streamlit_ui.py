"""Streamlit renderer for the simple evidence-first capstone workspace."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import streamlit as st

from finai_academy.capstone.models import ResearchRequest
from finai_academy.capstone.service import FinancialAnalystCopilot, build_copilot_for_request
from finai_academy.capstone.views import (
    CapstoneRunView,
    EvidenceComparisonView,
    HowItWorkedView,
    build_capstone_request,
    to_run_view,
)
from finai_academy.settings import Settings

ROOT = Path(__file__).resolve().parents[3]
_PROVIDERS = {"Recorded demo": "recorded", "Ollama": "ollama", "OpenAI": "openai"}
_DEFAULT_MODELS = {
    "Recorded demo": "recorded-capstone-v1",
    "Ollama": "qwen3:4b",
    "OpenAI": "gpt-5.6-luna",
}
_REFERENCE_MISSION = ResearchRequest.reference().question
_STATE_KEYS = ("capstone_run", "public_error")


def render_capstone(
    service_factory: Callable[[ResearchRequest], FinancialAnalystCopilot] | None = None,
    *,
    integration_status: Mapping[str, str] | None = None,
) -> None:
    """Render the capstone using display-only data saved as JSON in session state."""

    del integration_status
    st.set_page_config(page_title="Financial Document Analyst", page_icon="📊", layout="wide")
    _initialize_state()
    factory = service_factory or _default_service_factory

    st.title("Financial Document Analyst")
    st.write("Ask a financial question and see the exact report page and table behind the answer.")
    question = st.text_area("Question", value=_REFERENCE_MISSION, height=110)
    st.caption(
        "Evidence comes from NVIDIA's FY2026 annual report and Schneider Electric's FY2025 results."
    )
    run_clicked = st.button("Analyze the reports", key="analyze_reports", type="primary")
    with st.expander("Advanced settings", expanded=False):
        provider_label = st.selectbox(
            "Provider",
            tuple(_PROVIDERS),
            key="provider_selection",
            on_change=_clear_retained_outputs,
        )
        model = st.text_input(
            "Model",
            value=_DEFAULT_MODELS[provider_label],
            key=f"model_{_PROVIDERS[provider_label]}",
            on_change=_clear_retained_outputs,
        )
        st.caption(f"{provider_label} · {model} · certified document data")
        if st.button("Reset analysis", key="reset_capstone"):
            _reset_state()
            st.rerun()

    if run_clicked:
        _run_analysis(factory, question, _PROVIDERS[provider_label], model)
    _render_public_error()
    payload = st.session_state.capstone_run
    if isinstance(payload, dict):
        _render_run(CapstoneRunView.model_validate(payload))
    st.markdown(
        '<footer data-testid="capstone-footer"><hr>First Finance - Arnaud Demes</footer>',
        unsafe_allow_html=True,
    )


def _initialize_state() -> None:
    for key, value in {"capstone_run": None, "public_error": None}.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_state() -> None:
    for key in _STATE_KEYS:
        st.session_state.pop(key, None)


def _clear_retained_outputs() -> None:
    st.session_state.capstone_run = None
    st.session_state.public_error = None


def _default_service_factory(request: ResearchRequest) -> FinancialAnalystCopilot:
    return build_copilot_for_request(request, Settings.from_environment())


def _run_analysis(
    factory: Callable[[ResearchRequest], FinancialAnalystCopilot],
    question: str,
    provider: str,
    model: str,
) -> None:
    cleaned = question.strip()
    if not cleaned:
        st.session_state.public_error = "Enter a financial question before analyzing the reports."
        return
    request = build_capstone_request(
        mode="reference" if cleaned == _REFERENCE_MISSION else "custom",
        question=None if cleaned == _REFERENCE_MISSION else cleaned,
        provider=provider,
        model=model,
        data_mode="certified",
    )
    st.session_state.public_error = None
    try:
        result = factory(request).run(request)
        st.session_state.capstone_run = to_run_view(result).model_dump(mode="json")
    except Exception:  # noqa: BLE001 - raw dependency details must not enter the UI
        st.session_state.capstone_run = None
        st.session_state.public_error = "The selected route could not complete the certified analysis."


def _render_public_error() -> None:
    message = st.session_state.public_error
    if isinstance(message, str):
        st.error(message)


def _render_run(view: CapstoneRunView) -> None:
    st.divider()
    answer_tab, evidence_tab, process_tab = st.tabs(("Answer", "Evidence", "How it worked"))
    with answer_tab:
        _render_answer(view)
    with evidence_tab:
        _render_evidence(view.evidence)
    with process_tab:
        _render_process(view.how_it_worked)


def _render_answer(view: CapstoneRunView) -> None:
    if view.answer is None:
        st.error(view.outcome.assistant_message)
        st.info("Check the missing evidence below, then run the certified analysis again.")
        if view.release.missing_requirements:
            st.subheader("Missing evidence")
            for requirement in view.release.missing_requirements:
                st.markdown(f"- {requirement}")
        return

    answer = view.answer
    st.subheader("Conclusion")
    st.write(answer.conclusion)
    st.subheader("Company evidence")
    for section in answer.company_evidence:
        st.markdown(f"**{section.company}**")
        for claim in section.claims:
            st.markdown(f"- {claim}")
    st.subheader("Comparison limits")
    for limit in answer.comparison_limits:
        st.markdown(f"- {limit}")
    st.subheader("Citations")
    for citation in answer.citations:
        with st.container(border=True):
            st.markdown(f"**{citation.company}**")
            st.write(citation.claim)
            st.caption(f"Source: {citation.source}")
            st.caption(f"Evidence ID: {citation.evidence_id}")


def _render_evidence(evidence_items: tuple[EvidenceComparisonView, ...]) -> None:
    if not evidence_items:
        st.info("No certified report crop is available for this result.")
        return
    for evidence in evidence_items:
        st.subheader(evidence.company)
        left, right = st.columns((1, 1))
        with left:
            st.subheader("Original report")
            st.image(str(ROOT / evidence.crop_asset_key), caption=evidence.page_label)
        with right:
            st.subheader("Extracted table")
            st.markdown(evidence.extracted_markdown)
        st.subheader("Retrieved context")
        st.write(evidence.retrieved_chunk)
        st.subheader("Why this evidence was selected")
        st.write(evidence.selection_reason)
        with st.expander("Source details"):
            for label, value in evidence.source_details:
                st.markdown(f"**{label}:** {value}")


def _render_process(process: HowItWorkedView) -> None:
    st.subheader("How the analysis works")
    for index, step in enumerate(process.pipeline_steps, start=1):
        st.markdown(f"{index}. {step}")
    with st.expander("Retrieval details"):
        st.dataframe(
            [
                {
                    "Company": row.company,
                    "Chunk": row.chunk_id,
                    "Channel ranks": ", ".join(
                        f"{channel}: {rank}" for channel, rank in row.channel_ranks
                    ),
                    "Fused score": row.fused_score,
                }
                for row in process.retrieval_details
            ],
            hide_index=True,
        )
    with st.expander("Tool activity"):
        st.dataframe([row.model_dump() for row in process.tool_activity], hide_index=True)
    with st.expander("Technical trace"):
        st.dataframe([row.model_dump() for row in process.trace], hide_index=True)
    with st.expander("Evaluation and run details"):
        st.dataframe([row.model_dump() for row in process.scores], hide_index=True)
        st.caption(f"Model route: {process.model_route}")
        st.caption(f"Run duration: {process.total_duration}")
        st.caption(f"MLflow run: {process.mlflow_run_id or 'Not recorded'}")
        st.caption(f"MLflow trace: {process.mlflow_trace_id or 'Not recorded'}")


__all__ = ["render_capstone"]
