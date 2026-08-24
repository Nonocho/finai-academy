"""Streamlit renderer for the Financial Analyst Copilot reference application."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any

import streamlit as st

from finai_academy.capstone.models import ResearchRequest
from finai_academy.capstone.service import (
    FinancialAnalystCopilot,
    build_copilot_for_request,
)
from finai_academy.capstone.views import CapstoneRunView, to_run_view
from finai_academy.settings import Settings

_PROVIDERS = {
    "Recorded demo": "recorded",
    "Ollama": "ollama",
    "OpenAI": "openai",
}
_DATA_MODES = {
    "Certified snapshots": "certified",
    "Optional live enrichment": "live_enrichment",
}
_DEFAULT_MODELS = {
    "Recorded demo": "recorded-capstone-v1",
    "Ollama": "qwen3:4b",
    "OpenAI": "gpt-5-mini",
}
_STATE_KEYS = ("reference_run", "custom_run", "chat_history", "public_error")
_COMPANIES = ("NVIDIA", "Schneider Electric")


def render_capstone(
    service_factory: Callable[[ResearchRequest], FinancialAnalystCopilot] | None = None,
    *,
    integration_status: Mapping[str, str] | None = None,
) -> None:
    """Render the complete capstone using only public view data in session state."""

    st.set_page_config(page_title="Financial Analyst Copilot", page_icon="📊", layout="wide")
    _initialize_state()
    factory = service_factory or _default_service_factory
    integrations = dict(
        _default_integration_status() if integration_status is None else integration_status
    )

    with st.sidebar:
        st.title("Financial Analyst Copilot")
        st.caption("First Finance - Arnaud Demes")
        provider_label = st.selectbox(
            "Provider",
            tuple(_PROVIDERS),
            key="provider_selection",
        )
        selected_default = _DEFAULT_MODELS[provider_label]
        model = st.text_input(
            "Model",
            value=selected_default,
            key=f"model_{_PROVIDERS[provider_label]}",
            help="The exact model route used for a submitted run.",
        )
        data_mode_label = st.selectbox(
            "Data mode",
            tuple(_DATA_MODES),
            key="data_mode_selection",
        )
        st.caption(f"Tavily: {integrations.get('tavily', 'Not checked')}")
        st.caption(f"OpenAI: {integrations.get('openai', 'Not checked')}")
        st.caption(f"Ollama: {integrations.get('ollama', 'Not checked')}")
        st.info("Research support only. Not investment advice.")
        if st.button("Reset session", key="reset_capstone"):
            _reset_state()
            st.rerun()

    st.title("Financial Analyst Copilot")
    st.caption(
        "Bounded research for NVIDIA and Schneider Electric with visible evidence, "
        "execution, and release checks."
    )
    _render_readiness_strip(provider_label, model, data_mode_label, integrations)

    reference_tab, custom_tab = st.tabs(("Reference mission", "Ask the analyst"))
    with reference_tab:
        _render_reference_tab(
            factory=factory,
            provider=_PROVIDERS[provider_label],
            model=model,
            data_mode=_DATA_MODES[data_mode_label],
        )
    with custom_tab:
        _render_custom_tab(
            factory=factory,
            provider=_PROVIDERS[provider_label],
            model=model,
            data_mode=_DATA_MODES[data_mode_label],
        )


def _initialize_state() -> None:
    defaults: dict[str, Any] = {
        "reference_run": None,
        "custom_run": None,
        "chat_history": [],
        "public_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_state() -> None:
    for key in _STATE_KEYS:
        st.session_state.pop(key, None)


def _default_service_factory(request: ResearchRequest) -> FinancialAnalystCopilot:
    return build_copilot_for_request(request, Settings.from_environment())


def _default_integration_status() -> dict[str, str]:
    return {
        "tavily": "Available" if os.environ.get("TAVILY_API_KEY", "").strip() else "Unavailable",
        "openai": "Available" if os.environ.get("OPENAI_API_KEY", "").strip() else "Unavailable",
        "ollama": "Not checked",
    }


def _render_readiness_strip(
    provider: str,
    model: str,
    data_mode: str,
    integrations: Mapping[str, str],
) -> None:
    with st.container(border=True):
        columns = st.columns(4)
        columns[0].metric("Provider", provider)
        columns[1].metric("Model", model)
        columns[2].metric("Data", data_mode)
        columns[3].metric("Tavily", integrations.get("tavily", "Not checked"))


def _render_reference_tab(
    *,
    factory: Callable[[ResearchRequest], FinancialAnalystCopilot],
    provider: str,
    model: str,
    data_mode: str,
) -> None:
    request = ResearchRequest.reference(
        provider=provider,
        model=model,
        data_mode=data_mode,
        include_news=data_mode == "live_enrichment",
    )
    st.text_area(
        "Fixed reference mission",
        value=request.question,
        disabled=True,
        height=120,
    )
    st.caption("Universe: NVIDIA and Schneider Electric · Maximum 6 steps · Maximum 1 replan")
    if st.button("Run reference mission", key="run_reference", type="primary"):
        st.session_state.reference_run = _execute(factory, request)

    _render_public_error()
    payload = st.session_state.reference_run
    if isinstance(payload, dict):
        _render_run(CapstoneRunView.model_validate(payload))


def _render_custom_tab(
    *,
    factory: Callable[[ResearchRequest], FinancialAnalystCopilot],
    provider: str,
    model: str,
    data_mode: str,
) -> None:
    st.caption("Questions are limited to the NVIDIA and Schneider Electric research universe.")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    with st.form("custom_question_form", clear_on_submit=True):
        question = st.text_input(
            "Research question",
            max_chars=240,
            key="custom_question",
            placeholder="Compare the available operating-growth evidence.",
        )
        submitted = st.form_submit_button("Ask the analyst", key="ask_analyst")
    if submitted:
        cleaned = question.strip()
        if not cleaned:
            st.session_state.public_error = "Enter a research question before submitting."
        else:
            request = ResearchRequest(
                mode="custom",
                question=cleaned,
                companies=_COMPANIES,
                provider=provider,
                model=model,
                data_mode=data_mode,
                include_news=data_mode == "live_enrichment",
            )
            payload = _execute(factory, request)
            st.session_state.custom_run = payload
            st.session_state.chat_history = [
                *st.session_state.chat_history,
                {"role": "user", "content": cleaned},
                {
                    "role": "assistant",
                    "content": _assistant_message(payload),
                },
            ]

    _render_public_error()
    payload = st.session_state.custom_run
    if isinstance(payload, dict):
        _render_run(CapstoneRunView.model_validate(payload))


def _execute(
    factory: Callable[[ResearchRequest], FinancialAnalystCopilot],
    request: ResearchRequest,
) -> dict[str, Any] | None:
    st.session_state.public_error = None
    try:
        result = factory(request).run(request)
        return to_run_view(result).model_dump(mode="json")
    except Exception:  # noqa: BLE001 - raw dependency details must not enter the UI
        st.session_state.public_error = (
            "The selected route could not complete the bounded research run."
        )
        return None


def _assistant_message(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "The selected route did not complete. No briefing was released."
    release = payload.get("release")
    if isinstance(release, dict) and release.get("decision") == "Release passed":
        return "The evidence-backed research run completed. Review the public result below."
    return "The release checks did not pass. Review the evidence gate and execution trace below."


def _render_public_error() -> None:
    message = st.session_state.public_error
    if isinstance(message, str):
        st.error(message)


def _render_run(view: CapstoneRunView) -> None:
    st.divider()
    status_columns = st.columns(4)
    status_columns[0].metric("Run status", view.readiness.run_status)
    status_columns[1].metric("Evidence", view.release.evidence_gate)
    status_columns[2].metric("Replans", _replan_count(view))
    status_columns[3].metric("Duration", view.total_duration)

    st.subheader("Research plan")
    st.dataframe(
        [_plan_record(row.model_dump()) for row in view.plan],
        hide_index=True,
    )

    st.subheader("Tool activity")
    st.dataframe(
        [_tool_record(row.model_dump()) for row in view.tool_activity],
        hide_index=True,
    )
    errors = [row for row in view.tool_activity if row.status == "Error"]
    replans = [row for row in view.trace if row.phase == "Replanning"]
    if errors or replans:
        st.subheader("Typed errors and replan")
        for row in errors:
            st.warning(f"Attempt {row.attempt}: {row.outcome}")
        for row in replans:
            st.info(f"Revision {row.revision}: {row.summary}")

    if view.release.evidence_gate == "Evidence gate failed":
        st.error("Evidence gate failed")
        for requirement in view.release.missing_requirements:
            st.markdown(f"- Missing: {requirement}")
    else:
        st.success("Evidence gate passed")

    if view.briefing is not None:
        _render_briefing(view)

    st.subheader("Evidence and citations")
    if view.cited_facts:
        st.dataframe(
            [_fact_record(row.model_dump()) for row in view.cited_facts],
            hide_index=True,
        )
    else:
        st.caption("No cited facts were released.")
    if view.evidence:
        with st.expander("Collected document evidence"):
            st.dataframe(
                [_evidence_record(row.model_dump()) for row in view.evidence],
                hide_index=True,
            )

    with st.expander("Execution trace"):
        st.dataframe(
            [_trace_record(row.model_dump()) for row in view.trace],
            hide_index=True,
        )

    st.subheader("Deterministic release evaluation")
    st.dataframe(
        [
            {"Metric": row.metric, "Score": row.score, "Rationale": row.rationale}
            for row in view.scores
        ],
        hide_index=True,
    )
    if view.release.decision == "Release passed":
        st.success("Release passed")
    else:
        st.error("Release blocked")

    with st.container(border=True):
        st.subheader("Optional judge")
        st.markdown(f"**Status:** {view.judge.status}")
        st.write(view.judge.summary)
        st.caption(f"Judge score: {view.judge.score}")


def _render_briefing(view: CapstoneRunView) -> None:
    assert view.briefing is not None
    briefing = view.briefing
    st.subheader("Executive briefing")
    st.write(briefing.executive_briefing)

    st.subheader("Company evidence")
    for section in briefing.company_evidence:
        with st.container(border=True):
            st.markdown(f"**{section.company}**")
            for claim in section.claims:
                st.markdown(f"- {claim}")

    st.subheader("Cross-company comparison")
    for statement in briefing.cross_company_comparison:
        st.markdown(f"- {statement}")

    st.subheader("Limitations and open questions")
    for statement in briefing.limitations_and_open_questions:
        st.markdown(f"- {statement}")

    st.subheader("Sources and execution")
    for statement in briefing.sources_and_execution:
        st.markdown(f"- {statement}")


def _replan_count(view: CapstoneRunView) -> str:
    revisions = {row.revision for row in view.trace if row.phase == "Replanning"}
    return str(len(revisions))


def _plan_record(row: Mapping[str, str]) -> dict[str, str]:
    return {
        "Step": row["step"],
        "Capability": row["capability"],
        "Purpose": row["purpose"],
        "Expected evidence": row["expected_evidence"],
        "Depends on": row["depends_on"],
    }


def _tool_record(row: Mapping[str, str]) -> dict[str, str]:
    return {
        "Attempt": row["attempt"],
        "Capability": row["capability"],
        "Company": row["company"],
        "Status": row["status"],
        "Outcome": row["outcome"],
        "Provenance": row["provenance"],
        "Duration": row["duration"],
    }


def _fact_record(row: Mapping[str, str]) -> dict[str, str]:
    return {
        "Company": row["company"],
        "Provenance": row["provenance"],
        "Claim": row["claim"],
        "Source": row["source"],
        "Evidence ID": row["evidence_id"],
    }


def _evidence_record(row: Mapping[str, str]) -> dict[str, str]:
    return {
        "Company": row["company"],
        "Period": row["period"],
        "Section": row["section"],
        "Evidence": row["evidence"],
        "Evidence ID": row["evidence_id"],
        "Source": row["source"],
    }


def _trace_record(row: Mapping[str, str]) -> dict[str, str]:
    return {
        "#": row["index"],
        "Phase": row["phase"],
        "Capability": row["capability"],
        "Attempt": row["attempt"],
        "Revision": row["revision"],
        "Status": row["status"],
        "Error": row["error"],
        "Duration": row["duration"],
        "Failure owner": row["failure_owner"],
        "Summary": row["summary"],
    }


__all__ = ["render_capstone"]
