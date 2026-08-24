from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from finai_academy import capstone
from finai_academy.capstone import ResearchRequest, build_reference_copilot
from finai_academy.capstone.views import build_capstone_request


def _app(service_factory, integration_status):
    from finai_academy.capstone import render_capstone

    render_capstone(service_factory, integration_status=integration_status)


def _successful_factory(request: ResearchRequest):
    del request
    return build_reference_copilot(run_id_factory=lambda: "reference-run-001")


def _failed_factory(request: ResearchRequest):
    del request
    complete = build_reference_copilot()
    return build_reference_copilot(retriever=_MissingSchneiderRetriever(complete.retriever))


def _rendered_text(app: AppTest) -> str:
    element_types = (
        "title",
        "header",
        "subheader",
        "markdown",
        "caption",
        "info",
        "warning",
        "success",
        "error",
        "text",
    )
    return "\n".join(
        str(element.value)
        for element_type in element_types
        for element in app.get(element_type)
        if getattr(element, "value", None) is not None
    )


def _run_reference(app: AppTest) -> AppTest:
    button = next(item for item in app.button if item.key == "run_reference")
    return button.click().run()


def test_renderer_is_exported_and_both_product_paths_are_visible() -> None:
    assert hasattr(capstone, "render_capstone")
    app = AppTest.from_function(
        _app,
        args=(
            _successful_factory,
            {"tavily": "Unavailable", "openai": "Unavailable", "ollama": "Unavailable"},
        ),
    ).run()
    text = _rendered_text(app)

    assert [tab.label for tab in app.tabs] == ["Reference mission", "Ask the analyst"]
    assert {item.label for item in app.selectbox} >= {"Provider", "Data mode"}
    assert "Recorded demo" in app.selectbox[0].options
    assert "Ollama" in app.selectbox[0].options
    assert "OpenAI" in app.selectbox[0].options
    assert app.selectbox[1].options == ["Certified snapshots"]
    assert "First Finance - Arnaud Demes" in text
    assert "Research support only. Not investment advice." in text
    footer = next(
        item for item in app.main.markdown if 'data-testid="capstone-footer"' in item.value
    )
    assert "First Finance - Arnaud Demes" in footer.value


def test_route_changes_clear_retained_results_and_show_immutable_run_route() -> None:
    app = AppTest.from_function(
        _app,
        args=(
            _successful_factory,
            {"tavily": "Unavailable", "openai": "Unavailable", "ollama": "Unavailable"},
        ),
    ).run()
    app = _run_reference(app)

    assert "Run route: Recorded demo · recorded-capstone-v1 · Certified snapshots" in (
        _rendered_text(app)
    )
    assert app.session_state.filtered_state["reference_run"] is not None
    app.session_state["custom_run"] = app.session_state.filtered_state["reference_run"]
    app.session_state["chat_history"] = [
        {"role": "user", "content": "Retained question."},
        {"role": "assistant", "content": "Retained answer."},
    ]
    app.session_state["public_error"] = "Retained error."

    provider = next(item for item in app.selectbox if item.key == "provider_selection")
    app = provider.select("OpenAI").run()
    model = next(item for item in app.text_input if item.key == "model_openai")
    app = model.input("gpt-5-mini-reviewed").run()
    state = app.session_state.filtered_state
    assert state["reference_run"] is None
    assert state["custom_run"] is None
    assert state["chat_history"] == []
    assert state["public_error"] is None
    assert "Release passed" not in _rendered_text(app)


def test_readiness_and_run_status_values_use_wrapping_text_instead_of_metrics() -> None:
    app = AppTest.from_function(
        _app,
        args=(
            _successful_factory,
            {"tavily": "Unavailable", "openai": "Unavailable", "ollama": "Unavailable"},
        ),
    ).run()
    app = _run_reference(app)
    text = _rendered_text(app)

    assert not app.metric
    for value in (
        "Recorded demo",
        "recorded-capstone-v1",
        "Certified snapshots",
        "Completed",
        "Evidence gate passed",
    ):
        assert value in text


def test_citations_and_trace_render_as_wrapping_rows_instead_of_wide_dataframes() -> None:
    app = AppTest.from_function(
        _app,
        args=(
            _successful_factory,
            {"tavily": "Unavailable", "openai": "Unavailable", "ollama": "Unavailable"},
        ),
    ).run()
    app = _run_reference(app)
    text = _rendered_text(app)
    dataframe_columns = [set(frame.value.columns) for frame in app.dataframe]

    assert not any("Claim" in columns for columns in dataframe_columns)
    assert not any("Summary" in columns for columns in dataframe_columns)
    assert "Citation: First Finance controlled classroom fixture" in text
    assert "Initial research plan passed host validation." in text


def test_recorded_reference_click_renders_complete_release_evidence_and_trace() -> None:
    app = AppTest.from_function(
        _app,
        args=(
            _successful_factory,
            {"tavily": "Unavailable", "openai": "Unavailable", "ollama": "Unavailable"},
        ),
    ).run()

    app = _run_reference(app)
    text = _rendered_text(app)

    for heading in (
        "Executive briefing",
        "Company evidence",
        "Cross-company comparison",
        "Limitations and open questions",
        "Sources and execution",
    ):
        assert heading in text
    assert "Evidence and citations" in text
    assert "Execution trace" in [item.label for item in app.expander]
    assert "Deterministic release evaluation" in text
    assert "Optional judge" in text
    assert "Release passed" in text
    score_table = next(frame.value for frame in app.dataframe if "Score" in frame.value.columns)
    assert len(score_table) == 5
    assert "NVIDIA · Document" in text
    assert "Schneider Electric · Metric" in text
    assert "Citation: First Finance controlled classroom fixture" in text
    assert not app.exception


def test_result_view_preserves_every_exact_citation_pair_and_card_count() -> None:
    app = AppTest.from_function(
        _app,
        args=(
            _successful_factory,
            {"tavily": "Unavailable", "openai": "Unavailable", "ollama": "Unavailable"},
        ),
    ).run()
    app = _run_reference(app)
    expected = [
        (
            "NVIDIA · Metric",
            "NVIDIA P/E was 52.4 x as of 2026-08-20.",
            "Citation: First Finance controlled classroom fixture",
            "Evidence ID: Not applicable",
        ),
        (
            "Schneider Electric · Metric",
            "Schneider Electric P/E was 31.8 x as of 2026-08-20.",
            "Citation: First Finance controlled classroom fixture",
            "Evidence ID: Not applicable",
        ),
        (
            "NVIDIA · Document",
            "NVIDIA (FY2026): NVIDIA Gaming revenue grew 41% in fiscal 2026.",
            "Citation: assets/course-data/fixtures/nvidia_fy2026_excerpt.html",
            "Evidence ID: NVDA-FY2026-GAMING-001",
        ),
        (
            "NVIDIA · Document",
            "NVIDIA (FY2026): NVIDIA reported fiscal 2026 total revenue of $215.9 billion, including $193.7 billion from Data Center.",
            "Citation: assets/course-data/fixtures/nvidia_fy2026_excerpt.html",
            "Evidence ID: NVDA-FY2026-DATA-CENTER-001",
        ),
        (
            "Schneider Electric · Document",
            "Schneider Electric (FY2025): The Schneider Electric FY2025 extract reports Energy Management organic revenue growth of 10%.",
            "Citation: assets/course-data/fixtures/schneider_fy2025_excerpt.pdf",
            "Evidence ID: SU-FY2025-ENERGY-MANAGEMENT-002",
        ),
        (
            "Schneider Electric · Document",
            "Schneider Electric (FY2025): Schneider Electric reported FY2025 revenue of EUR 40.2 billion and an adjusted EBITA margin of 18.7%.",
            "Citation: assets/course-data/fixtures/schneider_fy2025_excerpt.pdf",
            "Evidence ID: SU-FY2025-ENERGY-MANAGEMENT-001",
        ),
    ]
    citation_headers = [
        item.value for item in app.caption if item.value in {row[0] for row in expected}
    ]
    citation_sources = [item.value for item in app.caption if item.value.startswith("Citation:")]
    citation_ids = [item.value for item in app.caption if item.value.startswith("Evidence ID:")][:6]
    claims = [item.value for item in app.markdown if item.value in {row[1] for row in expected}]

    assert len(citation_headers) == 6
    assert (
        list(zip(citation_headers, claims, citation_sources, citation_ids, strict=True)) == expected
    )


def test_result_view_preserves_exact_trace_order_and_typed_fields() -> None:
    app = AppTest.from_function(
        _app,
        args=(
            _successful_factory,
            {"tavily": "Unavailable", "openai": "Unavailable", "ollama": "Unavailable"},
        ),
    ).run()
    app = _run_reference(app)
    headers = [item.value for item in app.caption if item.value.startswith("Event ")]
    metadata = [item.value for item in app.caption if item.value.startswith("Capability:")]
    assert headers == [
        "Event 1 · Planning · Ok",
        "Event 2 · Policy · Ok",
        "Event 3 · Execution · Ok",
        "Event 4 · Execution · Ok",
        "Event 5 · Execution · Error",
        "Event 6 · Replanning · Ok",
        "Event 7 · Execution · Ok",
        "Event 8 · Execution · Ok",
        "Event 9 · Evidence gate · Ok",
        "Event 10 · Report · Ok",
    ]
    expected_fields = [
        ("Not applicable", "0", "None"),
        ("Not applicable", "0", "None"),
        ("Get company metric", "0", "None"),
        ("Get company metric", "0", "None"),
        ("Get company metric", "0", "unsupported_metric"),
        ("Get company metric", "1", "None"),
        ("Search financial documents", "1", "None"),
        ("Search financial documents", "1", "None"),
        ("Not applicable", "1", "None"),
        ("Not applicable", "1", "None"),
    ]
    assert len(metadata) == 10
    for value, (capability, revision, error) in zip(metadata, expected_fields, strict=True):
        assert value.startswith(f"Capability: {capability} ·")
        assert f" · Revision: {revision} · Error: {error} ·" in value


def test_insufficient_evidence_renders_gate_failure_without_briefing() -> None:
    app = AppTest.from_function(
        _app,
        args=(
            _failed_factory,
            {"tavily": "Unavailable", "openai": "Unavailable", "ollama": "Unavailable"},
        ),
    ).run()

    app = _run_reference(app)
    text = _rendered_text(app)

    assert "Evidence gate failed" in text
    assert "Schneider Electric document evidence" in text
    assert "Executive briefing" not in text
    assert "Release passed" not in text


def test_missing_integrations_are_truthful_and_never_claim_a_silent_route() -> None:
    app = AppTest.from_function(
        _app,
        args=(
            _successful_factory,
            {"tavily": "Unavailable", "openai": "Unavailable", "ollama": "Unavailable"},
        ),
    ).run()
    text = _rendered_text(app)

    assert "Tavily" not in text
    assert "OpenAI: Unavailable" in text
    assert "Ollama: Unavailable" in text
    assert "fallback" not in text.casefold()


def test_default_streamlit_factory_routes_openai_and_ollama_through_provider_builder(
    monkeypatch,
) -> None:
    from finai_academy.capstone import streamlit_ui

    routed: list[ResearchRequest] = []

    def recording_builder(request: ResearchRequest, settings: object):
        del settings
        routed.append(request)
        return build_reference_copilot()

    monkeypatch.setattr(streamlit_ui, "build_copilot_for_request", recording_builder)

    for expected in ("openai", "ollama"):
        request = build_capstone_request(
            mode="custom",
            question="Compare the operating-growth evidence.",
            provider=expected,
            model="gpt-5-mini" if expected == "openai" else "qwen3:4b",
            data_mode="certified",
        )
        service = streamlit_ui._default_service_factory(request)

        assert service is not None
        assert routed[-1].provider == expected


def test_reference_entrypoint_uses_the_default_routed_factory(monkeypatch) -> None:
    from finai_academy.capstone import streamlit_ui

    routed: list[ResearchRequest] = []

    def recording_builder(request: ResearchRequest, settings: object):
        del settings
        routed.append(request)
        return build_reference_copilot()

    monkeypatch.setattr(streamlit_ui, "build_copilot_for_request", recording_builder)
    entrypoint = Path(__file__).parents[1] / "final-project" / "reference" / "streamlit_app.py"
    app = AppTest.from_file(entrypoint).run()
    provider = next(item for item in app.selectbox if item.key == "provider_selection")
    app = provider.select("OpenAI").run()
    question = next(item for item in app.text_input if item.key == "custom_question")
    app = question.input("Compare the operating-growth evidence.").run()
    submit = next(item for item in app.button if item.key == "ask_analyst")
    app = submit.click().run()

    assert not app.exception
    assert routed and routed[-1].provider == "openai"


def test_session_state_is_json_compatible_and_custom_request_stays_bounded() -> None:
    requests: list[ResearchRequest] = []

    def recording_factory(request: ResearchRequest):
        requests.append(request)
        return build_reference_copilot(run_id_factory=lambda: "custom-run-001")

    app = AppTest.from_function(
        _app,
        args=(
            recording_factory,
            {"tavily": "Unavailable", "openai": "Unavailable", "ollama": "Unavailable"},
        ),
    ).run()
    question = next(item for item in app.text_input if item.key == "custom_question")
    app = question.input("How does the operating-growth evidence compare?").run()
    ask = next(item for item in app.button if item.key == "ask_analyst")
    app = ask.click().run()

    assert requests[-1].mode == "custom"
    assert requests[-1].companies == ("NVIDIA", "Schneider Electric")
    assert len(requests[-1].question) <= 240
    text = _rendered_text(app).casefold()
    assert " buy " not in f" {text} "
    assert " sell " not in f" {text} "
    assert "recommendation" not in text
    json.dumps(app.session_state.filtered_state)


class _MissingSchneiderRetriever:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def search(self, company: str, query: str, top_k: int = 2):
        if company == "Schneider Electric":
            return ()
        return self._wrapped.search(company, query, top_k)


def test_reference_entrypoint_loads_the_offline_application() -> None:
    entrypoint = Path(__file__).parents[1] / "final-project" / "reference" / "streamlit_app.py"

    app = AppTest.from_file(entrypoint).run()

    assert [tab.label for tab in app.tabs] == ["Reference mission", "Ask the analyst"]
    assert not app.exception
