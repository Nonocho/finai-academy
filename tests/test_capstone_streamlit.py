from __future__ import annotations

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

from finai_academy import capstone
from finai_academy.capstone import ResearchRequest, build_reference_copilot


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
    assert "Certified snapshots" in app.selectbox[1].options
    assert "Optional live enrichment" in app.selectbox[1].options
    assert "First Finance - Arnaud Demes" in text
    assert "Research support only. Not investment advice." in text


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
    evidence_table = next(
        frame.value for frame in app.dataframe if "Claim" in frame.value.columns
    )
    assert set(evidence_table["Provenance"]) == {"Document", "Metric"}
    assert not app.exception


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

    assert "Tavily: Unavailable" in text
    assert "OpenAI: Unavailable" in text
    assert "Ollama: Unavailable" in text
    assert "fallback" not in text.casefold()


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
    entrypoint = (
        Path(__file__).parents[1] / "final-project" / "reference" / "streamlit_app.py"
    )

    app = AppTest.from_file(entrypoint).run()

    assert [tab.label for tab in app.tabs] == ["Reference mission", "Ask the analyst"]
    assert not app.exception
