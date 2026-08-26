from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest

from finai_academy.capstone import ResearchRequest, build_reference_copilot


READINESS = {"openai": "Unavailable", "ollama": "Unavailable"}


def _app(service_factory, integration_status):
    from finai_academy.capstone import render_capstone

    render_capstone(service_factory, integration_status=integration_status)


def _successful_factory(request: ResearchRequest):
    del request
    return build_reference_copilot(run_id_factory=lambda: "reference-run-001")


class _MissingSchneiderRetriever:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def search(self, company: str, query: str, top_k: int = 2):
        if company == "Schneider Electric":
            return ()
        return self._wrapped.search(company, query, top_k)


def _blocked_factory(request: ResearchRequest):
    del request
    complete = build_reference_copilot()
    return build_reference_copilot(retriever=_MissingSchneiderRetriever(complete.retriever))


def _error_factory(request: ResearchRequest):
    del request
    raise RuntimeError("raw backend exception must never be displayed")


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


def _run_analysis(app: AppTest) -> AppTest:
    button = next(item for item in app.button if item.key == "analyze_reports")
    return button.click().run()


def test_initial_screen_leads_with_the_learning_job() -> None:
    """Restoring configuration-first controls must fail this test."""

    app = AppTest.from_function(_app, args=(_successful_factory, READINESS)).run()
    text = _rendered_text(app)

    assert app.title[0].value == "Financial Document Analyst"
    assert "Ask a financial question and see the exact report page and table behind the answer." in text
    assert next(item for item in app.button if item.key == "analyze_reports").label == (
        "Analyze the reports"
    )
    assert [item.label for item in app.expander] == ["Advanced settings"]
    assert not app.tabs


def test_completed_result_orders_answer_evidence_then_process() -> None:
    """Putting the plan before the answer must fail this test."""

    app = _run_analysis(AppTest.from_function(_app, args=(_successful_factory, READINESS)).run())

    assert [tab.label for tab in app.tabs] == ["Answer", "Evidence", "How it worked"]
    text = _rendered_text(app)
    assert "Conclusion" in text
    assert "Original report" in text
    assert "Extracted table" in text
    assert "Why this evidence was selected" in text
    assert "Source details" in [item.label for item in app.expander]
    assert "Research plan" not in text
    assert not app.exception


def test_advanced_settings_keep_route_controls_collapsed_and_default_to_recorded() -> None:
    """Making route controls primary-screen content must fail this test."""

    app = AppTest.from_function(_app, args=(_successful_factory, READINESS)).run()
    provider = next(item for item in app.selectbox if item.key == "provider_selection")

    assert provider.value == "Recorded demo"
    assert provider.options == ["Recorded demo", "Ollama", "OpenAI"]
    assert next(item for item in app.text_input if item.key == "model_recorded").value == (
        "recorded-capstone-v1"
    )


def test_question_uses_reference_mode_by_default_and_keeps_custom_requests_bounded() -> None:
    """Losing the fixed mission or company boundary must fail this test."""

    requests: list[ResearchRequest] = []

    def recording_factory(request: ResearchRequest):
        requests.append(request)
        return build_reference_copilot(run_id_factory=lambda: "captured-run")

    app = AppTest.from_function(_app, args=(recording_factory, READINESS)).run()
    app = _run_analysis(app)
    assert requests[-1].mode == "reference"

    question = next(item for item in app.text_area if item.label == "Question")
    app = question.input("How does the operating-growth evidence compare?").run()
    app = _run_analysis(app)

    assert requests[-1].mode == "custom"
    assert requests[-1].companies == ("NVIDIA", "Schneider Electric")
    json.dumps(app.session_state.filtered_state)


def test_insufficient_evidence_uses_safe_next_step_copy_without_internal_jargon() -> None:
    """Leaking evidence-gate terminology into the learner-facing result must fail this test."""

    app = _run_analysis(AppTest.from_function(_app, args=(_blocked_factory, READINESS)).run())
    text = _rendered_text(app)

    assert "The reports did not provide enough contextual evidence to release an answer." in text
    assert "Check the missing evidence below, then run the certified analysis again." in text
    assert "typed stop" not in text.casefold()
    assert "trajectory" not in text.casefold()
    assert "rrf" not in text.casefold()
    assert "evidence gate failed" not in text.casefold()
    assert not app.exception


def test_missing_crop_is_not_replaced_with_a_misleading_report_image() -> None:
    """Rendering a different page for evidence without a crop must fail this test."""

    app = _run_analysis(AppTest.from_function(_app, args=(_successful_factory, READINESS)).run())

    assert len(app.image) == 1
    assert "Schneider Electric · FY2025 · page 5" not in [
        str(caption.value) for caption in app.caption
    ]


def test_service_exception_is_safe_and_does_not_expose_the_raw_exception() -> None:
    """Displaying the factory exception must fail this test."""

    app = _run_analysis(AppTest.from_function(_app, args=(_error_factory, READINESS)).run())
    text = _rendered_text(app)

    assert "The selected route could not complete the certified analysis." in text
    assert "raw backend exception" not in text
    assert not app.exception
