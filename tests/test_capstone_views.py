from __future__ import annotations

import json
import subprocess
import sys

from finai_academy.capstone import ResearchRequest, build_reference_copilot
from finai_academy.capstone.views import to_run_view


def recorded_result():
    return build_reference_copilot(run_id_factory=lambda: "reference-run-001").run(
        ResearchRequest.reference()
    )


def test_run_view_is_json_serializable_and_leads_with_plain_language_answer() -> None:
    """Removing the learner-facing answer model must fail this test."""

    view = to_run_view(recorded_result())
    payload = view.model_dump(mode="json")
    encoded = json.dumps(payload)

    assert payload["run_id"] == "reference-run-001"
    assert view.answer is not None
    assert view.answer.conclusion == (
        "Evidence-backed comparison prepared for the bounded research request."
    )
    assert [section.company for section in view.answer.company_evidence] == [
        "NVIDIA",
        "Schneider Electric",
    ]
    assert view.answer.comparison_limits == (
        "The companies report in different currencies and periods.",
        "Their business mixes and disclosed operating measures are not directly comparable.",
        "Open question: Which aligned operating measure would be most useful for a later comparison?",
    )
    assert "arguments" not in encoded
    assert "private_reasoning" not in encoded
    assert "system_prompt" not in encoded
    assert "api_key" not in encoded.casefold()
    assert "/Users/" not in encoded


def test_run_view_maps_certified_document_evidence_to_side_by_side_display_data() -> None:
    """Dropping a crop, table, context, or provenance label must fail this test."""

    result = recorded_result()
    view = to_run_view(result)
    nvidia_hit = next(hit for hit in result.evidence_gate.evidence_hits if hit.company == "NVIDIA")
    evidence = next(item for item in view.evidence if item.company == "NVIDIA")

    assert evidence.page_label == "NVIDIA · FY2026 · page 165"
    assert evidence.crop_asset_key == (
        "assets/course-data/capstone/crops/nvidia_segment_table_page_165.png"
    )
    assert "Compute &" in evidence.extracted_markdown
    assert "Networking" in evidence.extracted_markdown
    assert evidence.retrieved_chunk == nvidia_hit.text
    assert evidence.selection_reason == nvidia_hit.selection_reason
    assert dict(evidence.source_details) == {
        "Report": "NVIDIA FY2026 annual report",
        "Section": (
            "AI Reinvents\nComputer Graphics > NVIDIA Corporation > Forward-Looking Statements "
            "> NOTICE OF 2026 ANNUAL MEETING OF STOCKHOLDERS > NCGC > Note 16 - "
            "Segment Information"
        ),
        "Reporting period": "FY2026",
        "Page": "165",
        "Unit": "USD millions",
        "Source": (
            "https://s201.q4cdn.com/141608511/files/doc_financials/2026/ar/"
            "2026-Annual-Report-Web.pdf"
        ),
        "Document hash": "0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c",
    }


def test_how_it_worked_keeps_technical_lineage_out_of_answer_and_evidence() -> None:
    """Putting ranks or trace data in the public answer must fail this test."""

    result = recorded_result()
    view = to_run_view(result)

    assert len(view.how_it_worked.pipeline_steps) == 5
    assert view.how_it_worked.retrieval_details[0].channel_ranks == (
        ("bm25", 1),
        ("dense", 4),
    )
    assert len(view.how_it_worked.tool_activity) == len(result.observations)
    assert len(view.how_it_worked.trace) == len(result.trajectory)
    assert [row.score for row in view.how_it_worked.scores] == ["100%"] * 5
    assert view.how_it_worked.model_route == "Recorded demo · recorded-capstone-v1"
    assert view.answer is not None
    assert "bm25" not in view.answer.model_dump_json().casefold()
    assert "dense" not in view.answer.model_dump_json().casefold()
    assert "trajectory" not in view.evidence[0].model_dump_json().casefold()


class _MissingSchneiderRetriever:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def search(self, company: str, query: str, top_k: int = 2):
        if company == "Schneider Electric":
            return ()
        return self._wrapped.search(company, query, top_k)


def test_blocked_run_withholds_the_answer_but_retains_safe_evidence_context() -> None:
    """Releasing a conclusion after insufficient evidence must fail this test."""

    complete = build_reference_copilot()
    result = build_reference_copilot(
        retriever=_MissingSchneiderRetriever(complete.retriever)
    ).run(ResearchRequest.reference())
    view = to_run_view(result)

    assert view.answer is None
    assert view.outcome.status == "blocked"
    assert view.release.missing_requirements == ("Schneider Electric contextual table evidence",)
    assert len(view.evidence) == 1


def test_core_capstone_import_does_not_require_optional_streamlit() -> None:
    probe = """
import sys

class BlockStreamlit:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'streamlit' or fullname.startswith('streamlit.'):
            raise ModuleNotFoundError('blocked optional streamlit import')
        return None

sys.meta_path.insert(0, BlockStreamlit())
import finai_academy.capstone
"""

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
