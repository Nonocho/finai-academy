from __future__ import annotations

import json
import subprocess
import sys

from finai_academy import capstone
from finai_academy.capstone import ResearchRequest, build_reference_copilot


def recorded_result():
    return build_reference_copilot(run_id_factory=lambda: "reference-run-001").run(
        ResearchRequest.reference()
    )


def test_run_view_is_json_serializable_and_contains_only_public_display_fields() -> None:
    assert hasattr(capstone, "to_run_view")
    view = capstone.to_run_view(recorded_result())

    payload = view.model_dump(mode="json")
    encoded = json.dumps(payload)

    assert payload["run_id"] == "reference-run-001"
    assert payload["readiness"]["provider"] == "Recorded demo"
    assert payload["readiness"]["data_mode"] == "Certified snapshots"
    assert payload["release"]["decision"] == "Release passed"
    assert "arguments" not in encoded
    assert "private_reasoning" not in encoded
    assert "system_prompt" not in encoded
    assert "api_key" not in encoded.casefold()
    assert "/Users/" not in encoded


def test_fact_rows_preserve_company_and_document_or_metric_provenance() -> None:
    view = capstone.to_run_view(recorded_result())

    metric = next(row for row in view.cited_facts if row.provenance == "Metric")
    document = next(row for row in view.cited_facts if row.provenance == "Document")

    assert metric.company == "NVIDIA"
    assert metric.source == "First Finance controlled classroom fixture"
    assert metric.evidence_id == "Not applicable"
    assert document.company == "NVIDIA"
    assert document.source == "assets/course-data/fixtures/nvidia_fy2026_excerpt.html"
    assert document.evidence_id == "NVDA-FY2026-GAMING-001"
    assert {row.company for row in view.evidence} == {"NVIDIA", "Schneider Electric"}
    assert all(row.provenance == "Document" for row in view.evidence)


def test_view_formats_plan_trace_scores_and_failed_evidence_gate() -> None:
    complete = recorded_result()
    complete_view = capstone.to_run_view(complete)
    insufficient = build_reference_copilot(
        retriever=_MissingSchneiderRetriever(build_reference_copilot().retriever)
    ).run(ResearchRequest.reference())
    failed_view = capstone.to_run_view(insufficient)

    assert complete_view.plan[0].step == "1"
    assert complete_view.replan_count == 1
    assert complete_view.trace[2].duration.endswith(" ms")
    assert [row.score for row in complete_view.scores] == ["100%"] * 5
    assert len(complete_view.scores) == 5
    assert complete_view.release.evidence_gate == "Evidence gate passed"
    assert complete_view.outcome.status == "passed"
    assert complete_view.outcome.message == "Release passed"
    assert complete_view.outcome.assistant_message == (
        "The evidence-backed research run completed. Review the public result below."
    )
    assert failed_view.release.decision == "Release blocked"
    assert failed_view.release.evidence_gate == "Evidence gate failed"
    assert failed_view.release.missing_requirements == (
        "Schneider Electric document evidence",
    )
    assert failed_view.briefing is None
    assert failed_view.outcome.status == "blocked"
    assert failed_view.outcome.message == "Release blocked"


def test_request_boundary_maps_data_mode_and_keeps_the_company_universe() -> None:
    assert hasattr(capstone, "build_capstone_request")

    reference = capstone.build_capstone_request(
        mode="reference",
        question=None,
        provider="recorded",
        model="recorded-capstone-v1",
        data_mode="certified",
    )
    custom = capstone.build_capstone_request(
        mode="custom",
        question="Compare operating-growth evidence.",
        provider="openai",
        model="gpt-5-mini",
        data_mode="live_enrichment",
    )

    assert reference.include_news is False
    assert custom.include_news is True
    assert custom.companies == ("NVIDIA", "Schneider Electric")


class _MissingSchneiderRetriever:
    def __init__(self, wrapped: object) -> None:
        self._wrapped = wrapped

    def search(self, company: str, query: str, top_k: int = 2):
        if company == "Schneider Electric":
            return ()
        return self._wrapped.search(company, query, top_k)


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
