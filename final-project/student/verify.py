"""Verify the four student seams and the certified recorded capstone route."""

from __future__ import annotations

import contextlib
import importlib
import io
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path

from integration import (
    StudentIntegrationIncomplete,
    assemble_public_briefing_view,
    evaluate_student_evidence_gate,
    register_analyst_capabilities,
    wire_retriever,
)

from finai_academy.capstone import ResearchRequest, build_reference_copilot
from finai_academy.capstone.models import CapstoneEvidenceHit, ResearchRunResult
from finai_academy.capstone.persistence import CapstoneRunStore
from finai_academy.capstone.tools import build_certified_retriever

_SUCCESS_MARKER = "CAPSTONE_PASS"
_METRIC_NAMES = (
    "tool_call_correctness",
    "tool_call_efficiency",
    "answer_relevance",
    "answer_completeness",
    "citation_integrity",
)


def _certified_hits() -> tuple[CapstoneEvidenceHit, ...]:
    retriever = build_certified_retriever()
    return (
        *retriever.search("NVIDIA", "operating growth", top_k=1),
        *retriever.search("Schneider Electric", "operating growth", top_k=1),
    )


def _check_retriever() -> None:
    hits = wire_retriever("NVIDIA", "operating growth")
    if not isinstance(hits, tuple) or not hits:
        raise ValueError("retriever contract failed")
    if any(hit.company != "NVIDIA" for hit in hits):
        raise ValueError("retriever contract failed")
    if any(not hit.evidence_id or not hit.source_reference for hit in hits):
        raise ValueError("retriever contract failed")


def _check_capabilities() -> None:
    capabilities = register_analyst_capabilities(
        ("place_order", "search_financial_documents", "get_company_metric")
    )
    if capabilities != ("get_company_metric", "search_financial_documents"):
        raise ValueError("capability contract failed")


def _check_evidence_gate() -> None:
    hits = _certified_hits()
    decision = evaluate_student_evidence_gate(hits)
    if not decision.passed or decision.evidence_hits != hits:
        raise ValueError("evidence gate contract failed")
    if decision.coverage != {
        "NVIDIA": ("document",),
        "Schneider Electric": ("document",),
    }:
        raise ValueError("evidence gate contract failed")
    nvidia_only = tuple(hit for hit in hits if hit.company == "NVIDIA")
    blocked = evaluate_student_evidence_gate(nvidia_only)
    if blocked.passed:
        raise ValueError("evidence gate contract failed")
    if blocked.missing_requirements != ("Schneider Electric document evidence",):
        raise ValueError("evidence gate contract failed")


def _check_public_view(result: ResearchRunResult) -> None:
    view = assemble_public_briefing_view(result)
    if view.release.decision != "Release passed" or view.briefing is None:
        raise ValueError("public view contract failed")
    encoded = view.model_dump_json()
    if '"arguments"' in encoded or "/Users/" in encoded:
        raise ValueError("public view contract failed")


def _check_reference_mission(result: ResearchRunResult) -> None:
    if result.request != ResearchRequest.reference():
        raise ValueError("reference mission contract failed")
    if result.status != "completed" or result.provider != "recorded":
        raise ValueError("reference mission contract failed")
    if result.replan_count > 1 or len(result.observations) > 6:
        raise ValueError("reference mission contract failed")
    if not result.evidence_gate.passed or result.briefing is None:
        raise ValueError("reference mission contract failed")


def _check_citation_integrity(result: ResearchRunResult) -> None:
    briefing = result.briefing
    if briefing is None:
        raise ValueError("citation contract failed")
    hits_by_id = {hit.evidence_id: hit for hit in result.evidence_gate.evidence_hits}
    if not hits_by_id or len(hits_by_id) != len(result.evidence_gate.evidence_hits):
        raise ValueError("citation contract failed")
    for fact in briefing.cited_facts:
        if not fact.source_reference:
            raise ValueError("citation contract failed")
        if fact.provenance_kind == "document":
            hit = hits_by_id.get(fact.evidence_id or "")
            if hit is None:
                raise ValueError("citation contract failed")
            if hit.company != fact.company or hit.source_reference != fact.source_reference:
                raise ValueError("citation contract failed")
    expected_sources = tuple(dict.fromkeys(fact.source_reference for fact in briefing.cited_facts))
    if briefing.aggregate_sources != expected_sources:
        raise ValueError("citation contract failed")


def _check_release(result: ResearchRunResult) -> None:
    evaluation = result.deterministic_evaluation
    if tuple(metric.name for metric in evaluation.metrics) != _METRIC_NAMES:
        raise ValueError("release contract failed")
    if tuple(metric.value for metric in evaluation.metrics) != (1.0,) * 5:
        raise ValueError("release contract failed")
    if not evaluation.release_passed:
        raise ValueError("release contract failed")


def _check_persistence(result: ResearchRunResult) -> None:
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with (
        tempfile.TemporaryDirectory(prefix="finai-capstone-") as directory,
        contextlib.redirect_stdout(captured_stdout),
        contextlib.redirect_stderr(captured_stderr),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        references = CapstoneRunStore(Path(directory)).persist(result)
        mlflow = importlib.import_module("mlflow")
        mlflow.flush_trace_async_logging(terminate=True)
    if references.tracking_status != "persisted":
        raise ValueError("persistence contract failed")
    if not references.run_id or not references.trace_id:
        raise ValueError("persistence contract failed")
    if references.analysis_run_id != result.run_id:
        raise ValueError("persistence contract failed")
    if "/Users/" in references.model_dump_json():
        raise ValueError("persistence contract failed")


def _run_check(name: str, check: Callable[[], None]) -> bool:
    try:
        check()
    except StudentIntegrationIncomplete as status:
        print(f"FAIL {name}: {status.seam} — {status.hint}")
        return False
    except Exception:  # noqa: BLE001 - verifier output must not expose dependency details
        print(f"FAIL {name}: contract did not complete.")
        return False
    print(f"PASS {name}")
    return True


def main() -> int:
    try:
        result = build_reference_copilot(run_id_factory=lambda: "student-verification-run").run(
            ResearchRequest.reference()
        )
    except Exception:  # noqa: BLE001 - verifier output must remain path-free
        result = None

    def with_result(check: Callable[[ResearchRunResult], None]) -> Callable[[], None]:
        def run() -> None:
            if result is None:
                raise ValueError("recorded route unavailable")
            check(result)

        return run

    checks: tuple[tuple[str, Callable[[], None]], ...] = (
        ("wire_retriever", _check_retriever),
        ("register_analyst_capabilities", _check_capabilities),
        ("evaluate_student_evidence_gate", _check_evidence_gate),
        ("assemble_public_briefing_view", with_result(_check_public_view)),
        ("reference_mission", with_result(_check_reference_mission)),
        ("citation_integrity", with_result(_check_citation_integrity)),
        ("deterministic_release", with_result(_check_release)),
        ("persistence", with_result(_check_persistence)),
    )
    passed = True
    for name, check in checks:
        passed = _run_check(name, check) and passed
    if not passed:
        return 1
    print(_SUCCESS_MARKER)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
