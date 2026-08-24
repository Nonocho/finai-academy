"""Verify the four student seams and the certified recorded capstone route."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

from finai_academy.capstone import ResearchRequest, build_reference_copilot
from finai_academy.capstone.models import (
    CapstoneEvidenceHit,
    EvidenceGateDecision,
    ResearchRunResult,
)
from finai_academy.capstone.persistence import CapstoneRunStore
from finai_academy.capstone.tools import build_certified_retriever
from finai_academy.capstone.views import CapstoneRunView

_SUCCESS_MARKER = "CAPSTONE_PASS"
_WORKER_FLAG = "--_isolated-worker-fd"
_RESULT_VERSION = 1
_OS_WRITE = os.write
StudentIntegrationIncomplete: type[BaseException]
assemble_public_briefing_view: Callable[[ResearchRunResult], object]
evaluate_student_evidence_gate: Callable[[Sequence[CapstoneEvidenceHit]], object]
register_analyst_capabilities: Callable[[Sequence[str]], object]
wire_retriever: Callable[[str, str], object]
_METRIC_NAMES = (
    "tool_call_correctness",
    "tool_call_efficiency",
    "answer_relevance",
    "answer_completeness",
    "citation_integrity",
)
_INCOMPLETE_HINTS = {
    "wire_retriever": "Connect the certified company-scoped retrieval boundary.",
    "register_analyst_capabilities": "Apply discovery through the approved read-tool policy.",
    "evaluate_student_evidence_gate": "Require document evidence for both companies.",
    "assemble_public_briefing_view": "Use the display-safe public view boundary.",
}
_RETRIEVER_CASES = (
    (
        "NVIDIA",
        "gaming revenue growth",
        ("NVDA-FY2026-GAMING-001", "NVDA-FY2026-DATA-CENTER-001"),
    ),
    (
        "NVIDIA",
        "data center growth",
        ("NVDA-FY2026-DATA-CENTER-001", "NVDA-FY2026-GAMING-001"),
    ),
    (
        "Schneider Electric",
        "energy management growth",
        ("SU-FY2025-ENERGY-MANAGEMENT-002", "SU-FY2025-ENERGY-MANAGEMENT-001"),
    ),
    (
        "Schneider Electric",
        "adjusted EBITA margin",
        ("SU-FY2025-ENERGY-MANAGEMENT-001", "SU-FY2025-ENERGY-MANAGEMENT-002"),
    ),
)
_CAPABILITY_CASES = (
    (
        ("place_order", "search_financial_documents", "get_company_metric"),
        ("get_company_metric", "search_financial_documents"),
    ),
    (("search_financial_documents", "place_order"), ("search_financial_documents",)),
    (("get_company_metric", "delete_portfolio"), ("get_company_metric",)),
    (("place_order", "place_order"), ()),
)


class _IntegratedContractFailure(Exception):
    def __init__(self, seam: str) -> None:
        self.seam = seam
        super().__init__(seam)


class _MissingSchneiderRetriever:
    def __init__(self) -> None:
        self._wrapped = build_certified_retriever()

    def search(
        self, company: str, query: str, top_k: int = 2
    ) -> tuple[CapstoneEvidenceHit, ...]:
        if company == "Schneider Electric":
            return ()
        return self._wrapped.search(company, query, top_k)


def _certified_hits() -> tuple[CapstoneEvidenceHit, ...]:
    retriever = build_certified_retriever()
    return (
        *retriever.search("NVIDIA", "operating growth", top_k=1),
        *retriever.search("Schneider Electric", "operating growth", top_k=1),
    )


def _check_retriever() -> None:
    for company, query, expected_ids in _RETRIEVER_CASES:
        _validate_retriever_hits(wire_retriever(company, query), company, expected_ids)


def _validate_retriever_hits(
    hits: object,
    company: str,
    expected_ids: tuple[str, ...],
) -> tuple[CapstoneEvidenceHit, ...]:
    if not isinstance(hits, tuple) or not hits:
        raise ValueError("retriever contract failed")
    if any(not isinstance(hit, CapstoneEvidenceHit) for hit in hits):
        raise ValueError("retriever contract failed")
    typed_hits = tuple(hit for hit in hits if isinstance(hit, CapstoneEvidenceHit))
    if any(hit.company != company for hit in typed_hits):
        raise ValueError("retriever contract failed")
    if tuple(hit.evidence_id for hit in typed_hits) != expected_ids:
        raise ValueError("retriever contract failed")
    if any(
        not hit.source_reference or not hit.document_id or not hit.section or not hit.period
        for hit in typed_hits
    ):
        raise ValueError("retriever contract failed")
    return typed_hits


def _check_capabilities() -> None:
    for discovered, expected in _CAPABILITY_CASES:
        _validate_capabilities(register_analyst_capabilities(discovered), expected)


def _validate_capabilities(capabilities: object, expected: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(capabilities, tuple) or capabilities != expected:
        raise ValueError("capability contract failed")
    if any(not isinstance(capability, str) for capability in capabilities):
        raise ValueError("capability contract failed")
    return capabilities


def _check_evidence_gate() -> None:
    hits = _certified_hits()
    nvidia_only = tuple(hit for hit in hits if hit.company == "NVIDIA")
    schneider_only = tuple(hit for hit in hits if hit.company == "Schneider Electric")
    cases = (
        (
            tuple(reversed(hits)),
            True,
            {"NVIDIA": ("document",), "Schneider Electric": ("document",)},
            (),
        ),
        (
            nvidia_only,
            False,
            {"NVIDIA": ("document",), "Schneider Electric": ()},
            ("Schneider Electric document evidence",),
        ),
        (
            schneider_only,
            False,
            {"NVIDIA": (), "Schneider Electric": ("document",)},
            ("NVIDIA document evidence",),
        ),
        (
            (),
            False,
            {"NVIDIA": (), "Schneider Electric": ()},
            ("NVIDIA document evidence", "Schneider Electric document evidence"),
        ),
    )
    for selected, passed, coverage, missing in cases:
        _validate_evidence_gate(
            evaluate_student_evidence_gate(selected),
            selected,
            passed=passed,
            coverage=coverage,
            missing=missing,
        )


def _validate_evidence_gate(
    decision: object,
    hits: Sequence[CapstoneEvidenceHit],
    *,
    passed: bool,
    coverage: dict[str, tuple[str, ...]],
    missing: tuple[str, ...],
) -> EvidenceGateDecision:
    if not isinstance(decision, EvidenceGateDecision):
        raise TypeError("evidence gate contract failed")
    if decision.passed is not passed:
        raise ValueError("evidence gate contract failed")
    if decision.coverage != coverage or decision.missing_requirements != missing:
        raise ValueError("evidence gate contract failed")
    if decision.evidence_hits != tuple(hits):
        raise ValueError("evidence gate contract failed")
    return decision


def _check_public_view(result: ResearchRunResult, stopped_result: ResearchRunResult) -> None:
    _validate_public_view(assemble_public_briefing_view(result), result, released=True)
    _validate_public_view(
        assemble_public_briefing_view(stopped_result), stopped_result, released=False
    )


def _validate_public_view(
    view: object,
    result: ResearchRunResult,
    *,
    released: bool,
) -> CapstoneRunView:
    if not isinstance(view, CapstoneRunView) or view.run_id != result.run_id:
        raise ValueError("public view contract failed")
    expected_decision = "Release passed" if released else "Release blocked"
    if view.release.decision != expected_decision:
        raise ValueError("public view contract failed")
    if (view.briefing is not None) is not released:
        raise ValueError("public view contract failed")
    encoded = view.model_dump_json()
    if (
        '"arguments"' in encoded
        or "/Users/" in encoded
        or "OPENAI_API_KEY" in encoded
        or "sk-" in encoded
    ):
        raise ValueError("public view contract failed")
    return view


def _integrated_stage(seam: str, operation: Callable[[], object]) -> object:
    try:
        return operation()
    except BaseException as error:
        raise _IntegratedContractFailure(seam) from error


def _check_integrated_student_path(result: ResearchRunResult) -> None:
    discovered = ("search_financial_documents", "place_order", "get_company_metric")
    _integrated_stage(
        "register_analyst_capabilities",
        lambda: _validate_capabilities(
            register_analyst_capabilities(discovered),
            ("get_company_metric", "search_financial_documents"),
        ),
    )
    query = "reference mission operating growth"
    nvidia_hits = _integrated_stage(
        "wire_retriever",
        lambda: _validate_retriever_hits(
            wire_retriever("NVIDIA", query),
            "NVIDIA",
            ("NVDA-FY2026-DATA-CENTER-001", "NVDA-FY2026-GAMING-001"),
        ),
    )
    schneider_hits = _integrated_stage(
        "wire_retriever",
        lambda: _validate_retriever_hits(
            wire_retriever("Schneider Electric", query),
            "Schneider Electric",
            ("SU-FY2025-ENERGY-MANAGEMENT-002", "SU-FY2025-ENERGY-MANAGEMENT-001"),
        ),
    )
    hits = tuple(nvidia_hits) + tuple(schneider_hits)
    decision = _integrated_stage(
        "evaluate_student_evidence_gate",
        lambda: _validate_evidence_gate(
            evaluate_student_evidence_gate(hits),
            hits,
            passed=True,
            coverage={"NVIDIA": ("document",), "Schneider Electric": ("document",)},
            missing=(),
        ),
    )
    payload = result.model_dump(mode="python")
    payload["evidence_gate"] = decision.model_dump(mode="python")
    integrated_result = ResearchRunResult.model_validate(payload)
    _integrated_stage(
        "assemble_public_briefing_view",
        lambda: _validate_public_view(
            assemble_public_briefing_view(integrated_result),
            integrated_result,
            released=True,
        ),
    )


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


def _run_check(name: str, check: Callable[[], None]) -> tuple[bool, str]:
    try:
        check()
    except BaseException as error:  # noqa: BLE001 - learner exits must be sanitized
        failure: BaseException | None = error
    else:
        failure = None

    if isinstance(failure, _IntegratedContractFailure):
        return False, f"FAIL {failure.seam}: integrated contract did not complete."
    if isinstance(failure, StudentIntegrationIncomplete):
        hint = _INCOMPLETE_HINTS.get(name, "Complete this integration seam.")
        return False, f"FAIL {name}: incomplete — {hint}"
    if failure is not None:
        return False, f"FAIL {name}: contract did not complete."
    return True, f"PASS {name}"


def _load_student_integration() -> None:
    student = importlib.import_module("integration")
    required = (
        "StudentIntegrationIncomplete",
        "assemble_public_briefing_view",
        "evaluate_student_evidence_gate",
        "register_analyst_capabilities",
        "wire_retriever",
    )
    if any(not hasattr(student, name) for name in required):
        raise ImportError("student integration contract is incomplete")

    global StudentIntegrationIncomplete
    global assemble_public_briefing_view
    global evaluate_student_evidence_gate
    global register_analyst_capabilities
    global wire_retriever
    StudentIntegrationIncomplete = student.StudentIntegrationIncomplete
    assemble_public_briefing_view = student.assemble_public_briefing_view
    evaluate_student_evidence_gate = student.evaluate_student_evidence_gate
    register_analyst_capabilities = student.register_analyst_capabilities
    wire_retriever = student.wire_retriever


def _worker_result() -> dict[str, object]:
    try:
        _load_student_integration()
    except BaseException:  # noqa: BLE001 - imports must fail without disclosure
        return {
            "version": _RESULT_VERSION,
            "passed": False,
            "lines": ["FAIL verifier: worker could not load integration."],
        }

    try:
        result = build_reference_copilot(run_id_factory=lambda: "student-verification-run").run(
            ResearchRequest.reference()
        )
        stopped_result = build_reference_copilot(
            retriever=_MissingSchneiderRetriever(),
            run_id_factory=lambda: "student-stopped-run",
        ).run(ResearchRequest.reference())
    except Exception:  # noqa: BLE001 - verifier output must remain path-free
        result = None
        stopped_result = None

    def with_result(check: Callable[[ResearchRunResult], None]) -> Callable[[], None]:
        def run() -> None:
            if result is None:
                raise ValueError("recorded route unavailable")
            check(result)

        return run

    def with_results(
        check: Callable[[ResearchRunResult, ResearchRunResult], None],
    ) -> Callable[[], None]:
        def run() -> None:
            if result is None or stopped_result is None:
                raise ValueError("recorded routes unavailable")
            check(result, stopped_result)

        return run

    seam_checks: tuple[tuple[str, Callable[[], None]], ...] = (
        ("wire_retriever", _check_retriever),
        ("register_analyst_capabilities", _check_capabilities),
        ("evaluate_student_evidence_gate", _check_evidence_gate),
        ("assemble_public_briefing_view", with_results(_check_public_view)),
    )
    other_checks: tuple[tuple[str, Callable[[], None]], ...] = (
        ("reference_mission", with_result(_check_reference_mission)),
        ("citation_integrity", with_result(_check_citation_integrity)),
        ("deterministic_release", with_result(_check_release)),
        ("persistence", with_result(_check_persistence)),
    )
    seam_results = tuple(_run_check(name, check) for name, check in seam_checks)
    lines = [line for _, line in seam_results]
    passed = all(status for status, _ in seam_results)
    if passed:
        integrated_passed, integrated_line = _run_check(
            "integrated_student_path",
            with_result(_check_integrated_student_path),
        )
        lines.append(integrated_line)
        passed = integrated_passed
    for name, check in other_checks:
        check_passed, line = _run_check(name, check)
        lines.append(line)
        passed = check_passed and passed
    return {"version": _RESULT_VERSION, "passed": passed, "lines": lines}


def _worker_main(result_fd: int) -> int:
    payload = json.dumps(_worker_result(), separators=(",", ":")).encode("utf-8")
    try:
        _OS_WRITE(result_fd, payload)
    except OSError:
        return 2
    finally:
        os.close(result_fd)
    return 0


def _allowed_result_lines() -> set[str]:
    check_names = (
        *_INCOMPLETE_HINTS,
        "reference_mission",
        "citation_integrity",
        "deterministic_release",
        "persistence",
        "integrated_student_path",
    )
    lines = {f"PASS {name}" for name in check_names}
    lines.update(f"FAIL {name}: contract did not complete." for name in check_names)
    lines.update(
        f"FAIL {name}: incomplete — {hint}"
        for name, hint in _INCOMPLETE_HINTS.items()
    )
    lines.update(
        f"FAIL {name}: integrated contract did not complete."
        for name in _INCOMPLETE_HINTS
    )
    lines.add("FAIL verifier: worker could not load integration.")
    return lines


def _validate_worker_result(raw: bytes) -> tuple[bool, list[str]] | None:
    if not raw or len(raw) > 16_384:
        return None
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or set(payload) != {"version", "passed", "lines"}:
        return None
    if payload["version"] != _RESULT_VERSION or type(payload["passed"]) is not bool:
        return None
    lines = payload["lines"]
    if not isinstance(lines, list) or not 1 <= len(lines) <= 9:
        return None
    allowed = _allowed_result_lines()
    if any(type(line) is not str or line not in allowed for line in lines):
        return None
    passed = payload["passed"]
    if passed != all(line.startswith("PASS ") for line in lines):
        return None
    expected_success = [
        *(f"PASS {name}" for name in _INCOMPLETE_HINTS),
        "PASS integrated_student_path",
        "PASS reference_mission",
        "PASS citation_integrity",
        "PASS deterministic_release",
        "PASS persistence",
    ]
    if passed and lines != expected_success:
        return None
    return passed, lines


def _parent_main() -> int:
    read_fd, write_fd = os.pipe()
    try:
        try:
            completed = subprocess.run(
                [sys.executable, __file__, _WORKER_FLAG, str(write_fd)],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                timeout=60,
                pass_fds=(write_fd,),
            )
        except (OSError, subprocess.SubprocessError):
            print("FAIL verifier: isolated worker failed safely.")
            return 1
        finally:
            os.close(write_fd)
        raw_result = os.read(read_fd, 16_385)
    finally:
        os.close(read_fd)

    if completed.stdout or completed.stderr:
        print("FAIL verifier: isolated worker emitted output.")
        return 1
    if completed.returncode != 0:
        print("FAIL verifier: isolated worker failed safely.")
        return 1
    validated = _validate_worker_result(raw_result)
    if validated is None:
        print("FAIL verifier: isolated worker returned an invalid result.")
        return 1
    passed, lines = validated
    for line in lines:
        print(line)
    if not passed:
        return 1
    print(_SUCCESS_MARKER)
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == _WORKER_FLAG:
        try:
            result_fd = int(sys.argv[2])
        except ValueError:
            return 2
        return _worker_main(result_fd)
    if len(sys.argv) != 1:
        print("FAIL verifier: unsupported invocation.")
        return 1
    return _parent_main()


if __name__ == "__main__":
    raise SystemExit(main())
