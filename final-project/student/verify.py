"""Verify the four student seams and the certified recorded capstone route."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import subprocess
import sys
import tempfile
import threading
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
_OPERATION_FLAG = "--_operation-worker"
_OUTPUT_LIMIT = 64 * 1024
_CANDIDATE_LIMIT = 64 * 1024
_WORKER_TIMEOUT_SECONDS = 15
_WORKER_INCOMPLETE = 20
_WORKER_ERROR = 21
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


class _DiagnosticRegression(Exception):
    pass


class _OperationIncomplete(Exception):
    pass


class _OperationOutput(Exception):
    pass


class _OperationFailure(Exception):
    pass


class _MissingSchneiderRetriever:
    def __init__(self) -> None:
        self._wrapped = build_certified_retriever()

    def search(
        self, company: str, query: str, top_k: int = 2
    ) -> tuple[CapstoneEvidenceHit, ...]:
        if company == "Schneider Electric":
            return ()
        return self._wrapped.search(company, query, top_k)


class _DiagnosticRetriever:
    def __init__(self, drop_company: str | None) -> None:
        self._drop_company = drop_company
        self._wrapped = build_certified_retriever()

    def search(
        self, company: str, query: str, top_k: int = 2
    ) -> tuple[CapstoneEvidenceHit, ...]:
        if company == self._drop_company:
            return ()
        return self._wrapped.search(company, query, top_k)


def _certified_hits() -> tuple[CapstoneEvidenceHit, ...]:
    retriever = build_certified_retriever()
    return (
        *retriever.search("NVIDIA", "operating growth", top_k=1),
        *retriever.search("Schneider Electric", "operating growth", top_k=1),
    )


def _check_retriever(run_operation: Callable[[str, dict[str, object]], object]) -> None:
    for company, query, expected_ids in _RETRIEVER_CASES:
        candidate = run_operation(
            "wire_retriever",
            {"company": company, "query": query},
        )
        _validate_retriever_hits(candidate, company, expected_ids)


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


def _check_capabilities(run_operation: Callable[[str, dict[str, object]], object]) -> None:
    for discovered, expected in _CAPABILITY_CASES:
        candidate = run_operation(
            "register_analyst_capabilities",
            {"discovered": list(discovered)},
        )
        _validate_capabilities(candidate, expected)


def _validate_capabilities(capabilities: object, expected: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(capabilities, tuple) or capabilities != expected:
        raise ValueError("capability contract failed")
    if any(not isinstance(capability, str) for capability in capabilities):
        raise ValueError("capability contract failed")
    return capabilities


def _check_evidence_gate(
    run_operation: Callable[[str, dict[str, object]], object],
) -> None:
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
        candidate = run_operation(
            "evaluate_student_evidence_gate",
            {"hits": [hit.model_dump(mode="json") for hit in selected]},
        )
        _validate_evidence_gate(
            candidate,
            selected,
            passed=passed,
            coverage=coverage,
            missing=missing,
        )
    nvidia = nvidia_only[0]
    forged = nvidia.model_copy(update={"company": "Schneider Electric"})
    selected = (nvidia, forged)
    candidate = run_operation(
        "evaluate_student_evidence_gate",
        {"hits": [hit.model_dump(mode="json") for hit in selected]},
    )
    _validate_evidence_gate(
        candidate,
        selected,
        passed=False,
        coverage={"NVIDIA": ("document",), "Schneider Electric": ()},
        missing=("Schneider Electric document evidence",),
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


def _check_public_view(
    run_operation: Callable[[str, dict[str, object]], object],
    result: ResearchRunResult,
    stopped_result: ResearchRunResult,
) -> None:
    complete = run_operation(
        "assemble_public_briefing_view",
        {"result": result.model_dump(mode="json")},
    )
    _validate_public_view(complete, result, released=True)
    stopped = run_operation(
        "assemble_public_briefing_view",
        {"result": stopped_result.model_dump(mode="json")},
    )
    _validate_public_view(
        stopped,
        stopped_result,
        released=False,
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
    except _OperationOutput:
        raise
    except BaseException as error:
        raise _IntegratedContractFailure(seam) from error


def _check_integrated_student_path(
    run_operation: Callable[[str, dict[str, object]], object],
    result: ResearchRunResult,
) -> None:
    discovered = ("search_financial_documents", "place_order", "get_company_metric")
    _integrated_stage(
        "register_analyst_capabilities",
        lambda: _validate_capabilities(
            run_operation(
                "register_analyst_capabilities",
                {"discovered": list(discovered)},
            ),
            ("get_company_metric", "search_financial_documents"),
        ),
    )
    query = "reference mission operating growth"
    nvidia_hits = _integrated_stage(
        "wire_retriever",
        lambda: _validate_retriever_hits(
            run_operation(
                "wire_retriever",
                {"company": "NVIDIA", "query": query},
            ),
            "NVIDIA",
            ("NVDA-FY2026-DATA-CENTER-001", "NVDA-FY2026-GAMING-001"),
        ),
    )
    schneider_hits = _integrated_stage(
        "wire_retriever",
        lambda: _validate_retriever_hits(
            run_operation(
                "wire_retriever",
                {"company": "Schneider Electric", "query": query},
            ),
            "Schneider Electric",
            ("SU-FY2025-ENERGY-MANAGEMENT-002", "SU-FY2025-ENERGY-MANAGEMENT-001"),
        ),
    )
    hits = tuple(nvidia_hits) + tuple(schneider_hits)
    decision = _integrated_stage(
        "evaluate_student_evidence_gate",
        lambda: _validate_evidence_gate(
            run_operation(
                "evaluate_student_evidence_gate",
                {"hits": [hit.model_dump(mode="json") for hit in hits]},
            ),
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
            run_operation(
                "assemble_public_briefing_view",
                {"result": integrated_result.model_dump(mode="json")},
            ),
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


def _check_diagnostic_case() -> None:
    payload = json.loads(Path(__file__).with_name("diagnostic_case.json").read_text())
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "drop_company"}:
        raise ValueError("diagnostic case contract failed")
    if payload["schema_version"] != 1 or payload["drop_company"] not in {
        None,
        "Schneider Electric",
    }:
        raise ValueError("diagnostic case contract failed")
    result = build_reference_copilot(
        retriever=_DiagnosticRetriever(payload["drop_company"]),
        run_id_factory=lambda: "student-diagnostic-gate",
    ).run(ResearchRequest.reference())
    if (
        result.status == "insufficient_evidence"
        and result.evidence_gate.missing_requirements
        == ("Schneider Electric document evidence",)
    ):
        raise _DiagnosticRegression
    if result.status != "completed" or not result.deterministic_evaluation.release_passed:
        raise ValueError("diagnostic case contract failed")


def _serialize_candidate(operation: str, value: object) -> dict[str, object] | None:
    if operation == "wire_retriever":
        if type(value) is not tuple or any(
            not isinstance(item, CapstoneEvidenceHit) for item in value
        ):
            return None
        return {"hits": [item.model_dump(mode="json") for item in value]}
    if operation == "register_analyst_capabilities":
        if type(value) is not tuple or any(type(item) is not str for item in value):
            return None
        return {"capabilities": list(value)}
    if operation == "evaluate_student_evidence_gate":
        if not isinstance(value, EvidenceGateDecision):
            return None
        return {"decision": value.model_dump(mode="json")}
    if operation == "assemble_public_briefing_view":
        if not isinstance(value, CapstoneRunView):
            return None
        return {"view": value.model_dump(mode="json")}
    return None


def _operation_worker(operation: str, input_path: Path, output_path: Path) -> int:
    try:
        raw_input = input_path.read_bytes()
        if not raw_input or len(raw_input) > _CANDIDATE_LIMIT:
            return _WORKER_ERROR
        payload = json.loads(raw_input)
        student = importlib.import_module("integration")
        incomplete_type = getattr(student, "StudentIntegrationIncomplete", None)
        if operation == "wire_retriever" and set(payload) == {"company", "query"}:
            invoke = lambda: student.wire_retriever(payload["company"], payload["query"])
        elif operation == "register_analyst_capabilities" and set(payload) == {
            "discovered"
        }:
            invoke = lambda: student.register_analyst_capabilities(payload["discovered"])
        elif operation == "evaluate_student_evidence_gate" and set(payload) == {"hits"}:
            hits = tuple(CapstoneEvidenceHit.model_validate(item) for item in payload["hits"])
            invoke = lambda: student.evaluate_student_evidence_gate(hits)
        elif operation == "assemble_public_briefing_view" and set(payload) == {"result"}:
            result = ResearchRunResult.model_validate(payload["result"])
            invoke = lambda: student.assemble_public_briefing_view(result)
        else:
            return _WORKER_ERROR
        try:
            value = invoke()
        except BaseException as error:  # noqa: BLE001 - learner failures stay private
            if isinstance(incomplete_type, type) and isinstance(error, incomplete_type):
                return _WORKER_INCOMPLETE
            return _WORKER_ERROR
        candidate = _serialize_candidate(operation, value)
        if candidate is None:
            return _WORKER_ERROR
        encoded = json.dumps(candidate, separators=(",", ":")).encode("utf-8")
        if len(encoded) > _CANDIDATE_LIMIT:
            return _WORKER_ERROR
        output_path.write_bytes(encoded)
    except BaseException:  # noqa: BLE001 - worker never discloses learner details
        return _WORKER_ERROR
    return 0


def _run_bounded_process(arguments: list[str]) -> tuple[int, bool, bool, bool]:
    try:
        process = subprocess.Popen(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError:
        return _WORKER_ERROR, False, False, False

    streams = (process.stdout, process.stderr)
    buffers = (bytearray(), bytearray())
    overflow = threading.Event()

    def drain(index: int) -> None:
        stream = streams[index]
        if stream is None:
            return
        try:
            while chunk := stream.read(4096):
                remaining = _OUTPUT_LIMIT - len(buffers[index])
                if remaining > 0:
                    buffers[index].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    overflow.set()
                    try:
                        process.kill()
                    except OSError:
                        pass
                    break
        finally:
            stream.close()

    threads = tuple(
        threading.Thread(target=drain, args=(index,), daemon=True) for index in range(2)
    )
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        return_code = process.wait(timeout=_WORKER_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        return_code = process.wait()
    for thread in threads:
        thread.join(timeout=1)
    return return_code, bool(buffers[0]), bool(buffers[1]), overflow.is_set() or timed_out


def _decode_candidate(operation: str, raw: bytes) -> object:
    if not raw or len(raw) > _CANDIDATE_LIMIT:
        raise _OperationFailure
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise _OperationFailure
        if operation == "wire_retriever" and set(payload) == {"hits"}:
            return tuple(CapstoneEvidenceHit.model_validate(item) for item in payload["hits"])
        if operation == "register_analyst_capabilities" and set(payload) == {
            "capabilities"
        }:
            capabilities = payload["capabilities"]
            if not isinstance(capabilities, list) or any(
                type(item) is not str for item in capabilities
            ):
                raise _OperationFailure
            return tuple(capabilities)
        if operation == "evaluate_student_evidence_gate" and set(payload) == {"decision"}:
            return EvidenceGateDecision.model_validate(payload["decision"])
        if operation == "assemble_public_briefing_view" and set(payload) == {"view"}:
            return CapstoneRunView.model_validate(payload["view"])
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise _OperationFailure from error
    raise _OperationFailure


def _operation_runner(directory: Path) -> Callable[[str, dict[str, object]], object]:
    counter = 0

    def run(operation: str, payload: dict[str, object]) -> object:
        nonlocal counter
        counter += 1
        input_path = directory / f"input-{counter}.json"
        output_path = directory / f"candidate-{counter}.json"
        input_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return_code, stdout_seen, stderr_seen, bounded_failure = _run_bounded_process(
            [
                sys.executable,
                __file__,
                _OPERATION_FLAG,
                operation,
                str(input_path),
                str(output_path),
            ]
        )
        if stdout_seen or stderr_seen or bounded_failure:
            raise _OperationOutput
        if return_code == _WORKER_INCOMPLETE:
            raise _OperationIncomplete
        if return_code != 0:
            raise _OperationFailure
        try:
            if output_path.stat().st_size > _CANDIDATE_LIMIT:
                raise _OperationFailure
            with output_path.open("rb") as candidate_file:
                raw_candidate = candidate_file.read(_CANDIDATE_LIMIT + 1)
        except OSError as error:
            raise _OperationFailure from error
        return _decode_candidate(operation, raw_candidate)

    return run


def _run_check(name: str, check: Callable[[], None]) -> tuple[bool, str]:
    try:
        check()
    except _OperationOutput:
        raise
    except _IntegratedContractFailure as error:
        return False, f"FAIL {error.seam}: integrated contract did not complete."
    except _OperationIncomplete:
        hint = _INCOMPLETE_HINTS.get(name, "Complete this integration seam.")
        return False, f"FAIL {name}: incomplete — {hint}"
    except _DiagnosticRegression:
        return (
            False,
            "FAIL diagnostic_case: regressed evidence routing still drops Schneider Electric.",
        )
    except BaseException:  # noqa: BLE001 - diagnostics must remain sanitized
        return False, f"FAIL {name}: contract did not complete."
    return True, f"PASS {name}"


def _parent_main() -> int:
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

    def require_result() -> ResearchRunResult:
        if result is None:
            raise ValueError("recorded route unavailable")
        return result

    def require_stopped_result() -> ResearchRunResult:
        if stopped_result is None:
            raise ValueError("stopped route unavailable")
        return stopped_result

    lines: list[str] = []
    passed = True
    try:
        with tempfile.TemporaryDirectory(prefix="finai-student-operations-") as directory:
            run_operation = _operation_runner(Path(directory))
            seam_checks: tuple[tuple[str, Callable[[], None]], ...] = (
                ("wire_retriever", lambda: _check_retriever(run_operation)),
                (
                    "register_analyst_capabilities",
                    lambda: _check_capabilities(run_operation),
                ),
                (
                    "evaluate_student_evidence_gate",
                    lambda: _check_evidence_gate(run_operation),
                ),
                (
                    "assemble_public_briefing_view",
                    lambda: _check_public_view(
                        run_operation,
                        require_result(),
                        require_stopped_result(),
                    ),
                ),
            )
            seam_results = tuple(_run_check(name, check) for name, check in seam_checks)
            lines.extend(line for _, line in seam_results)
            passed = all(status for status, _ in seam_results)
            if passed:
                integrated_passed, integrated_line = _run_check(
                    "integrated_student_path",
                    lambda: _check_integrated_student_path(
                        run_operation,
                        require_result(),
                    ),
                )
                lines.append(integrated_line)
                passed = integrated_passed
    except _OperationOutput:
        print("FAIL verifier: isolated worker emitted output.")
        return 1

    trusted_checks: tuple[tuple[str, Callable[[], None]], ...] = (
        ("diagnostic_case", _check_diagnostic_case),
        ("reference_mission", lambda: _check_reference_mission(require_result())),
        ("citation_integrity", lambda: _check_citation_integrity(require_result())),
        ("deterministic_release", lambda: _check_release(require_result())),
        ("persistence", lambda: _check_persistence(require_result())),
    )
    for name, check in trusted_checks:
        check_passed, line = _run_check(name, check)
        lines.append(line)
        passed = check_passed and passed
    for line in lines:
        print(line)
    if not passed:
        return 1
    print(_SUCCESS_MARKER)
    return 0


def main() -> int:
    if len(sys.argv) == 5 and sys.argv[1] == _OPERATION_FLAG:
        return _operation_worker(
            sys.argv[2],
            Path(sys.argv[3]),
            Path(sys.argv[4]),
        )
    if len(sys.argv) != 1:
        print("FAIL verifier: unsupported invocation.")
        return 1
    return _parent_main()


if __name__ == "__main__":
    raise SystemExit(main())
