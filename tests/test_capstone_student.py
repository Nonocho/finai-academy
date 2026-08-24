from __future__ import annotations

import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType

import pytest
from streamlit.testing.v1 import AppTest

from finai_academy.capstone import ResearchRequest, build_reference_copilot
from finai_academy.capstone.models import (
    CapstoneEvidenceHit,
    EvidenceGateDecision,
    ResearchRunResult,
)
from finai_academy.capstone.tools import AnalystToolRegistry, build_certified_retriever
from finai_academy.capstone.views import CapstoneRunView, to_run_view

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STUDENT_DIR = _PROJECT_ROOT / "final-project" / "student"
_INTEGRATION_PATH = _STUDENT_DIR / "integration.py"
_SEAM_NAMES = (
    "wire_retriever",
    "register_analyst_capabilities",
    "evaluate_student_evidence_gate",
    "assemble_public_briefing_view",
)


def _reference_wire_retriever(
    company: str, query: str
) -> tuple[CapstoneEvidenceHit, ...]:
    return build_certified_retriever().search(company, query)


def _reference_register_analyst_capabilities(
    discovered: Sequence[str],
) -> tuple[str, ...]:
    return AnalystToolRegistry(discovered=discovered).discover()


def _reference_evaluate_student_evidence_gate(
    hits: Sequence[CapstoneEvidenceHit],
) -> EvidenceGateDecision:
    companies = ("NVIDIA", "Schneider Electric")
    covered = {hit.company for hit in hits}
    missing = tuple(f"{company} document evidence" for company in companies if company not in covered)
    return EvidenceGateDecision(
        passed=not missing,
        coverage={company: (("document",) if company in covered else ()) for company in companies},
        missing_requirements=missing,
        evidence_hits=tuple(hits),
    )


def _reference_assemble_public_briefing_view(
    result: ResearchRunResult,
) -> CapstoneRunView:
    return to_run_view(result)


_REFERENCE_ADAPTERS = (
    _reference_wire_retriever,
    _reference_register_analyst_capabilities,
    _reference_evaluate_student_evidence_gate,
    _reference_assemble_public_briefing_view,
)
_ADAPTER_HEADER = '''"""Generated contract adapter; not a student answer source."""
from __future__ import annotations

import dataclasses
import sys
from collections.abc import Sequence

from finai_academy.capstone import ResearchRequest, build_reference_copilot
from finai_academy.capstone.models import (
    CapstoneEvidenceHit,
    EvidenceGateDecision,
    ResearchRunResult,
)
from finai_academy.capstone.tools import AnalystToolRegistry, build_certified_retriever
from finai_academy.capstone.views import CapstoneRunView, to_run_view


@dataclasses.dataclass(frozen=True, slots=True)
class StudentIntegrationIncomplete(Exception):
    seam: str
    hint: str
'''


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError("student integration module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _generated_adapter_source(overrides: Mapping[str, str] | None = None) -> str:
    replacements = dict(overrides or {})
    blocks: list[str] = []
    for reference in _REFERENCE_ADAPTERS:
        seam = reference.__name__.removeprefix("_reference_")
        source = inspect.getsource(reference).replace(
            f"def _reference_{seam}", f"def {seam}", 1
        )
        blocks.append(replacements.get(seam, source))
    return _ADAPTER_HEADER + "\n\n" + "\n\n".join(blocks)


def _copy_with_generated_adapter(
    destination: Path,
    *,
    overrides: Mapping[str, str] | None = None,
) -> Path:
    shutil.copytree(_STUDENT_DIR, destination)
    (destination / "integration.py").write_text(
        _generated_adapter_source(overrides), encoding="utf-8"
    )
    return destination


@pytest.fixture(scope="module")
def completed_student_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("completed-capstone") / "student"
    return _copy_with_generated_adapter(destination)


@pytest.fixture(scope="module")
def completed_module(completed_student_dir: Path) -> ModuleType:
    return _load_module(
        completed_student_dir / "integration.py",
        "completed_student_integration",
    )


@pytest.fixture(scope="module")
def recorded_result() -> ResearchRunResult:
    return build_reference_copilot(run_id_factory=lambda: "student-contract-run").run(
        ResearchRequest.reference()
    )


class _MissingSchneiderRetriever:
    def __init__(self) -> None:
        self._wrapped = build_certified_retriever()

    def search(
        self, company: str, query: str, top_k: int = 2
    ) -> tuple[CapstoneEvidenceHit, ...]:
        if company == "Schneider Electric":
            return ()
        return self._wrapped.search(company, query, top_k)


@pytest.fixture(scope="module")
def stopped_result() -> ResearchRunResult:
    return build_reference_copilot(retriever=_MissingSchneiderRetriever()).run(
        ResearchRequest.reference()
    )


def _document_hits() -> tuple[CapstoneEvidenceHit, ...]:
    retriever = build_certified_retriever()
    return (
        *retriever.search("NVIDIA", "operating growth", top_k=1),
        *retriever.search("Schneider Electric", "operating growth", top_k=1),
    )


class TestRetrieverContract:
    @pytest.mark.parametrize(
        ("company", "query", "expected_ids"),
        [
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
                (
                    "SU-FY2025-ENERGY-MANAGEMENT-002",
                    "SU-FY2025-ENERGY-MANAGEMENT-001",
                ),
            ),
            (
                "Schneider Electric",
                "adjusted EBITA margin",
                (
                    "SU-FY2025-ENERGY-MANAGEMENT-001",
                    "SU-FY2025-ENERGY-MANAGEMENT-002",
                ),
            ),
        ],
    )
    def test_returns_source_addressable_hits_inside_the_company_boundary(
        self,
        completed_module: ModuleType,
        company: str,
        query: str,
        expected_ids: tuple[str, ...],
    ) -> None:
        hits = completed_module.wire_retriever(company, query)

        assert isinstance(hits, tuple)
        assert all(isinstance(hit, CapstoneEvidenceHit) for hit in hits)
        assert {hit.company for hit in hits} == {company}
        assert tuple(hit.evidence_id for hit in hits) == expected_ids
        assert all(hit.source_reference and hit.document_id for hit in hits)


class TestCapabilitiesContract:
    @pytest.mark.parametrize(
        ("discovered", "expected"),
        [
            (
                ("place_order", "search_financial_documents", "get_company_metric"),
                ("get_company_metric", "search_financial_documents"),
            ),
            (("search_financial_documents", "place_order"), ("search_financial_documents",)),
            (("get_company_metric", "delete_portfolio"), ("get_company_metric",)),
            (("place_order", "place_order"), ()),
        ],
    )
    def test_keeps_only_discovered_certified_read_capabilities(
        self,
        completed_module: ModuleType,
        discovered: tuple[str, ...],
        expected: tuple[str, ...],
    ) -> None:
        capabilities = completed_module.register_analyst_capabilities(discovered)

        assert isinstance(capabilities, tuple)
        assert capabilities == expected


class TestEvidenceGateContract:
    def test_passes_with_reordered_document_evidence_for_both_companies(
        self, completed_module: ModuleType
    ) -> None:
        hits = tuple(reversed(_document_hits()))

        complete = completed_module.evaluate_student_evidence_gate(hits)

        assert isinstance(complete, EvidenceGateDecision)
        assert complete.passed
        assert complete.coverage == {
            "NVIDIA": ("document",),
            "Schneider Electric": ("document",),
        }
        assert complete.evidence_hits == hits

    @pytest.mark.parametrize(
        ("companies", "missing"),
        [
            (("NVIDIA",), ("Schneider Electric document evidence",)),
            (("Schneider Electric",), ("NVIDIA document evidence",)),
            (
                (),
                ("NVIDIA document evidence", "Schneider Electric document evidence"),
            ),
        ],
    )
    def test_blocks_each_missing_company_case(
        self,
        completed_module: ModuleType,
        companies: tuple[str, ...],
        missing: tuple[str, ...],
    ) -> None:
        selected = tuple(hit for hit in _document_hits() if hit.company in companies)

        decision = completed_module.evaluate_student_evidence_gate(selected)

        assert isinstance(decision, EvidenceGateDecision)
        assert not decision.passed
        assert decision.missing_requirements == missing
        assert decision.evidence_hits == selected


class TestPublicViewContract:
    @pytest.mark.parametrize("result_fixture", ["recorded_result", "stopped_result"])
    def test_uses_the_safe_public_view_boundary_for_completed_and_stopped_runs(
        self,
        completed_module: ModuleType,
        result_fixture: str,
        request: pytest.FixtureRequest,
    ) -> None:
        result = request.getfixturevalue(result_fixture)
        view = completed_module.assemble_public_briefing_view(result)
        encoded = view.model_dump_json()

        assert isinstance(view, CapstoneRunView)
        if result.status == "completed":
            assert view.release.decision == "Release passed"
            assert view.briefing is not None
        else:
            assert view.release.decision == "Release blocked"
            assert view.briefing is None
        assert '"arguments"' not in encoded
        assert "/Users/" not in encoded


def _starter_calls(recorded_result) -> dict[str, Callable[[], object]]:
    return {
        "wire_retriever": lambda: ("NVIDIA", "operating growth"),
        "register_analyst_capabilities": lambda: (
            "search_financial_documents",
            "get_company_metric",
        ),
        "evaluate_student_evidence_gate": _document_hits,
        "assemble_public_briefing_view": lambda: recorded_result,
    }


def test_starter_exposes_exactly_four_conceptual_incomplete_seams(recorded_result) -> None:
    module = _load_module(_INTEGRATION_PATH, "starter_student_integration")
    public_functions = {
        name
        for name, member in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith("_")
    }
    assert public_functions == set(_SEAM_NAMES)

    statuses = []
    for seam, argument_factory in _starter_calls(recorded_result).items():
        function = getattr(module, seam)
        arguments = argument_factory()
        if seam == "wire_retriever":
            arguments = tuple(arguments)
        else:
            arguments = (arguments,)
        with pytest.raises(module.StudentIntegrationIncomplete) as caught:
            function(*arguments)
        statuses.append(caught.value)

    assert tuple(status.seam for status in statuses) == _SEAM_NAMES
    assert all(status.hint and "(" not in status.hint for status in statuses)
    assert len({status.hint for status in statuses}) == 4


def _rendered_text(app: AppTest) -> str:
    return "\n".join(
        str(element.value)
        for element_type in (
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
        for element in app.get(element_type)
        if getattr(element, "value", None) is not None
    )


def test_starter_streamlit_app_launches_and_renders_all_four_statuses() -> None:
    app = AppTest.from_file(_STUDENT_DIR / "streamlit_app.py").run(timeout=20)
    text = _rendered_text(app)

    assert not app.exception
    assert "Student integration challenge" in text
    assert "Reference mission" in text
    assert "Reference output shape" in text
    assert "How to work" in text
    assert all(seam in text for seam in _SEAM_NAMES)
    assert text.count("Incomplete:") == 4


def test_student_app_sanitizes_generic_errors_and_rejects_wrong_types(tmp_path: Path) -> None:
    overrides = {
        "wire_retriever": '''def wire_retriever(company: str, query: str):
    raise RuntimeError("OPENAI_API_KEY=sk-secret at /Users/private")
''',
        "register_analyst_capabilities": '''def register_analyst_capabilities(discovered):
    return None
''',
        "evaluate_student_evidence_gate": '''def evaluate_student_evidence_gate(hits):
    return ()
''',
        "assemble_public_briefing_view": '''def assemble_public_briefing_view(result):
    return None
''',
    }
    student_dir = _copy_with_generated_adapter(tmp_path / "student", overrides=overrides)
    sys.modules.pop("integration", None)

    app = AppTest.from_file(student_dir / "streamlit_app.py").run(timeout=20)
    text = _rendered_text(app)

    assert not app.exception
    assert text.count("Error:") == 4
    assert "Ready" not in text
    assert "OPENAI_API_KEY" not in text
    assert "sk-secret" not in text
    assert "/Users/" not in text


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    python_path = [str(_PROJECT_ROOT / "src")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    for name in (
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "FINAI_MODEL_PROVIDER",
        "FINAI_CHAT_MODEL",
    ):
        environment.pop(name, None)
    return environment


def _run_verifier(student_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(student_dir / "verify.py")],
        cwd=_PROJECT_ROOT,
        env=_subprocess_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )


def _failure_lines(completed: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in completed.stdout.splitlines() if line.startswith("FAIL ")]


def test_starter_verifier_reports_only_the_four_incomplete_groups() -> None:
    completed = _run_verifier(_STUDENT_DIR)

    assert completed.returncode != 0
    assert len(_failure_lines(completed)) == 4
    assert all(seam in completed.stdout for seam in _SEAM_NAMES)
    assert "CAPSTONE_PASS" not in completed.stdout
    assert completed.stderr == ""


def test_completed_copy_passes_full_verifier_with_exactly_one_marker(
    completed_student_dir: Path,
) -> None:
    completed = _run_verifier(completed_student_dir)

    assert completed.returncode == 0, completed.stdout
    assert completed.stdout.splitlines().count("CAPSTONE_PASS") == 1
    assert completed.stderr == ""
    assert "/Users/" not in completed.stdout
    assert "OPENAI_API_KEY" not in completed.stdout


@pytest.mark.parametrize(
    "payload",
    [
        "CAPSTONE_PASS",
        "OPENAI_API_KEY=sk-malicious",
        "/Users/private/credentials.txt",
    ],
)
def test_verifier_captures_and_sanitizes_malicious_student_output(
    tmp_path: Path,
    payload: str,
) -> None:
    printing_retriever = f'''def wire_retriever(company: str, query: str):
    print({payload!r})
    print({payload!r}, file=sys.stderr)
    return build_certified_retriever().search(company, query)
'''
    student_dir = _copy_with_generated_adapter(
        tmp_path / "student",
        overrides={"wire_retriever": printing_retriever},
    )

    completed = _run_verifier(student_dir)

    assert completed.returncode != 0
    assert _failure_lines(completed) == [
        "FAIL wire_retriever: student output is not allowed."
    ]
    assert completed.stdout.splitlines().count("CAPSTONE_PASS") == 0
    assert "OPENAI_API_KEY" not in completed.stdout
    assert "sk-malicious" not in completed.stdout
    assert "/Users/" not in completed.stdout
    assert completed.stderr == ""


@pytest.mark.parametrize(
    ("seam", "canned_source"),
    [
        (
            "wire_retriever",
            '''def wire_retriever(company: str, query: str):
    return build_certified_retriever().search("NVIDIA", "operating growth")
''',
        ),
        (
            "register_analyst_capabilities",
            '''def register_analyst_capabilities(discovered):
    return ("get_company_metric", "search_financial_documents")
''',
        ),
        (
            "evaluate_student_evidence_gate",
            '''def evaluate_student_evidence_gate(hits):
    nvidia_only = bool(hits) and all(hit.company == "NVIDIA" for hit in hits)
    missing = ("Schneider Electric document evidence",) if nvidia_only else ()
    return EvidenceGateDecision(
        passed=not missing,
        coverage={
            "NVIDIA": ("document",),
            "Schneider Electric": (() if nvidia_only else ("document",)),
        },
        missing_requirements=missing,
        evidence_hits=tuple(hits),
    )
''',
        ),
    ],
)
def test_verifier_rejects_single_case_canned_answers(
    tmp_path: Path,
    seam: str,
    canned_source: str,
) -> None:
    student_dir = _copy_with_generated_adapter(
        tmp_path / "student",
        overrides={seam: canned_source},
    )

    completed = _run_verifier(student_dir)

    assert completed.returncode != 0
    assert _failure_lines(completed) == [f"FAIL {seam}: contract did not complete."]
    assert completed.stdout.splitlines().count("CAPSTONE_PASS") == 0
    assert completed.stderr == ""


def test_near_solved_copy_names_only_the_stopped_view_failure(tmp_path: Path) -> None:
    canned_view = '''def assemble_public_briefing_view(result):
    completed = build_reference_copilot().run(ResearchRequest.reference())
    return to_run_view(completed)
'''
    student_dir = _copy_with_generated_adapter(
        tmp_path / "student",
        overrides={"assemble_public_briefing_view": canned_view},
    )

    completed = _run_verifier(student_dir)

    assert completed.returncode != 0
    assert _failure_lines(completed) == [
        "FAIL assemble_public_briefing_view: contract did not complete."
    ]
    assert completed.stdout.splitlines().count("CAPSTONE_PASS") == 0
    assert completed.stderr == ""


def test_integrated_path_uses_student_retrieval_instead_of_reference_release_only(
    tmp_path: Path,
) -> None:
    integration_sensitive_retriever = '''def wire_retriever(company: str, query: str):
    if query == "reference mission operating growth":
        return ()
    return build_certified_retriever().search(company, query)
'''
    student_dir = _copy_with_generated_adapter(
        tmp_path / "student",
        overrides={"wire_retriever": integration_sensitive_retriever},
    )

    completed = _run_verifier(student_dir)

    assert completed.returncode != 0
    assert _failure_lines(completed) == [
        "FAIL wire_retriever: integrated contract did not complete."
    ]
    assert completed.stdout.splitlines().count("CAPSTONE_PASS") == 0
    assert completed.stderr == ""
