from __future__ import annotations

import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from streamlit.testing.v1 import AppTest

from finai_academy.capstone import ResearchRequest, build_reference_copilot
from finai_academy.capstone.models import CapstoneEvidenceHit
from finai_academy.capstone.tools import build_certified_retriever

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STUDENT_DIR = _PROJECT_ROOT / "final-project" / "student"
_INTEGRATION_PATH = _STUDENT_DIR / "integration.py"
_SEAM_NAMES = (
    "wire_retriever",
    "register_analyst_capabilities",
    "evaluate_student_evidence_gate",
    "assemble_public_briefing_view",
)
_COMPLETED_BODIES = {
    "    raise StudentIntegrationIncomplete(\n"
    '        seam="wire_retriever",\n'
    '        hint="Use the certified retriever and preserve the company boundary.",\n'
    "    )\n": (
        "    from finai_academy.capstone.tools import build_certified_retriever\n\n"
        "    return build_certified_retriever().search(company, query)\n"
    ),
    "    raise StudentIntegrationIncomplete(\n"
    '        seam="register_analyst_capabilities",\n'
    '        hint="Intersect discovered tools with the certified analyst allowlist.",\n'
    "    )\n": (
        "    from finai_academy.capstone.tools import AnalystToolRegistry\n\n"
        "    return AnalystToolRegistry(discovered=discovered).discover()\n"
    ),
    "    raise StudentIntegrationIncomplete(\n"
    '        seam="evaluate_student_evidence_gate",\n'
    '        hint="Require document evidence for both companies before release.",\n'
    "    )\n": (
        '    companies = ("NVIDIA", "Schneider Electric")\n'
        "    covered_companies = {hit.company for hit in hits}\n"
        "    coverage = {\n"
        '        company: (("document",) if company in covered_companies else ())\n'
        "        for company in companies\n"
        "    }\n"
        "    missing = tuple(\n"
        '        f"{company} document evidence"\n'
        "        for company in companies\n"
        "        if company not in covered_companies\n"
        "    )\n"
        "    return EvidenceGateDecision(\n"
        "        passed=not missing,\n"
        "        coverage=coverage,\n"
        "        missing_requirements=missing,\n"
        "        evidence_hits=tuple(hits),\n"
        "    )\n"
    ),
    "    raise StudentIntegrationIncomplete(\n"
    '        seam="assemble_public_briefing_view",\n'
    '        hint="Convert the run through the public presentation boundary.",\n'
    "    )\n": (
        "    from finai_academy.capstone.views import to_run_view\n\n"
        "    return to_run_view(result)\n"
    ),
}


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError("student integration module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _apply_completed_bodies(student_dir: Path) -> None:
    path = student_dir / "integration.py"
    source = path.read_text(encoding="utf-8")
    for incomplete, completed in _COMPLETED_BODIES.items():
        if incomplete in source:
            assert source.count(incomplete) == 1
            source = source.replace(incomplete, completed)
        else:
            assert source.count(completed) == 1
    path.write_text(source, encoding="utf-8")


@pytest.fixture(scope="module")
def completed_student_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    destination = tmp_path_factory.mktemp("completed-capstone") / "student"
    shutil.copytree(_STUDENT_DIR, destination)
    _apply_completed_bodies(destination)
    return destination


@pytest.fixture(scope="module")
def completed_module(completed_student_dir: Path) -> ModuleType:
    return _load_module(
        completed_student_dir / "integration.py",
        "completed_student_integration",
    )


@pytest.fixture(scope="module")
def recorded_result():
    return build_reference_copilot(run_id_factory=lambda: "student-contract-run").run(
        ResearchRequest.reference()
    )


def _document_hits() -> tuple[CapstoneEvidenceHit, ...]:
    retriever = build_certified_retriever()
    return (
        *retriever.search("NVIDIA", "operating growth", top_k=1),
        *retriever.search("Schneider Electric", "operating growth", top_k=1),
    )


class TestRetrieverContract:
    def test_returns_source_addressable_hits_inside_the_company_boundary(
        self, completed_module: ModuleType
    ) -> None:
        hits = completed_module.wire_retriever("NVIDIA", "operating growth")

        assert isinstance(hits, tuple)
        assert hits
        assert {hit.company for hit in hits} == {"NVIDIA"}
        assert all(hit.evidence_id and hit.source_reference for hit in hits)


class TestCapabilitiesContract:
    def test_keeps_only_discovered_certified_read_capabilities(
        self, completed_module: ModuleType
    ) -> None:
        capabilities = completed_module.register_analyst_capabilities(
            ("place_order", "search_financial_documents", "get_company_metric")
        )

        assert capabilities == ("get_company_metric", "search_financial_documents")


class TestEvidenceGateContract:
    def test_passes_only_with_document_evidence_for_both_companies(
        self, completed_module: ModuleType
    ) -> None:
        hits = _document_hits()

        complete = completed_module.evaluate_student_evidence_gate(hits)
        incomplete = completed_module.evaluate_student_evidence_gate(
            tuple(hit for hit in hits if hit.company == "NVIDIA")
        )

        assert complete.passed
        assert complete.coverage == {
            "NVIDIA": ("document",),
            "Schneider Electric": ("document",),
        }
        assert complete.evidence_hits == hits
        assert not incomplete.passed
        assert incomplete.missing_requirements == (
            "Schneider Electric document evidence",
        )


class TestPublicViewContract:
    def test_uses_the_safe_public_view_boundary(
        self, completed_module: ModuleType, recorded_result
    ) -> None:
        view = completed_module.assemble_public_briefing_view(recorded_result)
        encoded = view.model_dump_json()

        assert view.release.decision == "Release passed"
        assert view.briefing is not None
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


def test_starter_verifier_reports_only_the_four_incomplete_groups() -> None:
    completed = _run_verifier(_STUDENT_DIR)

    failure_lines = [line for line in completed.stdout.splitlines() if line.startswith("FAIL ")]
    assert completed.returncode != 0
    assert len(failure_lines) == 4
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
