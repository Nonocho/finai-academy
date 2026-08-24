from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPSTONE = ROOT / "final-project"
COMMANDS = (
    "uv sync --extra capstone --extra ai",
    "uv run streamlit run final-project/reference/streamlit_app.py",
    "uv run streamlit run final-project/student/streamlit_app.py",
    "uv run python final-project/student/verify.py",
)
MISSION = (
    "Compare NVIDIA and Schneider Electric using official documents and selected "
    "financial metrics. Identify the main operating-growth evidence, explain why "
    "direct comparison is limited, and cite every factual claim."
)
USER_FACING_DOCUMENTS = (
    CAPSTONE / "README.md",
    CAPSTONE / "STUDENT_BRIEF.md",
    CAPSTONE / "reference" / "README.md",
    CAPSTONE / "student" / "README.md",
    CAPSTONE / "student" / "CHECKLIST.md",
)


def _read(relative_path: str) -> str:
    return (CAPSTONE / relative_path).read_text(encoding="utf-8")


def test_canonical_readme_has_the_classroom_start_and_route_boundaries() -> None:
    text = _read("README.md")

    for command in COMMANDS:
        assert command in text
    for phrase in (
        "Recorded demo",
        "Certified snapshots",
        "Optional live enrichment",
        "Reference mission",
        "Ask the analyst",
        "Ollama",
        "OpenAI",
        "Tavily",
        ".env.example",
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "FINAI_MODEL_PROVIDER",
        "qwen3:4b",
        "macOS",
        "Windows PowerShell",
        "research support",
        "not investment advice",
        "First Finance - Arnaud Demes",
    ):
        assert phrase in text


def test_student_handout_defines_the_fixed_challenge_without_solution_bodies() -> None:
    text = _read("STUDENT_BRIEF.md")

    assert MISSION in text
    for seam in (
        "wire_retriever",
        "register_analyst_capabilities",
        "evaluate_student_evidence_gate",
        "assemble_public_briefing_view",
    ):
        assert seam in text
    for command in COMMANDS:
        assert command in text
    for phrase in (
        "CAPSTONE_PASS",
        "individual",
        "pair",
        "diagnostic",
        "15:30–15:40",
        "15:40–16:10",
        "16:10–16:25",
        "16:25–16:30",
        "16:30–17:00",
        "60-minute",
        "30-minute",
        "credential",
        "path-free",
    ):
        assert phrase in text
    assert "return build_certified_retriever().search" not in text
    assert "return AnalystToolRegistry(discovered=discovered).discover" not in text
    assert "return to_run_view(result)" not in text


def test_instructor_guide_covers_facilitation_correction_and_recovery() -> None:
    text = _read("INSTRUCTOR_GUIDE.md")

    for command in COMMANDS:
        assert command in text
    for phrase in (
        "Prerequisites and preflight",
        "Expected reference output",
        "Pair rotation",
        "Progressive hints",
        "Correction",
        "wire_retriever",
        "register_analyst_capabilities",
        "evaluate_student_evidence_gate",
        "assemble_public_briefing_view",
        "MLflow",
        "trace",
        "failure owner",
        "recorded fallback",
        "Windows recovery",
        "macOS recovery",
        "Skip-if-late route",
        "Reset procedure",
        "does not delete or overwrite learner work",
        "production non-goals",
    ):
        assert phrase in text
    assert "return build_certified_retriever().search(company, query)" in text
    assert "return AnalystToolRegistry(discovered=discovered).discover()" in text
    assert "return to_run_view(result)" in text


def test_supporting_readmes_keep_the_certified_offline_student_route() -> None:
    reference = _read("reference/README.md")
    student = _read("student/README.md")
    checklist = _read("student/CHECKLIST.md")

    assert "Recorded demo" in reference
    assert "Certified snapshots" in reference
    assert "qwen3:4b" in reference
    for document in (student, checklist):
        assert "uv run streamlit run final-project/student/streamlit_app.py" in document
        assert "uv run python final-project/student/verify.py" in document
        assert "CAPSTONE_PASS" in document
        assert "offline" in document.casefold()


def test_course_copy_has_no_secrets_or_em_dash() -> None:
    for document in USER_FACING_DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        assert "—" not in text
        assert not re.search(r"(?:OPENAI_API_KEY=sk-|TAVILY_API_KEY=tvly-)", text)
