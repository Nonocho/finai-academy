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
CAPSTONE_DOCUMENTS = (
    CAPSTONE / "README.md",
    CAPSTONE / "STUDENT_BRIEF.md",
    CAPSTONE / "INSTRUCTOR_GUIDE.md",
    CAPSTONE / "reference" / "README.md",
    CAPSTONE / "student" / "README.md",
    CAPSTONE / "student" / "CHECKLIST.md",
)
DOCUMENT_COMMANDS = {
    "README.md": COMMANDS,
    "STUDENT_BRIEF.md": COMMANDS,
    "INSTRUCTOR_GUIDE.md": COMMANDS,
    "reference/README.md": COMMANDS[:2],
    "student/README.md": (COMMANDS[0], *COMMANDS[2:]),
    "student/CHECKLIST.md": (COMMANDS[0], *COMMANDS[2:]),
}
TIMED_DOCUMENTS = (
    "STUDENT_BRIEF.md",
    "INSTRUCTOR_GUIDE.md",
    "student/CHECKLIST.md",
)
TIMETABLE = (
    "15:30–15:40",
    "15:40–16:10",
    "16:10–16:25",
    "16:25–16:30",
    "16:30–17:00",
)
OBSOLETE_COMMANDS = (
    ".venv/bin/streamlit run",
    ".venv/bin/python final-project/student/verify.py",
    "uv run streamlit run final-project/app.py",
    "uv run python final-project/app.py",
)


def _read(relative_path: str) -> str:
    return (CAPSTONE / relative_path).read_text(encoding="utf-8")


def test_relevant_documents_use_the_final_commands_and_timetable() -> None:
    for relative_path, commands in DOCUMENT_COMMANDS.items():
        text = _read(relative_path)
        for command in commands:
            assert command in text
    for relative_path in TIMED_DOCUMENTS:
        text = _read(relative_path)
        for time in TIMETABLE:
            assert time in text


def test_terminal_workflow_does_not_treat_streamlit_as_a_completed_command() -> None:
    for relative_path in (
        "README.md",
        "STUDENT_BRIEF.md",
        "INSTRUCTOR_GUIDE.md",
        "reference/README.md",
    ):
        text = _read(relative_path)
        assert "Terminal 1" in text
        assert "Terminal 2" in text
        assert "Terminal 3" in text
        assert "stays running" in text
        assert "Ctrl+C" in text
    for relative_path in ("student/README.md", "student/CHECKLIST.md"):
        text = _read(relative_path)
        assert "Terminal 2" in text
        assert "Terminal 3" in text
        assert "stays running" in text
        assert "Ctrl+C" in text


def test_capstone_documents_reject_obsolete_commands() -> None:
    for document in CAPSTONE_DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        for command in OBSOLETE_COMMANDS:
            assert command not in text


def test_canonical_readme_has_the_classroom_start_and_route_boundaries() -> None:
    text = _read("README.md")

    for phrase in (
        "Recorded demo",
        "Certified snapshots",
        "Reference mission",
        "Ask the analyst",
        "Ollama",
        "OpenAI",
        "Tavily",
        "not composed into this capstone Streamlit route",
        ".env.example",
        "OPENAI_API_KEY",
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
    for phrase in (
        "CAPSTONE_PASS",
        "individual",
        "pair",
        "diagnostic",
        "60-minute",
        "30-minute",
        "credential",
        "path-free",
        "diagnose.py run",
        "diagnose.py inspect",
        "diagnostic_case.json",
        "MLFLOW_RUN_ID",
        "evidence_gate",
    ):
        assert phrase in text
    assert "return build_certified_retriever().search" not in text
    assert "return AnalystToolRegistry(discovered=discovered).discover" not in text
    assert "return to_run_view(result)" not in text


def test_instructor_guide_covers_facilitation_correction_and_recovery() -> None:
    text = _read("INSTRUCTOR_GUIDE.md")

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
        "student_diagnostic_solution.json",
        "DIAGNOSTIC_STATUS=completed",
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
    for document in CAPSTONE_DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        assert "—" not in text
        assert not re.search(r"(?:OPENAI_API_KEY=sk-|TAVILY_API_KEY=tvly-)", text)
