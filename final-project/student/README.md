# Financial Analyst Copilot student challenge

Complete four bounded integration seams in a certified offline-first analyst workflow. The starter launches before the seams are complete and reports named, conceptual diagnostics without requiring API credentials, network access, Tavily, or Ollama.

## Start and verify

From the repository root:

```bash
uv sync --extra capstone --extra ai
uv run streamlit run final-project/student/streamlit_app.py
uv run python final-project/student/verify.py
```

The starter exits nonzero with one diagnostic per unfinished seam. A complete integration prints diagnostics followed by one standalone `CAPSTONE_PASS` line. Do not print that marker from your function bodies.

## Scope

Edit only the four function bodies in `integration.py`:

1. `wire_retriever`
2. `register_analyst_capabilities`
3. `evaluate_student_evidence_gate`
4. `assemble_public_briefing_view`

The recorded mission uses repository fixtures only. Keep output path-free and credential-free. Do not add shell commands, provider fallback, trading actions, or learner-controlled process execution.

Read [../STUDENT_BRIEF.md](../STUDENT_BRIEF.md) for the mission, timetable, public pass rule, pair guidance, and demonstration requirements. Use [CHECKLIST.md](CHECKLIST.md) to keep the challenge within 60 minutes.

This application provides research support, not investment advice.

First Finance - Arnaud Demes
